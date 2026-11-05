//! Aegist's second language core on the GPU (arch "hybrid-moe" with hc_count > 1), a port of the
//! lanes model in aegist/hybrid.py + aegist/hc.py:
//!
//!   x: hc lanes of d_model (the embedding repeated)
//!   x += ple(x)                       per-layer n-gram embedding (some blocks)
//!   h, inj = hc_attn(x);  x += mixer(h) * inj      gated DeltaNet or gated attention (+ QSA indexer)
//!   h, inj = hc_mlp(x);   x += (moe(h) + cores(h, engine(h))) * inj
//!   logits = lm_head(hc_out(x))
//!
//! Weights are kept as released: bf16 linears and norms in VRAM, FP8 experts (128 x 128 bf16 block
//! scales) paged from the mapped .nx through the same LRU pool as the first core, and the FP8
//! n-gram table read row by row from the file (16 rows of 160 bytes per token).
//!
//! The QSA indexer keeps each attention block's raw indexer keys and its pooled block keys. Below
//! idx_budget tokens of context every block is visible (plain causal attention); beyond, each query
//! attends to its idx_budget / idx_ratio best blocks plus its own incomplete block.
//!
//! On several GPUs each takes a run of consecutive layers (weights, caches, recurrent state and
//! an expert pool of its own); the lanes go from one GPU to the next once per pass, ~40 KB per
//! token. More GPUs means more experts already in VRAM, which is what decode speed waits on.
use crate::cuda::{Arg, Buf, Gpu, Kernel, Result};
use crate::hybrid::{env_usize, Pool, READ_THREADS};
use crate::model::{bytes_i32, upload, Lm, PTX};
use crate::nx::NxFile;
use crate::reader::{Prefetcher, Reader};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct NConfig {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub attn: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub hk: usize,
    pub hv: usize,
    pub dk: usize,
    pub dv: usize,
    pub conv_k: usize,
    pub experts: usize,
    pub top_k: usize,
    pub expert_hidden: usize,
    pub shared_hidden: usize,
    pub eps: f32,
    pub n1: usize,
    pub n2: usize,
    pub rank: usize,
    pub cores_active: usize,
    pub hc: usize,
    pub hc_rank: usize,
    pub ple_layers: Vec<usize>,
    pub ple_dim: usize,
    pub ple_conv_k: usize,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub ngram_base: u64,
    pub ngram_seed: u64,
    pub eos: u32,
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_budget: usize,
    pub idx_ratio: usize,
    pub sigmoid_gate: bool,
}

/// True for brains this engine runs (hyper-connection lanes).
pub fn is_next(c: &Value) -> bool {
    c["hc_count"].as_u64().map_or(false, |v| v > 1)
}

impl NConfig {
    pub fn from_header(c: &Value) -> Result<NConfig> {
        let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or(format!("config is missing {k}"));
        let f = |k: &str| c[k].as_f64().map(|v| v as f32).ok_or(format!("config is missing {k}"));
        let kinds = c["layer_types"].as_array().ok_or("config is missing layer_types")?;
        let grid = c["core_grid"].as_array().ok_or("config is missing core_grid")?;
        let ple_layers = c["ple_layers"].as_array().ok_or("config is missing ple_layers")?;
        if c["tie_embeddings"].as_bool() == Some(true) {
            return Err("this engine needs an untied output head".into());
        }
        if c["expert_format"].as_str() != Some("fp8") {
            return Err(format!("this engine needs FP8 experts, the brain has {}", c["expert_format"]));
        }
        let cfg = NConfig {
            vocab: u("vocab_size")?,
            d: u("d_model")?,
            layers: u("n_layers")?,
            attn: kinds.iter().map(|k| k.as_str() == Some("attn")).collect(),
            heads: u("n_heads")?,
            kv_heads: u("n_kv_heads")?,
            head_dim: u("head_dim")?,
            rotary_dim: u("rotary_dim")?,
            rope_theta: f("rope_theta")?,
            hk: u("lin_k_heads")?,
            hv: u("lin_v_heads")?,
            dk: u("lin_k_dim")?,
            dv: u("lin_v_dim")?,
            conv_k: u("conv_kernel")?,
            experts: u("n_experts")?,
            top_k: u("experts_active")?,
            expert_hidden: u("expert_hidden")?,
            shared_hidden: u("shared_hidden")?,
            eps: f("norm_eps")?,
            n1: grid[0].as_u64().ok_or("bad core_grid")? as usize,
            n2: grid[1].as_u64().ok_or("bad core_grid")? as usize,
            rank: u("core_rank")?,
            cores_active: u("cores_active")?,
            hc: u("hc_count")?,
            hc_rank: u("hc_lowrank")?,
            ple_layers: ple_layers.iter().map(|v| v.as_u64().unwrap_or(u64::MAX) as usize).collect(),
            ple_dim: u("ple_dim")?,
            ple_conv_k: u("ple_conv_kernel")?,
            ngram: u("ngram_size")?,
            heads_per_ngram: u("heads_per_ngram")?,
            ngram_base: u("ngram_vocab_base")? as u64,
            ngram_seed: u("ngram_seed")? as u64,
            eos: u("eos_id")? as u32,
            idx_heads: u("idx_heads")?,
            idx_dim: u("idx_head_dim")?,
            idx_budget: u("idx_budget")?,
            idx_ratio: u("idx_ratio")?,
            sigmoid_gate: c["out_gate"].as_str() == Some("sigmoid"),
        };
        cfg.check()?;
        Ok(cfg)
    }

    /// The shapes these kernels are written for, with clear errors otherwise.
    fn check(&self) -> Result<()> {
        let mut bad = vec![];
        if self.attn.len() != self.layers {
            bad.push("layer_types length".to_string());
        }
        if self.head_dim != 256 || self.rotary_dim > 128 || self.rotary_dim % 2 != 0 || self.heads % self.kv_heads != 0 {
            bad.push(format!("attention head_dim {} rotary {} heads {}/{}", self.head_dim, self.rotary_dim, self.heads, self.kv_heads));
        }
        if self.dk != 128 || self.dv != 128 || self.hv % self.hk != 0 || self.conv_k != 4 {
            bad.push(format!("DeltaNet dims {}x{} heads {}/{} conv {}", self.dk, self.dv, self.hk, self.hv, self.conv_k));
        }
        if self.experts > 1024 || self.top_k > 16 || self.expert_hidden % 128 != 0 || self.d % 128 != 0 {
            bad.push(format!("experts {} top-{} hidden {} d_model {}", self.experts, self.top_k, self.expert_hidden, self.d));
        }
        if self.shared_hidden % 32 != 0 || self.hc_rank % 32 != 0 || self.hc < 2 || self.hc > 16 {
            bad.push(format!("shared {} / lanes {} rank {}", self.shared_hidden, self.hc, self.hc_rank));
        }
        if !self.ple_layers.is_empty() && (self.ngram != 3 || self.ple_conv_k != 4 || self.ple_dim % 32 != 0
            || self.ple_layers.iter().any(|&l| l >= self.layers) || self.ple_dim % ((self.ngram - 1) * self.heads_per_ngram) != 0)
        {
            bad.push(format!("PLE n-gram {} conv {} dim {}", self.ngram, self.ple_conv_k, self.ple_dim));
        }
        if self.idx_heads > 0 && (self.idx_dim != 128 || self.idx_ratio == 0 || self.idx_budget < self.idx_ratio || self.rotary_dim > self.idx_dim) {
            bad.push(format!("indexer heads {} dim {} budget {} ratio {}", self.idx_heads, self.idx_dim, self.idx_budget, self.idx_ratio));
        }
        if self.n1 > 128 || self.n2 > 64 || self.rank > 64 || self.cores_active > 8 || self.cores_active > self.n1 * self.n2 {
            bad.push("core grid too large (rows <= 128; columns, rank <= 64; active <= 8)".into());
        }
        if bad.is_empty() { Ok(()) } else { Err(format!("this engine build does not support this brain's shape: {}", bad.join("; "))) }
    }

    fn n_cores(&self) -> usize {
        self.n1 * self.n2
    }
    fn lanes(&self) -> usize {
        self.hc * self.d
    }
    fn conv_dim(&self) -> usize {
        2 * self.hk * self.dk + self.hv * self.dv
    }
    fn n_attn(&self) -> usize {
        self.attn.iter().filter(|&&a| a).count()
    }
    fn ple_heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }
    fn ple_row(&self) -> usize {
        self.ple_dim / self.ple_heads()
    }
    fn idx_top(&self) -> usize {
        self.idx_budget / self.idx_ratio
    }
}

/// Byte offsets inside one FP8 expert (aegist/hc.py expert_layout_fp8).
#[derive(Clone, Copy, Debug)]
struct ExpertLayout {
    dn: usize,
    gu_s: usize,
    dn_s: usize,
    bytes: usize,
}

impl ExpertLayout {
    fn new(c: &NConfig) -> ExpertLayout {
        let (i, d) = (c.expert_hidden, c.d);
        let b = |n: usize| (n + 127) / 128;
        let dn = 2 * i * d;
        let gu_s = dn + d * i;
        let dn_s = gu_s + b(2 * i) * b(d) * 2;
        let end = dn_s + b(d) * b(i) * 2;
        ExpertLayout { dn, gu_s, dn_s, bytes: (end + 63) / 64 * 64 }
    }
}

/// Byte offsets inside one 4-bit copy of an expert ($AEGIST_EXPERT_BITS=4; kernels/next.cu
/// q4_from_f8): gate_up codes at 0, then its scales and mins, the down codes, its scales and mins.
#[derive(Clone, Copy, Debug)]
struct Q4Layout {
    s4g: usize,
    m4g: usize,
    q4d: usize,
    s4d: usize,
    m4d: usize,
    bytes: usize,
}

impl Q4Layout {
    fn new(c: &NConfig) -> Q4Layout {
        let (i, d) = (c.expert_hidden, c.d);
        let s4g = 2 * i * d / 2;
        let m4g = s4g + 2 * i * (d / 64) * 2;
        let q4d = m4g + 2 * i * (d / 64) * 2;
        let s4d = q4d + d * i / 2;
        let m4d = s4d + d * (i / 64) * 2;
        Q4Layout { s4g, m4g, q4d, s4d, m4d, bytes: (m4d + d * (i / 64) * 2 + 63) / 64 * 64 }
    }
}

/// The 4-bit expert file: every expert's 4-bit copy, exactly what q4_from_f8 makes of its FP8
/// bytes, so a miss reads 2.8 MB instead of 4.9 MB and needs no packing. `aegist q4-file` makes it
/// once (it is a cache of the brain's own experts, not new weights); it sits next to the base
/// brain as <name>.experts-q4 and is used only when its header matches the brain's base hash and
/// sizes. $AEGIST_Q4_FILE=off ignores it, $AEGIST_Q4_FILE=PATH puts it elsewhere.
const Q4_MAGIC: &[u8; 8] = b"AEGQ4X1\n";
const Q4_HEAD: usize = 4096;

/// Background readers warming the next layer's likely experts (two already fill D:'s bandwidth).
const PREFETCH_THREADS: usize = 2;
/// Prefetch only when a step feeds at most this many tokens (decoding); a prompt's chunks touch
/// most experts anyway.
const PREFETCH_TOKENS: usize = 1;
/// Read ahead only each token's this many heaviest guesses ($AEGIST_PREFETCH_TOP): the lighter
/// guesses are wrong more often, and a wrong one costs a read the disk had no time for.
const PREFETCH_TOP: usize = 2;
/// GB of the hot list's experts read into the OS file cache while idle ($AEGIST_WARM_GB; 0: none).
/// Windows gives that memory back to any app that needs it.
const WARM_GB: f64 = 12.0;
/// The hot list is saved every this many steps (and when the engine closes).
const HOT_SAVE_STEPS: usize = 64;

/// Queue the most used experts, up to $AEGIST_WARM_GB, to be read into the OS file cache while idle.
fn warm_hottest(qf: &Q4File, uses: &[u16]) {
    let Some(pre) = &qf.pre else { return };
    let gb = std::env::var("AEGIST_WARM_GB").ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(WARM_GB);
    let n = (gb.max(0.0) * 1e9 / qf.slot as f64) as usize;
    let mut order: Vec<usize> = (0..uses.len()).filter(|&k| uses[k] > 0).collect();
    order.sort_by(|&a, &b| uses[b].cmp(&uses[a]));
    order.truncate(n);
    pre.warm(order.into_iter().map(|k| (Q4_HEAD + k * qf.slot) as u64));
}

struct Q4File {
    reader: Reader,
    /// warms the OS file cache with the next layer's likely experts ($AEGIST_PREFETCH=off: none)
    pre: Option<Prefetcher>,
    /// the hot list: each expert's use count when last saved (u16 LE by layer * experts + expert)
    hot: PathBuf,
    /// bytes per expert (Q4Layout::bytes)
    slot: usize,
    experts: usize,
}

fn q4_file_path(nx: &NxFile) -> Option<PathBuf> {
    match std::env::var("AEGIST_Q4_FILE") {
        Ok(v) if ["off", "0", "no", "false"].contains(&v.trim().to_lowercase().as_str()) => None,
        Ok(v) if !v.trim().is_empty() => Some(PathBuf::from(v.trim())),
        _ => Some(Path::new(&nx.path).with_extension("experts-q4")),
    }
}

/// What the file's header must say for this brain: the base weights' hash and the sizes.
fn q4_file_head(nx: &NxFile, cfg: &NConfig, q: &Q4Layout) -> Option<Value> {
    use sha2::{Digest, Sha256};
    let base = &nx.header["sha256"]["base"];
    if base.is_null() {
        return None;
    }
    let key: String = Sha256::digest(base.to_string().as_bytes())[..12].iter().map(|b| format!("{b:02x}")).collect();
    Some(serde_json::json!({"key": key, "layers": cfg.layers, "experts": cfg.experts, "d": cfg.d,
                            "expert_hidden": cfg.expert_hidden, "slot": q.bytes}))
}

/// The 4-bit expert file for this brain, if there is one and it matches.
fn open_q4_file(nx: &NxFile, cfg: &NConfig, q: &Q4Layout) -> Option<Q4File> {
    let (path, want) = (q4_file_path(nx)?, q4_file_head(nx, cfg, q)?);
    let mut head = vec![0u8; Q4_HEAD];
    let mut f = std::fs::File::open(&path).ok()?;
    std::io::Read::read_exact(&mut f, &mut head).ok()?;
    if &head[..8] != Q4_MAGIC {
        return None;
    }
    let end = head[8..].iter().position(|&b| b == 0).map_or(Q4_HEAD, |k| k + 8);
    let got: Value = serde_json::from_slice(&head[8..end]).ok()?;
    let size = f.metadata().ok()?.len() as usize;
    if got != want || size < Q4_HEAD + cfg.layers * cfg.experts * q.bytes {
        eprintln!("{} does not match this brain: not used (`aegist q4-file` remakes it)", path.display());
        return None;
    }
    let reader = Reader::open(&path.to_string_lossy(), READ_THREADS).ok()?;
    let pre = match std::env::var("AEGIST_PREFETCH").ok().as_deref().map(str::trim) {
        Some("off") | Some("0") => None,
        _ => Prefetcher::open(&path.to_string_lossy(), PREFETCH_THREADS, q.bytes).ok(),
    };
    Some(Q4File { reader, pre, hot: path.with_extension("experts-hot"), slot: q.bytes, experts: cfg.experts })
}

/// The RAM tier: an expert pushed out of VRAM keeps its VRAM bytes (its 4-bit copy, or FP8) in
/// host memory, up to $AEGIST_RAM_TIER_GB (default RAM_TIER_GB; 0 turns it off), so its next use
/// copies them back (well under a millisecond) instead of reading the disk (about 9 ms per FP8
/// expert from D:). The bytes are the same, so the results are too. It holds only experts not in
/// VRAM (one taken back is freed here); when full, the one stored longest ago goes. It grows as
/// experts are pushed out and stops growing if memory runs out. Off by default: with the 4-bit
/// expert file, Windows' file cache already keeps 4-bit copies in RAM, and on 60 scored tokens
/// with 1,500 VRAM slots a 4 or 8 GB tier read 21-32% fewer experts from disk but took 113-120 s
/// against 81-96 s without it (the copies out of VRAM wait for the GPU; the results matched).
const RAM_TIER_GB: f64 = 0.0;

struct RamTier {
    size: usize,
    cap: usize,
    data: Vec<Vec<u8>>,
    owner: Vec<i32>,
    stamp: Vec<u64>,
    slot_of: Vec<i32>,
    free: Vec<usize>,
    clock: u64,
    hits: u64,
}

impl RamTier {
    fn from_env(size: usize, keys: usize) -> Option<RamTier> {
        let gb = std::env::var("AEGIST_RAM_TIER_GB").ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(RAM_TIER_GB);
        let cap = (gb.max(0.0) * 1e9 / size.max(1) as f64) as usize;
        (cap > 0).then(|| RamTier { size, cap, data: vec![], owner: vec![], stamp: vec![], slot_of: vec![-1; keys], free: vec![], clock: 0, hits: 0 })
    }

    fn has(&self, key: usize) -> bool {
        self.slot_of[key] >= 0
    }

    /// A slot to store into: a freed one, a new one, else the one stored longest ago.
    fn slot(&mut self) -> Option<usize> {
        if let Some(s) = self.free.pop() {
            return Some(s);
        }
        if self.data.len() < self.cap {
            let mut v = Vec::new();
            if v.try_reserve_exact(self.size).is_ok() {
                v.resize(self.size, 0);
                self.data.push(v);
                self.owner.push(-1);
                self.stamp.push(0);
                return Some(self.data.len() - 1);
            }
            self.cap = self.data.len();
        }
        let s = (0..self.data.len()).min_by_key(|&s| self.stamp[s])?;
        if self.owner[s] >= 0 {
            self.slot_of[self.owner[s] as usize] = -1;
            self.owner[s] = -1;
        }
        Some(s)
    }

    /// Keep the experts `pool`'s last assign pushed out, then copy in from here the `loads` held
    /// here (keys base + expert). Returns the loads left for the disk.
    fn swap(&mut self, base: usize, pool: &Pool, loads: &[(usize, usize)]) -> Result<Vec<(usize, usize)>> {
        for &(vslot, key) in &pool.evicted {
            if self.has(key) {
                continue;
            }
            let Some(s) = self.slot() else { break };
            if let Err(e) = pool.download_slot(vslot, &mut self.data[s]) {
                self.free.push(s);
                return Err(e);
            }
            self.clock += 1;
            (self.owner[s], self.stamp[s], self.slot_of[key]) = (key as i32, self.clock, s as i32);
        }
        let mut disk = Vec::with_capacity(loads.len());
        for &(vslot, e) in loads {
            let key = base + e;
            match self.slot_of[key] {
                s if s >= 0 => {
                    let s = s as usize;
                    pool.upload(vslot, &self.data[s])?;
                    (self.owner[s], self.slot_of[key]) = (-1, -1);
                    self.free.push(s);
                    self.hits += 1;
                }
                _ => disk.push((vslot, e)),
            }
        }
        Ok(disk)
    }
}

// --- the per-layer n-gram hash (aegist/hc.py) -----------------------------------------------------

const GAMMA: u64 = 0x9E3779B97F4A7C15;

fn splitmix64(v: u64) -> u64 {
    let mut v = v.wrapping_add(GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D049BB133111EB);
    v ^ (v >> 31)
}

fn ngram_multipliers(vocab: u64, n: usize, ple_index: u64, seed: u64) -> Vec<i64> {
    let half = ((i64::MAX as u64) / vocab.max(1) / 2).max(1);
    let base = seed.wrapping_add(10007u64.wrapping_mul(ple_index));
    (0..n as u64).map(|i| (2 * (splitmix64(base.wrapping_add(GAMMA.wrapping_mul(i + 1))) % half) + 1) as i64).collect()
}

fn is_prime(v: u64) -> bool {
    if v < 2 || v % 2 == 0 {
        return v == 2;
    }
    let mut d = 3;
    while d * d <= v {
        if v % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// Each head's table size: the (global head index + 1)-th prime from `base`.
fn ngram_head_sizes(base: u64, heads: usize, ple_index: usize) -> Vec<i64> {
    let (mut out, mut p, want, mut k) = (vec![], base - 1, ple_index * heads, 0);
    while out.len() < heads {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
        if k >= want {
            out.push(p as i64);
        }
        k += 1;
    }
    out
}

struct Lin {
    w: Buf,
    n: usize,
    k: usize,
    /// the tensor's name in the brain
    src: String,
    /// streamed: `w` is a view into a slot of its GPU's LayerStream, filled before its layer runs
    lazy: bool,
}

/// Layer streaming (AirLLM-style, $AEGIST_LAYER_STREAM=on): the big matrices of layers not kept
/// resident live in the brain file (the OS keeps it cached in RAM) and are copied into one of two
/// slots on their GPU just before the layer runs, so VRAM holds two layers' weights instead of all
/// 48; lm_head goes through the last GPU's slots in row chunks. The experts stream as always.
/// Capacity, not speed: every token copies every streamed layer again.
struct LayerStream {
    /// per GPU: two slots, each the size of its largest streamed layer
    slots: Vec<[Buf; 2]>,
    /// per GPU: which layer each slot holds (None: lm_head chunks passed through it)
    loaded: Vec<[Option<usize>; 2]>,
    /// per layer: its slot (layers streamed on its GPU alternate)
    slot_of: Vec<usize>,
    /// bytes copied in and seconds spent so far
    bytes: u64,
    secs: f64,
}

/// Matrices at least this big are streamed in layer-streaming mode; the small ones (norms,
/// router, cores) stay resident.
const STREAM_MIN: usize = 8 << 20;

/// Set while the layers of a streaming load are built: lin() then leaves big matrices in the file.
static LAZY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// $AEGIST_LAYER_STREAM: Some(true) for on, Some(false) for off, None (unset or "auto"): stream
/// only when the dense weights do not fit the GPUs.
fn layer_stream_setting() -> Option<bool> {
    let v = std::env::var("AEGIST_LAYER_STREAM").unwrap_or_default().trim().to_lowercase();
    if ["on", "1", "yes", "true"].contains(&v.as_str()) {
        Some(true)
    } else if ["off", "0", "no", "false"].contains(&v.as_str()) {
        Some(false)
    } else {
        None
    }
}

fn lins(l: &Layer) -> Vec<&Lin> {
    let mut v = vec![];
    for h in [&l.hc_attn, &l.hc_mlp] {
        v.push(&h.down);
        v.push(&h.up);
        if let Some(i) = &h.inject {
            v.push(i);
        }
    }
    match &l.mixer {
        Mixer::Attn { q, k, v: vv, o, idx, .. } => {
            v.extend([q, k, vv, o]);
            if let Some(i) = idx {
                v.push(&i.qk);
            }
        }
        Mixer::Delta { in_qkv, in_z, out, .. } => v.extend([in_qkv, in_z, out]),
    }
    if let Some(p) = &l.ple {
        v.push(&p.key);
        v.push(&p.value);
    }
    v.extend([&l.sh_gate, &l.sh_up, &l.sh_down]);
    v
}

fn lins_mut(l: &mut Layer) -> Vec<&mut Lin> {
    let mut v = vec![];
    for h in [&mut l.hc_attn, &mut l.hc_mlp] {
        v.push(&mut h.down);
        v.push(&mut h.up);
        if let Some(i) = &mut h.inject {
            v.push(i);
        }
    }
    match &mut l.mixer {
        Mixer::Attn { q, k, v: vv, o, idx, .. } => {
            v.extend([q, k, vv, o]);
            if let Some(i) = idx {
                v.push(&mut i.qk);
            }
        }
        Mixer::Delta { in_qkv, in_z, out, .. } => v.extend([in_qkv, in_z, out]),
    }
    if let Some(p) = &mut l.ple {
        v.push(&mut p.key);
        v.push(&mut p.value);
    }
    v.extend([&mut l.sh_gate, &mut l.sh_up, &mut l.sh_down]);
    v
}

/// Slot offsets are kept 256-byte aligned for the kernels.
fn align256(n: usize) -> usize {
    (n + 255) & !255
}

struct Hc {
    norm: Buf,
    down: Lin,
    up: Lin,
    inject: Option<Lin>,
}

struct Indexer {
    qk: Lin,
    q_norm: Buf,
    k_norm: Buf,
}

/// `slot`: the layer's place among its GPU's attention (or DeltaNet) layers.
enum Mixer {
    Attn { q: Lin, k: Lin, v: Lin, o: Lin, q_norm: Buf, k_norm: Buf, idx: Option<Indexer>, slot: usize },
    Delta { in_qkv: Lin, in_z: Lin, in_a: Buf, in_b: Buf, conv: Buf, a_log: Buf, dt_bias: Buf, norm: Buf, out: Lin, slot: usize },
}

struct Ple {
    key: Lin,
    value: Lin,
    norm_key: Buf,
    norm_query: Buf,
    norm_conv: Buf,
    conv: Buf,
    scale: Buf,
    table_at: usize,
    rows: usize,
    mult: Vec<i64>,
    sizes: Vec<i64>,
    offsets: Vec<i64>,
    /// its conv state's place among its GPU's PLE layers
    slot: usize,
}

struct Layer {
    /// the GPU (index into NextEngine::parts) this layer runs on
    dev: usize,
    hc_attn: Hc,
    hc_mlp: Hc,
    ple: Option<Ple>,
    mixer: Mixer,
    router: Buf,
    sh_gate: Lin,
    sh_up: Lin,
    sh_down: Lin,
    sh_gw: Buf,
    key1: Buf,
    key2: Buf,
    live: Buf,
    cores: Option<(Buf, Buf, Buf)>,
    experts_at: usize,
}

struct Kernels {
    gemv: Kernel,
    gemm: Kernel,
    gznorm: Kernel,
    scale_silu: Kernel,
    hc_mix: Kernel,
    hc_inject: Kernel,
    silu_mul: Kernel,
    attn_prep: Kernel,
    attention_gated: Kernel,
    attention_list: Kernel,
    delta_gates: Kernel,
    conv_silu: Kernel,
    l2norm_heads: Kernel,
    delta_scan: Kernel,
    gated_norm: Kernel,
    router: Kernel,
    gu_gemv: Kernel,
    dn_gemv: Kernel,
    gu_gemm: Kernel,
    dn_gemm: Kernel,
    gu4_gemv: Kernel,
    dn4_gemv: Kernel,
    gu4_gemm: Kernel,
    dn4_gemm: Kernel,
    q4_from_f8: Kernel,
    moe_reduce: Kernel,
    shared_gate_add: Kernel,
    cores_apply: Kernel,
    ple_embed: Kernel,
    ple_gate: Kernel,
    ple_conv: Kernel,
    idx_prep: Kernel,
    idx_blocks: Kernel,
    idx_scores: Kernel,
    idx_select: Kernel,
}

impl Kernels {
    fn new(gpu: &Gpu) -> Result<Kernels> {
        let f = |n: &str| gpu.function(n);
        Ok(Kernels {
            gemv: f("gemv_bf16a")?,
            gemm: f("gemm_bf16")?,
            gznorm: f("gznorm")?,
            scale_silu: f("scale_silu")?,
            hc_mix: f("hc_mix")?,
            hc_inject: f("hc_inject")?,
            silu_mul: f("silu_mul")?,
            attn_prep: f("attn_prep")?,
            attention_gated: f("attention_gated")?,
            attention_list: f("attention_gated_list")?,
            delta_gates: f("delta_gates")?,
            conv_silu: f("conv_silu")?,
            l2norm_heads: f("l2norm_heads")?,
            delta_scan: f("delta_scan")?,
            gated_norm: f("gated_norm2")?,
            router: f("router_topk_wide")?,
            gu_gemv: f("moe8_gu_gemv")?,
            dn_gemv: f("moe8_dn_gemv")?,
            gu_gemm: f("moe8_gu_gemm")?,
            dn_gemm: f("moe8_dn_gemm")?,
            gu4_gemv: f("moe_gu_gemv")?,
            dn4_gemv: f("moe4_dn_gemv")?,
            gu4_gemm: f("moe_gu_gemm")?,
            dn4_gemm: f("moe_dn_gemm")?,
            q4_from_f8: f("q4_from_f8")?,
            moe_reduce: f("moe_reduce")?,
            shared_gate_add: f("shared_gate_add")?,
            cores_apply: f("cores_apply")?,
            ple_embed: f("ple_embed")?,
            ple_gate: f("ple_gate")?,
            ple_conv: f("ple_conv")?,
            idx_prep: f("idx_prep")?,
            idx_blocks: f("idx_blocks")?,
            idx_scores: f("idx_scores")?,
            idx_select: f("idx_select")?,
        })
    }
}

const GEMV_TOKENS: usize = 8;
const STREAM_TOKENS: usize = 64;
const SNAPSHOTS: usize = 2;
const CHUNK: usize = 512;

/// The recurrent state that `crop` cannot recompute: DeltaNet conv + delta states, PLE conv
/// states (of one GPU's layers).
struct Recurrent {
    conv: Buf,
    state: Buf,
    ple: Buf,
}

impl Recurrent {
    fn bytes(c: &NConfig, n_delta: usize, n_ple: usize) -> usize {
        n_delta * (c.conv_dim() * 3 * 4 + c.hv * c.dk * c.dv * 4) + n_ple * c.lanes() * 9 * 4
    }

    fn new(c: &NConfig, n_delta: usize, n_ple: usize, zero: bool) -> Result<Recurrent> {
        let a = |n: usize| if zero { Buf::zeroed(n) } else { Buf::alloc(n) };
        Ok(Recurrent {
            conv: a(n_delta * c.conv_dim() * 3 * 4)?,
            state: a(n_delta * c.hv * c.dk * c.dv * 4)?,
            ple: a(n_ple * c.lanes() * 9 * 4)?,
        })
    }

    fn copy_from(&self, o: &Recurrent) -> Result<()> {
        self.conv.copy_from(&o.conv, o.conv.bytes)?;
        self.state.copy_from(&o.state, o.state.bytes)?;
        self.ple.copy_from(&o.ple, o.ple.bytes)
    }

    fn zero(&self) -> Result<()> {
        self.conv.zero()?;
        self.state.zero()?;
        self.ple.zero()
    }
}

/// Working memory for up to `chunk` tokens; each GPU has its own.
struct Scratch {
    x: Buf,
    xn: Buf,
    wl: Buf,
    gv: Buf,
    hr: Buf,
    il: Buf,
    h: Buf,
    ysub: Buf,
    pe: Buf,
    prow: Buf,
    big: Buf,
    kb: Buf,
    vb: Buf,
    qn: Buf,
    att: Buf,
    z: Buf,
    y: Buf,
    o: Buf,
    gb: Buf,
    iqk: Buf,
    qi: Buf,
    sc: Buf,
    blist: Buf,
    nsel: Buf,
    idx: Buf,
    wts: Buf,
    /// the next layer's router picks for the same h (prefetch guesses), one token's worth
    idx2: Buf,
    wts2: Buf,
    act: Buf,
    ypick: Buf,
    sh_g: Buf,
    sh_u: Buf,
    sh_out: Buf,
    meta: Buf,
}

impl Scratch {
    /// On the current GPU.
    fn new(c: &NConfig, ch: usize, cap: usize) -> Result<Scratch> {
        let (d, lanes, kk) = (c.d, c.lanes(), c.top_k);
        let f32b = |n: usize| Buf::alloc(n.max(1) * 4);
        let has_idx = c.idx_heads > 0 && c.n_attn() > 0;
        let nbmax = if has_idx { cap / c.idx_ratio } else { 0 };
        let has_ple = !c.ple_layers.is_empty();
        let big = (c.heads * 2 * c.head_dim).max(c.conv_dim());
        let wide = (c.heads * c.head_dim).max(c.hv * c.dv);
        Ok(Scratch {
            x: f32b(ch * lanes)?,
            xn: f32b(ch * lanes)?,
            wl: f32b(ch * lanes)?,
            gv: f32b(if has_ple { ch * lanes } else { 1 })?,
            hr: f32b(ch * c.hc_rank)?,
            il: f32b(ch * c.hc)?,
            h: f32b(ch * d)?,
            ysub: f32b(ch * d)?,
            pe: f32b(if has_ple { ch * c.ple_dim } else { 1 })?,
            prow: Buf::alloc(if has_ple { ch * c.ple_dim } else { 1 })?,
            big: f32b(ch * big)?,
            kb: f32b(ch * c.kv_heads * c.head_dim)?,
            vb: f32b(ch * c.kv_heads * c.head_dim)?,
            qn: f32b(ch * c.heads * c.head_dim)?,
            att: f32b(ch * wide)?,
            z: f32b(ch * c.hv * c.dv)?,
            y: f32b(ch * c.conv_dim())?,
            o: f32b(ch * c.hv * c.dv)?,
            gb: f32b(ch * 2 * c.hv)?,
            iqk: f32b(if has_idx { ch * (c.idx_heads + 1) * c.idx_dim } else { 1 })?,
            qi: f32b(if has_idx { ch * c.idx_heads * c.idx_dim } else { 1 })?,
            sc: f32b(if has_idx { ch * nbmax } else { 1 })?,
            blist: Buf::alloc(if has_idx { ch * c.idx_top() * 4 } else { 4 })?,
            nsel: Buf::alloc(ch * 4)?,
            idx: Buf::alloc(ch * kk * 4)?,
            wts: f32b(ch * kk)?,
            idx2: Buf::alloc(PREFETCH_TOKENS * kk * 4)?,
            wts2: f32b(PREFETCH_TOKENS * kk)?,
            act: f32b(ch * kk * c.expert_hidden)?,
            ypick: f32b(ch * kk * d)?,
            sh_g: f32b(ch * c.shared_hidden)?,
            sh_u: f32b(ch * c.shared_hidden)?,
            sh_out: f32b(ch * d)?,
            meta: Buf::alloc(meta_layout(c, ch).3)?,
        })
    }
}

/// One GPU's share of the brain: a run of consecutive layers (`first..end`) with their weights,
/// caches, recurrent state and expert pool, and its own working memory.
struct Part {
    gpu: Gpu,
    k: Kernels,
    s: Scratch,
    first: usize,
    end: usize,
    kc: Buf,
    vc: Buf,
    kraw: Option<Buf>,
    bk: Option<Buf>,
    rec: Recurrent,
    snaps: Vec<Recurrent>,
    pool: Pool,
    /// with 4-bit slots: where each FP8 expert read lands before q4_from_f8 packs it into its slot
    stage: Option<Buf>,
    /// most experts one layer runs at once (the pool's size, less a margin)
    batch: usize,
    tap_buf: Option<Buf>,
    /// core switch-on counts, [layers][cores] (zero for the other GPUs' layers)
    counts: Buf,
    weight_bytes: usize,
}

impl Part {
    fn mm(&self, w: &Lin, x: u64, y: u64, t: usize, accumulate: bool) -> Result<()> {
        let acc = Arg::I32(accumulate as i32);
        if t == 1 {
            self.k.gemv.launch((((w.n + 7) / 8) as u32, 1), 256, &[Arg::Ptr(x), Arg::Ptr(w.w.ptr), Arg::Ptr(y), Arg::I32(w.n as i32), Arg::I32(w.k as i32), acc])
        } else {
            self.k.gemm.launch(
                (((w.n + 63) / 64) as u32, ((t + 31) / 32) as u32),
                256,
                &[Arg::Ptr(x), Arg::Ptr(w.w.ptr), Arg::Ptr(y), Arg::I32(t as i32), Arg::I32(w.n as i32), Arg::I32(w.k as i32), acc],
            )
        }
    }

    /// Zero-centred RMSNorm of every d_model-wide group of `rows` rows of hc lanes.
    fn gznorm(&self, c: &NConfig, w: &Buf, src: u64, dst: u64, rows: usize) -> Result<()> {
        self.k.gznorm.launch(((rows * c.hc) as u32, 1), 256, &[Arg::Ptr(src), Arg::Ptr(w.ptr), Arg::Ptr(dst), Arg::I32(c.d as i32), Arg::I32(c.hc as i32), Arg::F32(c.eps)])
    }

    /// The sublayer input h (and, when `hc` injects, its lane weights in il) from rows of x at `xp`.
    fn hc_pre(&self, c: &NConfig, hc: &Hc, xp: u64, t: usize) -> Result<()> {
        let s = &self.s;
        self.gznorm(c, &hc.norm, xp, s.xn.ptr, t)?;
        self.mm(&hc.down, s.xn.ptr, s.hr.ptr, t, false)?;
        let n = t * c.hc_rank;
        self.k.scale_silu.launch((((n + 255) / 256) as u32, 1), 256, &[Arg::Ptr(s.hr.ptr), Arg::I32(n as i32), Arg::F32(1.0 / c.hc as f32)])?;
        self.mm(&hc.up, s.hr.ptr, s.wl.ptr, t, false)?;
        self.k.hc_mix.launch(
            (t as u32, ((c.d + 255) / 256) as u32),
            256,
            &[Arg::Ptr(s.wl.ptr), Arg::Ptr(s.xn.ptr), Arg::Ptr(s.h.ptr), Arg::I32(c.d as i32), Arg::I32(c.hc as i32)],
        )?;
        if let Some(inj) = &hc.inject {
            self.mm(inj, s.xn.ptr, s.il.ptr, t, false)?;
        }
        Ok(())
    }

    /// x += ysub in every lane, weighted by il.
    fn hc_post(&self, c: &NConfig, t: usize) -> Result<()> {
        let s = &self.s;
        self.k.hc_inject.launch(
            (t as u32, ((c.d + 255) / 256) as u32),
            256,
            &[Arg::Ptr(s.x.ptr), Arg::Ptr(s.ysub.ptr), Arg::Ptr(s.il.ptr), Arg::I32(c.d as i32), Arg::I32(c.hc as i32)],
        )
    }
}

pub struct NextEngine {
    pub cfg: NConfig,
    nx: NxFile,
    lay: ExpertLayout,
    /// the pool holds 4-bit copies of the experts ($AEGIST_EXPERT_BITS=4)
    q4: Option<Q4Layout>,
    embed_at: usize,
    /// on the last GPU, with the logits
    hc_out: Hc,
    lm_head: Lin,
    logits: Buf,
    layers: Vec<Layer>,
    parts: Vec<Part>,
    reader: Reader,
    /// 4-bit misses read from the 4-bit expert file instead of the brain (when it exists)
    q4file: Option<Q4File>,
    ram: Option<RamTier>,
    snap_pos: Vec<Option<usize>>,
    toks: Vec<u32>,
    pub cap: usize,
    pub length: usize,
    last_t: usize,
    chunk: usize,
    gemv_tokens: usize,
    stream_tokens: usize,
    /// Per token, skip picks of experts not in VRAM, lightest first, up to this much routing weight
    /// (0: never); the other picks are reweighted to sum to 1. $AEGIST_SKIP_MISS, default SKIP_MISS.
    skip_mass: f32,
    /// skip_mass while thinking (set_light): $AEGIST_SKIP_THINK, default SKIP_THINK
    skip_think: f32,
    light: bool,
    /// seconds in expert reads (disk and the copy into VRAM) so far
    read_secs: f64,
    /// steps since the hot list was saved
    hot_steps: usize,
    /// the last read-ahead guesses, and (guesses, guesses the next layer then read) so far
    guessed: Vec<u64>,
    guess_hits: (u64, u64),
    /// prompts longer than a chunk go layer by layer (feed_layers); $AEGIST_PREFILL_LAYERS=off: chunk by chunk
    prefill_layers: bool,
    /// (picks skipped, expert reads saved) so far
    pub skipped: (u64, u64),
    tap_layers: Vec<usize>,
    /// lanes_before: (layer, every token's lanes going into it so far)
    dump: Option<(usize, Vec<f32>)>,
    pub live_cores: usize,
    /// layer streaming ($AEGIST_LAYER_STREAM=on), else None
    stream: Option<LayerStream>,
}

/// Layer streaming: the output head for one position (in p.s.h) into `y`, lm_head going through
/// the last GPU's two slots in row chunks.
fn streamed_head(p: &Part, st: &mut LayerStream, lm: &Lin, nx: &NxFile, y: u64) -> Result<()> {
    let t0 = std::time::Instant::now();
    let (k, j) = (lm.k, st.slots.len() - 1);
    let all = nx.bytes(&lm.src)?;
    let rows = (st.slots[j][0].bytes / (k * 2)).max(1);
    for (c, r0) in (0..lm.n).step_by(rows).enumerate() {
        let n = rows.min(lm.n - r0);
        let slot = &st.slots[j][c % 2];
        slot.upload_at(0, &all[r0 * k * 2..(r0 + n) * k * 2])?;
        let w = Lin { w: Buf::view(slot.ptr, n * k * 2), n, k, src: String::new(), lazy: true };
        p.mm(&w, p.s.h.ptr, y + (r0 * 4) as u64, 1, false)?;
    }
    st.bytes += all.len() as u64;
    st.loaded[j] = [None, None];
    st.secs += t0.elapsed().as_secs_f64();
    Ok(())
}

fn lin(nx: &NxFile, name: &str, n: usize, k: usize, total: &mut usize) -> Result<Lin> {
    let t = nx.info(name)?;
    if t.dtype != "bf16" || t.shape != [n, k] {
        return Err(format!("{name} is {} {:?}, expected bf16 [{n}, {k}]", t.dtype, t.shape));
    }
    if LAZY.load(std::sync::atomic::Ordering::Relaxed) && t.nbytes >= STREAM_MIN {
        return Ok(Lin { w: Buf::view(0, t.nbytes), n, k, src: name.into(), lazy: true });
    }
    Ok(Lin { w: upload(nx, name, total)?, n, k, src: name.into(), lazy: false })
}

fn hc_weights(nx: &NxFile, p: &str, c: &NConfig, combine: bool, total: &mut usize) -> Result<Hc> {
    let lanes = c.lanes();
    Ok(Hc {
        norm: upload(nx, &format!("{p}.norm.weight"), total)?,
        down: lin(nx, &format!("{p}.down.weight"), c.hc_rank, lanes, total)?,
        up: lin(nx, &format!("{p}.up.weight"), lanes, c.hc_rank, total)?,
        inject: if combine { Some(lin(nx, &format!("{p}.inject.weight"), c.hc, lanes, total)?) } else { None },
    })
}

/// The layer's stored cores and the id -> index map for cores_apply (as in hybrid.rs).
fn live_bank(nx: &NxFile, layer: usize, cfg: &NConfig, total: &mut usize) -> Result<(Buf, Option<(Buf, Buf, Buf)>, usize)> {
    let name = |k: &str| format!("blocks.{layer}.cores.{k}");
    let ids_info = nx.info(&name("ids"))?;
    if ids_info.dtype != "i64" {
        return Err(format!("{} must be i64", name("ids")));
    }
    let n = ids_info.shape.first().copied().unwrap_or(0);
    for k in ["gate", "up", "down"] {
        let t = nx.info(&name(k))?;
        if t.dtype != "bf16" || t.shape.first().copied().unwrap_or(0) != n {
            return Err(format!("{} must be bf16 with {n} rows", name(k)));
        }
    }
    let mut map = vec![-1i32; cfg.n_cores()];
    for (j, c) in nx.bytes(&name("ids"))?.chunks_exact(8).enumerate() {
        let id = i64::from_le_bytes(c.try_into().unwrap());
        if id < 0 || id as usize >= cfg.n_cores() {
            return Err(format!("{}: core id {id} is outside the {} x {} grid", name("ids"), cfg.n1, cfg.n2));
        }
        map[id as usize] = j as i32;
    }
    let live = Buf::from_bytes(&bytes_i32(&map))?;
    if n == 0 {
        return Ok((live, None, 0));
    }
    let bank = (upload(nx, &name("gate"), total)?, upload(nx, &name("up"), total)?, upload(nx, &name("down"), total)?);
    Ok((live, Some(bank), n))
}

fn meta_layout(c: &NConfig, chunk: usize) -> (usize, usize, usize, usize) {
    let a16 = |n: usize| (n + 15) / 16 * 16;
    let picks = chunk * c.top_k * 4;
    let us = c.experts * 8;
    let pt = us + a16((c.experts + 1) * 4);
    let ps = pt + a16(picks);
    (us, pt, ps, ps + picks)
}

/// The GPUs to use: $AEGIST_GPUS ("0,1"; "0" for the first only), else every GPU the driver sees.
fn gpu_ordinals() -> Result<Vec<usize>> {
    match std::env::var("AEGIST_GPUS") {
        Ok(v) if !v.trim().is_empty() => {
            let mut out = vec![];
            for s in v.split(',') {
                let o: usize = s.trim().parse().map_err(|_| format!("AEGIST_GPUS={v}: expected GPU numbers like 0,1"))?;
                if !out.contains(&o) {
                    out.push(o);
                }
            }
            Ok(out)
        }
        _ => Ok((0..crate::cuda::device_count()?.max(1)).collect()),
    }
}

/// Layers per GPU, in order: each GPU takes a run of consecutive layers. `w`: each layer's bytes
/// in VRAM besides its experts; `avail`: each GPU's bytes for layers and experts; `eb`: one
/// expert's bytes. The split gives every GPU the same number of expert slots per layer it runs
/// (as near as whole layers allow), so each layer finds as many of its experts in VRAM as the
/// others. A GPU too small to help gets no layers.
fn split_layers(w: &[f64], avail: &[f64], eb: f64) -> Vec<usize> {
    let n = avail.len();
    if n <= 1 {
        return vec![w.len()];
    }
    // greedy: each GPU in turn takes layers while it keeps at least r slots per layer
    let take = |r: f64| -> Option<Vec<usize>> {
        let (mut out, mut l) = (vec![0usize; n], 0);
        for i in 0..n {
            let mut used = 0.0;
            while l < w.len() && (avail[i] - used - w[l]) / eb / (out[i] + 1) as f64 >= r {
                used += w[l];
                out[i] += 1;
                l += 1;
            }
        }
        (l == w.len()).then_some(out)
    };
    let Some(mut best) = take(0.0) else {
        // not even the weights fit: spread them evenly and let loading say what is short
        return (0..n).map(|i| (i + 1) * w.len() / n - i * w.len() / n).collect();
    };
    let (mut lo, mut hi) = (0.0, avail.iter().cloned().fold(0.0, f64::max) / eb + 1.0);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        match take(mid) {
            Some(v) => {
                best = v;
                lo = mid;
            }
            None => hi = mid,
        }
    }
    best
}

impl NextEngine {
    /// Load `nx` (kept open: experts, the embedding and the n-gram table are read from it).
    /// `max_context` 0 picks 32k tokens.
    ///
    /// Several GPUs ($AEGIST_GPUS, default all of them): the layers are split into one run per
    /// GPU (see split_layers; $AEGIST_SPLIT="26" starts the second GPU at layer 26), and the
    /// lanes go from one GPU to the next once per pass. On each GPU, the VRAM left after its
    /// layers becomes expert slots for those layers, minus $AEGIST_VRAM_HEADROOM_MB (default
    /// 1536 on a GPU other apps use, such as the one driving the display, else 512).
    /// $AEGIST_EXPERT_SLOTS caps the slot count over all GPUs.
    pub fn load(nx: NxFile, max_context: usize, mut progress: impl FnMut(usize, usize)) -> Result<NextEngine> {
        let cfg = NConfig::from_header(&nx.header["config"])?;
        let lay = ExpertLayout::new(&cfg);
        let q4 = expert_bits()?.eq(&4).then(|| Q4Layout::new(&cfg));
        if q4.is_some() && (cfg.d % 64 != 0 || cfg.expert_hidden % 64 != 0) {
            return Err(format!("4-bit experts need d_model and expert_hidden divisible by 64 ({} and {})", cfg.d, cfg.expert_hidden));
        }
        // bytes of one expert slot in VRAM
        let sb = q4.map_or(lay.bytes, |q| q.bytes);
        let cap = if max_context == 0 { 32768 } else { max_context };
        let ch = CHUNK;
        let (d, lanes, nl) = (cfg.d, cfg.lanes(), cfg.layers);
        let need = |name: &str, dtype: &str, shape: &[usize]| -> Result<()> {
            let t = nx.info(name)?;
            if t.dtype != dtype || t.shape != shape {
                return Err(format!("{name} is {} {:?}, expected {dtype} {shape:?}", t.dtype, t.shape));
            }
            Ok(())
        };
        need("embed.weight", "bf16", &[cfg.vocab, d])?;

        // every GPU, its kernels and working memory; then what is left on each
        let set_headroom = env_usize("AEGIST_VRAM_HEADROOM_MB");
        let mut gpus = vec![];
        for o in gpu_ordinals()? {
            let gpu = Gpu::on(o, PTX)?;
            let (free, total) = gpu.mem_info();
            // a GPU that other apps already use (the display, a game) keeps more room free
            let headroom = set_headroom.unwrap_or(if total - free > 1 << 30 { 1536 } else { 512 }) << 20;
            let k = Kernels::new(&gpu)?;
            gpu.bind()?;
            let s = Scratch::new(&cfg, ch, cap)?;
            let (free, _) = gpu.mem_info();
            gpus.push((gpu, k, s, free, headroom));
        }

        // each layer's VRAM besides its experts: weights (the n-gram table stays in the file) + caches
        let kv_layer = cfg.kv_heads * cap * cfg.head_dim * 2;
        let has_idx = cfg.idx_heads > 0 && cfg.n_attn() > 0;
        let nbmax = if has_idx { cap / cfg.idx_ratio } else { 0 };
        let mut w = vec![0f64; nl];
        let mut stream_w = vec![0f64; nl];
        let mut head = 0f64;
        let mut lm = 0f64;
        for (name, t) in &nx.tensors {
            if let Some(rest) = name.strip_prefix("blocks.") {
                let (i, sub) = rest.split_once('.').unwrap_or(("", ""));
                if let Ok(i) = i.parse::<usize>() {
                    if i < nl && !sub.ends_with("experts.fp8") && !sub.starts_with("ple.table.") && sub != "cores.ids" {
                        // a streamed layer's big matrices are in its GPU's two slots instead
                        if t.nbytes >= STREAM_MIN && t.shape.len() == 2 && t.dtype == "bf16" {
                            stream_w[i] += align256(t.nbytes) as f64;
                        } else {
                            w[i] += t.nbytes as f64;
                        }
                    }
                }
            } else if name.starts_with("hc_out.") {
                head += t.nbytes as f64;
            } else if name.starts_with("lm_head.") {
                lm += t.nbytes as f64;
            }
        }
        head += (cfg.vocab * 4) as f64;
        for i in 0..nl {
            w[i] += if cfg.attn[i] {
                (2 * kv_layer + if has_idx { (cap + nbmax) * cfg.idx_dim * 4 } else { 0 }) as f64
            } else {
                (Recurrent::bytes(&cfg, 1, 0) * (1 + SNAPSHOTS)) as f64
            };
            if cfg.ple_layers.contains(&i) {
                w[i] += (Recurrent::bytes(&cfg, 0, 1) * (1 + SNAPSHOTS)) as f64;
            }
        }
        // layer streaming: as asked, or (by default) when the dense weights plus one token's
        // experts do not fit what the GPUs have free; then as many layers as fit stay resident
        let room: f64 = gpus.iter().map(|g| g.3 as f64 - g.4 as f64).sum();
        let one_token = (nl * cfg.top_k) as f64 * sb as f64;
        let dense: f64 = w.iter().sum::<f64>() + stream_w.iter().sum::<f64>() + head + lm;
        let setting = layer_stream_setting();
        let streaming = setting.unwrap_or(dense + one_token > room);
        if !streaming {
            for i in 0..nl {
                w[i] += stream_w[i];
                stream_w[i] = 0.0;
            }
            head += lm;
        }
        let mut resident = vec![!streaming; nl];
        if streaming {
            // the first layers stay resident, up to $AEGIST_RESIDENT_GB, or by default (when not
            // forced on) what fits besides two slots, one token's experts and 1 GB to spare
            let fit = (room - w.iter().sum::<f64>() - head - one_token - 2.0 * stream_w.iter().cloned().fold(0.0, f64::max) - 1e9).max(0.0);
            let mut budget = match std::env::var("AEGIST_RESIDENT_GB").ok().and_then(|v| v.trim().parse::<f64>().ok()) {
                Some(g) => g * 1e9,
                None if setting.is_none() => fit,
                None => 0.0,
            };
            for i in 0..nl {
                if stream_w[i] <= budget {
                    budget -= stream_w[i];
                    resident[i] = true;
                    w[i] += stream_w[i];
                    stream_w[i] = 0.0;
                } else {
                    break;
                }
            }
        }
        let stream_max = stream_w.iter().cloned().fold(0.0, f64::max);
        let forced = std::env::var("AEGIST_SPLIT").ok().filter(|v| !v.trim().is_empty());
        let counts = loop {
            let avail: Vec<f64> = gpus
                .iter()
                .enumerate()
                .map(|(j, g)| g.3 as f64 - g.4 as f64 - if j + 1 == gpus.len() { head } else { 0.0 } - 2.0 * stream_max)
                .collect();
            let counts = match &forced {
                Some(v) => {
                    let mut starts = vec![0usize];
                    for s in v.split(',') {
                        starts.push(s.trim().parse().map_err(|_| format!("AEGIST_SPLIT={v}: expected layer numbers like 26"))?);
                    }
                    starts.push(nl);
                    if starts.len() != gpus.len() + 1 || starts.windows(2).any(|p| p[0] >= p[1]) {
                        return Err(format!("AEGIST_SPLIT={v}: needs {} increasing layer numbers below {nl} for {} GPUs", gpus.len() - 1, gpus.len()));
                    }
                    starts.windows(2).map(|p| p[1] - p[0]).collect()
                }
                None => split_layers(&w, &avail, sb as f64),
            };
            if counts.iter().all(|&c| c > 0) {
                break counts;
            }
            // a GPU too small to help: leave it out and split again
            let mut keep = counts.iter().map(|&c| c > 0);
            gpus.retain(|_| keep.next().unwrap());
        };
        let mut dev_of = vec![];
        for (j, &c) in counts.iter().enumerate() {
            dev_of.extend(std::iter::repeat(j).take(c));
        }

        // the layers, each on its GPU
        let mut totals = vec![0usize; gpus.len()];
        let (mut n_attn, mut n_delta, mut n_ple) = (vec![0usize; gpus.len()], vec![0usize; gpus.len()], vec![0usize; gpus.len()]);
        let mut live_cores = 0;
        let mut layers = Vec::with_capacity(nl);
        for i in 0..nl {
            progress(i, nl);
            let dev = dev_of[i];
            gpus[dev].0.bind()?;
            LAZY.store(!resident[i], std::sync::atomic::Ordering::Relaxed);
            let total = &mut totals[dev];
            let p = |s: &str| format!("blocks.{i}.{s}");
            let mixer = if cfg.attn[i] {
                n_attn[dev] += 1;
                let hd = cfg.head_dim;
                let idx = if cfg.idx_heads > 0 {
                    Some(Indexer {
                        q_norm: upload(&nx, &p("mixer.indexer.q_norm.weight"), total)?,
                        k_norm: upload(&nx, &p("mixer.indexer.k_norm.weight"), total)?,
                        qk: lin(&nx, &p("mixer.indexer.qk.weight"), (cfg.idx_heads + 1) * cfg.idx_dim, d, total)?,
                    })
                } else {
                    None
                };
                Mixer::Attn {
                    q_norm: upload(&nx, &p("mixer.q_norm.weight"), total)?,
                    k_norm: upload(&nx, &p("mixer.k_norm.weight"), total)?,
                    q: lin(&nx, &p("mixer.q.weight"), cfg.heads * hd * 2, d, total)?,
                    k: lin(&nx, &p("mixer.k.weight"), cfg.kv_heads * hd, d, total)?,
                    v: lin(&nx, &p("mixer.v.weight"), cfg.kv_heads * hd, d, total)?,
                    o: lin(&nx, &p("mixer.o.weight"), d, cfg.heads * hd, total)?,
                    idx,
                    slot: n_attn[dev] - 1,
                }
            } else {
                n_delta[dev] += 1;
                need(&p("mixer.conv"), "bf16", &[cfg.conv_dim(), cfg.conv_k])?;
                let vd = cfg.hv * cfg.dv;
                Mixer::Delta {
                    in_a: upload(&nx, &p("mixer.in_a.weight"), total)?,
                    in_b: upload(&nx, &p("mixer.in_b.weight"), total)?,
                    conv: upload(&nx, &p("mixer.conv"), total)?,
                    a_log: upload(&nx, &p("mixer.A_log"), total)?,
                    dt_bias: upload(&nx, &p("mixer.dt_bias"), total)?,
                    norm: upload(&nx, &p("mixer.norm.weight"), total)?,
                    in_qkv: lin(&nx, &p("mixer.in_qkv.weight"), cfg.conv_dim(), d, total)?,
                    in_z: lin(&nx, &p("mixer.in_z.weight"), vd, d, total)?,
                    out: lin(&nx, &p("mixer.out.weight"), d, vd, total)?,
                    slot: n_delta[dev] - 1,
                }
            };
            let ple = match cfg.ple_layers.iter().position(|&l| l == i) {
                Some(index) => {
                    n_ple[dev] += 1;
                    let heads = cfg.ple_heads();
                    let tname = p("ple.table.weight");
                    let t = nx.info(&tname)?;
                    if t.dtype != "f8e4m3" || t.shape.len() != 2 || t.shape[1] != cfg.ple_row() {
                        return Err(format!("{tname} is {} {:?}, expected f8e4m3 [rows, {}]", t.dtype, t.shape, cfg.ple_row()));
                    }
                    need(&p("ple.table_scale"), "bf16", &[1])?;
                    need(&p("ple.conv"), "bf16", &[lanes, cfg.ple_conv_k])?;
                    let sizes = ngram_head_sizes(cfg.ngram_base, heads, index);
                    let offsets: Vec<i64> = (0..heads).map(|j| sizes[..j].iter().sum()).collect();
                    if (offsets[heads - 1] + sizes[heads - 1]) as usize > t.shape[0] {
                        return Err(format!("{tname}: {} rows, the n-gram hash needs more", t.shape[0]));
                    }
                    Some(Ple {
                        key: lin(&nx, &p("ple.key.weight"), lanes, cfg.ple_dim, total)?,
                        value: lin(&nx, &p("ple.value.weight"), d, cfg.ple_dim, total)?,
                        norm_key: upload(&nx, &p("ple.norm_key.weight"), total)?,
                        norm_query: upload(&nx, &p("ple.norm_query.weight"), total)?,
                        norm_conv: upload(&nx, &p("ple.norm_conv.weight"), total)?,
                        conv: upload(&nx, &p("ple.conv"), total)?,
                        scale: upload(&nx, &p("ple.table_scale"), total)?,
                        table_at: nx.offset_of(&tname)?,
                        rows: t.shape[0],
                        mult: ngram_multipliers(cfg.vocab as u64, cfg.ngram, index as u64, cfg.ngram_seed),
                        sizes,
                        offsets,
                        slot: n_ple[dev] - 1,
                    })
                }
                None => None,
            };
            need(&p("moe.experts.fp8"), "u8", &[cfg.experts, lay.bytes])?;
            need(&p("moe.router.weight"), "bf16", &[cfg.experts, d])?;
            need(&p("moe.shared_gate.weight"), "bf16", &[1, d])?;
            need(&p("engine.key1.weight"), "bf16", &[cfg.n1, d])?;
            need(&p("engine.key2.weight"), "bf16", &[cfg.n2, d])?;
            let (live, cores, n_live) = live_bank(&nx, i, &cfg, total)?;
            live_cores += n_live;
            let sh = cfg.shared_hidden;
            layers.push(Layer {
                dev,
                hc_attn: hc_weights(&nx, &p("hc_attn"), &cfg, true, total)?,
                hc_mlp: hc_weights(&nx, &p("hc_mlp"), &cfg, true, total)?,
                ple,
                mixer,
                router: upload(&nx, &p("moe.router.weight"), total)?,
                sh_gate: lin(&nx, &p("moe.shared.gate.weight"), sh, d, total)?,
                sh_up: lin(&nx, &p("moe.shared.up.weight"), sh, d, total)?,
                sh_down: lin(&nx, &p("moe.shared.down.weight"), d, sh, total)?,
                sh_gw: upload(&nx, &p("moe.shared_gate.weight"), total)?,
                key1: upload(&nx, &p("engine.key1.weight"), total)?,
                key2: upload(&nx, &p("engine.key2.weight"), total)?,
                live,
                cores,
                experts_at: nx.offset_of(&p("moe.experts.fp8"))?,
            });
        }
        progress(nl, nl);
        let last = gpus.len() - 1;
        gpus[last].0.bind()?;
        LAZY.store(false, std::sync::atomic::Ordering::Relaxed);
        let hc_out = hc_weights(&nx, "hc_out", &cfg, false, &mut totals[last])?;
        LAZY.store(streaming, std::sync::atomic::Ordering::Relaxed);
        let lm_head = lin(&nx, "lm_head.weight", cfg.vocab, d, &mut totals[last]);
        LAZY.store(false, std::sync::atomic::Ordering::Relaxed);
        let lm_head = lm_head?;

        // layer streaming: two slots per GPU, and every streamed matrix a view into one of them
        let stream = if streaming {
            let mut slots = vec![];
            let mut slot_of = vec![0; nl];
            for (j, g) in gpus.iter().enumerate() {
                let mine: Vec<usize> = (0..nl).filter(|&i| dev_of[i] == j && !resident[i]).collect();
                let size = mine.iter().map(|&i| lins(&layers[i]).iter().filter(|l| l.lazy).map(|l| align256(l.w.bytes)).sum::<usize>()).max().unwrap_or(0);
                // the last GPU's slots also pass lm_head through, at least a few rows at a time
                let size = if j == last { size.max(64 * d * 2) } else { size };
                g.0.bind()?;
                let pair = [Buf::alloc(size.max(256))?, Buf::alloc(size.max(256))?];
                totals[j] += 2 * size;
                for (n, &i) in mine.iter().enumerate() {
                    slot_of[i] = n % 2;
                    let base = pair[n % 2].ptr;
                    let mut off = 0;
                    for l in lins_mut(&mut layers[i]).into_iter().filter(|l| l.lazy) {
                        l.w = Buf::view(base + off as u64, l.w.bytes);
                        off += align256(l.w.bytes);
                    }
                }
                slots.push(pair);
            }
            gpus[last].0.bind()?;
            let slot_gb: f64 = slots.iter().map(|p| 2.0 * p[0].bytes as f64).sum::<f64>() / 1e9;
            let streamed_gb = stream_w.iter().sum::<f64>() / 1e9;
            eprintln!(
                "layer streaming: {} of {nl} layers streamed ({streamed_gb:.2} GB of dense weights stay in the file), {slot_gb:.2} GB of slots on the GPUs",
                resident.iter().filter(|r| !**r).count()
            );
            Some(LayerStream { loaded: vec![[None, None]; slots.len()], slots, slot_of, bytes: 0, secs: 0.0 })
        } else {
            None
        };
        let logits = Buf::alloc(cfg.vocab * 4)?;

        // each GPU's caches and state, then its expert pool
        let mut parts = vec![];
        let mut first = 0;
        let n_gpus = gpus.len();
        for (j, (gpu, k, s, _, headroom)) in gpus.into_iter().enumerate() {
            gpu.bind()?;
            let kc = Buf::alloc(kv_layer * n_attn[j]).map_err(|e| format!("KV cache for {cap} tokens on {}: {e}", gpu.name))?;
            let vc = Buf::alloc(kv_layer * n_attn[j]).map_err(|e| format!("KV cache for {cap} tokens on {}: {e}", gpu.name))?;
            let (kraw, bk) = if has_idx && n_attn[j] > 0 {
                (Some(Buf::alloc(n_attn[j] * cap * cfg.idx_dim * 4)?), Some(Buf::alloc(n_attn[j] * nbmax.max(1) * cfg.idx_dim * 4)?))
            } else {
                (None, None)
            };
            let mut snaps = vec![];
            for _ in 0..SNAPSHOTS {
                snaps.push(Recurrent::new(&cfg, n_delta[j], n_ple[j], false)?);
            }
            let rec = Recurrent::new(&cfg, n_delta[j], n_ple[j], true)?;
            let counts_buf = Buf::zeroed(nl * cfg.n_cores() * 4)?;

            // whatever VRAM is left (minus headroom) holds this GPU's layers' experts
            let end = first + counts[j];
            let stage = match q4 {
                Some(_) => Some(Buf::alloc(lay.bytes)?),
                None => None,
            };
            let (free, _) = gpu.mem_info();
            let mine = counts[j] * cfg.experts;
            let min_slots = 2 * cfg.top_k + 8;
            let mut want = (free.saturating_sub(headroom) / sb).min(mine);
            if want < min_slots && set_headroom.is_none() {
                // other apps hold most of the GPU: run anyway on what is left, with a small margin
                want = (free.saturating_sub(256 << 20) / sb).min(mine);
            }
            if let Some(n) = env_usize("AEGIST_EXPERT_SLOTS") {
                want = want.min((n * counts[j] / nl).max(min_slots));
            }
            let per_chunk = 64;
            let (mut chunks, mut got) = (vec![], 0);
            while got < want {
                let n = per_chunk.min(want - got);
                match Buf::alloc(n * sb) {
                    Ok(b) => {
                        chunks.push(b);
                        got += n;
                    }
                    Err(_) => break,
                }
            }
            if got < min_slots {
                return Err(format!(
                    "not enough memory on {} for the experts of layers {first}-{}: room for {got} expert slots, at least \
                     {min_slots} needed ({:.2} GB free after the other weights). Close GPU-heavy apps, lower --context or \
                     $AEGIST_VRAM_HEADROOM_MB, or move layers with $AEGIST_SPLIT.",
                    gpu.name,
                    end - 1,
                    free as f64 / 1e9
                ));
            }
            let pool = Pool::new(chunks, per_chunk, got, sb, cfg.experts, nl * cfg.experts);
            parts.push(Part {
                weight_bytes: totals[j] + got * sb + stage.as_ref().map_or(0, |b| b.bytes),
                // a layer whose distinct experts do not fit in the pool runs them in batches that do
                batch: got - 8,
                gpu,
                k,
                s,
                first,
                end,
                kc,
                vc,
                kraw,
                bk,
                rec,
                snaps,
                pool,
                stage,
                tap_buf: None,
                counts: counts_buf,
            });
            first = end;
        }
        debug_assert_eq!(parts.len(), n_gpus);
        let reader = Reader::open(&nx.path, READ_THREADS)?;
        let q4file = q4.as_ref().and_then(|q| open_q4_file(&nx, &cfg, q));
        let mut eng = NextEngine {
            snap_pos: vec![None; SNAPSHOTS],
            toks: vec![],
            reader,
            q4file,
            ram: RamTier::from_env(parts[0].pool.slot_bytes(), cfg.layers * cfg.experts),
            embed_at: nx.offset_of("embed.weight")?,
            nx,
            lay,
            q4,
            hc_out,
            lm_head,
            logits,
            layers,
            parts,
            cap,
            length: 0,
            last_t: 0,
            chunk: ch,
            gemv_tokens: env_usize("AEGIST_GEMV_TOKENS").unwrap_or(GEMV_TOKENS),
            stream_tokens: env_usize("AEGIST_STREAM_TOKENS").unwrap_or(STREAM_TOKENS),
            skip_mass: std::env::var("AEGIST_SKIP_MISS").ok().and_then(|v| v.parse().ok()).unwrap_or(SKIP_MISS),
            skip_think: std::env::var("AEGIST_SKIP_THINK").ok().and_then(|v| v.parse().ok()).unwrap_or(SKIP_THINK),
            light: false,
            read_secs: 0.0,
            hot_steps: 0,
            guessed: vec![],
            guess_hits: (0, 0),
            skipped: (0, 0),
            prefill_layers: std::env::var("AEGIST_PREFILL_LAYERS").map_or(true, |v| v.trim() != "off"),
            tap_layers: vec![],
            dump: None,
            live_cores,
            stream,
            cfg,
        };
        eng.start_hot();
        Ok(eng)
    }

    /// Make the 4-bit expert file (see Q4File). Layer by layer: all its FP8 experts are read and
    /// packed on the first GPU by q4_from_f8 into host memory first (reads held to
    /// $AEGIST_Q4_READ_MBPS, default 250), then written in one go, held to $AEGIST_Q4_WRITE_MBPS
    /// (default 150) and flushed, so the drive never reads and writes at once: a USB-attached
    /// drive (D: here) resets under full-speed bulk reads and writes. A failed read or write waits
    /// and tries again (5 times); a stopped run goes on from its last finished layer. It is written
    /// to a .tmp file and renamed when complete. `progress(layers done, layers)`.
    pub fn write_q4_file(&mut self, mut progress: impl FnMut(usize, usize)) -> Result<PathBuf> {
        use std::io::{Seek, SeekFrom, Write};
        let q = self.q4.ok_or("the 4-bit expert file is made with $AEGIST_EXPERT_BITS=4")?;
        let path = q4_file_path(&self.nx).ok_or("$AEGIST_Q4_FILE=off")?;
        let head = q4_file_head(&self.nx, &self.cfg, &q).ok_or("this brain has no base weight hash")?;
        let tmp = path.with_extension("experts-q4.tmp");
        let done_path = path.with_extension("experts-q4.tmp.layers");
        let ioerr = |e: std::io::Error| format!("{}: {e}", tmp.display());
        let mut block = vec![0u8; Q4_HEAD];
        let text = head.to_string();
        block[..8].copy_from_slice(Q4_MAGIC);
        block[8..8 + text.len()].copy_from_slice(text.as_bytes());
        // a stopped run of the same file goes on after its last finished layer
        let mut start = 0;
        if let Ok(n) = std::fs::read_to_string(&done_path) {
            let mut h = vec![0u8; Q4_HEAD];
            let same = std::fs::File::open(&tmp).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut h)).is_ok() && h == block;
            if same {
                start = n.trim().parse().unwrap_or(0);
            }
        }
        if start == 0 {
            let mut f = std::fs::File::create(&tmp).map_err(ioerr)?;
            f.write_all(&block).map_err(ioerr)?;
            f.sync_all().map_err(ioerr)?;
        } else {
            eprintln!("going on after layer {start} of {}", tmp.display());
        }
        let mbps: f64 = std::env::var("AEGIST_Q4_WRITE_MBPS").ok().and_then(|v| v.parse().ok()).unwrap_or(150.0);
        let read_mbps: f64 = std::env::var("AEGIST_Q4_READ_MBPS").ok().and_then(|v| v.parse().ok()).unwrap_or(250.0);
        let (d, ih, n, nl, eb) = (self.cfg.d, self.cfg.expert_hidden, self.cfg.experts, self.cfg.layers, self.lay.bytes);
        let part = &self.parts[0];
        part.gpu.bind()?;
        let st = part.stage.as_ref().ok_or("no stage buffer")?;
        let dst = Buf::alloc(q.bytes)?;
        let (kq, lay) = (&part.k.q4_from_f8, &self.lay);
        let groups = 3 * ih * d / 64;
        let i = |v: usize| Arg::I32(v as i32);
        let mut buf = vec![0u8; n * q.bytes];
        progress(start, nl);
        for li in start..nl {
            let at = self.layers[li].experts_at;
            let jobs: Vec<(usize, usize)> = (0..n).map(|e| (e, at + e * eb)).collect();
            // read in small groups at $AEGIST_Q4_READ_MBPS: a full-speed bulk read alone reset D:
            for group in jobs.chunks(16) {
                let t = std::time::Instant::now();
                let mut tries = 0;
                loop {
                    let read = self.reader.load(group, eb, |e, bytes| {
                        st.upload(bytes)?;
                        kq.launch(
                            (((groups + 7) / 8) as u32, 1),
                            256,
                            &[Arg::Ptr(st.ptr), Arg::Ptr(dst.ptr), i(d), i(ih), i(lay.gu_s), i(lay.dn), i(lay.dn_s), i(q.s4g), i(q.m4g), i(q.q4d), i(q.s4d), i(q.m4d)],
                        )?;
                        buf[e * q.bytes..(e + 1) * q.bytes].copy_from_slice(&dst.download_at(0, q.bytes)?);
                        Ok(())
                    });
                    match read {
                        Ok(()) => break,
                        Err(e) if tries < 5 => {
                            tries += 1;
                            eprintln!("layer {li}: reading failed ({e}); trying again in 60 s ({tries}/5)");
                            std::thread::sleep(std::time::Duration::from_secs(60));
                        }
                        Err(e) => return Err(e),
                    }
                }
                let want = (group.len() * eb) as f64 / (read_mbps * 1e6);
                let took = t.elapsed().as_secs_f64();
                if took < want {
                    std::thread::sleep(std::time::Duration::from_secs_f64(want - took));
                }
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
            let mut tries = 0;
            loop {
                let wrote = (|| -> std::io::Result<()> {
                    let mut f = std::fs::OpenOptions::new().write(true).open(&tmp)?;
                    f.seek(SeekFrom::Start((Q4_HEAD + li * n * q.bytes) as u64))?;
                    let piece = 16 << 20;
                    for c in buf.chunks(piece) {
                        let t = std::time::Instant::now();
                        f.write_all(c)?;
                        let want = c.len() as f64 / (mbps * 1e6);
                        let took = t.elapsed().as_secs_f64();
                        if took < want {
                            std::thread::sleep(std::time::Duration::from_secs_f64(want - took));
                        }
                    }
                    f.sync_all()
                })();
                match wrote {
                    Ok(()) => break,
                    Err(e) if tries < 5 => {
                        tries += 1;
                        eprintln!("layer {li}: writing failed ({e}); trying again in 60 s ({tries}/5)");
                        std::thread::sleep(std::time::Duration::from_secs(60));
                    }
                    Err(e) => return Err(ioerr(e)),
                }
            }
            std::fs::write(&done_path, (li + 1).to_string()).map_err(|e| format!("{}: {e}", done_path.display()))?;
            progress(li + 1, nl);
        }
        std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
        let _ = std::fs::remove_file(&done_path);
        Ok(path)
    }

    pub fn expert_slots(&self) -> (usize, usize) {
        (self.parts.iter().map(|p| p.pool.slots()).sum(), self.cfg.layers * self.cfg.experts)
    }

    pub fn weight_bytes(&self) -> usize {
        self.parts.iter().map(|p| p.weight_bytes).sum()
    }

    /// Embedding rows looked up on the host (the table stays in the mapped file), in every lane,
    /// into the x of GPU `dev`.
    fn embed(&self, dev: usize, ids: &[u32]) -> Result<()> {
        self.parts[dev].s.x.upload(&self.embed_bytes(ids)?)
    }

    /// The first lanes of `ids`: each token's embedding in every lane, as f32 bytes.
    fn embed_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let (d, hc) = (self.cfg.d, self.cfg.hc);
        let mut bytes = Vec::with_capacity(ids.len() * d * hc * 4);
        for &id in ids {
            let id = id as usize;
            if id >= self.cfg.vocab {
                return Err(format!("token id {id} is outside the vocabulary ({})", self.cfg.vocab));
            }
            let mut row = Vec::with_capacity(d * 4);
            for p in self.nx.raw(self.embed_at + id * d * 2, d * 2).chunks_exact(2) {
                let bits = (u16::from_le_bytes([p[0], p[1]]) as u32) << 16;
                row.extend_from_slice(&bits.to_le_bytes());
            }
            for _ in 0..hc {
                bytes.extend_from_slice(&row);
            }
        }
        Ok(bytes)
    }

    /// Seed the pools' use counts from the hot list and queue its hottest experts to be read into
    /// the OS file cache while idle.
    fn start_hot(&mut self) {
        let Some(qf) = &self.q4file else { return };
        let keys = self.cfg.layers * self.cfg.experts;
        let Ok(b) = std::fs::read(&qf.hot) else { return };
        if b.len() != keys * 2 {
            return;
        }
        let uses: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        for p in &mut self.parts {
            p.pool.seed_uses(&uses);
        }
        warm_hottest(qf, &uses);
    }

    /// Write the pools' use counts as the hot list for the next run (a failed write only loses it).
    fn save_hot(&self) {
        let Some(qf) = &self.q4file else { return };
        let keys = self.cfg.layers * self.cfg.experts;
        let mut uses = vec![0u16; keys];
        for p in &self.parts {
            for (u, &v) in uses.iter_mut().zip(p.pool.uses()) {
                *u = (*u).max(v);
            }
        }
        if uses.iter().all(|&u| u == 0) {
            return;
        }
        let tmp = qf.hot.with_extension("experts-hot.tmp");
        let bytes: Vec<u8> = uses.iter().flat_map(|u| u.to_le_bytes()).collect();
        if std::fs::write(&tmp, bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &qf.hot);
        }
        // the next idle moment reads back what a game or another app pushed out of the file cache
        // (blocks still cached cost only a copy)
        warm_hottest(qf, &uses);
    }

    fn forward_chunk(&mut self, ids: &[u32]) -> Result<()> {
        self.hot_steps += 1;
        if self.hot_steps >= HOT_SAVE_STEPS {
            self.hot_steps = 0;
            self.save_hot();
        }
        let t = ids.len();
        self.toks.truncate(self.length);
        self.toks.extend_from_slice(ids);
        let mut cur = self.layers[0].dev;
        self.embed(cur, ids)?;
        let row = self.cfg.lanes() * 4;
        for li in 0..self.layers.len() {
            let dev = self.layers[li].dev;
            if dev != cur {
                // the lanes move on to the next GPU
                self.parts[dev].s.x.copy_range(0, &self.parts[cur].s.x, 0, t * row)?;
                cur = dev;
            }
            if let Some((l, out)) = &mut self.dump {
                if *l == li {
                    out.extend(self.parts[dev].s.x.download_f32(t * self.cfg.lanes())?);
                    break; // nothing above it is needed
                }
            }
            self.run_layer(li, t)?;
        }
        self.length += t;
        self.last_t = t;
        Ok(())
    }

    /// Layer `li` on the t tokens of lanes in its GPU's s.x (positions length .. length + t).
    fn run_layer(&mut self, li: usize, t: usize) -> Result<()> {
        self.stage(li)?;
        let dev = self.layers[li].dev;
        let row = self.cfg.lanes() * 4;
        if self.layers[li].ple.is_some() {
            self.ple(li, t)?;
        }
        {
            let (c, l, p) = (&self.cfg, &self.layers[li], &self.parts[dev]);
            p.hc_pre(c, &l.hc_attn, p.s.x.ptr, t)?;
        }
        self.mixer(li, t)?;
        {
            let (c, l, p) = (&self.cfg, &self.layers[li], &self.parts[dev]);
            p.hc_post(c, t)?;
            p.hc_pre(c, &l.hc_mlp, p.s.x.ptr, t)?;
        }
        self.moe(li, t)?;
        let p = &self.parts[dev];
        p.hc_post(&self.cfg, t)?;
        if let (Some(j), Some(buf)) = (self.tap_layers.iter().position(|&l| l == li), &p.tap_buf) {
            buf.copy_range(j * row, &p.s.x, (t - 1) * row, row)?;
        }
        Ok(())
    }

    /// The table rows of each new token (positions length .. length + t of self.toks).
    fn ple_rows(&self, p: &Ple, t: usize) -> Vec<usize> {
        let (n, eos) = (self.cfg.ngram, self.cfg.eos);
        let hpn = self.cfg.heads_per_ngram;
        let mut out = Vec::with_capacity(t * p.sizes.len());
        for pos in self.length..self.length + t {
            // the token s back within the current document (eos-delimited), else eos
            let back = |s: usize| -> i64 {
                if s > pos || (pos - s..pos).any(|q| self.toks[q] == eos) {
                    eos as i64
                } else {
                    self.toks[pos - s] as i64
                }
            };
            let sh: Vec<i64> = (0..n).map(back).collect();
            for order in 2..=n {
                let mut mixed = sh[0].wrapping_mul(p.mult[0]);
                for q in 1..order {
                    mixed ^= sh[q].wrapping_mul(p.mult[q]);
                }
                let a = (order - 2) * hpn;
                for j in a..a + hpn {
                    out.push((mixed.rem_euclid(p.sizes[j]) + p.offsets[j]) as usize);
                }
            }
        }
        out
    }

    fn ple(&self, li: usize, t: usize) -> Result<()> {
        let c = &self.cfg;
        let l = &self.layers[li];
        let part = &self.parts[l.dev];
        let (k, s) = (&part.k, &part.s);
        let p = l.ple.as_ref().unwrap();
        let rb = c.ple_row();
        let mut bytes = Vec::with_capacity(t * c.ple_dim);
        for r in self.ple_rows(p, t) {
            if r >= p.rows {
                return Err(format!("n-gram row {r} outside the table ({} rows)", p.rows));
            }
            bytes.extend_from_slice(self.nx.raw(p.table_at + r * rb, rb));
        }
        s.prow.upload(&bytes)?;
        let n = t * c.ple_dim;
        k.ple_embed.launch((((n + 255) / 256) as u32, 1), 256, &[Arg::Ptr(s.prow.ptr), Arg::Ptr(p.scale.ptr), Arg::Ptr(s.pe.ptr), Arg::I32(n as i32)])?;
        let lanes = c.lanes();
        part.mm(&p.key, s.pe.ptr, s.wl.ptr, t, false)?;
        part.gznorm(c, &p.norm_key, s.wl.ptr, s.wl.ptr, t)?;
        part.mm(&p.value, s.pe.ptr, s.ysub.ptr, t, false)?;
        part.gznorm(c, &p.norm_query, s.x.ptr, s.xn.ptr, t)?;
        k.ple_gate.launch(
            (t as u32, c.hc as u32),
            256,
            &[Arg::Ptr(s.wl.ptr), Arg::Ptr(s.xn.ptr), Arg::Ptr(s.ysub.ptr), Arg::Ptr(s.gv.ptr), Arg::I32(c.d as i32), Arg::I32(c.hc as i32)],
        )?;
        part.gznorm(c, &p.norm_conv, s.gv.ptr, s.wl.ptr, t)?;
        k.ple_conv.launch(
            (((lanes + 255) / 256) as u32, 1),
            256,
            &[
                Arg::Ptr(s.wl.ptr), Arg::Ptr(s.gv.ptr), Arg::Ptr(s.x.ptr), Arg::Ptr(p.conv.ptr),
                Arg::Ptr(part.rec.ple.at(p.slot * lanes * 9 * 4)), Arg::I32(t as i32), Arg::I32(lanes as i32),
            ],
        )
    }

    fn mixer(&self, li: usize, t: usize) -> Result<()> {
        let c = &self.cfg;
        let l = &self.layers[li];
        let part = &self.parts[l.dev];
        let (k, s) = (&part.k, &part.s);
        let pos0 = self.length;
        match &l.mixer {
            Mixer::Attn { q, k: wk, v, o, q_norm, k_norm, idx, slot } => {
                let kv_layer = c.kv_heads * self.cap * c.head_dim * 2;
                let (kc, vc) = (part.kc.at(slot * kv_layer), part.vc.at(slot * kv_layer));
                part.mm(q, s.h.ptr, s.big.ptr, t, false)?;
                part.mm(wk, s.h.ptr, s.kb.ptr, t, false)?;
                part.mm(v, s.h.ptr, s.vb.ptr, t, false)?;
                k.attn_prep.launch(
                    (t as u32, (c.heads + c.kv_heads) as u32),
                    c.head_dim as u32,
                    &[
                        Arg::Ptr(s.big.ptr), Arg::Ptr(s.kb.ptr), Arg::Ptr(s.vb.ptr), Arg::Ptr(q_norm.ptr), Arg::Ptr(k_norm.ptr),
                        Arg::Ptr(s.qn.ptr), Arg::Ptr(kc), Arg::Ptr(vc), Arg::I32(c.heads as i32), Arg::I32(c.kv_heads as i32),
                        Arg::I32(c.head_dim as i32), Arg::I32(c.rotary_dim as i32), Arg::I32(pos0 as i32), Arg::I32(self.cap as i32),
                        Arg::F32(c.eps), Arg::F32(c.rope_theta),
                    ],
                )?;
                let scale = Arg::F32(1.0 / (c.head_dim as f32).sqrt());
                let mut sparse = false;
                if let Some(ix) = idx {
                    let (ih, id, r) = (c.idx_heads, c.idx_dim, c.idx_ratio);
                    let nbmax = self.cap / r;
                    let kraw = part.kraw.as_ref().unwrap().at(slot * self.cap * id * 4);
                    let bk = part.bk.as_ref().unwrap().at(slot * nbmax * id * 4);
                    part.mm(&ix.qk, s.h.ptr, s.iqk.ptr, t, false)?;
                    k.idx_prep.launch(
                        (t as u32, (ih + 1) as u32),
                        id as u32,
                        &[
                            Arg::Ptr(s.iqk.ptr), Arg::Ptr(ix.q_norm.ptr), Arg::Ptr(s.qi.ptr), Arg::Ptr(kraw), Arg::I32(ih as i32),
                            Arg::I32(id as i32), Arg::I32(c.rotary_dim as i32), Arg::I32(pos0 as i32), Arg::F32(c.eps), Arg::F32(c.rope_theta),
                        ],
                    )?;
                    let (b0, b1) = (pos0 / r, (pos0 + t) / r);
                    if b1 > b0 {
                        k.idx_blocks.launch(
                            ((b1 - b0) as u32, 1),
                            id as u32,
                            &[
                                Arg::Ptr(kraw), Arg::Ptr(ix.k_norm.ptr), Arg::Ptr(bk), Arg::I32(b0 as i32), Arg::I32(r as i32), Arg::I32(id as i32),
                                Arg::I32(c.rotary_dim as i32), Arg::F32(c.eps), Arg::F32(c.rope_theta),
                            ],
                        )?;
                    }
                    if b1 > c.idx_top() {
                        k.idx_scores.launch(
                            (t as u32, ((b1 + 7) / 8) as u32),
                            256,
                            &[
                                Arg::Ptr(s.qi.ptr), Arg::Ptr(bk), Arg::Ptr(s.sc.ptr), Arg::I32(ih as i32), Arg::I32(id as i32),
                                Arg::I32(pos0 as i32), Arg::I32(r as i32), Arg::I32(nbmax as i32),
                            ],
                        )?;
                        k.idx_select.launch(
                            (t as u32, 1),
                            256,
                            &[
                                Arg::Ptr(s.sc.ptr), Arg::Ptr(s.blist.ptr), Arg::Ptr(s.nsel.ptr), Arg::I32(c.idx_top() as i32),
                                Arg::I32(r as i32), Arg::I32(pos0 as i32), Arg::I32(nbmax as i32),
                            ],
                        )?;
                        k.attention_list.launch(
                            (t as u32, c.heads as u32),
                            256,
                            &[
                                Arg::Ptr(s.qn.ptr), Arg::Ptr(kc), Arg::Ptr(vc), Arg::Ptr(s.big.ptr), Arg::Ptr(s.att.ptr),
                                Arg::I32(c.heads as i32), Arg::I32(c.kv_heads as i32), Arg::I32(pos0 as i32), Arg::I32(self.cap as i32), scale,
                                Arg::Ptr(s.blist.ptr), Arg::Ptr(s.nsel.ptr), Arg::I32(c.idx_top() as i32), Arg::I32(r as i32),
                            ],
                        )?;
                        sparse = true;
                    }
                }
                if !sparse {
                    k.attention_gated.launch(
                        (t as u32, c.heads as u32),
                        256,
                        &[
                            Arg::Ptr(s.qn.ptr), Arg::Ptr(kc), Arg::Ptr(vc), Arg::Ptr(s.big.ptr), Arg::Ptr(s.att.ptr),
                            Arg::I32(c.heads as i32), Arg::I32(c.kv_heads as i32), Arg::I32(pos0 as i32), Arg::I32(self.cap as i32), scale,
                        ],
                    )?;
                }
                part.mm(o, s.att.ptr, s.ysub.ptr, t, false)
            }
            Mixer::Delta { in_qkv, in_z, in_a, in_b, conv, a_log, dt_bias, norm, out, slot } => {
                let cd = c.conv_dim();
                let conv_state = part.rec.conv.at(slot * cd * 3 * 4);
                let state = part.rec.state.at(slot * c.hv * c.dk * c.dv * 4);
                part.mm(in_qkv, s.h.ptr, s.big.ptr, t, false)?;
                part.mm(in_z, s.h.ptr, s.z.ptr, t, false)?;
                k.delta_gates.launch(
                    (t as u32, 1),
                    256,
                    &[
                        Arg::Ptr(s.h.ptr), Arg::Ptr(in_b.ptr), Arg::Ptr(in_a.ptr), Arg::Ptr(a_log.ptr), Arg::Ptr(dt_bias.ptr),
                        Arg::Ptr(s.gb.ptr), Arg::I32(c.d as i32), Arg::I32(c.hv as i32),
                    ],
                )?;
                k.conv_silu.launch(
                    (((cd + 255) / 256) as u32, 1),
                    256,
                    &[Arg::Ptr(s.big.ptr), Arg::Ptr(s.y.ptr), Arg::Ptr(conv.ptr), Arg::Ptr(conv_state), Arg::I32(t as i32), Arg::I32(cd as i32)],
                )?;
                k.l2norm_heads.launch(
                    (t as u32, (2 * c.hk) as u32),
                    c.dk as u32,
                    &[Arg::Ptr(s.y.ptr), Arg::I32(cd as i32), Arg::I32(c.hk as i32), Arg::I32(c.dk as i32), Arg::F32(1.0 / (c.dk as f32).sqrt()), Arg::F32(1e-6)],
                )?;
                k.delta_scan.launch(
                    (c.hv as u32, 1),
                    512,
                    &[
                        Arg::Ptr(s.y.ptr), Arg::Ptr(s.gb.ptr), Arg::Ptr(state), Arg::Ptr(s.o.ptr), Arg::I32(t as i32), Arg::I32(cd as i32),
                        Arg::I32(c.hk as i32), Arg::I32(c.hv as i32),
                    ],
                )?;
                k.gated_norm.launch(
                    (t as u32, c.hv as u32),
                    c.dv as u32,
                    &[
                        Arg::Ptr(s.o.ptr), Arg::Ptr(s.z.ptr), Arg::Ptr(norm.ptr), Arg::Ptr(s.att.ptr), Arg::I32(c.hv as i32), Arg::I32(c.dv as i32),
                        Arg::F32(c.eps), Arg::I32(c.sigmoid_gate as i32),
                    ],
                )?;
                part.mm(out, s.att.ptr, s.ysub.ptr, t, false)
            }
        }
    }

    /// ysub = experts + gated shared expert + cores, all of h.
    fn moe(&mut self, li: usize, t: usize) -> Result<()> {
        let (d, kk, e_n, ih) = (self.cfg.d, self.cfg.top_k, self.cfg.experts, self.cfg.expert_hidden);
        let l = &self.layers[li];
        let part = &mut self.parts[l.dev];
        part.s.ysub.zero_bytes(t * d * 4)?;
        let rb = ((e_n + 31) / 32 * 32).max(64) as u32;
        part.k.router.launch(
            (t as u32, 1),
            rb,
            &[Arg::Ptr(part.s.h.ptr), Arg::Ptr(l.router.ptr), Arg::Ptr(part.s.idx.ptr), Arg::Ptr(part.s.wts.ptr), Arg::I32(d as i32), Arg::I32(e_n as i32), Arg::I32(kk as i32)],
        )?;
        // the next layer's router on this h guesses its experts, to read ahead while the GPU works
        let ahead = li + 1 < self.layers.len()
            && self.layers[li + 1].dev == l.dev
            && t <= PREFETCH_TOKENS
            && self.q4file.as_ref().is_some_and(|q| q.pre.is_some());
        if ahead {
            part.k.router.launch(
                (t as u32, 1),
                rb,
                &[Arg::Ptr(part.s.h.ptr), Arg::Ptr(self.layers[li + 1].router.ptr), Arg::Ptr(part.s.idx2.ptr), Arg::Ptr(part.s.wts2.ptr), Arg::I32(d as i32), Arg::I32(e_n as i32), Arg::I32(kk as i32)],
            )?;
        }
        // group the picks by expert (this download is the one host sync per layer)
        let idx = part.s.idx.download_i32(t * kk)?;
        let guess = if ahead { Some((part.s.idx2.download_i32(t * kk)?, part.s.wts2.download_f32(t * kk)?)) } else { None };
        // reads ahead not started yet are dropped: this layer's own reads go first
        if let Some(pre) = self.q4file.as_ref().and_then(|q| q.pre.as_ref()) {
            pre.settle();
        }
        if let Some(&e) = idx.iter().find(|&&e| e < 0 || e as usize >= e_n) {
            return Err(format!("router returned expert {e}"));
        }
        // picks to leave out: light picks of experts that would have to be read from disk
        let mut keep = vec![true; t * kk];
        let skip_mass = if self.light { self.skip_think.max(self.skip_mass) } else { self.skip_mass };
        if skip_mass > 0.0 {
            let mut w = part.s.wts.download_f32(t * kk)?;
            // Each token may lose its lightest picks of experts not in VRAM, up to skip_mass of its
            // routing weight in all (its first, heaviest pick always stays). An expert is skipped
            // only when every token that picked it may lose it.
            let mut light = vec![true; e_n];
            for tok in 0..t {
                let r = tok * kk..(tok + 1) * kk;
                let near = |e: usize| part.pool.has(li, e) || self.ram.as_ref().is_some_and(|m| m.has(li * e_n + e));
                let mut miss: Vec<usize> = r.clone().skip(1).filter(|&p| !near(idx[p] as usize)).collect();
                miss.sort_by(|&a, &b| w[a].total_cmp(&w[b]));
                let (mut mass, mut may) = (0.0, vec![false; kk]);
                for p in miss {
                    if mass + w[p] > skip_mass {
                        break;
                    }
                    mass += w[p];
                    may[p - r.start] = true;
                }
                for p in r.clone() {
                    if !may[p - r.start] {
                        light[idx[p] as usize] = false;
                    }
                }
            }
            let mut dropped = false;
            for (p, &e) in idx.iter().enumerate() {
                if light[e as usize] {
                    keep[p] = false;
                    dropped = true;
                    self.skipped.0 += 1;
                }
            }
            if dropped {
                let mut saved: Vec<usize> = idx.iter().map(|&e| e as usize).filter(|&e| light[e]).collect();
                saved.sort_unstable();
                saved.dedup();
                self.skipped.1 += saved.len() as u64;
                for tok in 0..t {
                    let r = tok * kk..(tok + 1) * kk;
                    let sum: f32 = r.clone().filter(|&p| keep[p]).map(|p| w[p]).sum();
                    for p in r {
                        w[p] = if keep[p] { w[p] / sum } else { 0.0 };
                    }
                }
                part.s.wts.upload(&w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                part.s.ypick.zero_bytes(t * kk * d * 4)?;
            }
        }
        let mut count = vec![0usize; e_n];
        for (p, &e) in idx.iter().enumerate() {
            if keep[p] {
                count[e as usize] += 1;
            }
        }
        let uniq: Vec<usize> = (0..e_n).filter(|&e| count[e] > 0).collect();
        let mut cursor = vec![0usize; e_n];
        let mut ustart = Vec::with_capacity(uniq.len() + 1);
        let mut acc = 0;
        for &e in &uniq {
            ustart.push(acc as i32);
            cursor[e] = acc;
            acc += count[e];
        }
        ustart.push(acc as i32);
        let mut ptok = vec![0i32; t * kk];
        let mut psrc = vec![0i32; t * kk];
        for (p, &e) in idx.iter().enumerate() {
            if !keep[p] {
                continue;
            }
            let pos = cursor[e as usize];
            cursor[e as usize] += 1;
            ptok[pos] = (p / kk) as i32;
            psrc[pos] = p as i32;
        }
        let (at, eb) = (l.experts_at, self.lay.bytes);
        let (o_us, o_pt, o_ps, _) = meta_layout(&self.cfg, self.chunk.max(t));
        // the expert pointers go in per batch; the pick lists once
        let mut meta = vec![0u8; o_ps + t * kk * 4];
        meta[o_us..o_us + ustart.len() * 4].copy_from_slice(&bytes_i32(&ustart));
        meta[o_pt..o_pt + t * kk * 4].copy_from_slice(&bytes_i32(&ptok));
        meta[o_ps..o_ps + t * kk * 4].copy_from_slice(&bytes_i32(&psrc));
        part.s.meta.upload(&meta)?;

        // Batches of experts that fit in the pool. Every copy and kernel runs in order on one
        // stream, so a batch's experts are not overwritten before the kernels using them finish.
        let (h, act, wts, yp) = (Arg::Ptr(part.s.h.ptr), Arg::Ptr(part.s.act.ptr), Arg::Ptr(part.s.wts.ptr), Arg::Ptr(part.s.ypick.ptr));
        let (di, ii) = (Arg::I32(d as i32), Arg::I32(ih as i32));
        let (pt, ps) = (Arg::Ptr(part.s.meta.at(o_pt)), Arg::Ptr(part.s.meta.at(o_ps)));
        let streaming = t > self.stream_tokens;
        for (bi, chunk) in uniq.chunks(part.batch).enumerate() {
            let first = bi * part.batch;
            let (ptrs, loads) = part.pool.assign(li, chunk, streaming)?;
            // what was pushed out goes to the RAM tier; what it holds comes back from there
            let loads = match self.ram.as_mut().map_or(Ok(loads.clone()), |m| m.swap(li * e_n, &part.pool, &loads)) {
                Ok(disk) => disk,
                Err(err) => {
                    part.pool.forget(&loads);
                    return Err(err);
                }
            };
            // the experts not in VRAM or RAM, read in parallel and copied in as each arrives
            let jobs: Vec<(usize, usize)> = loads.iter().map(|&(slot, e)| (slot, at + e * eb)).collect();
            let (pool, stage, kq) = (&part.pool, &part.stage, &part.k.q4_from_f8);
            let (lay, q4) = (&self.lay, &self.q4);
            let sink = |slot: usize, bytes: &[u8]| match (q4, stage) {
                // in stream order: the next read into the stage waits for this pack
                (Some(q), Some(st)) => {
                    st.upload(bytes)?;
                    let groups = 3 * ih * d / 64;
                    let i = |v: usize| Arg::I32(v as i32);
                    kq.launch(
                        (((groups + 7) / 8) as u32, 1),
                        256,
                        &[Arg::Ptr(st.ptr), Arg::Ptr(pool.slot_ptr(slot)), i(d), i(ih), i(lay.gu_s), i(lay.dn), i(lay.dn_s), i(q.s4g), i(q.m4g), i(q.q4d), i(q.s4d), i(q.m4d)],
                    )
                }
                _ => pool.upload(slot, bytes),
            };
            let t_read = std::time::Instant::now();
            let read = match &mut self.q4file {
                // the 4-bit copies straight from the 4-bit expert file
                Some(qf) => {
                    let (n, sz) = (qf.experts, qf.slot);
                    let jobs: Vec<(usize, usize)> = loads.iter().map(|&(slot, e)| (slot, Q4_HEAD + (li * n + e) * sz)).collect();
                    self.guess_hits.1 += jobs.iter().filter(|j| self.guessed.contains(&(j.1 as u64))).count() as u64;
                    qf.reader.load(&jobs, sz, |slot, bytes| pool.upload(slot, bytes))
                }
                None => self.reader.load(&jobs, eb, sink),
            };
            self.read_secs += t_read.elapsed().as_secs_f64();
            if let Err(err) = read {
                part.pool.forget(&loads);
                return Err(err);
            }
            part.s.meta.upload_at(first * 8, &ptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>())?;
            let (k, lay) = (&part.k, &self.lay);
            let (eptr, us) = (Arg::Ptr(part.s.meta.at(first * 8)), Arg::Ptr(part.s.meta.at(o_us + first * 4)));
            let nu = chunk.len() as u32;
            if let Some(q) = &self.q4 {
                let i = |v: usize| Arg::I32(v as i32);
                if t <= self.gemv_tokens {
                    k.gu4_gemv.launch((((ih + 7) / 8) as u32, nu), 256, &[h, eptr, us, pt, act, di, ii, i(q.s4g), i(q.m4g)])?;
                    k.dn4_gemv.launch((((d + 7) / 8) as u32, nu), 256, &[act, eptr, us, ps, wts, yp, di, ii, i(q.q4d), i(q.s4d), i(q.m4d)])?;
                } else {
                    k.gu4_gemm.launch(((ih / 32) as u32, nu), 256, &[h, eptr, us, pt, act, di, ii, i(q.s4g), i(q.m4g)])?;
                    k.dn4_gemm.launch(((d / 64) as u32, nu), 256, &[act, eptr, us, ps, wts, yp, di, ii, i(q.q4d), i(q.s4d), i(q.m4d)])?;
                }
            } else if t <= self.gemv_tokens {
                k.gu_gemv.launch((((ih + 7) / 8) as u32, nu), 256, &[h, eptr, us, pt, act, di, ii, Arg::I32(lay.gu_s as i32)])?;
                k.dn_gemv.launch((((d + 7) / 8) as u32, nu), 256, &[act, eptr, us, ps, wts, yp, di, ii, Arg::I32(lay.dn as i32), Arg::I32(lay.dn_s as i32)])?;
            } else {
                k.gu_gemm.launch(((ih / 32) as u32, nu), 256, &[h, eptr, us, pt, act, di, ii, Arg::I32(lay.gu_s as i32)])?;
                k.dn_gemm.launch(((d / 64) as u32, nu), 256, &[act, eptr, us, ps, wts, yp, di, ii, Arg::I32(lay.dn as i32), Arg::I32(lay.dn_s as i32)])?;
            }
        }
        if let (Some((gi, gw)), Some(qf)) = (&guess, &self.q4file) {
            if let Some(pre) = &qf.pre {
                // the guessed picks the skip rule would keep, of experts not in VRAM (or the RAM tier)
                let nl = li + 1;
                let mut offs = vec![];
                for tok in 0..t {
                    let mut r: Vec<usize> = (tok * kk..(tok + 1) * kk).collect();
                    r.sort_by(|&a, &b| gw[b].total_cmp(&gw[a]));
                    let mut light = 0.0;
                    let mut keep_n = kk;
                    while keep_n > 1 && light + gw[r[keep_n - 1]] <= skip_mass {
                        light += gw[r[keep_n - 1]];
                        keep_n -= 1;
                    }
                    let top = env_usize("AEGIST_PREFETCH_TOP").unwrap_or(PREFETCH_TOP);
                    for &p in &r[..keep_n.min(top)] {
                        let e = gi[p] as usize;
                        if e < e_n && !part.pool.has(nl, e) && !self.ram.as_ref().is_some_and(|m| m.has(nl * e_n + e)) {
                            let off = (Q4_HEAD + (nl * qf.experts + e) * qf.slot) as u64;
                            if !offs.contains(&off) {
                                offs.push(off);
                            }
                        }
                    }
                }
                self.guess_hits.0 += offs.len() as u64;
                self.guessed = offs.clone();
                pre.want(offs);
            }
        }
        let (k, s) = (&part.k, &part.s);
        k.moe_reduce.launch((t as u32, ((d + 255) / 256) as u32), 256, &[yp, Arg::Ptr(s.ysub.ptr), di, Arg::I32(kk as i32)])?;

        // shared expert, gated
        let sh = self.cfg.shared_hidden;
        part.mm(&l.sh_gate, s.h.ptr, s.sh_g.ptr, t, false)?;
        part.mm(&l.sh_up, s.h.ptr, s.sh_u.ptr, t, false)?;
        let n = (t * sh) as i32;
        k.silu_mul.launch((((n + 255) / 256) as u32, 1), 256, &[Arg::Ptr(s.sh_g.ptr), Arg::Ptr(s.sh_u.ptr), Arg::Ptr(s.sh_g.ptr), Arg::I32(n)])?;
        part.mm(&l.sh_down, s.sh_g.ptr, s.sh_out.ptr, t, false)?;
        k.shared_gate_add.launch((t as u32, 1), 256, &[Arg::Ptr(s.ysub.ptr), Arg::Ptr(s.sh_out.ptr), h, Arg::Ptr(l.sh_gw.ptr), di])?;

        // the core engine and the switchable cores
        let c = &self.cfg;
        let zero = Arg::Ptr(0);
        let (cg, cu, cdn) = match &l.cores {
            Some((g, u, dn)) => (Arg::Ptr(g.ptr), Arg::Ptr(u.ptr), Arg::Ptr(dn.ptr)),
            None => (zero, zero, zero),
        };
        k.cores_apply.launch(
            (t as u32, 1),
            256,
            &[
                h, Arg::Ptr(l.key1.ptr), Arg::Ptr(l.key2.ptr), Arg::Ptr(l.live.ptr), cg, cu, cdn, Arg::Ptr(s.ysub.ptr),
                Arg::Ptr(part.counts.at(li * c.n_cores() * 4)), di, Arg::I32(c.n1 as i32), Arg::I32(c.n2 as i32),
                Arg::I32(c.cores_active as i32), Arg::I32(c.rank as i32),
            ],
        )
    }

    /// Every token's hidden lanes going into `layer` ([tokens][hc * d] fp32), from an empty
    /// context: what learning time needs to train the layers from `layer` up without running the
    /// ones below. The engine is left empty.
    pub fn lanes_before(&mut self, ids: &[u32], layer: usize) -> Result<Vec<f32>> {
        if layer == 0 || layer >= self.cfg.layers {
            return Err(format!("layer {layer}: must be 1 to {}", self.cfg.layers - 1));
        }
        if ids.len() > self.cap {
            return Err(format!("{} tokens > context {}", ids.len(), self.cap));
        }
        self.crop(0);
        self.dump = Some((layer, Vec::with_capacity(ids.len() * self.cfg.lanes())));
        let mut r = Ok(());
        for c in ids.chunks(self.chunk) {
            r = self.forward_chunk(c);
            if r.is_err() {
                break;
            }
        }
        let out = self.dump.take().map(|d| d.1).unwrap_or_default();
        self.crop(0);
        r.map(|_| out)
    }

    /// As lanes_before for many prompts, layer by layer: every prompt goes through layer 0, then
    /// every prompt through layer 1, and so on, with the lanes kept in host memory between
    /// layers. A layer's experts are read from disk once for all the prompts instead of once per
    /// prompt (as AirLLM streams layers). The results are the same as lanes_before's.
    pub fn lanes_before_many(&mut self, prompts: &[Vec<u32>], layer: usize, progress: impl FnMut(usize)) -> Result<Vec<Vec<f32>>> {
        if layer == 0 || layer >= self.cfg.layers {
            return Err(format!("layer {layer}: must be 1 to {}", self.cfg.layers - 1));
        }
        let xs = self.lanes_many(prompts, layer, progress)?;
        Ok(xs.into_iter().map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()).collect())
    }

    pub fn layers(&self) -> usize {
        self.cfg.layers
    }

    /// Each item's summed loss (nats) on ids[start..] and how many tokens that is, as `score`
    /// would get feeding the tokens one at a time, but with every item read layer by layer as in
    /// lanes_before_many (each layer's experts once for all the items), then the output head on
    /// each position. The engine is left empty.
    pub fn score_many(&mut self, items: &[(Vec<u32>, usize)], progress: impl FnMut(usize)) -> Result<Vec<(f64, usize)>> {
        // the last token is only ever a target
        let prompts: Vec<Vec<u32>> = items.iter().map(|(ids, _)| ids[..ids.len().saturating_sub(1)].to_vec()).collect();
        let lanes = self.lanes_many(&prompts, self.cfg.layers, progress)?;
        let (v, row) = (self.cfg.vocab, self.cfg.lanes() * 4);
        // a streamed lm_head goes one position at a time
        let ht = if self.lm_head.lazy { 1 } else { 16.min(self.chunk) };
        let p = self.parts.last().unwrap();
        p.gpu.bind()?;
        let lg = Buf::alloc(ht * v * 4)?;
        let mut out = vec![];
        for ((ids, start), x) in items.iter().zip(&lanes) {
            let s = (*start).clamp(1, ids.len().max(1));
            let mut nll = 0f64;
            // lanes row j predicts ids[j + 1]
            let mut j = s - 1;
            while j + 1 < ids.len() {
                let t = ht.min(ids.len() - 1 - j);
                p.s.x.upload(&x[j * row..(j + t) * row])?;
                p.hc_pre(&self.cfg, &self.hc_out, p.s.x.ptr, t)?;
                match &mut self.stream {
                    Some(st) if self.lm_head.lazy => streamed_head(p, st, &self.lm_head, &self.nx, lg.ptr)?,
                    _ => p.mm(&self.lm_head, p.s.h.ptr, lg.ptr, t, false)?,
                }
                let l = lg.download_f32(t * v)?;
                let rows: Vec<f64> = std::thread::scope(|sc| {
                    let hs: Vec<_> = (0..t)
                        .map(|r| {
                            let (lr, target) = (&l[r * v..(r + 1) * v], ids[j + r + 1] as usize);
                            sc.spawn(move || {
                                let m = lr.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                                let z: f64 = lr.iter().map(|&a| (a as f64 - m).exp()).sum();
                                m + z.ln() - lr[target] as f64
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                nll += rows.iter().sum::<f64>();
                j += t;
            }
            out.push((nll, ids.len() - s));
        }
        Ok(out)
    }

    /// Every prompt's lanes (fp32 bytes) after layers 0..layer, layer by layer; the engine is left empty.
    fn lanes_many(&mut self, prompts: &[Vec<u32>], layer: usize, mut progress: impl FnMut(usize)) -> Result<Vec<Vec<u8>>> {
        if let Some(p) = prompts.iter().find(|p| p.len() > self.cap || p.is_empty()) {
            return Err(format!("a prompt of {} tokens (context {})", p.len(), self.cap));
        }
        let row = self.cfg.lanes() * 4;
        let mut xs: Vec<Vec<u8>> = prompts.iter().map(|p| self.embed_bytes(p)).collect::<Result<_>>()?;
        // every expert of a layer fits in the pool: keep them all while the prompts pass through
        let stream = std::mem::replace(&mut self.stream_tokens, usize::MAX);
        let mut run = || -> Result<()> {
            for li in 0..layer {
                let dev = self.layers[li].dev;
                for (ids, x) in prompts.iter().zip(xs.iter_mut()) {
                    self.parts[dev].rec.zero()?;
                    self.toks = ids.clone();
                    self.length = 0;
                    for c in (0..ids.len()).step_by(self.chunk) {
                        let t = self.chunk.min(ids.len() - c);
                        let part = &self.parts[dev];
                        part.s.x.upload(&x[c * row..(c + t) * row])?;
                        self.run_layer(li, t)?;
                        x[c * row..(c + t) * row].copy_from_slice(&self.parts[dev].s.x.download_at(0, t * row)?);
                        self.length += t;
                    }
                }
                progress(li + 1);
            }
            Ok(())
        };
        let r = run();
        self.stream_tokens = stream;
        self.length = 0;
        self.toks.clear();
        self.snap_pos = vec![None; SNAPSHOTS];
        for p in &self.parts {
            p.rec.zero()?;
        }
        r?;
        Ok(xs)
    }

    /// forward_chunk over every chunk of `ids`, but layer by layer: all the chunks through layer
    /// 0, then all through layer 1, ..., with the lanes in host memory between layers. Each layer's
    /// experts are then read once, not once per chunk (a 1,000-token prompt is 2 chunks). The same
    /// kernels run on the same inputs, so with nothing skipped the result is bit-identical.
    fn feed_layers(&mut self, ids: &[u32]) -> Result<()> {
        let (l0, n, row, ch) = (self.length, ids.len(), self.cfg.lanes() * 4, self.chunk);
        self.toks.truncate(l0);
        self.toks.extend_from_slice(ids);
        let mut x = self.embed_bytes(ids)?;
        let stream = std::mem::replace(&mut self.stream_tokens, usize::MAX);
        let mut run = || -> Result<()> {
            let nl = self.layers.len();
            for li in 0..nl {
                let dev = self.layers[li].dev;
                for c in (0..n).step_by(ch) {
                    let t = ch.min(n - c);
                    self.length = l0 + c;
                    self.parts[dev].s.x.upload(&x[c * row..(c + t) * row])?;
                    self.run_layer(li, t)?;
                    // the last chunk out of the last layer stays in s.x for head()
                    if li + 1 < nl || c + t < n {
                        x[c * row..(c + t) * row].copy_from_slice(&self.parts[dev].s.x.download_at(0, t * row)?);
                    }
                }
            }
            Ok(())
        };
        let r = run();
        self.stream_tokens = stream;
        match r {
            Ok(()) => {
                self.length = l0 + n;
                self.last_t = n - (n - 1) / ch * ch;
                Ok(())
            }
            Err(e) => {
                // some layers ran ahead: start over from an empty context
                self.length = 0;
                self.toks.clear();
                self.snap_pos = vec![None; SNAPSHOTS];
                for p in &self.parts {
                    let _ = p.rec.zero();
                }
                Err(e)
            }
        }
    }

    /// Layer streaming: copy layer `li`'s streamed matrices into its slot, unless they are there.
    fn stage(&mut self, li: usize) -> Result<()> {
        let Some(st) = &mut self.stream else { return Ok(()) };
        let l = &self.layers[li];
        let slot = st.slot_of[li];
        if st.loaded[l.dev][slot] == Some(li) || !lins(l).iter().any(|w| w.lazy) {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        self.parts[l.dev].gpu.bind()?;
        for w in lins(l).into_iter().filter(|w| w.lazy) {
            let b = self.nx.bytes(&w.src)?;
            w.w.upload(b)?;
            st.bytes += b.len() as u64;
        }
        st.loaded[l.dev][slot] = Some(li);
        st.secs += t0.elapsed().as_secs_f64();
        Ok(())
    }

    fn head(&mut self) -> Result<Vec<f32>> {
        let p = self.parts.last().unwrap();
        let last = p.s.x.at((self.last_t - 1) * self.cfg.lanes() * 4);
        p.hc_pre(&self.cfg, &self.hc_out, last, 1)?;
        match &mut self.stream {
            Some(st) if self.lm_head.lazy => streamed_head(p, st, &self.lm_head, &self.nx, self.logits.ptr)?,
            _ => p.mm(&self.lm_head, p.s.h.ptr, self.logits.ptr, 1, false)?,
        }
        self.logits.download_f32(self.cfg.vocab)
    }
}

const STATE_MAGIC: &[u8; 8] = b"AEGSTATE";

/// The default $AEGIST_SKIP_MISS. On 450 tokens of text: 0.1 changed the loss by -0.4% (within
/// noise) and read 27% fewer experts (1.4x faster from D:); 0.2: +1.6%, 2.1x; 0.3: +2.4%, 2.9x.
const SKIP_MISS: f32 = 0.1;
/// The default $AEGIST_SKIP_THINK: the skip mass while the brain thinks (Lm::set_light). Thinking
/// is never shown as the answer, so it trades a little accuracy (+2.4% loss at 0.3 above) for
/// about twice the speed of the answer's setting; the answer itself keeps SKIP_MISS.
const SKIP_THINK: f32 = 0.3;

/// $AEGIST_EXPERT_BITS: 4 (default) packs each expert read into a 4-bit copy in VRAM (q4_from_f8),
/// so nearly twice as many fit and fewer are read from disk; 8 keeps the FP8 experts as they are.
/// What learning time and its checks read is computed with 8 (main.rs sets it for those commands).
fn expert_bits() -> Result<u32> {
    match std::env::var("AEGIST_EXPERT_BITS").ok().as_deref().map(str::trim) {
        Some("8") => Ok(8),
        None | Some("") | Some("4") => Ok(4),
        Some(v) => Err(format!("AEGIST_EXPERT_BITS={v}: expected 8 or 4")),
    }
}

impl Drop for NextEngine {
    fn drop(&mut self) {
        self.save_hot();
    }
}

impl NextEngine {
    /// What the first `n` tokens left behind, layer by layer, as (buffer, offset, bytes): the
    /// attention layers' keys and values (and indexer keys), the recurrent layers' states. Kept
    /// per layer, not per GPU, so a state saved on one GPU loads on two.
    fn state_pieces(&self, n: usize) -> Vec<(&Buf, usize, usize)> {
        let c = &self.cfg;
        let mut v = vec![];
        for l in &self.layers {
            let part = &self.parts[l.dev];
            match &l.mixer {
                Mixer::Attn { idx, slot, .. } => {
                    let row = c.head_dim * 2;
                    for h in 0..c.kv_heads {
                        let at = (slot * c.kv_heads + h) * self.cap * row;
                        v.push((&part.kc, at, n * row));
                        v.push((&part.vc, at, n * row));
                    }
                    if idx.is_some() {
                        let (id, nbmax) = (c.idx_dim, self.cap / c.idx_ratio);
                        v.push((part.kraw.as_ref().unwrap(), slot * self.cap * id * 4, n * id * 4));
                        v.push((part.bk.as_ref().unwrap(), slot * nbmax * id * 4, n.div_ceil(c.idx_ratio).min(nbmax) * id * 4));
                    }
                }
                Mixer::Delta { slot, .. } => {
                    let (a, b) = (c.conv_dim() * 3 * 4, c.hv * c.dk * c.dv * 4);
                    v.push((&part.rec.conv, slot * a, a));
                    v.push((&part.rec.state, slot * b, b));
                }
            }
            if let Some(p) = &l.ple {
                let a = c.lanes() * 9 * 4;
                v.push((&part.rec.ple, p.slot * a, a));
            }
        }
        v
    }

    fn read_state(&mut self, path: &std::path::Path) -> Result<Vec<u32>> {
        use std::io::Read;
        let err = |e: std::io::Error| format!("{}: {e}", path.display());
        let mut f = std::io::BufReader::new(std::fs::File::open(path).map_err(err)?);
        let mut magic = [0u8; 16];
        f.read_exact(&mut magic).map_err(err)?;
        if &magic[..8] != STATE_MAGIC {
            return Err(format!("{}: not a saved state", path.display()));
        }
        let mut head = vec![0u8; u64::from_le_bytes(magic[8..].try_into().unwrap()) as usize];
        f.read_exact(&mut head).map_err(err)?;
        let head: Value = serde_json::from_slice(&head).map_err(|e| format!("{}: {e}", path.display()))?;
        if head["key"].as_str() != self.state_key().as_deref() {
            return Err(format!("{}: saved by another brain", path.display()));
        }
        let toks: Vec<u32> = head["toks"].as_array().map(|a| a.iter().filter_map(|t| t.as_u64()).map(|t| t as u32).collect()).unwrap_or_default();
        let n = toks.len();
        if n == 0 || n >= self.cap {
            return Err(format!("{}: {n} tokens do not fit a context of {}", path.display(), self.cap));
        }
        let mut buf = vec![];
        for (b, at, len) in self.state_pieces(n) {
            buf.resize(len, 0);
            f.read_exact(&mut buf).map_err(err)?;
            b.upload_at(at, &buf)?;
        }
        self.length = n;
        self.toks = toks.clone();
        self.snap_pos = vec![None; SNAPSHOTS];
        self.snapshot()?;
        Ok(toks)
    }
}

impl Lm for NextEngine {
    fn state_key(&self) -> Option<String> {
        let c = &self.cfg;
        // a context read with 4-bit experts is kept apart from one read with the FP8 ones
        let bits = if self.q4.is_some() { "-q4" } else { "" };
        self.nx.fingerprint().map(|f| format!("{f}-{}x{}x{}{bits}", c.layers, c.kv_heads, c.head_dim))
    }

    fn save_state(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let n = self.length;
        let head = serde_json::json!({"key": self.state_key(), "toks": &self.toks[..n]}).to_string();
        let tmp = path.with_extension("tmp");
        let err = |e: std::io::Error| format!("{}: {e}", tmp.display());
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp).map_err(err)?);
        f.write_all(STATE_MAGIC).map_err(err)?;
        f.write_all(&(head.len() as u64).to_le_bytes()).map_err(err)?;
        f.write_all(head.as_bytes()).map_err(err)?;
        for (b, at, len) in self.state_pieces(n) {
            f.write_all(&b.download_at(at, len)?).map_err(err)?;
        }
        f.into_inner().map_err(|e| format!("{}: {e}", tmp.display()))?.sync_all().map_err(err)?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn load_state(&mut self, path: &std::path::Path) -> Result<Vec<u32>> {
        let r = self.read_state(path);
        if r.is_err() {
            for p in &self.parts {
                p.rec.zero()?;
            }
            self.length = 0;
            self.toks.clear();
            self.snap_pos = vec![None; SNAPSHOTS];
        }
        r
    }

    fn feed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        assert!(!ids.is_empty());
        if self.length + ids.len() > self.cap {
            return Err(format!("context full: {} + {} tokens > {}", self.length, ids.len(), self.cap));
        }
        if ids.len() > self.chunk && self.dump.is_none() && self.prefill_layers {
            self.feed_layers(ids)?;
        } else {
            for c in ids.chunks(self.chunk) {
                self.forward_chunk(c)?;
            }
        }
        self.head()
    }

    fn crop(&mut self, n: usize) -> usize {
        if n >= self.length {
            return self.length;
        }
        let best = (0..SNAPSHOTS).filter(|&i| self.snap_pos[i].map_or(false, |p| p <= n)).max_by_key(|&i| self.snap_pos[i]);
        let restored = match best {
            Some(i) => self.parts.iter().try_for_each(|p| p.rec.copy_from(&p.snaps[i])).map(|_| self.snap_pos[i].unwrap()),
            None => Err(String::new()),
        };
        match restored {
            Ok(p) => {
                self.length = p;
                for s in self.snap_pos.iter_mut() {
                    if s.map_or(false, |q| q > p) {
                        *s = None;
                    }
                }
            }
            Err(_) => {
                for p in &self.parts {
                    let _ = p.rec.zero();
                }
                self.length = 0;
                self.snap_pos = vec![None; SNAPSHOTS];
            }
        }
        self.toks.truncate(self.length);
        self.length
    }

    fn snapshot(&mut self) -> Result<()> {
        if self.snap_pos.contains(&Some(self.length)) {
            return Ok(());
        }
        let i = (0..SNAPSHOTS).min_by_key(|&i| self.snap_pos[i].map_or(-1, |p| p as i64)).unwrap();
        for p in &self.parts {
            p.snaps[i].copy_from(&p.rec)?;
        }
        self.snap_pos[i] = Some(self.length);
        Ok(())
    }

    fn length(&self) -> usize {
        self.length
    }

    fn cap(&self) -> usize {
        self.cap
    }

    fn set_taps(&mut self, layers: &[usize]) -> Result<()> {
        if let Some(&l) = layers.iter().find(|&&l| l >= self.cfg.layers) {
            return Err(format!("tap layer {l} out of range (the brain has {} layers)", self.cfg.layers));
        }
        for p in &mut self.parts {
            p.tap_buf = None;
            if !layers.is_empty() {
                p.gpu.bind()?;
                p.tap_buf = Some(Buf::zeroed(layers.len() * self.cfg.lanes() * 4)?);
            }
        }
        self.tap_layers = layers.to_vec();
        Ok(())
    }

    /// The mean of the lanes after each tapped layer (each GPU fills the rows of its own layers).
    fn taps(&self) -> Result<Vec<f32>> {
        let (d, hc) = (self.cfg.d, self.cfg.hc);
        if self.tap_layers.is_empty() {
            return Ok(vec![]);
        }
        let n = self.tap_layers.len() * hc * d;
        let mut all = vec![0f32; n];
        for p in &self.parts {
            if let Some(b) = &p.tap_buf {
                for (a, v) in all.iter_mut().zip(b.download_f32(n)?) {
                    *a += v;
                }
            }
        }
        Ok(all.chunks_exact(hc * d).flat_map(|r| (0..d).map(move |i| (0..hc).map(|l| r[l * d + i]).sum::<f32>() / hc as f32)).collect())
    }

    fn core_counts(&self) -> Result<Vec<i32>> {
        let n = self.cfg.layers * self.cfg.n_cores();
        let mut all = vec![0i32; n];
        for p in &self.parts {
            for (a, v) in all.iter_mut().zip(p.counts.download_i32(n)?) {
                *a += v;
            }
        }
        Ok(all)
    }

    fn reset_counts(&self) -> Result<()> {
        self.parts.iter().try_for_each(|p| p.counts.zero())
    }

    fn skipped(&self) -> (u64, u64) {
        self.skipped
    }

    fn set_light(&mut self, light: bool) {
        self.light = light;
    }

    fn read_seconds(&self) -> f64 {
        self.read_secs
    }

    fn guess_hits(&self) -> (u64, u64) {
        self.guess_hits
    }

    fn hold_background(&self, on: bool) {
        if let Some(pre) = self.q4file.as_ref().and_then(|q| q.pre.as_ref()) {
            pre.hold(on);
        }
    }

    fn prefetched(&self) -> u64 {
        self.q4file.as_ref().and_then(|q| q.pre.as_ref()).map_or(0, |p| p.done())
    }

    fn expert_counters(&self) -> (u64, u64) {
        // RAM tier hits count as hits: the misses are the disk reads
        let ram = self.ram.as_ref().map_or(0, |m| m.hits);
        let (h, m) = self.parts.iter().fold((0, 0), |(h, m), p| (h + p.pool.hits, m + p.pool.page_ins));
        (h + ram, m - ram)
    }

    fn describe(&self) -> String {
        let (slots, all) = self.expert_slots();
        let gpus = if self.parts.len() == 1 {
            format!("GPU {} {}", self.parts[0].gpu.ordinal, self.parts[0].gpu.name)
        } else {
            self.parts.iter().map(|p| format!("GPU {} {} [layers {}-{}]", p.gpu.ordinal, p.gpu.name, p.first, p.end - 1)).collect::<Vec<_>>().join(" + ")
        };
        format!(
            "{gpus}: {:.2} GB in VRAM ({slots} of {all} experts cached {}{}), {} live cores, context {} tokens",
            self.weight_bytes() as f64 / 1e9,
            match (&self.q4, &self.q4file) {
                (Some(_), Some(_)) => "as 4-bit copies, read from the 4-bit expert file",
                (Some(_), None) => "as 4-bit copies",
                _ => "in FP8",
            },
            match &self.ram {
                Some(m) => format!(", up to {} more in RAM", m.cap),
                None => String::new(),
            },
            self.live_cores,
            self.cap
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_constants_match_python() {
        // aegist/hc.py for vocab 97, n 3, ple_index 0, seed 1234 and base 50 (tests/test_hc.py's model)
        assert!(is_prime(53) && !is_prime(51) && is_prime(2) && !is_prime(1));
        assert_eq!(ngram_head_sizes(50, 4, 0), vec![53, 59, 61, 67]);
        assert_eq!(ngram_head_sizes(50, 2, 1), vec![61, 67]);
        assert_eq!(ngram_multipliers(97, 3, 0, 1234), vec![3923727992659049, 40654649216656725, 76114243727631751]);
        assert_eq!(ngram_multipliers(248320, 3, 0, 1234), vec![23703573157769, 20109073645365, 8052911324071]);
    }

    #[test]
    fn split_balances_slots_per_layer() {
        let eb = 5.0;
        // two equal GPUs, equal layers: half each
        assert_eq!(split_layers(&[10.0; 48], &[2000.0, 2000.0], eb), vec![24, 24]);
        // one GPU with twice the room takes about twice the layers
        let s = split_layers(&[10.0; 48], &[4000.0, 2000.0], eb);
        assert_eq!(s.iter().sum::<usize>(), 48);
        assert!((31..=33).contains(&s[0]), "{s:?}");
        // a GPU that cannot hold even one layer gets none
        assert_eq!(split_layers(&[10.0; 48], &[3000.0, 5.0], eb), vec![48, 0]);
        // one GPU: everything
        assert_eq!(split_layers(&[10.0; 8], &[100.0], eb), vec![8]);
        // three GPUs
        let s = split_layers(&[10.0; 30], &[1000.0, 1000.0, 1000.0], eb);
        assert_eq!(s, vec![10, 10, 10]);
    }
}
