//! Aegist's hybrid-MoE brain on the GPU (arch "hybrid-moe"), a port of aegist/hybrid.py:
//!
//!   x += mixer(norm1(x))        gated DeltaNet (3 of every 4 blocks) or gated attention
//!   h = norm2(x)
//!   x += moe(h) + cores(h, engine(h))
//!
//! The always-on weights (int8 linears, bf16 norms, router, gates) live in VRAM. The embedding
//! table stays in the memory-mapped .nx (rows are looked up on the host). The experts (40 x 256,
//! 4-bit) also stay in the file; a pool of expert slots sized to free VRAM holds the recently used
//! ones, least recently used out. Each layer the router runs on the GPU, its picks come back to the
//! host, missing experts are copied in, and the expert kernels read them through a pointer table.
//! A long prompt streams its misses through the pool's coldest slots, so it does not evict the warm
//! set that decoding relies on.
//!
//! DeltaNet blocks keep a fixed-size state instead of a KV cache. A state cannot be cut back to an
//! earlier position, so `snapshot` records it and `crop` returns to the latest snapshot at or before
//! the requested length; the chat runtime snapshots the conversation before each reply.
use crate::cuda::{Arg, Buf, Gpu, Kernel, Result};
use crate::model::{bytes_i32, qweight, upload, Lm, PTX, QW};
use crate::nx::NxFile;
use crate::reader::Reader;
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct HConfig {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub attn: Vec<bool>, // per layer: gated attention (true) or gated DeltaNet
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
    pub group: usize,
    pub eps: f32,
    pub n1: usize,
    pub n2: usize,
    pub rank: usize,
    pub cores_active: usize,
}

impl HConfig {
    pub fn from_header(c: &Value) -> Result<HConfig> {
        let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or(format!("config is missing {k}"));
        let f = |k: &str| c[k].as_f64().map(|v| v as f32).ok_or(format!("config is missing {k}"));
        let kinds = c["layer_types"].as_array().ok_or("config is missing layer_types")?;
        let attn = kinds.iter().map(|k| k.as_str() == Some("attn")).collect::<Vec<_>>();
        let grid = c["core_grid"].as_array().ok_or("config is missing core_grid")?;
        let cfg = HConfig {
            vocab: u("vocab_size")?,
            d: u("d_model")?,
            layers: u("n_layers")?,
            attn,
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
            group: u("expert_group")?,
            eps: f("norm_eps")?,
            n1: grid[0].as_u64().ok_or("bad core_grid")? as usize,
            n2: grid[1].as_u64().ok_or("bad core_grid")? as usize,
            rank: u("core_rank")?,
            cores_active: u("cores_active")?,
        };
        if c["tie_embeddings"].as_bool() == Some(true) {
            return Err("this engine build needs an untied output head".into());
        }
        cfg.check()?;
        Ok(cfg)
    }

    /// The shapes these kernels are written for (Qwen3.6-35B-A3B's), with clear errors otherwise.
    fn check(&self) -> Result<()> {
        let mut bad = vec![];
        if self.attn.len() != self.layers {
            bad.push("layer_types length".to_string());
        }
        if self.head_dim != 256 || self.rotary_dim > self.head_dim || self.rotary_dim % 2 != 0 {
            bad.push(format!("head_dim {} / rotary_dim {} (needs 256 / even <= 256)", self.head_dim, self.rotary_dim));
        }
        if self.heads % self.kv_heads != 0 {
            bad.push("n_heads not a multiple of n_kv_heads".into());
        }
        if self.dk != 128 || self.dv != 128 || self.hv % self.hk != 0 || self.conv_k != 4 {
            bad.push(format!("DeltaNet dims {}x{} heads {}/{} conv {} (needs 128x128, hv % hk == 0, conv 4)", self.dk, self.dv, self.hk, self.hv, self.conv_k));
        }
        if self.experts > 256 || self.top_k > 8 || self.expert_hidden != 512 || self.group != 64 || self.d % 64 != 0 {
            bad.push(format!(
                "experts {} top-{} hidden {} group {} d_model {} (needs <= 256, <= 8, 512, 64, d % 64 == 0)",
                self.experts, self.top_k, self.expert_hidden, self.group, self.d
            ));
        }
        if self.shared_hidden % 16 != 0 || self.d % 32 != 0 {
            bad.push("shared_hidden / d_model alignment".into());
        }
        if self.n1 > 128 || self.n2 > 64 || self.rank > 64 || self.cores_active > 8 || self.cores_active > self.n1 * self.n2 {
            bad.push("core grid too large (rows <= 128; columns, rank <= 64; active <= 8)".into());
        }
        if bad.is_empty() { Ok(()) } else { Err(format!("this engine build does not support this brain's shape: {}", bad.join("; "))) }
    }

    pub fn n_cores(&self) -> usize {
        self.n1 * self.n2
    }
    fn conv_dim(&self) -> usize {
        2 * self.hk * self.dk + self.hv * self.dv
    }
    fn n_attn(&self) -> usize {
        self.attn.iter().filter(|&&a| a).count()
    }
    fn n_delta(&self) -> usize {
        self.layers - self.n_attn()
    }
}

/// Byte offsets of the sections inside one 4-bit expert (aegist/hybrid.py expert_layout).
#[derive(Clone, Copy, Debug)]
struct ExpertLayout {
    gu_s: usize,
    gu_m: usize,
    dn_q: usize,
    dn_s: usize,
    dn_m: usize,
    bytes: usize,
}

impl ExpertLayout {
    fn new(c: &HConfig) -> ExpertLayout {
        let (i, d, g) = (c.expert_hidden, c.d, c.group);
        let gu_q = 2 * i * d / 2;
        let gu_sm = 2 * i * (d / g) * 2;
        let dn_q = d * i / 2;
        let dn_sm = d * (i / g) * 2;
        let gu_s = gu_q;
        let gu_m = gu_s + gu_sm;
        let dn_qo = gu_m + gu_sm;
        let dn_s = dn_qo + dn_q;
        let dn_m = dn_s + dn_sm;
        ExpertLayout { gu_s, gu_m, dn_q: dn_qo, dn_s, dn_m, bytes: dn_m + dn_sm }
    }
}

/// $AEGIST_ROUTE_TRACE=FILE: append each layer's expert picks to FILE as lines "layer e1 e2 ...",
/// for trying cache policies offline (scripts/sim_expert_cache.py).
fn trace(layer: usize, experts: &[usize]) {
    use std::io::Write;
    static FILE: std::sync::OnceLock<Option<std::sync::Mutex<std::io::BufWriter<std::fs::File>>>> = std::sync::OnceLock::new();
    let f = FILE.get_or_init(|| {
        let p = std::env::var("AEGIST_ROUTE_TRACE").ok()?;
        std::fs::OpenOptions::new().create(true).append(true).open(p).ok().map(|f| std::sync::Mutex::new(std::io::BufWriter::new(f)))
    });
    if let Some(m) = f {
        let mut w = m.lock().unwrap();
        let _ = write!(w, "{layer}");
        for e in experts {
            let _ = write!(w, " {e}");
        }
        let _ = writeln!(w);
        let _ = w.flush();
    }
}

/// Expert uses between halvings of the use counts.
const AGE: usize = 1 << 16;

/// VRAM slots for experts. Keys are layer * n_experts + expert. Out goes the least used of the
/// `window` least recently used slots (LRU with use counts, after FlashMoE, arXiv 2601.17063):
/// an expert many tokens pick outlasts one a single token picked a moment ago. The counts halve
/// every AGE uses. $AEGIST_POOL_WINDOW=1 is plain LRU.
pub(crate) struct Pool {
    chunks: Vec<Buf>,
    per_chunk: usize,
    eb: usize,
    n_experts: usize,
    owner: Vec<i32>,
    slot_of: Vec<i32>,
    prev: Vec<i32>,
    next: Vec<i32>,
    head: i32,
    tail: i32,
    mark: Vec<u32>,
    epoch: u32,
    uses: Vec<u16>,
    window: usize,
    since_aging: usize,
    pub(crate) hits: u64,
    pub(crate) page_ins: u64,
    /// (slot, key) of the experts the last `assign` pushed out, their bytes still in the slots
    pub(crate) evicted: Vec<(usize, usize)>,
}

impl Pool {
    pub(crate) fn new(chunks: Vec<Buf>, per_chunk: usize, n: usize, eb: usize, n_experts: usize, keys: usize) -> Pool {
        Pool {
            chunks,
            per_chunk,
            eb,
            n_experts,
            owner: vec![-1; n],
            slot_of: vec![-1; keys],
            prev: (0..n as i32).map(|i| i - 1).collect(),
            next: (0..n as i32).map(|i| if i + 1 < n as i32 { i + 1 } else { -1 }).collect(),
            head: if n > 0 { 0 } else { -1 },
            tail: n as i32 - 1,
            mark: vec![0; n],
            epoch: 0,
            uses: vec![0; keys],
            window: env_usize("AEGIST_POOL_WINDOW").unwrap_or(16).max(1),
            since_aging: 0,
            hits: 0,
            page_ins: 0,
            evicted: vec![],
        }
    }

    pub(crate) fn slots(&self) -> usize {
        self.owner.len()
    }

    fn ptr(&self, s: usize) -> u64 {
        self.chunks[s / self.per_chunk].at((s % self.per_chunk) * self.eb)
    }

    /// Device address of slot `s`.
    pub(crate) fn slot_ptr(&self, s: usize) -> u64 {
        self.ptr(s)
    }

    fn unlink(&mut self, s: usize) {
        let (p, n) = (self.prev[s], self.next[s]);
        if p >= 0 { self.next[p as usize] = n } else { self.head = n }
        if n >= 0 { self.prev[n as usize] = p } else { self.tail = p }
    }

    fn push_front(&mut self, s: usize) {
        self.prev[s] = -1;
        self.next[s] = self.head;
        if self.head >= 0 {
            self.prev[self.head as usize] = s as i32;
        }
        self.head = s as i32;
        if self.tail < 0 {
            self.tail = s as i32;
        }
    }

    /// Slots for `experts` of `layer` (claimed as if loaded) and their device addresses, in order,
    /// plus the (slot, expert) pairs whose bytes the caller must `upload` before use. Streaming
    /// (long prompts): hits are not promoted and misses go to the coldest slots and stay cold.
    pub(crate) fn assign(&mut self, layer: usize, experts: &[usize], streaming: bool) -> Result<(Vec<u64>, Vec<(usize, usize)>)> {
        self.epoch = self.epoch.wrapping_add(1).max(1);
        self.evicted.clear();
        let base = layer * self.n_experts;
        trace(layer, experts);
        for &e in experts {
            let s = self.slot_of[base + e];
            if s >= 0 {
                self.mark[s as usize] = self.epoch;
            }
            self.uses[base + e] = self.uses[base + e].saturating_add(1);
        }
        self.since_aging += experts.len();
        if self.since_aging >= AGE {
            self.since_aging = 0;
            for u in self.uses.iter_mut() {
                *u /= 2;
            }
        }
        let mut out = Vec::with_capacity(experts.len());
        let mut loads = vec![];
        for &e in experts {
            let key = base + e;
            let s = if self.slot_of[key] >= 0 {
                self.hits += 1;
                self.slot_of[key] as usize
            } else {
                let Some(v) = self.victim() else {
                    self.forget(&loads);
                    return Err("expert pool too small for one layer's experts".into());
                };
                let was_empty = self.owner[v] < 0;
                if !was_empty {
                    self.slot_of[self.owner[v] as usize] = -1;
                    self.evicted.push((v, self.owner[v] as usize));
                }
                loads.push((v, e));
                self.owner[v] = key as i32;
                self.slot_of[key] = v as i32;
                self.mark[v] = self.epoch;
                self.page_ins += 1;
                if streaming && was_empty {
                    // a free slot evicts nothing: keep what lands there
                    self.unlink(v);
                    self.push_front(v);
                }
                v
            };
            if !streaming {
                self.unlink(s);
                self.push_front(s);
            }
            out.push(self.ptr(s));
        }
        Ok((out, loads))
    }

    /// The slot to load into: an empty one, else the least used of the `window` least recently
    /// used slots this layer is not using.
    fn victim(&self) -> Option<usize> {
        let (mut v, mut best, mut seen) = (self.tail, None::<(u16, usize)>, 0);
        while v >= 0 && seen < self.window {
            let s = v as usize;
            if self.mark[s] != self.epoch {
                if self.owner[s] < 0 {
                    return Some(s);
                }
                let u = self.uses[self.owner[s] as usize];
                if best.map_or(true, |(b, _)| u < b) {
                    best = Some((u, s));
                }
                seen += 1;
            }
            v = self.prev[s];
        }
        best.map(|(_, s)| s)
    }

    /// Each expert's recent use count (halved every AGE uses), by layer * experts + expert.
    pub(crate) fn uses(&self) -> &[u16] {
        &self.uses
    }

    /// Start the use counts from an earlier run's (the hot list), so eviction knows the experts
    /// that were used most before this run has used them.
    pub(crate) fn seed_uses(&mut self, uses: &[u16]) {
        for (u, &v) in self.uses.iter_mut().zip(uses) {
            *u = (*u).max(v);
        }
    }

    /// Whether this expert of this layer is in VRAM now.
    pub(crate) fn has(&self, layer: usize, e: usize) -> bool {
        self.slot_of[layer * self.n_experts + e] >= 0
    }

    /// Bytes per slot.
    pub(crate) fn slot_bytes(&self) -> usize {
        self.eb
    }

    /// Copy slot `slot`'s bytes out (after the work queued before).
    pub(crate) fn download_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        self.chunks[slot / self.per_chunk].download_into((slot % self.per_chunk) * self.eb, out)
    }

    /// Copy one expert's bytes into its slot.
    pub(crate) fn upload(&self, slot: usize, bytes: &[u8]) -> Result<()> {
        self.chunks[slot / self.per_chunk].upload_at((slot % self.per_chunk) * self.eb, bytes)
    }

    /// Mark these slots empty again (their load failed or never happened).
    pub(crate) fn forget(&mut self, loads: &[(usize, usize)]) {
        for &(slot, _) in loads {
            if self.owner[slot] >= 0 {
                self.slot_of[self.owner[slot] as usize] = -1;
                self.owner[slot] = -1;
            }
        }
    }
}

enum Mixer {
    Attn { q: QW, k: QW, v: QW, o: QW, q_norm: Buf, k_norm: Buf, slot: usize },
    Delta { in_qkv: QW, in_z: QW, in_a: Buf, in_b: Buf, conv: Buf, a_log: Buf, dt_bias: Buf, norm: Buf, out: QW, slot: usize },
}

struct Layer {
    norm1: Buf,
    norm2: Buf,
    mixer: Mixer,
    router: Buf,
    sh_gate: QW,
    sh_up: QW,
    sh_down: QW,
    sh_gw: Buf,
    key1: Buf,
    key2: Buf,
    live: Buf, // [n_cores] i32: index into the stored (live) cores, or -1
    cores: Option<(Buf, Buf, Buf)>,
    experts_at: usize, // absolute offset of blocks.N.moe.experts.q4 in the mapped file
}

struct Kernels {
    znorm: Kernel,
    gemv_i8: Kernel,
    gemv_i8_gateup: Kernel,
    gemm_i8: Kernel,
    silu_mul: Kernel,
    attn_prep: Kernel,
    attention_gated: Kernel,
    delta_gates: Kernel,
    conv_silu: Kernel,
    l2norm_heads: Kernel,
    delta_scan: Kernel,
    gated_norm: Kernel,
    router_topk: Kernel,
    moe_gu_gemv: Kernel,
    moe_dn_gemv: Kernel,
    moe_gu_gemm: Kernel,
    moe_dn_gemm: Kernel,
    moe_reduce: Kernel,
    shared_gate_add: Kernel,
    cores_apply: Kernel,
}

/// Threads reading experts from the brain file ($AEGIST_READ_THREADS).
pub(crate) const READ_THREADS: usize = 4;
/// At most this many tokens per layer pass use the per-pick expert kernels; more use the tiled ones.
const GEMV_TOKENS: usize = 8;
/// Feeds longer than this stream their expert misses (see Pool::ensure).
const STREAM_TOKENS: usize = 64;
const SNAPSHOTS: usize = 2;

pub struct HybridEngine {
    pub gpu: Gpu,
    pub cfg: HConfig,
    k: Kernels,
    nx: NxFile,
    lay: ExpertLayout,
    embed_at: usize,
    norm: Buf,
    lm_head: QW,
    layers: Vec<Layer>,
    pool: Pool,
    reader: Reader,
    kc: Buf,
    vc: Buf,
    conv: Buf,
    state: Buf,
    snap_conv: Vec<Buf>,
    snap_state: Vec<Buf>,
    snap_pos: Vec<Option<usize>>,
    pub cap: usize,
    pub length: usize,
    last_t: usize,
    chunk: usize,
    gemv_tokens: usize,
    stream_tokens: usize,
    tap_layers: Vec<usize>,
    tap_buf: Option<Buf>,
    x: Buf,
    h: Buf,
    big: Buf,
    kb: Buf,
    vb: Buf,
    qn: Buf,
    att: Buf,
    z: Buf,
    y: Buf,
    o: Buf,
    gb: Buf,
    idx: Buf,
    wts: Buf,
    act: Buf,
    ypick: Buf,
    sh_g: Buf,
    sh_u: Buf,
    sh_out: Buf,
    meta: Buf,
    logits: Buf,
    counts: Buf,
    pub weight_bytes: usize,
    pub live_cores: usize,
}

pub(crate) fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

impl HybridEngine {
    /// Load `nx` (kept open: experts and the embedding are read from it while running).
    /// `max_context` 0 picks 32k tokens. VRAM left after everything else becomes expert slots,
    /// minus a headroom for other apps ($AEGIST_VRAM_HEADROOM_MB, default 1536);
    /// $AEGIST_EXPERT_SLOTS caps the slot count.
    pub fn load(nx: NxFile, max_context: usize, progress: impl FnMut(usize, usize)) -> Result<HybridEngine> {
        HybridEngine::load_on(nx, max_context, 0, None, progress)
    }

    /// `load` on GPU `ordinal`, with at most `pool_bytes` of expert slots (the fast brain beside
    /// the big one: it shares that GPU, so it takes only part of it).
    pub fn load_on(nx: NxFile, max_context: usize, ordinal: usize, pool_bytes: Option<usize>, mut progress: impl FnMut(usize, usize)) -> Result<HybridEngine> {
        let cfg = HConfig::from_header(&nx.header["config"])?;
        let lay = ExpertLayout::new(&cfg);
        let gpu = Gpu::on(ordinal, PTX)?;
        let f = |n: &str| gpu.function(n);
        let k = Kernels {
            znorm: f("znorm")?,
            gemv_i8: f("gemv_i8")?,
            gemv_i8_gateup: f("gemv_i8_gateup")?,
            gemm_i8: f("gemm_i8")?,
            silu_mul: f("silu_mul")?,
            attn_prep: f("attn_prep")?,
            attention_gated: f("attention_gated")?,
            delta_gates: f("delta_gates")?,
            conv_silu: f("conv_silu")?,
            l2norm_heads: f("l2norm_heads")?,
            delta_scan: f("delta_scan")?,
            gated_norm: f("gated_norm")?,
            router_topk: f("router_topk")?,
            moe_gu_gemv: f("moe_gu_gemv")?,
            moe_dn_gemv: f("moe_dn_gemv")?,
            moe_gu_gemm: f("moe_gu_gemm")?,
            moe_dn_gemm: f("moe_dn_gemm")?,
            moe_reduce: f("moe_reduce")?,
            shared_gate_add: f("shared_gate_add")?,
            cores_apply: f("cores_apply")?,
        };

        let need = |name: &str, dtype: &str, shape: &[usize]| -> Result<()> {
            let t = nx.info(name)?;
            if t.dtype != dtype || t.shape != shape {
                return Err(format!("{name} is {} {:?}, expected {dtype} {shape:?}", t.dtype, t.shape));
            }
            Ok(())
        };
        need("embed.weight", "bf16", &[cfg.vocab, cfg.d])?;
        let mut total = 0usize;
        let norm = upload(&nx, "norm.weight", &mut total)?;
        let lm_head = qweight(&nx, "lm_head", &mut total)?;
        let (mut n_attn, mut n_delta, mut live_cores) = (0, 0, 0);
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            progress(i, cfg.layers);
            let p = |s: &str| format!("blocks.{i}.{s}");
            let mut up = |s: &str| upload(&nx, &p(s), &mut total);
            let mixer = if cfg.attn[i] {
                n_attn += 1;
                Mixer::Attn {
                    q_norm: up("mixer.q_norm.weight")?,
                    k_norm: up("mixer.k_norm.weight")?,
                    q: qweight(&nx, &p("mixer.q"), &mut total)?,
                    k: qweight(&nx, &p("mixer.k"), &mut total)?,
                    v: qweight(&nx, &p("mixer.v"), &mut total)?,
                    o: qweight(&nx, &p("mixer.o"), &mut total)?,
                    slot: n_attn - 1,
                }
            } else {
                n_delta += 1;
                need(&p("mixer.conv"), "bf16", &[cfg.conv_dim(), cfg.conv_k])?;
                Mixer::Delta {
                    in_a: up("mixer.in_a.weight")?,
                    in_b: up("mixer.in_b.weight")?,
                    conv: up("mixer.conv")?,
                    a_log: up("mixer.A_log")?,
                    dt_bias: up("mixer.dt_bias")?,
                    norm: up("mixer.norm.weight")?,
                    in_qkv: qweight(&nx, &p("mixer.in_qkv"), &mut total)?,
                    in_z: qweight(&nx, &p("mixer.in_z"), &mut total)?,
                    out: qweight(&nx, &p("mixer.out"), &mut total)?,
                    slot: n_delta - 1,
                }
            };
            need(&p("moe.experts.q4"), "u8", &[cfg.experts, lay.bytes])?;
            need(&p("moe.router.weight"), "bf16", &[cfg.experts, cfg.d])?;
            need(&p("engine.key1.weight"), "bf16", &[cfg.n1, cfg.d])?;
            need(&p("engine.key2.weight"), "bf16", &[cfg.n2, cfg.d])?;
            let (live, cores, n_live) = live_bank(&nx, i, &cfg, &mut total)?;
            live_cores += n_live;
            layers.push(Layer {
                norm1: upload(&nx, &p("norm1.weight"), &mut total)?,
                norm2: upload(&nx, &p("norm2.weight"), &mut total)?,
                mixer,
                router: upload(&nx, &p("moe.router.weight"), &mut total)?,
                sh_gate: qweight(&nx, &p("moe.shared.gate"), &mut total)?,
                sh_up: qweight(&nx, &p("moe.shared.up"), &mut total)?,
                sh_down: qweight(&nx, &p("moe.shared.down"), &mut total)?,
                sh_gw: upload(&nx, &p("moe.shared_gate.weight"), &mut total)?,
                key1: upload(&nx, &p("engine.key1.weight"), &mut total)?,
                key2: upload(&nx, &p("engine.key2.weight"), &mut total)?,
                live,
                cores,
                experts_at: nx.offset_of(&p("moe.experts.q4"))?,
            });
        }
        progress(cfg.layers, cfg.layers);

        // context, recurrent state, snapshots, workspace
        let cap = if max_context == 0 { 32768 } else { max_context };
        let chunk = 1024;
        let (d, kk) = (cfg.d, cfg.top_k);
        let f32b = |n: usize| Buf::alloc(n * 4);
        let kv_layer = cfg.kv_heads * cap * cfg.head_dim * 2;
        let conv_bytes = cfg.n_delta() * cfg.conv_dim() * 3 * 4;
        let state_bytes = cfg.n_delta() * cfg.hv * cfg.dk * cfg.dv * 4;
        let big = (cfg.heads * 2 * cfg.head_dim).max(cfg.conv_dim());
        let wide = (cfg.heads * cfg.head_dim).max(cfg.hv * cfg.dv);
        let meta_bytes = meta_layout(&cfg, chunk).3;
        let mut snap_conv = vec![];
        let mut snap_state = vec![];
        for _ in 0..SNAPSHOTS {
            snap_conv.push(Buf::alloc(conv_bytes)?);
            snap_state.push(Buf::alloc(state_bytes)?);
        }
        let kc = Buf::alloc(kv_layer * n_attn).map_err(|e| format!("KV cache for {cap} tokens: {e}"))?;
        let vc = Buf::alloc(kv_layer * n_attn).map_err(|e| format!("KV cache for {cap} tokens: {e}"))?;
        let reader = Reader::open(&nx.path, READ_THREADS)?;
        let mut eng = HybridEngine {
            conv: Buf::zeroed(conv_bytes)?,
            state: Buf::zeroed(state_bytes)?,
            x: f32b(chunk * d)?,
            h: f32b(chunk * d)?,
            big: f32b(chunk * big)?,
            kb: f32b(chunk * cfg.kv_heads * cfg.head_dim)?,
            vb: f32b(chunk * cfg.kv_heads * cfg.head_dim)?,
            qn: f32b(chunk * cfg.heads * cfg.head_dim)?,
            att: f32b(chunk * wide)?,
            z: f32b(chunk * cfg.hv * cfg.dv)?,
            y: f32b(chunk * cfg.conv_dim())?,
            o: f32b(chunk * cfg.hv * cfg.dv)?,
            gb: f32b(chunk * 2 * cfg.hv)?,
            idx: Buf::alloc(chunk * kk * 4)?,
            wts: f32b(chunk * kk)?,
            act: f32b(chunk * kk * cfg.expert_hidden)?,
            ypick: f32b(chunk * kk * d)?,
            sh_g: f32b(chunk * cfg.shared_hidden)?,
            sh_u: f32b(chunk * cfg.shared_hidden)?,
            sh_out: f32b(chunk * d)?,
            meta: Buf::alloc(meta_bytes)?,
            logits: f32b(cfg.vocab)?,
            counts: Buf::zeroed(cfg.layers * cfg.n_cores() * 4)?,
            kc,
            vc,
            snap_conv,
            snap_state,
            snap_pos: vec![None; SNAPSHOTS],
            pool: Pool::new(vec![], 1, 0, lay.bytes, cfg.experts, cfg.layers * cfg.experts),
            reader,
            embed_at: nx.offset_of("embed.weight")?,
            gpu,
            k,
            nx,
            lay,
            norm,
            lm_head,
            layers,
            cap,
            length: 0,
            last_t: 0,
            chunk,
            gemv_tokens: env_usize("AEGIST_GEMV_TOKENS").unwrap_or(GEMV_TOKENS),
            stream_tokens: env_usize("AEGIST_STREAM_TOKENS").unwrap_or(STREAM_TOKENS),
            tap_layers: vec![],
            tap_buf: None,
            weight_bytes: total,
            live_cores,
            cfg,
        };

        // whatever VRAM is left (minus headroom) holds experts
        let (free, _) = eng.gpu.mem_info();
        let headroom = env_usize("AEGIST_VRAM_HEADROOM_MB").unwrap_or(1536) << 20;
        let all = eng.cfg.layers * eng.cfg.experts;
        let mut want = (free.saturating_sub(headroom) / lay.bytes).min(all);
        if let Some(n) = env_usize("AEGIST_EXPERT_SLOTS").or(pool_bytes.map(|b| b / lay.bytes)) {
            want = want.min(n);
        }
        let min_slots = eng.cfg.experts + 32;
        let per_chunk = 256;
        let mut chunks = vec![];
        let mut got = 0;
        while got < want {
            let n = per_chunk.min(want - got);
            match Buf::alloc(n * lay.bytes) {
                Ok(b) => {
                    chunks.push(b);
                    got += n;
                }
                Err(_) => break,
            }
        }
        if got < min_slots {
            return Err(format!(
                "not enough GPU memory for the experts: room for {got} expert slots, at least {min_slots} needed \
                 ({:.2} GB free after the other weights). Close GPU-heavy apps or lower $AEGIST_VRAM_HEADROOM_MB.",
                free as f64 / 1e9
            ));
        }
        eng.pool = Pool::new(chunks, per_chunk, got, lay.bytes, eng.cfg.experts, all);
        eng.weight_bytes += got * lay.bytes;
        Ok(eng)
    }

    pub fn expert_slots(&self) -> (usize, usize) {
        (self.pool.slots(), self.cfg.layers * self.cfg.experts)
    }

    fn mm(&self, w: &QW, x: &Buf, y: &Buf, t: usize, accumulate: bool) -> Result<()> {
        let acc = Arg::I32(accumulate as i32);
        if t == 1 {
            self.k.gemv_i8.launch(
                (((w.n + 7) / 8) as u32, 1),
                256,
                &[Arg::Ptr(x.ptr), Arg::Ptr(w.w.ptr), Arg::Ptr(w.s.ptr), Arg::Ptr(y.ptr), Arg::I32(w.n as i32), Arg::I32(w.k as i32), acc],
            )
        } else {
            self.k.gemm_i8.launch(
                (((w.n + 63) / 64) as u32, ((t + 31) / 32) as u32),
                256,
                &[
                    Arg::Ptr(x.ptr), Arg::Ptr(w.w.ptr), Arg::Ptr(w.s.ptr), Arg::Ptr(y.ptr), Arg::I32(t as i32), Arg::I32(w.n as i32),
                    Arg::I32(w.k as i32), acc,
                ],
            )
        }
    }

    fn znorm(&self, w: &Buf, src: &Buf, dst: &Buf, rows: usize) -> Result<()> {
        let c = &self.cfg;
        self.k.znorm.launch((rows as u32, 1), 256, &[Arg::Ptr(src.ptr), Arg::Ptr(w.ptr), Arg::Ptr(dst.ptr), Arg::I32(c.d as i32), Arg::F32(c.eps)])
    }

    /// Embedding rows looked up on the host (the table stays in the mapped file), widened to fp32.
    fn embed(&self, ids: &[u32]) -> Result<()> {
        let d = self.cfg.d;
        let mut bytes = Vec::with_capacity(ids.len() * d * 4);
        for &id in ids {
            let id = id as usize;
            if id >= self.cfg.vocab {
                return Err(format!("token id {id} is outside the vocabulary ({})", self.cfg.vocab));
            }
            let row = self.nx.raw(self.embed_at + id * d * 2, d * 2);
            for p in row.chunks_exact(2) {
                let bits = (u16::from_le_bytes([p[0], p[1]]) as u32) << 16;
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
        }
        self.x.upload(&bytes)
    }

    fn forward_chunk(&mut self, ids: &[u32]) -> Result<()> {
        let t = ids.len();
        self.embed(ids)?;
        let row = self.cfg.d * 4;
        for li in 0..self.layers.len() {
            self.mixer(li, t)?;
            self.moe(li, t)?;
            if let (Some(j), Some(buf)) = (self.tap_layers.iter().position(|&l| l == li), &self.tap_buf) {
                buf.copy_range(j * row, &self.x, (t - 1) * row, row)?;
            }
        }
        // for tests: $AEGIST_DUMP_FINAL=file appends every row's hidden state after the last layer (fp32)
        if let Ok(path) = std::env::var("AEGIST_DUMP_FINAL") {
            use std::io::Write;
            let all: Vec<u8> = self.x.download_f32(t * self.cfg.d)?.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .and_then(|mut f| f.write_all(&all))
                .map_err(|e| format!("{path}: {e}"))?;
        }
        self.length += t;
        self.last_t = t;
        Ok(())
    }

    fn mixer(&self, li: usize, t: usize) -> Result<()> {
        let c = &self.cfg;
        let k = &self.k;
        let l = &self.layers[li];
        let pos0 = self.length as i32;
        self.znorm(&l.norm1, &self.x, &self.h, t)?;
        match &l.mixer {
            Mixer::Attn { q, k: wk, v, o, q_norm, k_norm, slot } => {
                let kv_layer = c.kv_heads * self.cap * c.head_dim * 2;
                let (kc, vc) = (self.kc.at(slot * kv_layer), self.vc.at(slot * kv_layer));
                self.mm(q, &self.h, &self.big, t, false)?;
                self.mm(wk, &self.h, &self.kb, t, false)?;
                self.mm(v, &self.h, &self.vb, t, false)?;
                k.attn_prep.launch(
                    (t as u32, (c.heads + c.kv_heads) as u32),
                    c.head_dim as u32,
                    &[
                        Arg::Ptr(self.big.ptr), Arg::Ptr(self.kb.ptr), Arg::Ptr(self.vb.ptr), Arg::Ptr(q_norm.ptr), Arg::Ptr(k_norm.ptr),
                        Arg::Ptr(self.qn.ptr), Arg::Ptr(kc), Arg::Ptr(vc), Arg::I32(c.heads as i32), Arg::I32(c.kv_heads as i32),
                        Arg::I32(c.head_dim as i32), Arg::I32(c.rotary_dim as i32), Arg::I32(pos0), Arg::I32(self.cap as i32),
                        Arg::F32(c.eps), Arg::F32(c.rope_theta),
                    ],
                )?;
                k.attention_gated.launch(
                    (t as u32, c.heads as u32),
                    256,
                    &[
                        Arg::Ptr(self.qn.ptr), Arg::Ptr(kc), Arg::Ptr(vc), Arg::Ptr(self.big.ptr), Arg::Ptr(self.att.ptr),
                        Arg::I32(c.heads as i32), Arg::I32(c.kv_heads as i32), Arg::I32(pos0), Arg::I32(self.cap as i32),
                        Arg::F32(1.0 / (c.head_dim as f32).sqrt()),
                    ],
                )?;
                self.mm(o, &self.att, &self.x, t, true)
            }
            Mixer::Delta { in_qkv, in_z, in_a, in_b, conv, a_log, dt_bias, norm, out, slot } => {
                let cd = c.conv_dim();
                let conv_state = self.conv.at(slot * cd * 3 * 4);
                let state = self.state.at(slot * c.hv * c.dk * c.dv * 4);
                self.mm(in_qkv, &self.h, &self.big, t, false)?;
                self.mm(in_z, &self.h, &self.z, t, false)?;
                k.delta_gates.launch(
                    (t as u32, 1),
                    256,
                    &[
                        Arg::Ptr(self.h.ptr), Arg::Ptr(in_b.ptr), Arg::Ptr(in_a.ptr), Arg::Ptr(a_log.ptr), Arg::Ptr(dt_bias.ptr),
                        Arg::Ptr(self.gb.ptr), Arg::I32(c.d as i32), Arg::I32(c.hv as i32),
                    ],
                )?;
                k.conv_silu.launch(
                    (((cd + 255) / 256) as u32, 1),
                    256,
                    &[Arg::Ptr(self.big.ptr), Arg::Ptr(self.y.ptr), Arg::Ptr(conv.ptr), Arg::Ptr(conv_state), Arg::I32(t as i32), Arg::I32(cd as i32)],
                )?;
                k.l2norm_heads.launch(
                    (t as u32, (2 * c.hk) as u32),
                    c.dk as u32,
                    &[Arg::Ptr(self.y.ptr), Arg::I32(cd as i32), Arg::I32(c.hk as i32), Arg::I32(c.dk as i32), Arg::F32(1.0 / (c.dk as f32).sqrt()), Arg::F32(1e-6)],
                )?;
                k.delta_scan.launch(
                    (c.hv as u32, 1),
                    512,
                    &[
                        Arg::Ptr(self.y.ptr), Arg::Ptr(self.gb.ptr), Arg::Ptr(state), Arg::Ptr(self.o.ptr), Arg::I32(t as i32), Arg::I32(cd as i32),
                        Arg::I32(c.hk as i32), Arg::I32(c.hv as i32),
                    ],
                )?;
                k.gated_norm.launch(
                    (t as u32, c.hv as u32),
                    c.dv as u32,
                    &[Arg::Ptr(self.o.ptr), Arg::Ptr(self.z.ptr), Arg::Ptr(norm.ptr), Arg::Ptr(self.att.ptr), Arg::I32(c.hv as i32), Arg::I32(c.dv as i32), Arg::F32(c.eps)],
                )?;
                self.mm(out, &self.att, &self.x, t, true)
            }
        }
    }

    fn moe(&mut self, li: usize, t: usize) -> Result<()> {
        let (d, kk, e_n, ih) = (self.cfg.d, self.cfg.top_k, self.cfg.experts, self.cfg.expert_hidden);
        {
            let l = &self.layers[li];
            self.znorm(&l.norm2, &self.x, &self.h, t)?;
            self.k.router_topk.launch(
                (t as u32, 1),
                256,
                &[Arg::Ptr(self.h.ptr), Arg::Ptr(l.router.ptr), Arg::Ptr(self.idx.ptr), Arg::Ptr(self.wts.ptr), Arg::I32(d as i32), Arg::I32(e_n as i32), Arg::I32(kk as i32)],
            )?;
        }
        // group the picks by expert (this download is the one host sync per layer)
        let idx = self.idx.download_i32(t * kk)?;
        let mut count = vec![0usize; e_n];
        for &e in &idx {
            if e < 0 || e as usize >= e_n {
                return Err(format!("router returned expert {e}"));
            }
            count[e as usize] += 1;
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
            let pos = cursor[e as usize];
            cursor[e as usize] += 1;
            ptok[pos] = (p / kk) as i32;
            psrc[pos] = p as i32;
        }
        let (ptrs, loads) = self.pool.assign(li, &uniq, t > self.stream_tokens)?;
        // the experts not in VRAM, read in parallel and copied in as each arrives
        let (at, eb) = (self.layers[li].experts_at, self.lay.bytes);
        let jobs: Vec<(usize, usize)> = loads.iter().map(|&(slot, e)| (slot, at + e * eb)).collect();
        let pool = &self.pool;
        if let Err(err) = self.reader.load(&jobs, eb, |slot, bytes| pool.upload(slot, bytes)) {
            self.pool.forget(&loads);
            return Err(err);
        }

        let (o_us, o_pt, o_ps, _) = meta_layout(&self.cfg, self.chunk);
        let mut meta = vec![0u8; o_ps + t * kk * 4];
        for (i, p) in ptrs.iter().enumerate() {
            meta[i * 8..i * 8 + 8].copy_from_slice(&p.to_le_bytes());
        }
        meta[o_us..o_us + ustart.len() * 4].copy_from_slice(&bytes_i32(&ustart));
        meta[o_pt..o_pt + t * kk * 4].copy_from_slice(&bytes_i32(&ptok));
        meta[o_ps..o_ps + t * kk * 4].copy_from_slice(&bytes_i32(&psrc));
        self.meta.upload(&meta)?;

        let l = &self.layers[li];
        let k = &self.k;
        let lay = &self.lay;
        let (eptr, us, pt, ps) = (self.meta.ptr, self.meta.at(o_us), self.meta.at(o_pt), self.meta.at(o_ps));
        let nu = uniq.len() as u32;
        if t <= self.gemv_tokens {
            k.moe_gu_gemv.launch(
                (((ih + 7) / 8) as u32, nu),
                256,
                &[Arg::Ptr(self.h.ptr), Arg::Ptr(eptr), Arg::Ptr(us), Arg::Ptr(pt), Arg::Ptr(self.act.ptr), Arg::I32(d as i32), Arg::I32(ih as i32), Arg::I32(lay.gu_s as i32), Arg::I32(lay.gu_m as i32)],
            )?;
            k.moe_dn_gemv.launch(
                (((d + 15) / 16) as u32, nu),
                256,
                &[
                    Arg::Ptr(self.act.ptr), Arg::Ptr(eptr), Arg::Ptr(us), Arg::Ptr(ps), Arg::Ptr(self.wts.ptr), Arg::Ptr(self.ypick.ptr), Arg::I32(d as i32),
                    Arg::I32(ih as i32), Arg::I32(lay.dn_q as i32), Arg::I32(lay.dn_s as i32), Arg::I32(lay.dn_m as i32),
                ],
            )?;
        } else {
            k.moe_gu_gemm.launch(
                ((ih / 32) as u32, nu),
                256,
                &[Arg::Ptr(self.h.ptr), Arg::Ptr(eptr), Arg::Ptr(us), Arg::Ptr(pt), Arg::Ptr(self.act.ptr), Arg::I32(d as i32), Arg::I32(ih as i32), Arg::I32(lay.gu_s as i32), Arg::I32(lay.gu_m as i32)],
            )?;
            k.moe_dn_gemm.launch(
                ((d / 64) as u32, nu),
                256,
                &[
                    Arg::Ptr(self.act.ptr), Arg::Ptr(eptr), Arg::Ptr(us), Arg::Ptr(ps), Arg::Ptr(self.wts.ptr), Arg::Ptr(self.ypick.ptr), Arg::I32(d as i32),
                    Arg::I32(ih as i32), Arg::I32(lay.dn_q as i32), Arg::I32(lay.dn_s as i32), Arg::I32(lay.dn_m as i32),
                ],
            )?;
        }
        k.moe_reduce.launch((t as u32, ((d + 255) / 256) as u32), 256, &[Arg::Ptr(self.ypick.ptr), Arg::Ptr(self.x.ptr), Arg::I32(d as i32), Arg::I32(kk as i32)])?;

        // shared expert, gated
        let sh = self.cfg.shared_hidden;
        if t == 1 {
            k.gemv_i8_gateup.launch(
                (((sh + 7) / 8) as u32, 1),
                256,
                &[
                    Arg::Ptr(self.h.ptr), Arg::Ptr(l.sh_gate.w.ptr), Arg::Ptr(l.sh_gate.s.ptr), Arg::Ptr(l.sh_up.w.ptr), Arg::Ptr(l.sh_up.s.ptr),
                    Arg::Ptr(self.sh_g.ptr), Arg::I32(sh as i32), Arg::I32(d as i32),
                ],
            )?;
        } else {
            self.mm(&l.sh_gate, &self.h, &self.sh_g, t, false)?;
            self.mm(&l.sh_up, &self.h, &self.sh_u, t, false)?;
            let n = (t * sh) as i32;
            k.silu_mul.launch((((n + 255) / 256) as u32, 1), 256, &[Arg::Ptr(self.sh_g.ptr), Arg::Ptr(self.sh_u.ptr), Arg::Ptr(self.sh_g.ptr), Arg::I32(n)])?;
        }
        self.mm(&l.sh_down, &self.sh_g, &self.sh_out, t, false)?;
        k.shared_gate_add.launch((t as u32, 1), 256, &[Arg::Ptr(self.x.ptr), Arg::Ptr(self.sh_out.ptr), Arg::Ptr(self.h.ptr), Arg::Ptr(l.sh_gw.ptr), Arg::I32(d as i32)])?;

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
                Arg::Ptr(self.h.ptr), Arg::Ptr(l.key1.ptr), Arg::Ptr(l.key2.ptr), Arg::Ptr(l.live.ptr), cg, cu, cdn, Arg::Ptr(self.x.ptr),
                Arg::Ptr(self.counts.at(li * c.n_cores() * 4)), Arg::I32(d as i32), Arg::I32(c.n1 as i32), Arg::I32(c.n2 as i32),
                Arg::I32(c.cores_active as i32), Arg::I32(c.rank as i32),
            ],
        )
    }

    fn head(&mut self) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let last = self.x.at((self.last_t - 1) * c.d * 4);
        self.k.znorm.launch((1, 1), 256, &[Arg::Ptr(last), Arg::Ptr(self.norm.ptr), Arg::Ptr(self.h.ptr), Arg::I32(c.d as i32), Arg::F32(c.eps)])?;
        self.mm(&self.lm_head, &self.h, &self.logits, 1, false)?;
        self.logits.download_f32(c.vocab)
    }
}

/// Byte offsets inside the per-layer expert table upload: (ustart, ptok, psrc, total), after the
/// device addresses (8 bytes per unique expert) at 0.
fn meta_layout(c: &HConfig, chunk: usize) -> (usize, usize, usize, usize) {
    let a16 = |n: usize| (n + 15) / 16 * 16;
    let picks = chunk * c.top_k * 4;
    let us = c.experts * 8;
    let pt = us + a16((c.experts + 1) * 4);
    let ps = pt + a16(picks);
    (us, pt, ps, ps + picks)
}

impl Lm for HybridEngine {
    fn feed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        assert!(!ids.is_empty());
        // another brain in this process may have left its GPU current
        self.gpu.bind()?;
        if self.length + ids.len() > self.cap {
            return Err(format!("context full: {} + {} tokens > {}", self.length, ids.len(), self.cap));
        }
        for c in ids.chunks(self.chunk) {
            self.forward_chunk(c)?;
        }
        self.head()
    }

    fn crop(&mut self, n: usize) -> usize {
        if n >= self.length {
            return self.length;
        }
        let best = (0..SNAPSHOTS).filter(|&i| self.snap_pos[i].map_or(false, |p| p <= n)).max_by_key(|&i| self.snap_pos[i]);
        let restored = match best {
            Some(i) => self.conv.copy_from(&self.snap_conv[i], self.conv.bytes).and_then(|_| self.state.copy_from(&self.snap_state[i], self.state.bytes)).map(|_| self.snap_pos[i].unwrap()),
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
                // no usable snapshot (or the copy failed): start over
                let _ = self.conv.zero();
                let _ = self.state.zero();
                self.length = 0;
                self.snap_pos = vec![None; SNAPSHOTS];
            }
        }
        self.length
    }

    fn snapshot(&mut self) -> Result<()> {
        if self.snap_pos.contains(&Some(self.length)) {
            return Ok(());
        }
        // reuse an empty slot, else the oldest snapshot
        let i = (0..SNAPSHOTS).min_by_key(|&i| self.snap_pos[i].map_or(-1, |p| p as i64)).unwrap();
        self.snap_conv[i].copy_from(&self.conv, self.conv.bytes)?;
        self.snap_state[i].copy_from(&self.state, self.state.bytes)?;
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
        self.gpu.bind()?;
        self.tap_buf = if layers.is_empty() { None } else { Some(Buf::zeroed(layers.len() * self.cfg.d * 4)?) };
        self.tap_layers = layers.to_vec();
        Ok(())
    }

    fn taps(&self) -> Result<Vec<f32>> {
        match &self.tap_buf {
            Some(b) => b.download_f32(self.tap_layers.len() * self.cfg.d),
            None => Ok(vec![]),
        }
    }

    fn core_counts(&self) -> Result<Vec<i32>> {
        self.counts.download_i32(self.cfg.layers * self.cfg.n_cores())
    }

    fn reset_counts(&self) -> Result<()> {
        self.counts.zero()
    }

    fn expert_counters(&self) -> (u64, u64) {
        (self.pool.hits, self.pool.page_ins)
    }

    fn describe(&self) -> String {
        let (slots, all) = self.expert_slots();
        format!(
            "{}: {:.2} GB in VRAM ({} of {} experts cached), {} live cores, context {} tokens",
            self.gpu.name,
            self.weight_bytes as f64 / 1e9,
            slots,
            all,
            self.live_cores,
            self.cap
        )
    }
}

/// The layer's stored cores (sparse: ids + gate/up/down) and the id -> index map for cores_apply.
fn live_bank(nx: &NxFile, layer: usize, cfg: &HConfig, total: &mut usize) -> Result<(Buf, Option<(Buf, Buf, Buf)>, usize)> {
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
    let ids = nx.bytes(&name("ids"))?;
    for (j, c) in ids.chunks_exact(8).enumerate() {
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
