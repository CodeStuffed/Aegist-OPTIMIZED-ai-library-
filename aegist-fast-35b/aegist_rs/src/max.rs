//! Aegist's deep-think brain on the GPU (arch "max-moe"): Qwen3.8-2.4T-A95B, the same block math
//! as the hybrid brain (gated DeltaNet 3 of every 4 blocks, else gated attention; 512 experts,
//! top-10, plus a gated shared expert), without the core engine:
//!
//!   x += mixer(norm1(x));  h = norm2(x);  x += moe(h)
//!
//! The brain is far bigger than VRAM, so everything is tiered:
//!   * dense weights (int8 linears, bf16 norms/router/gates, ~45 GB) live in aegist-max.nx on D:.
//!     Each GPU takes a run of consecutive layers; as many as fit stay resident, the rest are
//!     copied in from the mapped file every pass (the OS file cache keeps recent ones in RAM).
//!   * experts (3-bit, 20.4 MB each, 47k of them) live in per-layer files on D: and E:
//!     (aegist/fetch_max.py). Each GPU keeps an LRU pool of them in VRAM; a layer that needs more
//!     experts than the pool holds runs them in batches.
//!   * the embedding stays in the file (rows looked up on the host); the output head is resident.
//!
//! Recurrent DeltaNet state cannot be cut back, so `snapshot`/`crop` work as in the hybrid engine.
use crate::cuda::{Arg, Buf, Gpu, Kernel, Result};
use crate::hybrid::{env_usize, Pool, READ_THREADS};
use crate::model::{bytes_i32, Lm, PTX, QW};
use crate::nx::NxFile;
use crate::reader::Reader;
use serde_json::Value;

/// Bytes of one 3-bit expert record and the offsets inside it (aegist/fetch_max.py MATRIX_AT).
const EXPERT_BYTES: usize = 20_447_232;
const GU_SCALES: i32 = 6_291_456;
const UP_CODES: i32 = 6_815_744;
const UP_SCALES: i32 = 13_107_200;
const DN_CODES: i32 = 13_631_488;
const DN_SCALES: i32 = 19_922_944;

const GEMV_TOKENS: usize = 8;
const STREAM_TOKENS: usize = 64;
const SNAPSHOTS: usize = 1;
/// Prefill chunk: bigger chunks read each expert once for more tokens ($AEGIST_MAX_CHUNK).
const CHUNK: usize = 512;
/// VRAM per GPU kept for experts before deciding which dense layers stay resident
/// ($AEGIST_MAX_POOL_GB).
const POOL_GB: f64 = 2.0;

#[derive(Clone, Debug)]
pub struct MConfig {
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
    pub experts: usize,
    pub top_k: usize,
    pub expert_hidden: usize,
    pub shared_hidden: usize,
    pub eps: f32,
    /// per layer: the file holding its experts
    pub files: Vec<String>,
}

pub fn is_max(c: &Value) -> bool {
    c["arch"].as_str() == Some("max-moe")
}

impl MConfig {
    pub fn from_header(c: &Value) -> Result<MConfig> {
        let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or(format!("config is missing {k}"));
        let f = |k: &str| c[k].as_f64().map(|v| v as f32).ok_or(format!("config is missing {k}"));
        let kinds = c["layer_types"].as_array().ok_or("config is missing layer_types")?;
        let layers = u("n_layers")?;
        let files = (0..layers)
            .map(|i| c["expert_files"][format!("L{i:02}")].as_str().map(String::from).ok_or(format!("config has no expert file for layer {i}")))
            .collect::<Result<Vec<_>>>()?;
        let cfg = MConfig {
            vocab: u("vocab_size")?,
            d: u("d_model")?,
            layers,
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
            experts: u("n_experts")?,
            top_k: u("experts_active")?,
            expert_hidden: u("expert_hidden")?,
            shared_hidden: u("shared_hidden")?,
            eps: f("norm_eps")?,
            files,
        };
        if c["expert_format"].as_str() != Some("q3g64") || u("expert_bytes")? != EXPERT_BYTES {
            return Err("this engine reads 3-bit q3g64 experts of 20,447,232 bytes".into());
        }
        let mut bad = vec![];
        if cfg.attn.len() != layers {
            bad.push("layer_types length".to_string());
        }
        if cfg.head_dim != 256 || cfg.heads % cfg.kv_heads != 0 || cfg.rotary_dim % 2 != 0 {
            bad.push("attention head shape".into());
        }
        if cfg.dk != 128 || cfg.dv != 128 || cfg.hv % cfg.hk != 0 || u("conv_kernel")? != 4 {
            bad.push("DeltaNet shape".into());
        }
        if cfg.experts > 1024 || cfg.top_k > 16 || cfg.d != 8192 || cfg.expert_hidden != 2048 {
            bad.push(format!("experts {} top-{} d {} hidden {} (needs <= 1024, <= 16, 8192, 2048)", cfg.experts, cfg.top_k, cfg.d, cfg.expert_hidden));
        }
        if bad.is_empty() { Ok(cfg) } else { Err(format!("this engine build does not support this brain's shape: {}", bad.join("; "))) }
    }

    fn conv_dim(&self) -> usize {
        2 * self.hk * self.dk + self.hv * self.dv
    }
}

enum Mixer {
    Attn { q: QW, k: QW, v: QW, o: QW, q_norm: Buf, k_norm: Buf },
    Delta { in_qkv: QW, in_z: QW, in_a: Buf, in_b: Buf, conv: Buf, a_log: Buf, dt_bias: Buf, norm: Buf, out: QW },
}

/// One layer's dense weights on one GPU.
struct Dense {
    norm1: Buf,
    norm2: Buf,
    mixer: Mixer,
    router: Buf,
    sh_gate: QW,
    sh_up: QW,
    sh_down: QW,
    sh_gw: Buf,
}

const ATTN_PLAIN: [&str; 2] = ["mixer.q_norm.weight", "mixer.k_norm.weight"];
const ATTN_INT8: [&str; 4] = ["mixer.q", "mixer.k", "mixer.v", "mixer.o"];
const DELTA_PLAIN: [&str; 6] = ["mixer.in_a.weight", "mixer.in_b.weight", "mixer.conv", "mixer.A_log", "mixer.dt_bias", "mixer.norm.weight"];
const DELTA_INT8: [&str; 3] = ["mixer.in_qkv", "mixer.in_z", "mixer.out"];
const PLAIN: [&str; 4] = ["norm1.weight", "norm2.weight", "moe.router.weight", "moe.shared_gate.weight"];
const INT8: [&str; 3] = ["moe.shared.gate", "moe.shared.up", "moe.shared.down"];

impl Dense {
    /// Layer `li`'s weights, uploaded (`fill`) or only allocated to its sizes (a streaming slot).
    fn alloc(nx: &NxFile, li: usize, attn: bool, fill: bool, total: &mut usize) -> Result<Dense> {
        let plain = |s: &str, total: &mut usize| -> Result<Buf> {
            let name = format!("blocks.{li}.{s}");
            let t = nx.info(&name)?;
            *total += t.nbytes;
            if fill { Buf::from_bytes(nx.bytes(&name)?) } else { Buf::alloc(t.nbytes) }
        };
        let mixer = if attn {
            let p = ATTN_PLAIN.iter().map(|s| plain(s, total)).collect::<Result<Vec<_>>>()?;
            let q = ATTN_INT8.iter().map(|s| qw(nx, li, s, fill, total)).collect::<Result<Vec<_>>>()?;
            let (mut p, mut q) = (p.into_iter(), q.into_iter());
            Mixer::Attn {
                q_norm: p.next().unwrap(),
                k_norm: p.next().unwrap(),
                q: q.next().unwrap(),
                k: q.next().unwrap(),
                v: q.next().unwrap(),
                o: q.next().unwrap(),
            }
        } else {
            let p = DELTA_PLAIN.iter().map(|s| plain(s, total)).collect::<Result<Vec<_>>>()?;
            let q = DELTA_INT8.iter().map(|s| qw(nx, li, s, fill, total)).collect::<Result<Vec<_>>>()?;
            let (mut p, mut q) = (p.into_iter(), q.into_iter());
            Mixer::Delta {
                in_a: p.next().unwrap(),
                in_b: p.next().unwrap(),
                conv: p.next().unwrap(),
                a_log: p.next().unwrap(),
                dt_bias: p.next().unwrap(),
                norm: p.next().unwrap(),
                in_qkv: q.next().unwrap(),
                in_z: q.next().unwrap(),
                out: q.next().unwrap(),
            }
        };
        let mut p = PLAIN.iter().map(|s| plain(s, total)).collect::<Result<Vec<_>>>()?.into_iter();
        let mut q = INT8.iter().map(|s| qw(nx, li, s, fill, total)).collect::<Result<Vec<_>>>()?.into_iter();
        Ok(Dense {
            norm1: p.next().unwrap(),
            norm2: p.next().unwrap(),
            router: p.next().unwrap(),
            sh_gw: p.next().unwrap(),
            mixer,
            sh_gate: q.next().unwrap(),
            sh_up: q.next().unwrap(),
            sh_down: q.next().unwrap(),
        })
    }

    /// Copy layer `li`'s weights into this slot (same kind of layer, same shapes).
    fn load(&self, nx: &NxFile, li: usize) -> Result<()> {
        let up = |b: &Buf, s: &str| -> Result<()> {
            let bytes = nx.bytes(&format!("blocks.{li}.{s}"))?;
            if bytes.len() != b.bytes {
                return Err(format!("blocks.{li}.{s}: {} bytes, slot holds {}", bytes.len(), b.bytes));
            }
            b.upload(bytes)
        };
        let upq = |w: &QW, s: &str| -> Result<()> {
            up(&w.w, &format!("{s}.qweight"))?;
            up(&w.s, &format!("{s}.scale"))
        };
        match &self.mixer {
            Mixer::Attn { q, k, v, o, q_norm, k_norm } => {
                up(q_norm, ATTN_PLAIN[0])?;
                up(k_norm, ATTN_PLAIN[1])?;
                for (w, s) in [q, k, v, o].into_iter().zip(ATTN_INT8) {
                    upq(w, s)?;
                }
            }
            Mixer::Delta { in_qkv, in_z, in_a, in_b, conv, a_log, dt_bias, norm, out } => {
                for (b, s) in [in_a, in_b, conv, a_log, dt_bias, norm].into_iter().zip(DELTA_PLAIN) {
                    up(b, s)?;
                }
                for (w, s) in [in_qkv, in_z, out].into_iter().zip(DELTA_INT8) {
                    upq(w, s)?;
                }
            }
        }
        for (b, s) in [&self.norm1, &self.norm2, &self.router, &self.sh_gw].into_iter().zip(PLAIN) {
            up(b, s)?;
        }
        for (w, s) in [&self.sh_gate, &self.sh_up, &self.sh_down].into_iter().zip(INT8) {
            upq(w, s)?;
        }
        Ok(())
    }
}

fn qw(nx: &NxFile, li: usize, s: &str, fill: bool, total: &mut usize) -> Result<QW> {
    let (wn, sn) = (format!("blocks.{li}.{s}.qweight"), format!("blocks.{li}.{s}.scale"));
    let (wi, si) = (nx.info(&wn).map_err(|_| format!("{wn} missing: rebuild with aegist.convert_max"))?, nx.info(&sn)?);
    *total += wi.nbytes + si.nbytes;
    let (n, k) = (wi.shape[0], wi.shape[1]);
    if fill {
        Ok(QW { w: Buf::from_bytes(nx.bytes(&wn)?)?, s: Buf::from_bytes(nx.bytes(&sn)?)?, n, k })
    } else {
        Ok(QW { w: Buf::alloc(wi.nbytes)?, s: Buf::alloc(si.nbytes)?, n, k })
    }
}

/// Bytes of layer `li`'s dense weights in the file.
fn dense_bytes(nx: &NxFile, li: usize) -> usize {
    nx.tensors.range(format!("blocks.{li}.")..format!("blocks.{li}/")).map(|(_, t)| t.nbytes).sum()
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
    router: Kernel,
    gu_gemv: Kernel,
    dn_gemv: Kernel,
    gu_gemm: Kernel,
    dn_gemm: Kernel,
    moe_reduce: Kernel,
    shared_gate_add: Kernel,
}

impl Kernels {
    fn new(gpu: &Gpu) -> Result<Kernels> {
        let f = |n: &str| gpu.function(n);
        Ok(Kernels {
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
            router: f("router_topk_wide")?,
            gu_gemv: f("moe3_gu_gemv")?,
            dn_gemv: f("moe3_dn_gemv")?,
            gu_gemm: f("moe3_gu_gemm")?,
            dn_gemm: f("moe3_dn_gemm")?,
            moe_reduce: f("moe_reduce")?,
            shared_gate_add: f("shared_gate_add")?,
        })
    }
}

struct Scratch {
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
}

/// Byte offsets inside the expert table upload: (ustart, ptok, psrc, total), after the device
/// addresses (8 bytes per expert of the batch) at 0.
fn meta_layout(c: &MConfig, chunk: usize) -> (usize, usize, usize, usize) {
    let a16 = |n: usize| (n + 15) / 16 * 16;
    let picks = chunk * c.top_k * 4;
    let us = c.experts * 8;
    let pt = us + a16((c.experts + 1) * 4);
    let ps = pt + a16(picks);
    (us, pt, ps, ps + picks)
}

impl Scratch {
    fn new(c: &MConfig, ch: usize) -> Result<Scratch> {
        let f32b = |n: usize| Buf::alloc(n.max(1) * 4);
        let big = (c.heads * 2 * c.head_dim).max(c.conv_dim());
        let wide = (c.heads * c.head_dim).max(c.hv * c.dv);
        Ok(Scratch {
            x: f32b(ch * c.d)?,
            h: f32b(ch * c.d)?,
            big: f32b(ch * big)?,
            kb: f32b(ch * c.kv_heads * c.head_dim)?,
            vb: f32b(ch * c.kv_heads * c.head_dim)?,
            qn: f32b(ch * c.heads * c.head_dim)?,
            att: f32b(ch * wide)?,
            z: f32b(ch * c.hv * c.dv)?,
            y: f32b(ch * c.conv_dim())?,
            o: f32b(ch * c.hv * c.dv)?,
            gb: f32b(ch * 2 * c.hv)?,
            idx: Buf::alloc(ch * c.top_k * 4)?,
            wts: f32b(ch * c.top_k)?,
            act: f32b(ch * c.top_k * c.expert_hidden)?,
            ypick: f32b(ch * c.top_k * c.d)?,
            sh_g: f32b(ch * c.shared_hidden)?,
            sh_u: f32b(ch * c.shared_hidden)?,
            sh_out: f32b(ch * c.d)?,
            meta: Buf::alloc(meta_layout(c, ch).3)?,
        })
    }
}

/// One GPU's run of layers `first..end`: resident dense weights for some, streaming slots for the
/// rest, KV cache and DeltaNet state for its layers, an expert pool and working memory.
struct Part {
    gpu: Gpu,
    k: Kernels,
    s: Scratch,
    first: usize,
    end: usize,
    kc: Buf,
    vc: Buf,
    conv: Buf,
    state: Buf,
    snap_conv: Vec<Buf>,
    snap_state: Vec<Buf>,
    spare_attn: Option<Dense>,
    spare_delta: Option<Dense>,
    pool: Pool,
    batch: usize,
    weight_bytes: usize,
}

struct Layer {
    dev: usize,
    /// resident weights, or None: copied into the part's spare slot each pass
    dense: Option<Dense>,
    /// index of this layer among its part's attention (KV) or DeltaNet (state) layers
    slot: usize,
}

pub struct MaxEngine {
    pub cfg: MConfig,
    nx: NxFile,
    parts: Vec<Part>,
    layers: Vec<Layer>,
    readers: Vec<Reader>,
    embed_at: usize,
    norm: Buf,
    lm_head: QW,
    logits: Buf,
    snap_pos: Vec<Option<usize>>,
    pub cap: usize,
    pub length: usize,
    last_t: usize,
    chunk: usize,
    stream_tokens: usize,
    gemv_tokens: usize,
    streamed_layers: usize,
    pub read_secs: f64,
}

impl MaxEngine {
    /// Load on every GPU (layers split in proportion to free VRAM). `max_context` 0 picks 8k.
    pub fn load(nx: NxFile, max_context: usize, mut progress: impl FnMut(usize, usize)) -> Result<MaxEngine> {
        let cfg = MConfig::from_header(&nx.header["config"])?;
        for (i, f) in cfg.files.iter().enumerate() {
            let want = (cfg.experts * EXPERT_BYTES) as u64;
            let have = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
            if have < want {
                return Err(format!("expert file for layer {i} ({f}) is {have} bytes, needs {want}: is the download finished?"));
            }
        }
        let cap = if max_context == 0 { 8192 } else { max_context };
        let chunk = env_usize("AEGIST_MAX_CHUNK").unwrap_or(CHUNK).clamp(1, 4096);
        let n_gpu = crate::cuda::device_count()?.max(1);
        let pool_floor = (std::env::var("AEGIST_MAX_POOL_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(POOL_GB) * 1e9) as usize;
        let headroom = env_usize("AEGIST_VRAM_HEADROOM_MB").unwrap_or(1024) << 20;

        // split the layers between the GPUs in proportion to their free memory
        let gpus: Vec<Gpu> = (0..n_gpu).map(|g| Gpu::on(g, PTX)).collect::<Result<_>>()?;
        let free: Vec<usize> = gpus.iter().map(|g| g.mem_info().0).collect();
        let all: usize = free.iter().sum();
        let mut bounds = vec![0];
        let mut acc = 0;
        for f in &free[..n_gpu - 1] {
            acc += f;
            bounds.push(((cfg.layers * acc + all / 2) / all).clamp(*bounds.last().unwrap(), cfg.layers));
        }
        bounds.push(cfg.layers);

        let mut parts = vec![];
        let mut layers = vec![];
        let mut done = 0;
        let mut total_head = 0usize;
        let (norm, lm_head) = {
            // the output head lives on the last GPU
            gpus[n_gpu - 1].bind()?;
            (Buf::from_bytes(nx.bytes("norm.weight")?)?, qhead(&nx, &mut total_head)?)
        };
        for (g, gpu) in gpus.into_iter().enumerate() {
            gpu.bind()?;
            let (first, end) = (bounds[g], bounds[g + 1]);
            let n_attn = (first..end).filter(|&i| cfg.attn[i]).count();
            let n_delta = end - first - n_attn;
            let kv = cfg.kv_heads * cap * cfg.head_dim * 2;
            let conv_b = n_delta * cfg.conv_dim() * 3 * 4;
            let state_b = n_delta * cfg.hv * cfg.dk * cfg.dv * 4;
            let mut wb = 0;
            let kc = Buf::alloc((kv * n_attn).max(16)).map_err(|e| format!("KV cache for {cap} tokens: {e}"))?;
            let vc = Buf::alloc((kv * n_attn).max(16))?;
            let conv = Buf::zeroed(conv_b.max(16))?;
            let state = Buf::zeroed(state_b.max(16))?;
            let mut snap_conv = vec![];
            let mut snap_state = vec![];
            for _ in 0..SNAPSHOTS {
                snap_conv.push(Buf::alloc(conv_b.max(16))?);
                snap_state.push(Buf::alloc(state_b.max(16))?);
            }
            let s = Scratch::new(&cfg, chunk)?;
            // one spare slot per kind of layer, then resident layers while VRAM lasts
            let first_attn = (first..end).find(|&i| cfg.attn[i]);
            let first_delta = (first..end).find(|&i| !cfg.attn[i]);
            let spare_attn = first_attn.map(|i| Dense::alloc(&nx, i, true, false, &mut wb)).transpose()?;
            let spare_delta = first_delta.map(|i| Dense::alloc(&nx, i, false, false, &mut wb)).transpose()?;
            let (mut ai, mut di) = (0, 0);
            for li in first..end {
                let slot = if cfg.attn[li] { ai += 1; ai - 1 } else { di += 1; di - 1 };
                layers.push(Layer { dev: g, dense: None, slot });
            }
            let k = Kernels::new(&gpu)?;
            // resident dense layers: as many as leave pool_floor + headroom free
            for li in first..end {
                progress(done, cfg.layers);
                done += 1;
                let (free_now, _) = gpu.mem_info();
                if free_now < dense_bytes(&nx, li) + pool_floor + headroom {
                    break;
                }
                layers[li].dense = Some(Dense::alloc(&nx, li, cfg.attn[li], true, &mut wb)?);
            }
            done = end;
            // the rest of the memory: expert slots
            let (free_now, _) = gpu.mem_info();
            let want = free_now.saturating_sub(headroom) / EXPERT_BYTES;
            let per_chunk = 16;
            let mut chunks = vec![];
            let mut got = 0;
            while got < want {
                let n = per_chunk.min(want - got);
                match Buf::alloc(n * EXPERT_BYTES) {
                    Ok(b) => {
                        chunks.push(b);
                        got += n;
                    }
                    Err(_) => break,
                }
            }
            if got < cfg.top_k {
                return Err(format!("GPU {g}: room for only {got} expert slots (needs at least {}); close GPU-heavy apps", cfg.top_k));
            }
            let pool = Pool::new(chunks, per_chunk, got, EXPERT_BYTES, cfg.experts, cfg.layers * cfg.experts);
            wb += got * EXPERT_BYTES;
            parts.push(Part {
                gpu,
                k,
                s,
                first,
                end,
                kc,
                vc,
                conv,
                state,
                snap_conv,
                snap_state,
                spare_attn,
                spare_delta,
                batch: got,
                pool,
                weight_bytes: wb,
            });
        }
        parts.last_mut().unwrap().weight_bytes += total_head;
        progress(cfg.layers, cfg.layers);
        let readers = cfg.files.iter().map(|f| Reader::open(f, READ_THREADS)).collect::<Result<Vec<_>>>()?;
        let last = parts.last().unwrap();
        last.gpu.bind()?;
        let logits = Buf::alloc(cfg.vocab * 4)?;
        let streamed_layers = layers.iter().filter(|l| l.dense.is_none()).count();
        Ok(MaxEngine {
            embed_at: nx.offset_of("embed.weight")?,
            nx,
            parts,
            layers,
            readers,
            norm,
            lm_head,
            logits,
            snap_pos: vec![None; SNAPSHOTS],
            cap,
            length: 0,
            last_t: 0,
            chunk,
            stream_tokens: env_usize("AEGIST_STREAM_TOKENS").unwrap_or(STREAM_TOKENS),
            gemv_tokens: env_usize("AEGIST_GEMV_TOKENS").unwrap_or(GEMV_TOKENS),
            streamed_layers,
            read_secs: 0.0,
            cfg,
        })
    }
}

fn qhead(nx: &NxFile, total: &mut usize) -> Result<QW> {
    crate::model::qweight(nx, "lm_head", total)
}

impl Part {
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

    fn znorm(&self, c: &MConfig, w: &Buf, src: u64, dst: u64, rows: usize) -> Result<()> {
        self.k.znorm.launch((rows as u32, 1), 256, &[Arg::Ptr(src), Arg::Ptr(w.ptr), Arg::Ptr(dst), Arg::I32(c.d as i32), Arg::F32(c.eps)])
    }
}

impl MaxEngine {
    /// The weights of layer `li`: resident, or its part's spare slot (filled by `forward_chunk`).
    fn weights(&self, li: usize) -> &Dense {
        let l = &self.layers[li];
        let p = &self.parts[l.dev];
        match &l.dense {
            Some(d) => d,
            None if self.cfg.attn[li] => p.spare_attn.as_ref().unwrap(),
            None => p.spare_delta.as_ref().unwrap(),
        }
    }

    /// Embedding rows looked up on the host (the table stays in the mapped file), widened to fp32.
    fn embed_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let d = self.cfg.d;
        let mut bytes = Vec::with_capacity(ids.len() * d * 4);
        for &id in ids {
            let id = id as usize;
            if id >= self.cfg.vocab {
                return Err(format!("token id {id} is outside the vocabulary ({})", self.cfg.vocab));
            }
            for p in self.nx.raw(self.embed_at + id * d * 2, d * 2).chunks_exact(2) {
                let bits = (u16::from_le_bytes([p[0], p[1]]) as u32) << 16;
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
        }
        Ok(bytes)
    }

    fn forward_chunk(&mut self, ids: &[u32]) -> Result<()> {
        let t = ids.len();
        let row = self.cfg.d * 4;
        let mut cur = self.layers[0].dev;
        self.parts[cur].gpu.bind()?;
        self.parts[cur].s.x.upload(&self.embed_bytes(ids)?)?;
        for li in 0..self.cfg.layers {
            let dev = self.layers[li].dev;
            if dev != cur {
                let x = self.parts[cur].s.x.download_at(0, t * row)?;
                self.parts[dev].gpu.bind()?;
                self.parts[dev].s.x.upload(&x)?;
                cur = dev;
            }
            if self.layers[li].dense.is_none() {
                self.weights(li).load(&self.nx, li)?;
            }
            self.mixer(li, t)?;
            self.moe(li, t)?;
        }
        let last = self.parts.len() - 1;
        if cur != last {
            let x = self.parts[cur].s.x.download_at(0, t * row)?;
            self.parts[last].gpu.bind()?;
            self.parts[last].s.x.upload(&x)?;
        }
        self.length += t;
        self.last_t = t;
        Ok(())
    }

    fn mixer(&self, li: usize, t: usize) -> Result<()> {
        let c = &self.cfg;
        let l = &self.layers[li];
        let p = &self.parts[l.dev];
        let (k, s) = (&p.k, &p.s);
        let w = self.weights(li);
        let pos0 = self.length as i32;
        p.znorm(c, &w.norm1, s.x.ptr, s.h.ptr, t)?;
        match &w.mixer {
            Mixer::Attn { q, k: wk, v, o, q_norm, k_norm } => {
                let kv_layer = c.kv_heads * self.cap * c.head_dim * 2;
                let (kc, vc) = (p.kc.at(l.slot * kv_layer), p.vc.at(l.slot * kv_layer));
                p.mm(q, &s.h, &s.big, t, false)?;
                p.mm(wk, &s.h, &s.kb, t, false)?;
                p.mm(v, &s.h, &s.vb, t, false)?;
                k.attn_prep.launch(
                    (t as u32, (c.heads + c.kv_heads) as u32),
                    c.head_dim as u32,
                    &[
                        Arg::Ptr(s.big.ptr), Arg::Ptr(s.kb.ptr), Arg::Ptr(s.vb.ptr), Arg::Ptr(q_norm.ptr), Arg::Ptr(k_norm.ptr), Arg::Ptr(s.qn.ptr),
                        Arg::Ptr(kc), Arg::Ptr(vc), Arg::I32(c.heads as i32), Arg::I32(c.kv_heads as i32), Arg::I32(c.head_dim as i32),
                        Arg::I32(c.rotary_dim as i32), Arg::I32(pos0), Arg::I32(self.cap as i32), Arg::F32(c.eps), Arg::F32(c.rope_theta),
                    ],
                )?;
                k.attention_gated.launch(
                    (t as u32, c.heads as u32),
                    256,
                    &[
                        Arg::Ptr(s.qn.ptr), Arg::Ptr(kc), Arg::Ptr(vc), Arg::Ptr(s.big.ptr), Arg::Ptr(s.att.ptr), Arg::I32(c.heads as i32),
                        Arg::I32(c.kv_heads as i32), Arg::I32(pos0), Arg::I32(self.cap as i32), Arg::F32(1.0 / (c.head_dim as f32).sqrt()),
                    ],
                )?;
                p.mm(o, &s.att, &s.x, t, true)
            }
            Mixer::Delta { in_qkv, in_z, in_a, in_b, conv, a_log, dt_bias, norm, out } => {
                let cd = c.conv_dim();
                let conv_state = p.conv.at(l.slot * cd * 3 * 4);
                let state = p.state.at(l.slot * c.hv * c.dk * c.dv * 4);
                p.mm(in_qkv, &s.h, &s.big, t, false)?;
                p.mm(in_z, &s.h, &s.z, t, false)?;
                k.delta_gates.launch(
                    (t as u32, 1),
                    256,
                    &[
                        Arg::Ptr(s.h.ptr), Arg::Ptr(in_b.ptr), Arg::Ptr(in_a.ptr), Arg::Ptr(a_log.ptr), Arg::Ptr(dt_bias.ptr), Arg::Ptr(s.gb.ptr),
                        Arg::I32(c.d as i32), Arg::I32(c.hv as i32),
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
                // output_gate_type "swish": the silu gate of gated_norm
                k.gated_norm.launch(
                    (t as u32, c.hv as u32),
                    c.dv as u32,
                    &[Arg::Ptr(s.o.ptr), Arg::Ptr(s.z.ptr), Arg::Ptr(norm.ptr), Arg::Ptr(s.att.ptr), Arg::I32(c.hv as i32), Arg::I32(c.dv as i32), Arg::F32(c.eps)],
                )?;
                p.mm(out, &s.att, &s.x, t, true)
            }
        }
    }

    fn moe(&mut self, li: usize, t: usize) -> Result<()> {
        let (d, kk, e_n, ih) = (self.cfg.d, self.cfg.top_k, self.cfg.experts, self.cfg.expert_hidden);
        let dev = self.layers[li].dev;
        {
            let (p, w) = (&self.parts[dev], self.weights(li));
            p.znorm(&self.cfg, &w.norm2, p.s.x.ptr, p.s.h.ptr, t)?;
            let rb = ((e_n + 31) / 32 * 32).max(64) as u32;
            p.k.router.launch(
                (t as u32, 1),
                rb,
                &[Arg::Ptr(p.s.h.ptr), Arg::Ptr(w.router.ptr), Arg::Ptr(p.s.idx.ptr), Arg::Ptr(p.s.wts.ptr), Arg::I32(d as i32), Arg::I32(e_n as i32), Arg::I32(kk as i32)],
            )?;
        }
        // group the picks by expert (the one host sync per layer)
        let idx = self.parts[dev].s.idx.download_i32(t * kk)?;
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
        let (o_us, o_pt, o_ps, _) = meta_layout(&self.cfg, self.chunk);
        {
            let m = &self.parts[dev].s.meta;
            m.upload_at(o_pt, &bytes_i32(&ptok))?;
            m.upload_at(o_ps, &bytes_i32(&psrc))?;
        }
        let streaming = t > self.stream_tokens;
        let batch = self.parts[dev].batch;
        // experts in batches the pool can hold at once: each batch is read, copied in, then run
        for a in (0..uniq.len()).step_by(batch) {
            let b = (a + batch).min(uniq.len());
            let (parts, readers) = (&mut self.parts, &mut self.readers);
            let p = &mut parts[dev];
            let (ptrs, loads) = p.pool.assign(li, &uniq[a..b], streaming)?;
            let jobs: Vec<(usize, usize)> = loads.iter().map(|&(slot, e)| (slot, e * EXPERT_BYTES)).collect();
            let t0 = std::time::Instant::now();
            let pool = &p.pool;
            if let Err(err) = readers[li].load(&jobs, EXPERT_BYTES, |slot, bytes| pool.upload(slot, bytes)) {
                p.pool.forget(&loads);
                return Err(err);
            }
            self.read_secs += t0.elapsed().as_secs_f64();
            let p = &self.parts[dev];
            let eptr: Vec<u8> = ptrs.iter().flat_map(|v| v.to_le_bytes()).collect();
            p.s.meta.upload_at(0, &eptr)?;
            p.s.meta.upload_at(o_us, &bytes_i32(&ustart[a..=b]))?;
            let s = &p.s;
            let (eptr, us, pt, ps) = (s.meta.ptr, s.meta.at(o_us), s.meta.at(o_pt), s.meta.at(o_ps));
            let nu = (b - a) as u32;
            let (di, ii) = (Arg::I32(d as i32), Arg::I32(ih as i32));
            let gu_args = [
                Arg::Ptr(s.h.ptr), Arg::Ptr(eptr), Arg::Ptr(us), Arg::Ptr(pt), Arg::Ptr(s.act.ptr), di, ii, Arg::I32(GU_SCALES), Arg::I32(UP_CODES),
                Arg::I32(UP_SCALES),
            ];
            let dn_args = [
                Arg::Ptr(s.act.ptr), Arg::Ptr(eptr), Arg::Ptr(us), Arg::Ptr(ps), Arg::Ptr(s.wts.ptr), Arg::Ptr(s.ypick.ptr), di, ii, Arg::I32(DN_CODES),
                Arg::I32(DN_SCALES),
            ];
            if t <= self.gemv_tokens {
                p.k.gu_gemv.launch((((ih + 7) / 8) as u32, nu), 256, &gu_args)?;
                p.k.dn_gemv.launch((((d + 7) / 8) as u32, nu), 256, &dn_args)?;
            } else {
                p.k.gu_gemm.launch(((ih / 32) as u32, nu), 256, &gu_args)?;
                p.k.dn_gemm.launch(((d / 64) as u32, nu), 256, &dn_args)?;
            }
        }
        let (p, w) = (&self.parts[dev], self.weights(li));
        let (k, s) = (&p.k, &p.s);
        k.moe_reduce.launch((t as u32, ((d + 255) / 256) as u32), 256, &[Arg::Ptr(s.ypick.ptr), Arg::Ptr(s.x.ptr), Arg::I32(d as i32), Arg::I32(kk as i32)])?;

        // shared expert, gated
        let sh = self.cfg.shared_hidden;
        if t == 1 {
            k.gemv_i8_gateup.launch(
                (((sh + 7) / 8) as u32, 1),
                256,
                &[
                    Arg::Ptr(s.h.ptr), Arg::Ptr(w.sh_gate.w.ptr), Arg::Ptr(w.sh_gate.s.ptr), Arg::Ptr(w.sh_up.w.ptr), Arg::Ptr(w.sh_up.s.ptr),
                    Arg::Ptr(s.sh_g.ptr), Arg::I32(sh as i32), Arg::I32(d as i32),
                ],
            )?;
        } else {
            p.mm(&w.sh_gate, &s.h, &s.sh_g, t, false)?;
            p.mm(&w.sh_up, &s.h, &s.sh_u, t, false)?;
            let n = (t * sh) as i32;
            k.silu_mul.launch((((n + 255) / 256) as u32, 1), 256, &[Arg::Ptr(s.sh_g.ptr), Arg::Ptr(s.sh_u.ptr), Arg::Ptr(s.sh_g.ptr), Arg::I32(n)])?;
        }
        p.mm(&w.sh_down, &s.sh_g, &s.sh_out, t, false)?;
        k.shared_gate_add.launch((t as u32, 1), 256, &[Arg::Ptr(s.x.ptr), Arg::Ptr(s.sh_out.ptr), Arg::Ptr(s.h.ptr), Arg::Ptr(w.sh_gw.ptr), Arg::I32(d as i32)])
    }

    fn head(&mut self) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let p = self.parts.last().unwrap();
        p.gpu.bind()?;
        let last = p.s.x.at((self.last_t - 1) * c.d * 4);
        p.znorm(c, &self.norm, last, p.s.h.ptr, 1)?;
        p.mm(&self.lm_head, &p.s.h, &self.logits, 1, false)?;
        self.logits.download_f32(c.vocab)
    }

    fn each_part(&self, mut f: impl FnMut(&Part) -> Result<()>) -> Result<()> {
        for p in &self.parts {
            p.gpu.bind()?;
            f(p)?;
        }
        Ok(())
    }
}

impl Lm for MaxEngine {
    fn feed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        assert!(!ids.is_empty());
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
            Some(i) => self
                .each_part(|p| {
                    p.conv.copy_from(&p.snap_conv[i], p.conv.bytes)?;
                    p.state.copy_from(&p.snap_state[i], p.state.bytes)
                })
                .map(|_| self.snap_pos[i].unwrap()),
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
                // no usable snapshot: start over
                let _ = self.each_part(|p| {
                    p.conv.zero()?;
                    p.state.zero()
                });
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
        let i = (0..SNAPSHOTS).min_by_key(|&i| self.snap_pos[i].map_or(-1, |p| p as i64)).unwrap();
        self.each_part(|p| {
            p.snap_conv[i].copy_from(&p.conv, p.conv.bytes)?;
            p.snap_state[i].copy_from(&p.state, p.state.bytes)
        })?;
        self.snap_pos[i] = Some(self.length);
        Ok(())
    }

    fn length(&self) -> usize {
        self.length
    }

    fn cap(&self) -> usize {
        self.cap
    }

    fn core_counts(&self) -> Result<Vec<i32>> {
        Ok(vec![])
    }

    fn reset_counts(&self) -> Result<()> {
        Ok(())
    }

    fn expert_counters(&self) -> (u64, u64) {
        self.parts.iter().fold((0, 0), |(h, m), p| (h + p.pool.hits, m + p.pool.page_ins))
    }

    fn read_seconds(&self) -> f64 {
        self.read_secs
    }

    fn describe(&self) -> String {
        let gb: f64 = self.parts.iter().map(|p| p.weight_bytes as f64 / 1e9).sum();
        let parts: Vec<String> =
            self.parts.iter().map(|p| format!("GPU {} {} layers {}-{} with {} expert slots", p.gpu.ordinal, p.gpu.name, p.first, p.end - 1, p.pool.slots())).collect();
        format!(
            "{:.1} GB in VRAM; {}; {} of {} layers' dense weights streamed per pass; context {} tokens",
            gb,
            parts.join("; "),
            self.streamed_layers,
            self.cfg.layers,
            self.cap
        )
    }
}
