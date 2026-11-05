//! Aegist's deep-think brain on the GPU (arch "kimi-mla"): Kimi-K2.6 (Moonshot AI), DeepSeek-V3-style:
//!
//!   x += o_proj(MLA(norm1(x)));  h = norm2(x);  x += moe(h) + shared(h)   (layer 0: a dense MLP)
//!
//! MLA attention runs in the 512-wide latent space: the cache holds only the normalised latent and the
//! roped 64-wide shared key per position (576 floats), and kv_b_proj is applied to queries and outputs.
//!
//! Tiers: the dense weights (int8 linears, bf16 norms / router / kv_b, ~11 GB) are resident, split
//! across the GPUs by layer runs; the 23,040 experts (int4 as published, 24.8 MB each, in per-layer
//! files from aegist/fetch_k2.py) are paged through an LRU pool in each GPU's spare memory. The
//! embedding stays in the mapped file; the head is resident on the last GPU.
use crate::cuda::{Arg, Buf, Gpu, Kernel, Result};
use crate::hybrid::{env_usize, Pool, READ_THREADS};
use crate::model::{bytes_i32, Lm, PTX, QW};
use crate::nx::NxFile;
use crate::reader::Reader;
use serde_json::Value;

/// One int4 expert record and the offsets inside it (aegist/fetch_k2.py).
const EXPERT_BYTES: usize = 24_772_608;
const GU_SCALES: i32 = 7_340_032;
const UP_CODES: i32 = 8_257_536;
const UP_SCALES: i32 = 15_597_568;
const DN_CODES: i32 = 16_515_072;
const DN_SCALES: i32 = 23_855_104;
/// The latent cache per position: c (kv_lora) then kpe (qk_rope).
const LATENT: usize = 512;
const ROPE: usize = 64;
const NOPE: usize = 128;
const VHEAD: usize = 128;

const GEMV_TOKENS: usize = 8;
const STREAM_TOKENS: usize = 64;
/// Prefill chunk ($AEGIST_KIMI_CHUNK).
const CHUNK: usize = 256;

#[derive(Clone, Debug)]
pub struct KConfig {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub first_dense: usize,
    pub dense_hidden: usize,
    pub heads: usize,
    pub q_lora: usize,
    pub experts: usize,
    pub top_k: usize,
    pub expert_hidden: usize,
    pub routed_scale: f32,
    pub eps: f32,
    pub softmax_scale: f32,
    /// rotary frequencies after YaRN correction (32 = ROPE / 2)
    pub inv_freq: Vec<f32>,
    /// per layer: the file holding its experts ("" for dense layers)
    pub files: Vec<String>,
}

pub fn is_kimi(c: &Value) -> bool {
    c["arch"].as_str() == Some("kimi-mla")
}

fn yarn(theta: f64, rs: &Value) -> Result<(Vec<f32>, f64)> {
    let g = |k: &str| rs[k].as_f64().ok_or(format!("rope_scaling is missing {k}"));
    let (factor, orig) = (g("factor")?, g("original_max_position_embeddings")?);
    let (fast, slow) = (g("beta_fast")?, g("beta_slow")?);
    let (mscale, mscale_all) = (g("mscale")?, g("mscale_all_dim")?);
    let dim = ROPE as f64;
    let corr = |rot: f64| dim * (orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * theta.ln());
    let low = corr(fast).floor().max(0.0);
    let mut high = corr(slow).ceil().min(dim - 1.0);
    if low == high {
        high += 0.001;
    }
    let inv: Vec<f32> = (0..ROPE / 2)
        .map(|i| {
            let extra = 1.0 / theta.powf(2.0 * i as f64 / dim);
            let inter = extra / factor;
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let keep = 1.0 - ramp; // inv_freq_mask: 1 keeps the original (extrapolated) frequency
            (inter * (1.0 - keep) + extra * keep) as f32
        })
        .collect();
    let ms = |m: f64| if factor <= 1.0 { 1.0 } else { 0.1 * m * factor.ln() + 1.0 };
    if (ms(mscale) / ms(mscale_all) - 1.0).abs() > 1e-9 {
        return Err("this engine assumes YaRN cos/sin are not rescaled (mscale == mscale_all_dim)".into());
    }
    Ok((inv, ms(mscale_all)))
}

impl KConfig {
    pub fn from_header(c: &Value) -> Result<KConfig> {
        let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or(format!("config is missing {k}"));
        let f = |k: &str| c[k].as_f64().ok_or(format!("config is missing {k}"));
        let layers = u("n_layers")?;
        let first_dense = u("first_dense")?;
        let files = (0..layers)
            .map(|i| {
                if i < first_dense {
                    return Ok(String::new());
                }
                c["expert_files"][format!("L{i:02}")].as_str().map(String::from).ok_or(format!("config has no expert file for layer {i}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let (inv_freq, mscale) = yarn(f("rope_theta")?, &c["rope_scaling"])?;
        let qk = (u("qk_nope")? + u("qk_rope")?) as f64;
        let cfg = KConfig {
            vocab: u("vocab_size")?,
            d: u("d_model")?,
            layers,
            first_dense,
            dense_hidden: u("dense_hidden")?,
            heads: u("n_heads")?,
            q_lora: u("q_lora")?,
            experts: u("n_experts")?,
            top_k: u("experts_active")?,
            expert_hidden: u("expert_hidden")?,
            routed_scale: f("routed_scale")? as f32,
            eps: f("norm_eps")? as f32,
            softmax_scale: (mscale * mscale / qk.sqrt()) as f32,
            inv_freq,
            files,
        };
        let mut bad = vec![];
        if c["expert_format"].as_str() != Some("q4g32s") || u("expert_bytes")? != EXPERT_BYTES {
            bad.push("experts (needs q4g32s records of 24,772,608 bytes)".to_string());
        }
        if u("kv_lora")? != LATENT || u("qk_rope")? != ROPE || u("qk_nope")? != NOPE || u("v_head")? != VHEAD {
            bad.push("MLA shape (needs latent 512, rope 64, nope 128, v 128)".into());
        }
        if cfg.d != 7168 || cfg.expert_hidden != 2048 || cfg.experts > 1024 || cfg.top_k > 16 || c["scoring"].as_str() != Some("sigmoid") {
            bad.push(format!("MoE d {} hidden {} experts {} top-{}", cfg.d, cfg.expert_hidden, cfg.experts, cfg.top_k));
        }
        if c["norm_topk"].as_bool() != Some(true) || u("n_shared")? != 1 {
            bad.push("routing (needs normalised top-k and one shared expert)".into());
        }
        if bad.is_empty() { Ok(cfg) } else { Err(format!("this engine build does not support this brain's shape: {}", bad.join("; "))) }
    }
}

enum Mlp {
    Dense { gate: QW, up: QW, down: QW },
    Moe { router: Buf, bias: Buf, sh_gate: QW, sh_up: QW, sh_down: QW },
}

/// One layer's dense weights (resident on its GPU).
struct Dense {
    norm1: Buf,
    norm2: Buf,
    q_a: QW,
    q_norm: Buf,
    q_b: QW,
    kv_a: QW,
    kv_norm: Buf,
    kv_b: Buf,
    o: QW,
    mlp: Mlp,
}

impl Dense {
    fn load(nx: &NxFile, li: usize, total: &mut usize) -> Result<Dense> {
        let plain = |s: &str, total: &mut usize| -> Result<Buf> {
            let name = format!("blocks.{li}.{s}");
            let t = nx.info(&name).map_err(|_| format!("{name} missing: rebuild with aegist.convert_k2"))?;
            *total += t.nbytes;
            Buf::from_bytes(nx.bytes(&name)?)
        };
        let q = |s: &str, total: &mut usize| crate::model::qweight(nx, &format!("blocks.{li}.{s}"), total);
        let mlp = if nx.info(&format!("blocks.{li}.mlp.gate.weight")).is_ok() {
            if nx.info(&format!("blocks.{li}.mlp.gate.e_score_correction_bias"))?.dtype != "F32" {
                return Err("router bias must be float32: rebuild with aegist.convert_k2".into());
            }
            Mlp::Moe {
                router: plain("mlp.gate.weight", total)?,
                bias: plain("mlp.gate.e_score_correction_bias", total)?,
                sh_gate: q("mlp.shared_experts.gate_proj", total)?,
                sh_up: q("mlp.shared_experts.up_proj", total)?,
                sh_down: q("mlp.shared_experts.down_proj", total)?,
            }
        } else {
            Mlp::Dense { gate: q("mlp.gate_proj", total)?, up: q("mlp.up_proj", total)?, down: q("mlp.down_proj", total)? }
        };
        Ok(Dense {
            norm1: plain("input_layernorm.weight", total)?,
            norm2: plain("post_attention_layernorm.weight", total)?,
            q_a: q("self_attn.q_a_proj", total)?,
            q_norm: plain("self_attn.q_a_layernorm.weight", total)?,
            q_b: q("self_attn.q_b_proj", total)?,
            kv_a: q("self_attn.kv_a_proj_with_mqa", total)?,
            kv_norm: plain("self_attn.kv_a_layernorm.weight", total)?,
            kv_b: plain("self_attn.kv_b_proj.weight", total)?,
            o: q("self_attn.o_proj", total)?,
            mlp,
        })
    }
}

struct Kernels {
    rmsnorm: Kernel,
    gemv_i8: Kernel,
    gemv_i8_gateup: Kernel,
    gemm_i8: Kernel,
    silu_mul: Kernel,
    mla_kv: Kernel,
    mla_q: Kernel,
    mla_attn: Kernel,
    mla_out: Kernel,
    router: Kernel,
    gu_gemv: Kernel,
    dn_gemv: Kernel,
    gu_gemm: Kernel,
    dn_gemm: Kernel,
    moe_reduce: Kernel,
}

impl Kernels {
    fn new(gpu: &Gpu) -> Result<Kernels> {
        let f = |n: &str| gpu.function(n);
        Ok(Kernels {
            rmsnorm: f("rmsnorm")?,
            gemv_i8: f("gemv_i8")?,
            gemv_i8_gateup: f("gemv_i8_gateup")?,
            gemm_i8: f("gemm_i8")?,
            silu_mul: f("silu_mul")?,
            mla_kv: f("mla_kv")?,
            mla_q: f("mla_q")?,
            mla_attn: f("mla_attn")?,
            mla_out: f("mla_out")?,
            router: f("k2_router")?,
            gu_gemv: f("k2q4_gu_gemv")?,
            dn_gemv: f("k2q4_dn_gemv")?,
            gu_gemm: f("k2q4_gu_gemm")?,
            dn_gemm: f("k2q4_dn_gemm")?,
            moe_reduce: f("moe_reduce")?,
        })
    }
}

struct Scratch {
    x: Buf,
    h: Buf,
    qa: Buf,
    qan: Buf,
    q: Buf,
    kv: Buf,
    qlat: Buf,
    olat: Buf,
    att: Buf,
    g: Buf,
    u: Buf,
    idx: Buf,
    wts: Buf,
    act: Buf,
    ypick: Buf,
    meta: Buf,
}

/// Byte offsets inside the expert table upload: (ustart, ptok, psrc, total), after the device
/// addresses (8 bytes per expert of the batch) at 0.
fn meta_layout(c: &KConfig, chunk: usize) -> (usize, usize, usize, usize) {
    let a16 = |n: usize| (n + 15) / 16 * 16;
    let picks = chunk * c.top_k * 4;
    let us = c.experts * 8;
    let pt = us + a16((c.experts + 1) * 4);
    let ps = pt + a16(picks);
    (us, pt, ps, ps + picks)
}

impl Scratch {
    fn new(c: &KConfig, ch: usize, dense_mlp: bool) -> Result<Scratch> {
        let f32b = |n: usize| Buf::alloc(n.max(1) * 4);
        let hid = if dense_mlp { c.dense_hidden } else { c.expert_hidden };
        Ok(Scratch {
            x: f32b(ch * c.d)?,
            h: f32b(ch * c.d)?,
            qa: f32b(ch * c.q_lora)?,
            qan: f32b(ch * c.q_lora)?,
            q: f32b(ch * c.heads * (NOPE + ROPE))?,
            kv: f32b(ch * (LATENT + ROPE))?,
            qlat: f32b(ch * c.heads * LATENT)?,
            olat: f32b(ch * c.heads * LATENT)?,
            att: f32b(ch * c.heads * VHEAD)?,
            g: f32b(ch * hid)?,
            u: f32b(ch * hid)?,
            idx: Buf::alloc(ch * c.top_k * 4)?,
            wts: f32b(ch * c.top_k)?,
            act: f32b(ch * c.top_k * c.expert_hidden)?,
            ypick: f32b(ch * c.top_k * c.d)?,
            meta: Buf::alloc(meta_layout(c, ch).3)?,
        })
    }
}

/// One GPU's run of layers `first..end`: their weights, latent caches, an expert pool, working memory.
struct Part {
    gpu: Gpu,
    k: Kernels,
    s: Scratch,
    first: usize,
    end: usize,
    /// [layers of this part][cap][512] and [..][cap][64], fp32
    ckv: Buf,
    kpe: Buf,
    inv_freq: Buf,
    pool: Pool,
    batch: usize,
    weight_bytes: usize,
}

pub struct KimiEngine {
    pub cfg: KConfig,
    nx: NxFile,
    parts: Vec<Part>,
    /// per layer: (GPU part, weights)
    layers: Vec<(usize, Dense)>,
    readers: Vec<Option<Reader>>,
    embed_at: usize,
    norm: Buf,
    lm_head: QW,
    logits: Buf,
    pub cap: usize,
    pub length: usize,
    last_t: usize,
    chunk: usize,
    stream_tokens: usize,
    gemv_tokens: usize,
    pub read_secs: f64,
}

impl KimiEngine {
    /// Load on every GPU (layers split in proportion to free VRAM). `max_context` 0 picks 8k.
    pub fn load(nx: NxFile, max_context: usize, mut progress: impl FnMut(usize, usize)) -> Result<KimiEngine> {
        let cfg = KConfig::from_header(&nx.header["config"])?;
        for (i, f) in cfg.files.iter().enumerate().filter(|(_, f)| !f.is_empty()) {
            let want = (cfg.experts * EXPERT_BYTES) as u64;
            let have = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
            if have < want {
                return Err(format!("expert file for layer {i} ({f}) is {have} bytes, needs {want}: is the download finished?"));
            }
        }
        let cap = if max_context == 0 { 8192 } else { max_context };
        let chunk = env_usize("AEGIST_KIMI_CHUNK").unwrap_or(CHUNK).clamp(1, 4096);
        let n_gpu = crate::cuda::device_count()?.max(1);
        let headroom = env_usize("AEGIST_VRAM_HEADROOM_MB").unwrap_or(1024) << 20;

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
        let mut head_bytes = 0usize;
        let (norm, lm_head) = {
            gpus[n_gpu - 1].bind()?;
            (Buf::from_bytes(nx.bytes("norm.weight")?)?, crate::model::qweight(&nx, "lm_head", &mut head_bytes)?)
        };
        for (g, gpu) in gpus.into_iter().enumerate() {
            gpu.bind()?;
            let (first, end) = (bounds[g], bounds[g + 1]);
            let n = end - first;
            let ckv = Buf::alloc((n * cap * LATENT * 4).max(16)).map_err(|e| format!("latent cache for {cap} tokens: {e}"))?;
            let kpe = Buf::alloc((n * cap * ROPE * 4).max(16))?;
            let inv_freq = Buf::from_bytes(&cfg.inv_freq.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            let s = Scratch::new(&cfg, chunk, first < cfg.first_dense)?;
            let mut wb = 0;
            for li in first..end {
                progress(li, cfg.layers);
                let (free_now, _) = gpu.mem_info();
                let d = Dense::load(&nx, li, &mut wb).map_err(|e| format!("GPU {g} layer {li} ({} MB free): {e}", free_now >> 20))?;
                layers.push((g, d));
            }
            let k = Kernels::new(&gpu)?;
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
            if got < cfg.top_k && end > first.max(cfg.first_dense) {
                return Err(format!("GPU {g}: room for only {got} expert slots (needs at least {}); close GPU-heavy apps", cfg.top_k));
            }
            let pool = Pool::new(chunks, per_chunk, got, EXPERT_BYTES, cfg.experts, cfg.layers * cfg.experts);
            wb += got * EXPERT_BYTES;
            parts.push(Part { gpu, k, s, first, end, ckv, kpe, inv_freq, pool, batch: got.max(1), weight_bytes: wb });
        }
        parts.last_mut().unwrap().weight_bytes += head_bytes;
        progress(cfg.layers, cfg.layers);
        let readers = cfg.files.iter().map(|f| if f.is_empty() { Ok(None) } else { Reader::open(f, READ_THREADS).map(Some) }).collect::<Result<Vec<_>>>()?;
        parts.last().unwrap().gpu.bind()?;
        let logits = Buf::alloc(cfg.vocab * 4)?;
        Ok(KimiEngine {
            embed_at: nx.offset_of("embed.weight")?,
            nx,
            parts,
            layers,
            readers,
            norm,
            lm_head,
            logits,
            cap,
            length: 0,
            last_t: 0,
            chunk,
            stream_tokens: env_usize("AEGIST_STREAM_TOKENS").unwrap_or(STREAM_TOKENS),
            gemv_tokens: env_usize("AEGIST_GEMV_TOKENS").unwrap_or(GEMV_TOKENS),
            read_secs: 0.0,
            cfg,
        })
    }
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

    fn norm(&self, w: &Buf, src: u64, dst: u64, rows: usize, d: usize, eps: f32) -> Result<()> {
        self.k.rmsnorm.launch((rows as u32, 1), 256, &[Arg::Ptr(src), Arg::Ptr(w.ptr), Arg::Ptr(dst), Arg::I32(d as i32), Arg::F32(eps)])
    }

    /// silu(gate . h) * (up . h) into s.g, then x += down . that.
    fn swiglu(&self, gate: &QW, up: &QW, down: &QW, t: usize) -> Result<()> {
        let (k, s) = (&self.k, &self.s);
        let n = gate.n;
        if t == 1 {
            k.gemv_i8_gateup.launch(
                (((n + 7) / 8) as u32, 1),
                256,
                &[
                    Arg::Ptr(s.h.ptr), Arg::Ptr(gate.w.ptr), Arg::Ptr(gate.s.ptr), Arg::Ptr(up.w.ptr), Arg::Ptr(up.s.ptr), Arg::Ptr(s.g.ptr),
                    Arg::I32(n as i32), Arg::I32(gate.k as i32),
                ],
            )?;
        } else {
            self.mm(gate, &s.h, &s.g, t, false)?;
            self.mm(up, &s.h, &s.u, t, false)?;
            let m = (t * n) as i32;
            k.silu_mul.launch((((m + 255) / 256) as u32, 1), 256, &[Arg::Ptr(s.g.ptr), Arg::Ptr(s.u.ptr), Arg::Ptr(s.g.ptr), Arg::I32(m)])?;
        }
        self.mm(down, &s.g, &s.x, t, true)
    }
}

impl KimiEngine {
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
        let mut cur = self.layers[0].0;
        self.parts[cur].gpu.bind()?;
        self.parts[cur].s.x.upload(&self.embed_bytes(ids)?)?;
        for li in 0..self.cfg.layers {
            let dev = self.layers[li].0;
            if dev != cur {
                let x = self.parts[cur].s.x.download_at(0, t * row)?;
                self.parts[dev].gpu.bind()?;
                self.parts[dev].s.x.upload(&x)?;
                cur = dev;
            }
            self.attention(li, t)?;
            self.mlp(li, t)?;
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

    fn attention(&self, li: usize, t: usize) -> Result<()> {
        let c = &self.cfg;
        let (dev, w) = (&self.layers[li].0, &self.layers[li].1);
        let p = &self.parts[*dev];
        let (k, s) = (&p.k, &p.s);
        let pos0 = self.length as i32;
        let slot = li - p.first;
        let (ckv, kpe) = (p.ckv.at(slot * self.cap * LATENT * 4), p.kpe.at(slot * self.cap * ROPE * 4));
        let h = c.heads as u32;
        p.norm(&w.norm1, s.x.ptr, s.h.ptr, t, c.d, c.eps)?;
        p.mm(&w.q_a, &s.h, &s.qa, t, false)?;
        p.norm(&w.q_norm, s.qa.ptr, s.qan.ptr, t, c.q_lora, c.eps)?;
        p.mm(&w.q_b, &s.qan, &s.q, t, false)?;
        p.mm(&w.kv_a, &s.h, &s.kv, t, false)?;
        k.mla_kv.launch(
            (t as u32, 1),
            256,
            &[Arg::Ptr(s.kv.ptr), Arg::Ptr(w.kv_norm.ptr), Arg::Ptr(p.inv_freq.ptr), Arg::Ptr(ckv), Arg::Ptr(kpe), Arg::I32(pos0), Arg::F32(c.eps)],
        )?;
        k.mla_q.launch(
            (t as u32, h),
            256,
            &[Arg::Ptr(s.q.ptr), Arg::Ptr(w.kv_b.ptr), Arg::Ptr(p.inv_freq.ptr), Arg::Ptr(s.qlat.ptr), Arg::I32(c.heads as i32), Arg::I32(pos0)],
        )?;
        k.mla_attn.launch(
            (t as u32, h),
            256,
            &[
                Arg::Ptr(s.qlat.ptr), Arg::Ptr(s.q.ptr), Arg::Ptr(ckv), Arg::Ptr(kpe), Arg::Ptr(s.olat.ptr), Arg::I32(c.heads as i32), Arg::I32(pos0),
                Arg::F32(c.softmax_scale),
            ],
        )?;
        k.mla_out.launch((t as u32, h), 256, &[Arg::Ptr(s.olat.ptr), Arg::Ptr(w.kv_b.ptr), Arg::Ptr(s.att.ptr), Arg::I32(c.heads as i32)])?;
        p.mm(&w.o, &s.att, &s.x, t, true)
    }

    fn mlp(&mut self, li: usize, t: usize) -> Result<()> {
        let (d, kk, e_n, ih) = (self.cfg.d, self.cfg.top_k, self.cfg.experts, self.cfg.expert_hidden);
        let dev = self.layers[li].0;
        {
            let (p, w) = (&self.parts[dev], &self.layers[li].1);
            p.norm(&w.norm2, p.s.x.ptr, p.s.h.ptr, t, d, self.cfg.eps)?;
            match &w.mlp {
                Mlp::Dense { gate, up, down } => return p.swiglu(gate, up, down, t),
                Mlp::Moe { router, bias, .. } => p.k.router.launch(
                    (t as u32, 1),
                    256,
                    &[
                        Arg::Ptr(p.s.h.ptr), Arg::Ptr(router.ptr), Arg::Ptr(bias.ptr), Arg::Ptr(p.s.idx.ptr), Arg::Ptr(p.s.wts.ptr), Arg::I32(d as i32),
                        Arg::I32(e_n as i32), Arg::I32(kk as i32), Arg::F32(self.cfg.routed_scale),
                    ],
                )?,
            }
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
        for a in (0..uniq.len()).step_by(batch) {
            let b = (a + batch).min(uniq.len());
            let (parts, readers) = (&mut self.parts, &mut self.readers);
            let p = &mut parts[dev];
            let (ptrs, loads) = p.pool.assign(li, &uniq[a..b], streaming)?;
            let jobs: Vec<(usize, usize)> = loads.iter().map(|&(slot, e)| (slot, e * EXPERT_BYTES)).collect();
            let t0 = std::time::Instant::now();
            let pool = &p.pool;
            let reader = readers[li].as_mut().ok_or(format!("layer {li} has no expert file"))?;
            if let Err(err) = reader.load(&jobs, EXPERT_BYTES, |slot, bytes| pool.upload(slot, bytes)) {
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
        let (p, w) = (&self.parts[dev], &self.layers[li].1);
        p.k.moe_reduce.launch((t as u32, ((d + 255) / 256) as u32), 256, &[Arg::Ptr(p.s.ypick.ptr), Arg::Ptr(p.s.x.ptr), Arg::I32(d as i32), Arg::I32(kk as i32)])?;
        match &w.mlp {
            Mlp::Moe { sh_gate, sh_up, sh_down, .. } => p.swiglu(sh_gate, sh_up, sh_down, t),
            Mlp::Dense { .. } => unreachable!(),
        }
    }

    fn head(&mut self) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let p = self.parts.last().unwrap();
        p.gpu.bind()?;
        let last = p.s.x.at((self.last_t - 1) * c.d * 4);
        p.norm(&self.norm, last, p.s.h.ptr, 1, c.d, c.eps)?;
        p.mm(&self.lm_head, &p.s.h, &self.logits, 1, false)?;
        self.logits.download_f32(c.vocab)
    }
}

impl Lm for KimiEngine {
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

    /// Attention only: positions past n are simply overwritten later.
    fn crop(&mut self, n: usize) -> usize {
        self.length = self.length.min(n);
        self.length
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
        format!("{:.1} GB in VRAM; {}; context {} tokens (MLA latent cache)", gb, parts.join("; "), self.cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yarn_matches_reference() {
        // Kimi-K2.6: theta 50000, factor 64, orig 4096, beta 32/1, mscale 1/1
        let rs = serde_json::json!({"factor": 64.0, "original_max_position_embeddings": 4096, "beta_fast": 32.0, "beta_slow": 1.0,
                                    "mscale": 1.0, "mscale_all_dim": 1.0});
        let (inv, ms) = yarn(50000.0, &rs).unwrap();
        assert_eq!(inv.len(), 32);
        assert!((inv[0] - 1.0).abs() < 1e-6, "fastest frequency kept: {}", inv[0]);
        let slowest = 1.0 / 50000f64.powf(62.0 / 64.0) / 64.0;
        assert!(((inv[31] as f64) / slowest - 1.0).abs() < 1e-4, "slowest frequency interpolated: {}", inv[31]);
        assert!((ms - (0.1 * 64f64.ln() + 1.0)).abs() < 1e-9);
    }
}
