//! Aegist on the GPU: uploads an .nx brain and runs the forward pass with Aegist's own kernels.
//!
//! Every block: x += attn(norm(x)); h = norm(x); x += ffn(h) + cores(h, engine(h)).
//! The always-on base is int8 (+ bf16 row scales). The core engine routes every token. Only
//! live cores (ones learning time has trained, i.e. with a non-zero output projection) are
//! uploaded; a silent core contributes exactly zero, so routing never needs a host round trip.
use crate::cuda::{Arg, Buf, Gpu, Kernel, Result};
use crate::nx::NxFile;
use serde_json::Value;

pub(crate) const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/aegist.ptx"));

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub n1: usize,
    pub n2: usize,
    pub rank: usize,
    pub active: usize,
}

impl Config {
    pub fn from_header(c: &Value) -> Result<Config> {
        let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or(format!("config is missing {k}"));
        let grid = c["core_grid"].as_array().ok_or("config is missing core_grid")?;
        Ok(Config {
            vocab: u("vocab_size")?,
            d: u("d_model")?,
            layers: u("n_layers")?,
            heads: u("n_heads")?,
            kv_heads: u("n_kv_heads")?,
            head_dim: u("head_dim")?,
            ffn: u("ffn_hidden")?,
            rope_theta: c["rope_theta"].as_f64().ok_or("config is missing rope_theta")? as f32,
            eps: c["norm_eps"].as_f64().ok_or("config is missing norm_eps")? as f32,
            n1: grid[0].as_u64().unwrap() as usize,
            n2: grid[1].as_u64().unwrap() as usize,
            rank: u("core_rank")?,
            active: u("cores_active")?,
        })
    }
    pub fn n_cores(&self) -> usize {
        self.n1 * self.n2
    }
}

pub(crate) struct QW {
    pub w: Buf,
    pub s: Buf,
    pub n: usize,
    pub k: usize,
}

struct Layer {
    attn_norm: Buf,
    ffn_norm: Buf,
    q_norm: Buf,
    k_norm: Buf,
    q: QW,
    k: QW,
    v: QW,
    o: QW,
    gate: QW,
    up: QW,
    down: QW,
    key1: Buf,
    key2: Buf,
    live: Buf, // [n_cores] i32: index into the compact live bank, or -1 (silent)
    cores: Option<(Buf, Buf, Buf)>,
}

struct Kernels {
    embed: Kernel,
    rmsnorm: Kernel,
    gemv_i8: Kernel,
    gemv_i8_gateup: Kernel,
    gemv_bf16: Kernel,
    gemm_i8: Kernel,
    silu_mul: Kernel,
    qk_norm_rope: Kernel,
    attention: Kernel,
    cores_apply: Kernel,
}

pub struct Engine {
    pub gpu: Gpu,
    pub cfg: Config,
    k: Kernels,
    embed: Buf,
    norm: Buf,
    layers: Vec<Layer>,
    kc: Buf,
    vc: Buf,
    pub cap: usize,
    pub length: usize,
    last_t: usize,
    chunk: usize,
    ids: Buf,
    x: Buf,
    h: Buf,
    q: Buf,
    kb: Buf,
    vb: Buf,
    att: Buf,
    g: Buf,
    u: Buf,
    logits: Buf,
    counts: Buf,
    pub weight_bytes: usize,
    pub live_cores: usize,
}

pub(crate) fn upload(nx: &NxFile, name: &str, total: &mut usize) -> Result<Buf> {
    let b = nx.bytes(name)?;
    *total += b.len();
    if std::env::var_os("AEGIST_DEBUG_VRAM").is_some() {
        eprintln!("vram {} {name}", b.len());
    }
    Buf::from_bytes(b)
}

pub(crate) fn qweight(nx: &NxFile, prefix: &str, total: &mut usize) -> Result<QW> {
    let name = format!("{prefix}.qweight");
    let info = nx.info(&name).map_err(|_| format!("{prefix} is not int8; rebuild the brain with --quant int8"))?;
    let (n, k) = (info.shape[0], info.shape[1]);
    Ok(QW { w: upload(nx, &name, total)?, s: upload(nx, &format!("{prefix}.scale"), total)?, n, k })
}

const KV_BYTES_PER: usize = 2; // bf16 cache

impl Engine {
    /// Bytes of KV cache needed per token of context.
    pub fn kv_bytes_per_token(cfg: &Config) -> usize {
        2 * cfg.layers * cfg.kv_heads * cfg.head_dim * KV_BYTES_PER
    }

    /// Load `nx` onto the GPU. `max_context` 0 picks the largest context (up to 32k) that fits.
    pub fn load(nx: &NxFile, max_context: usize, mut progress: impl FnMut(usize, usize)) -> Result<Engine> {
        let cfg = Config::from_header(&nx.header["config"])?;
        if cfg.head_dim != 128 {
            return Err(format!("this engine supports head_dim 128, the brain has {}", cfg.head_dim));
        }
        if cfg.n1 > 128 || cfg.n2 > 64 || cfg.rank > 64 || cfg.active > 8 || cfg.active > cfg.n1 * cfg.n2 {
            return Err("core grid too large for this engine build (n1 <= 128; n2, rank <= 64; active <= 8)".into());
        }
        let gpu = Gpu::new(PTX)?;
        let f = |n: &str| gpu.function(n);
        let k = Kernels {
            embed: f("embed")?,
            rmsnorm: f("rmsnorm")?,
            gemv_i8: f("gemv_i8")?,
            gemv_i8_gateup: f("gemv_i8_gateup")?,
            gemv_bf16: f("gemv_bf16")?,
            gemm_i8: f("gemm_i8")?,
            silu_mul: f("silu_mul")?,
            qk_norm_rope: f("qk_norm_rope")?,
            attention: f("attention")?,
            cores_apply: f("cores_apply")?,
        };
        let silent: Vec<bool> = nx.header["cores_silent"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_bool().unwrap_or(false)).collect())
            .unwrap_or_else(|| vec![false; cfg.layers]);

        let mut total = 0usize;
        let embed = upload(nx, "embed.weight", &mut total)?;
        let norm = upload(nx, "norm.weight", &mut total)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        let mut live_cores = 0;
        for i in 0..cfg.layers {
            progress(i, cfg.layers);
            let p = |s: &str| format!("blocks.{i}.{s}");
            let n_cores = cfg.n_cores();
            let (live, cores, n_live) = if silent.get(i).copied().unwrap_or(false) {
                (Buf::from_bytes(&bytes_i32(&vec![-1; n_cores]))?, None, 0)
            } else {
                live_bank(nx, i, &cfg, &mut total)?
            };
            live_cores += n_live;
            layers.push(Layer {
                attn_norm: upload(nx, &p("attn_norm.weight"), &mut total)?,
                ffn_norm: upload(nx, &p("ffn_norm.weight"), &mut total)?,
                q_norm: upload(nx, &p("attn.q_norm.weight"), &mut total)?,
                k_norm: upload(nx, &p("attn.k_norm.weight"), &mut total)?,
                q: qweight(nx, &p("attn.q"), &mut total)?,
                k: qweight(nx, &p("attn.k"), &mut total)?,
                v: qweight(nx, &p("attn.v"), &mut total)?,
                o: qweight(nx, &p("attn.o"), &mut total)?,
                gate: qweight(nx, &p("ffn.gate"), &mut total)?,
                up: qweight(nx, &p("ffn.up"), &mut total)?,
                down: qweight(nx, &p("ffn.down"), &mut total)?,
                key1: upload(nx, &p("engine.key1.weight"), &mut total)?,
                key2: upload(nx, &p("engine.key2.weight"), &mut total)?,
                live,
                cores,
            });
        }
        progress(cfg.layers, cfg.layers);

        let chunk = 256;
        let (free, _) = gpu.mem_info();
        let per = Self::kv_bytes_per_token(&cfg);
        let reserve = 300 << 20; // activations, logits, driver slack
        let fits = free.saturating_sub(reserve) / per;
        let cap = if max_context == 0 { fits.min(32768) } else { max_context.min(fits) };
        if cap < 512 {
            return Err(format!(
                "not enough GPU memory for a KV cache: {:.2} GB free after weights. Close other GPU-heavy apps.",
                free as f64 / 1e9
            ));
        }
        let kv_layer = cfg.kv_heads * cap * cfg.head_dim * KV_BYTES_PER;
        let f32b = |n: usize| Buf::alloc(n * 4);
        let qd = cfg.heads * cfg.head_dim;
        let kvd = cfg.kv_heads * cfg.head_dim;
        Ok(Engine {
            kc: Buf::alloc(kv_layer * cfg.layers)?,
            vc: Buf::alloc(kv_layer * cfg.layers)?,
            ids: Buf::alloc(chunk * 4)?,
            x: f32b(chunk * cfg.d)?,
            h: f32b(chunk * cfg.d)?,
            q: f32b(chunk * qd)?,
            kb: f32b(chunk * kvd)?,
            vb: f32b(chunk * kvd)?,
            att: f32b(chunk * qd)?,
            g: f32b(chunk * cfg.ffn)?,
            u: f32b(chunk * cfg.ffn)?,
            logits: f32b(cfg.vocab)?,
            counts: Buf::zeroed(cfg.layers * cfg.n_cores() * 4)?,
            gpu,
            k,
            embed,
            norm,
            layers,
            cap,
            length: 0,
            last_t: 0,
            chunk,
            weight_bytes: total,
            live_cores,
            cfg,
        })
    }

    fn forward_chunk(&mut self, ids: &[u32]) -> Result<()> {
        let c = &self.cfg;
        let t = ids.len();
        let (d, hd, h_, kv) = (c.d as i32, c.head_dim as i32, c.heads as i32, c.kv_heads as i32);
        let pos0 = self.length as i32;
        let ids_i32: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        self.ids.upload(&bytes_i32(&ids_i32))?;
        let k = &self.k;
        k.embed.launch((t as u32, 1), 256, &[Arg::Ptr(self.embed.ptr), Arg::Ptr(self.ids.ptr), Arg::Ptr(self.x.ptr), Arg::I32(d)])?;
        let kv_layer = c.kv_heads * self.cap * c.head_dim * KV_BYTES_PER;
        let n_cores = c.n_cores();
        for (li, l) in self.layers.iter().enumerate() {
            let norm = |w: &Buf, src: &Buf, dst: &Buf| {
                k.rmsnorm.launch((t as u32, 1), 256, &[Arg::Ptr(src.ptr), Arg::Ptr(w.ptr), Arg::Ptr(dst.ptr), Arg::I32(d), Arg::F32(c.eps)])
            };
            norm(&l.attn_norm, &self.x, &self.h)?;
            self.matmul(&l.q, &self.h, &self.q, t, false)?;
            self.matmul(&l.k, &self.h, &self.kb, t, false)?;
            self.matmul(&l.v, &self.h, &self.vb, t, false)?;
            let (kc, vc) = (self.kc.at(li * kv_layer), self.vc.at(li * kv_layer));
            k.qk_norm_rope.launch(
                (t as u32, (c.heads + c.kv_heads) as u32),
                c.head_dim as u32,
                &[
                    Arg::Ptr(self.q.ptr), Arg::Ptr(self.kb.ptr), Arg::Ptr(self.vb.ptr), Arg::Ptr(l.q_norm.ptr), Arg::Ptr(l.k_norm.ptr),
                    Arg::Ptr(kc), Arg::Ptr(vc), Arg::I32(h_), Arg::I32(kv), Arg::I32(hd), Arg::I32(pos0),
                    Arg::I32(self.cap as i32), Arg::F32(c.eps), Arg::F32(c.rope_theta),
                ],
            )?;
            k.attention.launch(
                (t as u32, c.heads as u32),
                256,
                &[
                    Arg::Ptr(self.q.ptr), Arg::Ptr(kc), Arg::Ptr(vc), Arg::Ptr(self.att.ptr), Arg::I32(h_), Arg::I32(kv),
                    Arg::I32(pos0), Arg::I32(self.cap as i32), Arg::F32(1.0 / (c.head_dim as f32).sqrt()),
                ],
            )?;
            self.matmul(&l.o, &self.att, &self.x, t, true)?;
            norm(&l.ffn_norm, &self.x, &self.h)?;
            if t == 1 {
                k.gemv_i8_gateup.launch(
                    (((c.ffn + 7) / 8) as u32, 1),
                    256,
                    &[
                        Arg::Ptr(self.h.ptr), Arg::Ptr(l.gate.w.ptr), Arg::Ptr(l.gate.s.ptr), Arg::Ptr(l.up.w.ptr),
                        Arg::Ptr(l.up.s.ptr), Arg::Ptr(self.g.ptr), Arg::I32(c.ffn as i32), Arg::I32(d),
                    ],
                )?;
            } else {
                self.matmul(&l.gate, &self.h, &self.g, t, false)?;
                self.matmul(&l.up, &self.h, &self.u, t, false)?;
                let n = (t * c.ffn) as i32;
                k.silu_mul.launch((((n + 255) / 256) as u32, 1), 256, &[Arg::Ptr(self.g.ptr), Arg::Ptr(self.u.ptr), Arg::Ptr(self.g.ptr), Arg::I32(n)])?;
            }
            self.matmul(&l.down, &self.g, &self.x, t, true)?;
            let zero = Arg::Ptr(0);
            let (cg, cu, cd) = match &l.cores {
                Some((g, u, dn)) => (Arg::Ptr(g.ptr), Arg::Ptr(u.ptr), Arg::Ptr(dn.ptr)),
                None => (zero, zero, zero),
            };
            k.cores_apply.launch(
                (t as u32, 1),
                256,
                &[
                    Arg::Ptr(self.h.ptr), Arg::Ptr(l.key1.ptr), Arg::Ptr(l.key2.ptr), Arg::Ptr(l.live.ptr), cg, cu, cd,
                    Arg::Ptr(self.x.ptr), Arg::Ptr(self.counts.at(li * n_cores * 4)), Arg::I32(d), Arg::I32(c.n1 as i32),
                    Arg::I32(c.n2 as i32), Arg::I32(c.active as i32), Arg::I32(c.rank as i32),
                ],
            )?;
        }
        self.length += t;
        self.last_t = t;
        Ok(())
    }

    fn head(&mut self) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let last = self.x.at((self.last_t - 1) * c.d * 4);
        self.k.rmsnorm.launch((1, 1), 256, &[Arg::Ptr(last), Arg::Ptr(self.norm.ptr), Arg::Ptr(self.h.ptr), Arg::I32(c.d as i32), Arg::F32(c.eps)])?;
        self.k.gemv_bf16.launch(
            (((c.vocab + 7) / 8) as u32, 1),
            256,
            &[Arg::Ptr(self.h.ptr), Arg::Ptr(self.embed.ptr), Arg::Ptr(self.logits.ptr), Arg::I32(c.vocab as i32), Arg::I32(c.d as i32)],
        )?;
        self.logits.download_f32(c.vocab)
    }

    fn matmul(&self, w: &QW, x: &Buf, y: &Buf, t: usize, accumulate: bool) -> Result<()> {
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

}

/// What the chat runtime needs from an engine (the dense one here, the hybrid-MoE one in hybrid.rs).
pub trait Lm {
    /// Feed tokens (any number) and return the logits after the last one.
    fn feed(&mut self, ids: &[u32]) -> Result<Vec<f32>>;
    /// Go back to at most `n` tokens and return the length actually kept: a recurrent state can
    /// only return to a snapshot. The caller re-feeds the tokens after it.
    fn crop(&mut self, n: usize) -> usize;
    /// Remember the state at the current length so `crop` can come back to it.
    fn snapshot(&mut self) -> Result<()> {
        Ok(())
    }
    fn length(&self) -> usize;
    fn cap(&self) -> usize;
    /// Names this brain's computation for saved states (None: this engine cannot save them).
    fn state_key(&self) -> Option<String> {
        None
    }
    /// Write the state after the tokens fed so far to `path`.
    fn save_state(&self, _path: &std::path::Path) -> Result<()> {
        Err("this engine cannot save its state".into())
    }
    /// Read a state written by save_state (with the same state_key); returns its tokens. On an
    /// error the engine is left empty.
    fn load_state(&mut self, _path: &std::path::Path) -> Result<Vec<u32>> {
        Err("this engine cannot load a state".into())
    }
    /// Keep the hidden state after each of these layers (for the last token of every feed), for
    /// the confidence probe. An empty list turns taps off.
    fn set_taps(&mut self, layers: &[usize]) -> Result<()> {
        if layers.is_empty() { Ok(()) } else { Err("this engine has no hidden-state taps".into()) }
    }
    /// The tapped hidden states after the last feed: [taps][d] fp32, in set_taps order.
    fn taps(&self) -> Result<Vec<f32>> {
        Ok(vec![])
    }
    /// Per-layer switch-on counts since the last reset: [layers][n_cores].
    fn core_counts(&self) -> Result<Vec<i32>>;
    fn reset_counts(&self) -> Result<()>;
    /// (expert uses served from VRAM, experts copied in) so far; (0, 0) without experts.
    fn expert_counters(&self) -> (u64, u64) {
        (0, 0)
    }
    /// (expert picks skipped, expert reads saved) by $AEGIST_SKIP_MISS so far.
    fn skipped(&self) -> (u64, u64) {
        (0, 0)
    }
    /// Light effort (thinking) or full effort (the answer the user reads). An engine that streams
    /// experts from disk may skip more of the ones it would have to read while light.
    fn set_light(&mut self, _light: bool) {}
    /// Seconds spent reading experts from disk (and copying them in) so far.
    fn read_seconds(&self) -> f64 {
        0.0
    }
    /// (read-ahead guesses, guesses then used) so far
    fn guess_hits(&self) -> (u64, u64) {
        (0, 0)
    }
    /// experts read ahead into the OS file cache so far
    fn prefetched(&self) -> u64 {
        0
    }
    /// Pause (true) or resume background disk reads (another brain is answering).
    fn hold_background(&self, _on: bool) {}
    /// One line for the load message: GPU, weights, cores, context.
    fn describe(&self) -> String;
}

impl Lm for Engine {
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

    /// A KV cache can be cut anywhere.
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
        self.counts.download_i32(self.cfg.layers * self.cfg.n_cores())
    }

    fn reset_counts(&self) -> Result<()> {
        self.counts.zero()
    }

    fn describe(&self) -> String {
        format!(
            "{}: {:.2} GB of weights, {} live cores, context {} tokens",
            self.gpu.name,
            self.weight_bytes as f64 / 1e9,
            self.live_cores,
            self.cap
        )
    }
}

pub(crate) fn bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Find the layer's live cores (non-zero output projection) and upload only those, compacted.
fn live_bank(nx: &NxFile, layer: usize, cfg: &Config, total: &mut usize) -> Result<(Buf, Option<(Buf, Buf, Buf)>, usize)> {
    let name = |k: &str| format!("blocks.{layer}.cores.{k}");
    for k in ["gate", "up", "down"] {
        if nx.info(&name(k))?.dtype != "bf16" {
            return Err(format!("{} must be bf16", name(k)));
        }
    }
    let (n, d, r) = (cfg.n_cores(), cfg.d, cfg.rank);
    let per = d * r * 2; // bytes per core per matrix
    let down = nx.bytes(&name("down"))?;
    let mut map = vec![-1i32; n];
    let mut ids = Vec::new();
    for c in 0..n {
        let core = &down[c * per..(c + 1) * per];
        if core.chunks_exact(2).any(|w| u16::from_le_bytes([w[0], w[1]]) & 0x7fff != 0) {
            map[c] = ids.len() as i32;
            ids.push(c);
        }
    }
    let live = Buf::from_bytes(&bytes_i32(&map))?;
    if ids.is_empty() {
        return Ok((live, None, 0));
    }
    let gather = |k: &str| -> Result<Buf> {
        let src = nx.bytes(&name(k))?;
        let buf = Buf::alloc(ids.len() * per)?;
        for (j, &c) in ids.iter().enumerate() {
            buf.upload_at(j * per, &src[c * per..(c + 1) * per])?;
        }
        Ok(buf)
    };
    let bank = (gather("gate")?, gather("up")?, gather("down")?);
    *total += 3 * ids.len() * per;
    Ok((live, Some(bank), ids.len()))
}
