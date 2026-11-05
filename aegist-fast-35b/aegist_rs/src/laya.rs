//! The reading core: Laya, rebuilt in Rust on the CPU, reads each prompt before Aegist answers it.
//!
//! Laya (convaiinnovations/laya, Apache-2.0, by Convai Innovations) is a 421M-parameter decision
//! model: a ModernBERT-large encoder, a type embedding, two transformer layers and an option scorer.
//! This is our own implementation of that forward pass, run straight from the released fp16 weights
//! (memory-mapped, converted row by row), so the GPU stays free for the brain. It matches
//! `aegist/laya.py` (tests/test_laya_rust.py checks the probabilities).
//!
//! Its job here is one choice per user turn: does the prompt ask for a fact, for something creative,
//! or for neither? Measured on 45 prompts written after the framing was fixed: 80% right, and no
//! fact prompt read as creative or the other way round (scripts/eval_prompt_reader.py).
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

type Result<T> = std::result::Result<T, String>;

const D: usize = 1024;
const HEADS: usize = 16;
const HD: usize = D / HEADS;

/// What a prompt asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Fact,
    Creative,
    Other,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Fact, Kind::Creative, Kind::Other];
    pub fn name(self) -> &'static str {
        match self {
            Kind::Fact => "fact",
            Kind::Creative => "creative",
            Kind::Other => "other",
        }
    }
}

/// The framing chosen on the dev prompts (scripts/eval_prompt_reader.py).
pub const READ_QUESTION: &str = "Is this request asking for a fact, something creative, or other help?";

struct T {
    off: usize,
    shape: Vec<usize>,
}

pub struct Laya {
    map: Mmap,
    t: HashMap<String, T>,
    tok: tokenizers::Tokenizer,
    cls: u32,
    sep: u32,
    mask: u32,
    specials: Vec<String>,
    layers: usize,
    head_layers: usize,
    inter: usize,
    window: usize,
    global_every: usize,
    max_len: usize,
    head_max: usize,
    temp: [f32; 3],
    temp_by_options: HashMap<String, f32>,
    threads: usize,
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: renormalise
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (31, 0) => sign | 0x7f80_0000,
        (31, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// fp16 bytes to f32, with the F16C instruction when the CPU has it.
fn convert(src: &[u8], dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx") {
            unsafe { convert_f16c(src, dst) };
            return;
        }
    }
    for (i, d) in dst.iter_mut().enumerate() {
        *d = f16_to_f32(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn convert_f16c(src: &[u8], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = dst.len();
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(src.as_ptr().add(2 * i) as *const __m128i);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        i += 8;
    }
    for j in i..n {
        dst[j] = f16_to_f32(u16::from_le_bytes([src[2 * j], src[2 * j + 1]]));
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    let mut s = [0f32; 8];
    let n = a.len() / 8 * 8;
    for i in (0..n).step_by(8) {
        for k in 0..8 {
            s[k] += a[i + k] * b[i + k];
        }
    }
    s.iter().sum::<f32>() + (n..a.len()).map(|i| a[i] * b[i]).sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0;
    while i + 16 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), _mm256_loadu_ps(b.as_ptr().add(i + 8)), s1);
        i += 16;
    }
    let mut v = [0f32; 8];
    _mm256_storeu_ps(v.as_mut_ptr(), _mm256_add_ps(s0, s1));
    let mut s: f32 = v.iter().sum();
    for j in i..n {
        s += a[j] * b[j];
    }
    s
}

fn layer_norm(x: &mut [f32], w: &[f32], b: Option<&[f32]>) {
    for row in x.chunks_mut(D) {
        let mean = row.iter().sum::<f32>() / D as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / D as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv * w[i] + b.map_or(0.0, |b| b[i]);
        }
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// erf to f32 precision (Abramowitz-Stegun 7.1.26 is too coarse; this is the Chebyshev fit of erfc from Numerical Recipes).
fn erf(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = 1.0
        - t * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

impl Laya {
    pub fn open(dir: &Path) -> Result<Laya> {
        let cfg: Value = serde_json::from_slice(&std::fs::read(dir.join("rl_agent_config.json")).map_err(|e| format!("Laya config: {e}"))?)
            .map_err(|e| e.to_string())?;
        let enc: Value = serde_json::from_slice(&std::fs::read(dir.join("encoder").join("config.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let file = std::fs::File::open(dir.join("model.safetensors")).map_err(|e| format!("Laya weights: {e}"))?;
        let map = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
        let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let head: Value = serde_json::from_slice(&map[8..8 + n]).map_err(|e| e.to_string())?;
        let mut t = HashMap::new();
        for (k, v) in head.as_object().ok_or("bad safetensors header")? {
            if k == "__metadata__" {
                continue;
            }
            let shape = v["shape"].as_array().ok_or("bad shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
            let off = 8 + n + v["data_offsets"][0].as_u64().ok_or("bad offsets")? as usize;
            if v["dtype"] != "F16" && k != "temperature" {
                return Err(format!("Laya tensor {k} is {}, expected F16", v["dtype"]));
            }
            t.insert(k.clone(), T { off, shape });
        }
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer").join("tokenizer.json")).map_err(|e| format!("Laya tokenizer: {e}"))?;
        let id = |s: &str| tok.token_to_id(s).ok_or(format!("Laya tokenizer has no {s}"));
        let (cls, sep, mask) = (id("[CLS]")?, id("[SEP]")?, id("[MASK]")?);
        let specials = ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"].iter().map(|s| s.to_string()).collect();
        let u = |v: &Value, k: &str| v[k].as_u64().map(|x| x as usize).ok_or(format!("Laya config has no {k}"));
        let temps: Vec<f32> = cfg["temperature"].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(1.0) as f32).collect()).unwrap_or_default();
        let laya = Laya {
            layers: u(&enc, "num_hidden_layers")?,
            inter: u(&enc, "intermediate_size")?,
            window: u(&enc, "local_attention")? / 2,
            global_every: u(&enc, "global_attn_every_n_layers")?,
            head_layers: u(&cfg, "head_layers")?,
            max_len: u(&cfg, "max_len")?,
            head_max: u(&cfg, "head_max_len")?,
            temp: [temps.first().copied().unwrap_or(1.0), temps.get(1).copied().unwrap_or(1.0), temps.get(2).copied().unwrap_or(1.0)],
            temp_by_options: cfg["temperature_by_options"].as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(1.0) as f32)).collect()).unwrap_or_default(),
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16),
            map,
            t,
            tok,
            cls,
            sep,
            mask,
            specials,
        };
        if u(&enc, "hidden_size")? != D || u(&enc, "num_attention_heads")? != HEADS {
            return Err("Laya: unexpected encoder size".into());
        }
        Ok(laya)
    }

    fn raw(&self, name: &str) -> Result<(&[u8], &[usize])> {
        let t = self.t.get(name).ok_or(format!("Laya has no tensor {name}"))?;
        let n: usize = t.shape.iter().product();
        Ok((&self.map[t.off..t.off + 2 * n], &t.shape))
    }

    fn vec(&self, name: &str) -> Result<Vec<f32>> {
        let (b, s) = self.raw(name)?;
        let mut v = vec![0f32; s.iter().product()];
        convert(b, &mut v);
        Ok(v)
    }

    /// y = x W^T (+ b) for `rows` tokens, W an (out, in) fp16 matrix; output rows split over threads.
    fn linear(&self, x: &[f32], rows: usize, name: &str, bias: Option<&str>) -> Result<Vec<f32>> {
        // nn.MultiheadAttention names its input projection "in_proj_weight"
        let wname = if name.ends_with("_weight") { name.to_string() } else { format!("{name}.weight") };
        let (w, s) = self.raw(&wname)?;
        let (out, inp) = (s[0], s[1]);
        debug_assert_eq!(x.len(), rows * inp);
        let b = bias.map(|b| self.vec(b)).transpose()?;
        let mut y = vec![0f32; rows * out];
        let per = out.div_ceil(self.threads);
        // each thread fills columns [o0, o1) of y, written through its own buffer
        let parts: Vec<(usize, Vec<f32>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..self.threads)
                .map(|k| (k * per, ((k + 1) * per).min(out)))
                .filter(|(a, b)| a < b)
                .map(|(o0, o1)| {
                    let b = b.as_deref();
                    sc.spawn(move || {
                        let mut row = vec![0f32; inp];
                        let mut part = vec![0f32; rows * (o1 - o0)];
                        for o in o0..o1 {
                            convert(&w[2 * o * inp..2 * (o + 1) * inp], &mut row);
                            let bo = b.map_or(0.0, |b| b[o]);
                            for r in 0..rows {
                                part[r * (o1 - o0) + o - o0] = dot(&x[r * inp..(r + 1) * inp], &row) + bo;
                            }
                        }
                        (o0, part)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (o0, part) in parts {
            let w = part.len() / rows;
            for r in 0..rows {
                y[r * out + o0..r * out + o0 + w].copy_from_slice(&part[r * w..(r + 1) * w]);
            }
        }
        Ok(y)
    }

    /// Softmax attention over `n` tokens; q, k, v are (n, D) with heads side by side.
    /// `window`: only keys with |i - j| <= window are seen.
    fn attend(q: &[f32], k: &[f32], v: &[f32], n: usize, window: Option<usize>) -> Vec<f32> {
        let mut out = vec![0f32; n * D];
        let scale = 1.0 / (HD as f32).sqrt();
        let mut s = vec![0f32; n];
        for h in 0..HEADS {
            for i in 0..n {
                let qi = &q[i * D + h * HD..i * D + (h + 1) * HD];
                let (lo, hi) = match window {
                    Some(w) => (i.saturating_sub(w), (i + w + 1).min(n)),
                    None => (0, n),
                };
                let mut m = f32::NEG_INFINITY;
                for j in lo..hi {
                    s[j] = dot(qi, &k[j * D + h * HD..j * D + (h + 1) * HD]) * scale;
                    m = m.max(s[j]);
                }
                let mut z = 0f32;
                for j in lo..hi {
                    s[j] = (s[j] - m).exp();
                    z += s[j];
                }
                let o = &mut out[i * D + h * HD..i * D + (h + 1) * HD];
                for j in lo..hi {
                    let p = s[j] / z;
                    for (d, vv) in o.iter_mut().zip(&v[j * D + h * HD..j * D + (h + 1) * HD]) {
                        *d += p * vv;
                    }
                }
            }
        }
        out
    }

    fn rope(x: &mut [f32], n: usize, theta: f32) {
        let half = HD / 2;
        for p in 0..n {
            for h in 0..HEADS {
                let base = p * D + h * HD;
                for i in 0..half {
                    let f = (p as f64) / (theta as f64).powf(2.0 * i as f64 / HD as f64);
                    let (sin, cos) = (f as f32).sin_cos();
                    let (a, b) = (x[base + i], x[base + i + half]);
                    x[base + i] = a * cos - b * sin;
                    x[base + i + half] = b * cos + a * sin;
                }
            }
        }
    }

    fn enc(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec())
    }

    /// Text from the user cannot add Laya's own markers.
    fn plain(&self, s: &str) -> String {
        self.specials.iter().fold(s.to_string(), |s, m| s.replace(m.as_str(), " "))
    }

    /// The sequence and the position of each option's [MASK]; the same layout as aegist/laya.py.
    pub fn build(&self, qtype: &str, instructions: &str, options: &[String], state: &str) -> Result<(Vec<u32>, Vec<usize>)> {
        let head = self.enc(&format!("{qtype} question: {instructions}"))?;
        let mut opt_ids: Vec<Vec<u32>> = options
            .iter()
            .map(|o| self.enc(&format!(" {}", self.plain(o))).map(|e| [vec![self.mask], e.into_iter().take(48).collect()].concat()))
            .collect::<Result<_>>()?;
        let mut budget = self.head_max as isize - opt_ids.iter().map(|o| o.len()).sum::<usize>() as isize;
        if budget < 16 {
            let share = 4.max((self.head_max - 16) / opt_ids.len());
            for o in &mut opt_ids {
                o.truncate(share);
            }
            budget = self.head_max as isize - opt_ids.iter().map(|o| o.len()).sum::<usize>() as isize;
        }
        let mut ids = vec![self.cls];
        ids.extend(head.iter().take(8.max(budget.max(0) as usize)));
        ids.push(self.sep);
        let mut markers = vec![];
        for o in &opt_ids {
            markers.push(ids.len());
            ids.extend(o);
        }
        ids.push(self.sep);
        let st = self.enc(&self.plain(state))?;
        let room = self.max_len.saturating_sub(ids.len() + 1);
        ids.extend(st.into_iter().take(room));
        ids.push(self.sep);
        markers.retain(|&m| m < self.max_len);
        if markers.len() != options.len() {
            return Err("the options do not fit in Laya's head".into());
        }
        ids.truncate(self.max_len);
        Ok((ids, markers))
    }

    /// Option logits for one sequence.
    pub fn logits(&self, ids: &[u32], markers: &[usize], qtype: usize) -> Result<Vec<f32>> {
        let n = ids.len();
        let (emb, _) = self.raw("encoder.embeddings.tok_embeddings.weight")?;
        let mut h = vec![0f32; n * D];
        for (p, &id) in ids.iter().enumerate() {
            convert(&emb[2 * id as usize * D..2 * (id as usize + 1) * D], &mut h[p * D..(p + 1) * D]);
        }
        layer_norm(&mut h, &self.vec("encoder.embeddings.norm.weight")?, None);
        for l in 0..self.layers {
            let pre = format!("encoder.layers.{l}");
            let mut x = h.clone();
            if l > 0 {
                layer_norm(&mut x, &self.vec(&format!("{pre}.attn_norm.weight"))?, None);
            }
            let qkv = self.linear(&x, n, &format!("{pre}.attn.Wqkv"), None)?;
            let (mut q, mut k, mut v) = (vec![0f32; n * D], vec![0f32; n * D], vec![0f32; n * D]);
            for p in 0..n {
                q[p * D..(p + 1) * D].copy_from_slice(&qkv[p * 3 * D..p * 3 * D + D]);
                k[p * D..(p + 1) * D].copy_from_slice(&qkv[p * 3 * D + D..p * 3 * D + 2 * D]);
                v[p * D..(p + 1) * D].copy_from_slice(&qkv[p * 3 * D + 2 * D..(p + 1) * 3 * D]);
            }
            let global = l % self.global_every == 0;
            let theta = if global { 160000.0 } else { 10000.0 };
            Self::rope(&mut q, n, theta);
            Self::rope(&mut k, n, theta);
            let a = Self::attend(&q, &k, &v, n, (!global).then_some(self.window));
            let o = self.linear(&a, n, &format!("{pre}.attn.Wo"), None)?;
            for (hv, ov) in h.iter_mut().zip(&o) {
                *hv += ov;
            }
            let mut x = h.clone();
            layer_norm(&mut x, &self.vec(&format!("{pre}.mlp_norm.weight"))?, None);
            let wi = self.linear(&x, n, &format!("{pre}.mlp.Wi"), None)?;
            let f = self.inter;
            let mut g = vec![0f32; n * f];
            for p in 0..n {
                for i in 0..f {
                    g[p * f + i] = gelu(wi[p * 2 * f + i]) * wi[p * 2 * f + f + i];
                }
            }
            let o = self.linear(&g, n, &format!("{pre}.mlp.Wo"), None)?;
            for (hv, ov) in h.iter_mut().zip(&o) {
                *hv += ov;
            }
        }
        layer_norm(&mut h, &self.vec("encoder.final_norm.weight")?, None);
        let te = self.vec("type_emb.weight")?;
        for row in h.chunks_mut(D) {
            for (v, t) in row.iter_mut().zip(&te[qtype * D..(qtype + 1) * D]) {
                *v += t;
            }
        }
        for l in 0..self.head_layers {
            let pre = format!("head.layers.{l}");
            let mut x = h.clone();
            layer_norm(&mut x, &self.vec(&format!("{pre}.norm1.weight"))?, Some(&self.vec(&format!("{pre}.norm1.bias"))?));
            let qkv = self.linear(&x, n, &format!("{pre}.self_attn.in_proj_weight"), Some(&format!("{pre}.self_attn.in_proj_bias")))?;
            let (mut q, mut k, mut v) = (vec![0f32; n * D], vec![0f32; n * D], vec![0f32; n * D]);
            for p in 0..n {
                q[p * D..(p + 1) * D].copy_from_slice(&qkv[p * 3 * D..p * 3 * D + D]);
                k[p * D..(p + 1) * D].copy_from_slice(&qkv[p * 3 * D + D..p * 3 * D + 2 * D]);
                v[p * D..(p + 1) * D].copy_from_slice(&qkv[p * 3 * D + 2 * D..(p + 1) * 3 * D]);
            }
            let a = Self::attend(&q, &k, &v, n, None);
            let o = self.linear(&a, n, &format!("{pre}.self_attn.out_proj"), Some(&format!("{pre}.self_attn.out_proj.bias")))?;
            for (hv, ov) in h.iter_mut().zip(&o) {
                *hv += ov;
            }
            let mut x = h.clone();
            layer_norm(&mut x, &self.vec(&format!("{pre}.norm2.weight"))?, Some(&self.vec(&format!("{pre}.norm2.bias"))?));
            let mut f = self.linear(&x, n, &format!("{pre}.linear1"), Some(&format!("{pre}.linear1.bias")))?;
            for v in &mut f {
                *v = v.max(0.0);
            }
            let o = self.linear(&f, n, &format!("{pre}.linear2"), Some(&format!("{pre}.linear2.bias")))?;
            for (hv, ov) in h.iter_mut().zip(&o) {
                *hv += ov;
            }
        }
        // the scorer reads only the option markers
        let mut m: Vec<f32> = markers.iter().flat_map(|&p| h[p * D..(p + 1) * D].to_vec()).collect();
        layer_norm(&mut m, &self.vec("scorer.0.weight")?, Some(&self.vec("scorer.0.bias")?));
        let mut s = self.linear(&m, markers.len(), "scorer.1", Some("scorer.1.bias"))?;
        for v in &mut s {
            *v = gelu(*v);
        }
        self.linear(&s, markers.len(), "scorer.3", Some("scorer.3.bias"))
    }

    /// Probabilities over `options` (Laya's "choice" question), after its fitted temperature.
    pub fn choice(&self, question: &str, options: &[String], state: &str) -> Result<Vec<f32>> {
        let (ids, markers) = self.build("choice", question, options, state)?;
        let z = self.logits(&ids, &markers, 0)?;
        let k = options.len();
        let bucket = format!("choice:{}", if k <= 2 { "2" } else if k <= 5 { "3-5" } else if k <= 10 { "6-10" } else { "11+" });
        let t = self.temp_by_options.get(&bucket).copied().unwrap_or(self.temp[0]);
        let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = z.iter().map(|v| ((v - m) / t).exp()).collect();
        let s: f32 = e.iter().sum();
        Ok(e.into_iter().map(|v| v / s).collect())
    }

    /// Read a prompt: what does it ask for, and how sure is the reading.
    pub fn read(&self, prompt: &str) -> Result<(Kind, [f32; 3])> {
        let opts: Vec<String> = Kind::ALL.iter().map(|k| k.name().to_string()).collect();
        let p = self.choice(READ_QUESTION, &opts, prompt)?;
        let best = (0..3).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap_or(2);
        Ok((Kind::ALL[best], [p[0], p[1], p[2]]))
    }
}

/// Where Laya lives: `$AEGIST_LAYA`, else `<home>/models/laya`.
pub fn find(home: &Path) -> Option<std::path::PathBuf> {
    let p = std::env::var("AEGIST_LAYA").map(std::path::PathBuf::from).unwrap_or_else(|_| home.join("models").join("laya"));
    p.join("model.safetensors").exists().then_some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats() {
        for (h, f) in [(0x3c00u16, 1.0f32), (0xc000, -2.0), (0x0000, 0.0), (0x7bff, 65504.0), (0x0001, 5.9604645e-8), (0x3555, 0.33325195)] {
            assert_eq!(f16_to_f32(h), f, "{h:#x}");
            let mut d = [0f32; 9];
            convert(&[h.to_le_bytes(); 9].concat(), &mut d);
            assert!(d.iter().all(|&x| x == f), "{h:#x}");
        }
    }

    #[test]
    fn erf_and_gelu() {
        for (x, e) in [(0.0f32, 0.0f32), (0.5, 0.5204999), (1.0, 0.8427008), (-2.0, -0.9953223), (3.0, 0.9999779)] {
            assert!((erf(x) - e).abs() < 2e-6, "erf({x}) = {}", erf(x));
        }
        assert!((gelu(1.0) - 0.8413447).abs() < 2e-6);
    }

    #[test]
    fn dot_matches_a_plain_sum() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.25 - 3.0).collect();
        let b: Vec<f32> = (0..37).map(|i| (i % 5) as f32 - 1.5).collect();
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - want).abs() < 1e-4);
    }
}
