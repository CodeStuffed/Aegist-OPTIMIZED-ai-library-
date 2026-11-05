//! The confidence probe (trained by aegist/confidence.py) and the sentence tracker that uses it
//! to mark answer sentences that are likely wrong. The probe is a linear head on the hidden state
//! plus four logit statistics; it flags likely mistakes, it does not check facts.
use crate::cuda::Result;
use crate::nx::NxFile;
use serde_json::Value;
use std::path::Path;

pub struct Probe {
    /// The layer whose hidden state it reads; None = logit statistics only.
    pub layer: Option<usize>,
    pub threshold: f32,
    pub path: String,
    pub metrics: Value,
    /// Sentences whose mean squared standardised feature is above this are unlike anything the
    /// probe was trained on, so it does not judge them.
    pub ood_max: Option<f32>,
    mean: Vec<f32>,
    std: Vec<f32>,
    w: Vec<f32>,
    b: f32,
}

fn f32s(nx: &NxFile, name: &str) -> Result<Vec<f32>> {
    let t = nx.info(name)?;
    if t.dtype != "f32" {
        return Err(format!("{}: {name} is {}, expected f32", nx.path, t.dtype));
    }
    Ok(nx.bytes(name)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

impl Probe {
    /// Load a probe and check that it fits a brain with `d_model` and `layers`.
    pub fn load(path: &Path, d_model: usize, layers: usize) -> Result<Probe> {
        let nx = NxFile::open(&path.to_string_lossy())?;
        let c = &nx.header["config"];
        if c["kind"].as_str() != Some("confidence-probe") {
            return Err(format!("{} is not a confidence probe", path.display()));
        }
        let layer = c["layer"].as_u64().map(|l| l as usize);
        let (mean, std, w) = (f32s(&nx, "probe.mean")?, f32s(&nx, "probe.std")?, f32s(&nx, "probe.w")?);
        let b = *f32s(&nx, "probe.b")?.first().ok_or("probe.b is empty")?;
        let want = layer.map_or(0, |_| d_model) + 4;
        if mean.len() != want || std.len() != want || w.len() != want {
            return Err(format!("{} expects {} features; this brain gives {want}", path.display(), w.len()));
        }
        if let Some(l) = layer.filter(|&l| l >= layers) {
            return Err(format!("{} reads layer {l}; this brain has {layers}", path.display()));
        }
        Ok(Probe {
            layer,
            threshold: c["threshold"].as_f64().unwrap_or(0.5) as f32,
            path: path.display().to_string(),
            metrics: nx.header["metrics"].clone(),
            ood_max: c["ood_max"].as_f64().map(|v| v as f32),
            mean,
            std,
            w,
            b,
        })
    }

    /// P(wrong) for one sentence from its mean hidden state (when the probe uses one) and
    /// [mean log-prob, min log-prob, mean entropy, max entropy] of its tokens; None when the
    /// sentence is out of the probe's range (not judged).
    pub fn score(&self, hidden_mean: &[f32], stats: [f32; 4]) -> Option<f32> {
        let x = if self.layer.is_some() { hidden_mean } else { &[] };
        let (mut z, mut sq, mut n) = (0f32, 0f32, 0usize);
        for (i, &v) in x.iter().chain(stats.iter()).enumerate() {
            let s = (v - self.mean[i]) / self.std[i];
            z += s * self.w[i];
            sq += s * s;
            n += 1;
        }
        if self.ood_max.is_some_and(|m| sq / n.max(1) as f32 > m) {
            return None;
        }
        Some(1.0 / (1.0 + (-(z + self.b)).exp()))
    }
}

/// (log-sum-exp, entropy) of a distribution given as logits, at temperature 1.
pub fn logit_stats(logits: &[f32]) -> (f64, f64) {
    let top = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    // with e_i = exp(l_i - top): lse = top + ln S, entropy = ln S - sum(e_i (l_i - top)) / S
    let (s, t) = logits.iter().fold((0f64, 0f64), |(s, t), &l| {
        let x = l as f64 - top;
        let e = x.exp();
        (s + e, t + e * x)
    });
    (top + s.ln(), s.ln() - t / s)
}

/// Splits the streamed answer into sentences and scores each as it ends. Code blocks are skipped.
#[derive(Default)]
pub struct Tracker {
    sum: Vec<f32>,
    n_hidden: usize,
    lps: Vec<f64>,
    ents: Vec<f64>,
    text: String,
    fences: usize,
    pending_end: bool,
    took_last: bool,
    pub scored: usize,
    pub flagged: usize,
    /// Sentences out of the probe's range: neither marked nor counted as scored.
    pub unjudged: usize,
    /// The flagged sentences and their P(wrong).
    pub marked: Vec<(String, f32)>,
    /// The highest P(wrong) of any scored sentence.
    pub highest: f32,
    /// When the engine taps more layers than the probe reads: (the probe's tap, number of taps).
    pub slice: Option<(usize, usize)>,
    /// Some: keep every sentence the probe looked at (probe training data from chat).
    pub record: Option<Vec<Sentence>>,
}

/// A sentence as the probe saw it, kept by `aegist exam --record` for probe training.
#[derive(Debug, Clone)]
pub struct Sentence {
    pub text: String,
    /// The mean over its tokens of every tapped layer's hidden state, tap after tap.
    pub mean: Vec<f32>,
    /// [mean log-prob, min log-prob, mean entropy, max entropy].
    pub stats: [f32; 4],
    /// The probe's P(wrong); None when the sentence was out of its range.
    pub p: Option<f32>,
    pub tokens: usize,
}

const MIN_TOKENS: usize = 3;

/// A heading or a label ("**Step 2:**", "## Results") introduces claims but makes none.
fn is_label(text: &str) -> bool {
    let t = text.trim();
    t.starts_with('#') || t.trim_end_matches(['*', '_', ' ']).ends_with(':')
}

impl Tracker {
    /// slice: see Tracker::slice; record: keep the sentences (Tracker::record).
    pub fn new(slice: Option<(usize, usize)>, record: bool) -> Tracker {
        Tracker { slice, record: record.then(Vec::new), ..Default::default() }
    }

    fn close(&mut self, probe: &Probe) -> Option<f32> {
        let ok = self.lps.len() >= MIN_TOKENS && self.text.chars().any(|c| c.is_alphanumeric())
            && !is_label(&self.text) && (probe.layer.is_none() || self.n_hidden > 0);
        let p = ok.then(|| {
            let n = self.lps.len() as f64;
            let stats = [
                (self.lps.iter().sum::<f64>() / n) as f32,
                self.lps.iter().cloned().fold(f64::INFINITY, f64::min) as f32,
                (self.ents.iter().sum::<f64>() / n) as f32,
                self.ents.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as f32,
            ];
            let mean: Vec<f32> = self.sum.iter().map(|v| v / self.n_hidden.max(1) as f32).collect();
            let x = match self.slice {
                Some((i, k)) if !mean.is_empty() => {
                    let d = mean.len() / k;
                    &mean[i * d..(i + 1) * d]
                }
                _ => &mean[..],
            };
            let p = probe.score(x, stats);
            if let Some(rec) = &mut self.record {
                rec.push(Sentence { text: self.text.trim().to_string(), mean, stats, p, tokens: self.lps.len() });
            }
            p
        });
        let text = std::mem::take(&mut self.text);
        self.close_silently();
        if p == Some(None) {
            self.unjudged += 1;
        }
        if let Some(Some(p)) = p {
            self.scored += 1;
            self.highest = self.highest.max(p);
            if p >= probe.threshold {
                self.flagged += 1;
                self.marked.push((text.trim().to_string(), p));
                return Some(p);
            }
        }
        None
    }

    /// Call for each answer token before showing its text `delta`, with its log-prob and the
    /// entropy it was drawn from. Returns where in `delta` an unsure mark goes (byte index) and
    /// the sentence's P(wrong), when the sentence before that point ends and was flagged.
    pub fn token(&mut self, probe: &Probe, delta: &str, logprob: f64, entropy: f64) -> Option<(usize, f32)> {
        let mut mark = None;
        let fences_before = self.fences;
        self.fences += delta.matches("```").count();
        if fences_before % 2 == 1 || self.fences % 2 == 1 {
            // inside a code block (or entering / leaving one): drop the sentence, score nothing
            if fences_before % 2 == 0 {
                mark = self.close(probe).map(|p| (0, p));
            }
            self.close_silently();
            self.took_last = false;
            return mark;
        }
        let starts_space = delta.starts_with(char::is_whitespace);
        if let Some(k) = delta.find('\n') {
            // a line break ends the sentence; this token's own numbers are left out
            self.text.push_str(&delta[..k]);
            mark = self.close(probe).map(|p| (k, p));
            self.took_last = false;
            return mark;
        }
        if self.pending_end && starts_space {
            mark = self.close(probe).map(|p| (0, p));
        }
        if !delta.is_empty() {
            let t = delta.trim_end_matches(['"', '\'', ')', ']', '*', '\u{201D}']);
            self.pending_end = t.ends_with(['.', '!', '?']);
        }
        self.text.push_str(delta);
        self.lps.push(logprob);
        self.ents.push(entropy);
        self.took_last = true;
        mark
    }

    /// The hidden state at the token just passed to `token` (after the engine fed it).
    pub fn hidden(&mut self, h: &[f32]) {
        if !self.took_last || h.is_empty() {
            return;
        }
        if self.sum.len() != h.len() {
            self.sum = vec![0.0; h.len()];
        }
        self.sum.iter_mut().zip(h).for_each(|(s, v)| *s += v);
        self.n_hidden += 1;
    }

    /// The reply ended: score what is left.
    pub fn finish(&mut self, probe: &Probe) -> Option<f32> {
        if self.fences % 2 == 1 {
            self.close_silently();
            return None;
        }
        self.close(probe)
    }

    /// The answer pauses here (a tool call): score the sentence so far, and attach nothing that
    /// follows to it.
    pub fn break_here(&mut self, probe: &Probe) -> Option<f32> {
        let m = self.finish(probe);
        self.took_last = false;
        m
    }

    fn close_silently(&mut self) {
        self.sum.iter_mut().for_each(|v| *v = 0.0);
        self.n_hidden = 0;
        self.lps.clear();
        self.ents.clear();
        self.text.clear();
        self.pending_end = false;
    }
}

/// The probe to use with `brain`: a probe reads that brain's own hidden states, so
/// probes/<brain file stem>/current.txt comes first. Then probes/current.txt (the 35B brain's),
/// else probes/confidence.nx; None if none exists.
pub fn find(home: &Path, brain: &Path) -> Option<std::path::PathBuf> {
    let dir = home.join("probes");
    let mut own: Vec<std::path::PathBuf> = brain.file_stem().map(|s| dir.join(s)).into_iter().collect();
    // a learned cores file uses its base brain's probe
    if let Ok(nx) = crate::nx::NxFile::open(&brain.to_string_lossy()) {
        if let Some(s) = Path::new(&nx.path).file_stem() {
            own.push(dir.join(s));
        }
    }
    for d in own.iter().chain(std::iter::once(&dir)) {
        if let Ok(rel) = std::fs::read_to_string(d.join("current.txt")) {
            let p = d.join(rel.trim());
            if p.exists() {
                return Some(p);
            }
        }
    }
    Some(dir.join("confidence.nx")).filter(|p| p.exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats_probe(threshold: f32) -> Probe {
        // P(wrong) rises as the mean log-prob falls
        Probe { layer: None, threshold, path: String::new(), metrics: Value::Null, ood_max: None, mean: vec![0.0; 4], std: vec![1.0; 4],
                w: vec![-4.0, 0.0, 0.0, 0.0], b: -1.0 }
    }

    fn feed(t: &mut Tracker, p: &Probe, parts: &[(&str, f64)]) -> Vec<(usize, usize, f32)> {
        let mut marks = vec![];
        for (i, (d, lp)) in parts.iter().enumerate() {
            if let Some((at, s)) = t.token(p, d, *lp, 0.5) {
                marks.push((i, at, s));
            }
        }
        marks
    }

    #[test]
    fn flags_the_unsure_sentence_only() {
        let p = stats_probe(0.5);
        let mut t = Tracker::default();
        let m = feed(&mut t, &p, &[("The", -0.01), (" sky", -0.01), (" is", -0.01), (" blue", -0.01), (".", -0.01),
                                    (" It", -2.0), (" was", -2.0), (" named", -2.0), (" in", -2.0), (" 1712", -2.0), (".", -2.0)]);
        assert!(m.is_empty(), "{m:?}");
        let last = t.finish(&p).expect("second sentence flagged");
        assert!(last > 0.9);
        assert_eq!((t.scored, t.flagged), (2, 1));
        assert_eq!(t.marked[0].0, "It was named in 1712.");
    }

    #[test]
    fn decimals_do_not_end_sentences_and_newlines_do() {
        let p = stats_probe(0.5);
        let mut t = Tracker::default();
        let m = feed(&mut t, &p, &[("Pi", -3.0), (" is", -3.0), (" 3", -3.0), (".", -3.0), ("14", -3.0), (".\n", -0.01),
                                    ("Done", -0.01), (" and", -0.01), (" dusted", -0.01)]);
        assert_eq!(m.len(), 1);
        assert_eq!((m[0].0, m[0].1), (5, 1)); // the mark goes before the line break
        assert!(t.finish(&p).is_none());
        assert_eq!(t.scored, 2);
    }

    #[test]
    fn sentences_out_of_range_are_not_judged() {
        let mut p = stats_probe(0.5);
        p.ood_max = Some(4.0);
        let mut t = Tracker::default();
        // standardised stats far beyond anything seen in training
        let m = feed(&mut t, &p, &[("Odd", -40.0), (" text", -40.0), (" here", -40.0), (".", -40.0)]);
        assert!(m.is_empty());
        assert!(t.finish(&p).is_none());
        assert_eq!((t.scored, t.flagged, t.unjudged), (0, 0, 1));
    }

    #[test]
    fn headings_and_labels_are_not_scored() {
        let p = stats_probe(0.5);
        let mut t = Tracker::default();
        let m = feed(&mut t, &p, &[("**", -3.0), ("Multiplication", -3.0), (" of", -3.0), (" 4", -3.0), (":**", -3.0), ("\n", -3.0),
                                    ("##", -3.0), (" Big", -3.0), (" picture", -3.0), ("\n", -3.0)]);
        assert!(m.is_empty(), "{m:?}");
        assert!(t.finish(&p).is_none());
        assert_eq!(t.scored, 0);
    }

    #[test]
    fn code_blocks_are_skipped() {
        let p = stats_probe(0.5);
        let mut t = Tracker::default();
        let m = feed(&mut t, &p, &[("```", -5.0), ("rust", -5.0), ("\n", -5.0), ("let", -5.0), (" x", -5.0), (" =", -5.0), (" 1", -5.0),
                                    (";\n", -5.0), ("```", -5.0), ("\n", -5.0)]);
        assert!(m.is_empty(), "{m:?}");
        assert!(t.finish(&p).is_none());
        assert_eq!(t.scored, 0);
    }
}
