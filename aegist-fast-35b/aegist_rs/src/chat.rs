//! Chat runtime: streaming generation, adaptive thinking, modes, shield. A port of aegist/chat.py.
use crate::cuda::Result;
use crate::governor::{Governor, IdeaStat};
use crate::laya::{Kind, Laya};
use crate::model::Lm;
use crate::persona;
use crate::probe::{self, Probe, Sentence, Tracker};
use crate::shield;
use crate::tokenizer::{Tok, THINK_CLOSE};
use crate::tools;
use regex::{Regex, RegexBuilder};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct GenConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_new_tokens: usize,
    pub think_temperature: f32,
    pub think_top_p: f32,
    /// Subtracted from the logit of every token already generated in this reply (0 = off).
    pub presence_penalty: f32,
    /// When thinking, the prompt itself opens the think block (newer chat templates do).
    pub open_think: bool,
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            max_new_tokens: 2048,
            think_temperature: 0.6,
            think_top_p: 0.95,
            presence_penalty: 0.0,
            open_think: false,
        }
    }
}

impl GenConfig {
    /// Built-in defaults per architecture ("hybrid-moe" uses its base model's recommended settings).
    pub fn for_arch(arch: &str) -> GenConfig {
        let d = GenConfig::default();
        if arch == "hybrid-moe" {
            GenConfig { think_temperature: 1.0, think_top_p: 0.95, presence_penalty: 1.5, open_think: true, ..d }
        } else {
            d
        }
    }

    /// Settings from the brain's header ("generation"), falling back to the architecture's
    /// defaults. A hybrid header with only temperature / top_p (no think_* keys) predates the
    /// split; there those two are the thinking settings.
    pub fn from_header(g: &Value, arch: &str) -> GenConfig {
        let d = GenConfig::for_arch(arch);
        let f = |k: &str, v: f32| g[k].as_f64().map(|x| x as f32).unwrap_or(v);
        let u = |k: &str, v: usize| g[k].as_u64().map(|x| x as usize).unwrap_or(v);
        let split = !g["think_temperature"].is_null() || arch != "hybrid-moe";
        GenConfig {
            temperature: if split { f("temperature", d.temperature) } else { d.temperature },
            top_p: if split { f("top_p", d.top_p) } else { d.top_p },
            top_k: u("top_k", d.top_k).max(1),
            max_new_tokens: u("max_new_tokens", d.max_new_tokens),
            think_temperature: f(if split { "think_temperature" } else { "temperature" }, d.think_temperature),
            think_top_p: f(if split { "think_top_p" } else { "top_p" }, d.think_top_p),
            presence_penalty: f("presence_penalty", d.presence_penalty),
            open_think: g["open_think"].as_bool().unwrap_or(d.open_think),
        }
    }
}

pub struct ThinkPlan {
    pub think: bool,
    pub budget: usize,
    pub reason: String,
}

const REASONING: &str = r"(\d\s*[-+*/^=]\s*\d|prove|proof|solve|calculate|equation|integral|derivative|probability|how many|step by step|algorithm|debug|bug|code|function|python|rust|sql|regex|why does|explain why|logic|puzzle|riddle|compare|plan|design|optimi[sz]e|complexity|\bfix\b)";

/// Cues that do not call for thinking on their own in a short question.
const WEAK: &[&str] = &["how many", "why does", "explain why", "compare", "plan", "design", "code", "function", "python", "rust", "sql"];

/// Most thinking tokens an auto plan allows (thinking is about half of all tokens otherwise).
const THINK_MAX: usize = 1536;

/// setting: "off" | "on" | "auto".
pub fn plan_thinking(text: &str, setting: &str) -> ThinkPlan {
    static R: OnceLock<Regex> = OnceLock::new();
    match setting {
        "off" => return ThinkPlan { think: false, budget: 0, reason: "thinking off".into() },
        "on" => return ThinkPlan { think: true, budget: 4096, reason: "thinking on".into() },
        _ => {}
    }
    let found: Vec<String> =
        R.get_or_init(|| RegexBuilder::new(REASONING).case_insensitive(true).build().unwrap()).find_iter(text).map(|m| m.as_str().to_lowercase()).collect();
    let hits = found.len();
    let len = text.chars().count();
    if hits == 0 && len < 200 {
        return ThinkPlan { think: false, budget: 0, reason: "auto: conversational".into() };
    }
    // a short question with only a weak cue ("how many legs does a spider have?") is answered
    // straight: thinking spent 600 tokens on that one
    if len < 120 && found.iter().all(|f| WEAK.contains(&f.as_str())) {
        return ThinkPlan { think: false, budget: 0, reason: format!("auto: short question, weak cues x{hits}") };
    }
    ThinkPlan { think: true, budget: THINK_MAX.min(256 + 384 * hits + len / 4), reason: format!("auto: reasoning cues x{hits}") }
}

/// Prompts that need the big brain's depth: proofs, hard maths, algorithms, long multi-part tasks.
const HARD: &str = r"(prove|proof|integral|derivative|theorem|algorithm|complexity|step by step|riddle|puzzle|optimi[sz]e|architecture|research|in depth|detailed analysis)";

/// Which brain answers when a fast one is loaded: (big, why). setting: "auto" | "big" | "fast".
/// Auto gives the fast brain everything but prompts with a hard cue or a long text; the fast
/// brain's unsure answers go to the big one afterwards (Chat::reply).
pub fn pick_brain(text: &str, setting: &str) -> (bool, String) {
    static R: OnceLock<Regex> = OnceLock::new();
    match setting {
        "big" => return (true, "big brain chosen".into()),
        "fast" => return (false, "fast brain chosen".into()),
        _ => {}
    }
    let hits = R.get_or_init(|| RegexBuilder::new(HARD).case_insensitive(true).build().unwrap()).find_iter(text).count();
    let len = text.chars().count();
    if hits > 0 {
        (true, format!("auto: hard cues x{hits}"))
    } else if len > 800 {
        (true, "auto: long prompt".into())
    } else {
        (false, "auto: everyday prompt".into())
    }
}

#[derive(Default, Debug)]
pub struct TurnStats {
    pub prompt_tokens: usize,
    pub reused_tokens: usize,
    pub thinking_tokens: usize,
    pub answer_tokens: usize,
    pub seconds: f64,
    pub prefill_seconds: f64,
    pub think_plan: String,
    pub shield: String,
    pub cores_used: usize,
    /// Answer sentences the confidence probe scored, and how many it marked unsure.
    pub sentences_scored: usize,
    pub sentences_unsure: usize,
    /// Answer sentences unlike what the probe was trained on, so not judged.
    pub sentences_unjudged: usize,
    /// The sentences marked unsure, with their P(wrong).
    pub unsure: Vec<(String, f32)>,
    /// The highest P(wrong) of any scored sentence.
    pub highest_unsure: f32,
    /// What the reading core read the prompt as asking for, and its probabilities (fact, creative, other).
    pub kind: Option<Kind>,
    pub kind_p: [f32; 3],
    /// A fact answer went greedy after its first unsure sentence.
    pub cooled: bool,
    /// The tools it called, one line each ("calculator: 87707 * 2233 = 195849731").
    pub tools_used: Vec<String>,
    /// The first answer was marked unsure, so it answered again (fact prompts).
    pub second_look: bool,
    /// The final answer's unsure marks (the second look's, when there was one).
    pub final_marks: Vec<f32>,
    /// When recording (Chat::record_sentences): each answer pass's text and the sentences the
    /// probe looked at (the first answer, then the second look's).
    pub recorded: Vec<(String, Vec<Sentence>)>,
    /// Which brain answered and why ("" when only one brain is loaded).
    pub brain: String,
}

impl TurnStats {
    /// Generation speed, prefill excluded.
    pub fn tok_per_s(&self) -> f64 {
        let n = (self.thinking_tokens + self.answer_tokens) as f64;
        let s = self.seconds - self.prefill_seconds;
        if s > 0.0 { n / s } else { 0.0 }
    }
}

pub struct Session {
    pub mode: String,
    pub think: String,
    /// Which brain answers when a fast one is loaded: "auto" | "big" | "fast".
    pub brain: String,
    pub turns: Vec<(String, String)>,
    pub memory: Vec<String>,
}

impl Session {
    pub fn new(mode: &str) -> Session {
        Session { mode: mode.into(), think: "auto".into(), brain: "auto".into(), turns: vec![], memory: vec![] }
    }
}

pub enum Event<'a> {
    Shield(&'a str),
    Think(&'a str),
    Answer(&'a str),
    /// The answer sentence just shown is likely wrong (the probe's P(wrong)).
    Unsure(f32),
    /// A new part of the reply begins (idea mode: each wild idea, then the review).
    Phase(&'a str),
    /// A note for the user, e.g. the governor moved a knob.
    Note(&'a str),
    /// The reading core read the prompt: what it asks for, how sure it is, how the answer is sampled.
    Read(Kind, f32, &'a str),
    /// The reply ended with sentences marked unsure: what Aegist says about that.
    UnsureSummary(&'a str),
    /// A tool ran: which, and what it returned.
    Tool(&'a str),
}

/// splitmix64: small, fast and good enough for sampling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Top-k, then top-p (nucleus) sampling; temperature 0 is greedy.
pub fn sample(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut Rng) -> u32 {
    let argmax = || logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
    if temperature <= 0.0 {
        return argmax();
    }
    let k = top_k.min(logits.len());
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let desc = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, desc);
        idx.truncate(k);
    }
    idx.sort_by(desc);
    let top = logits[idx[0] as usize];
    let mut p: Vec<f64> = idx.iter().map(|&i| (((logits[i as usize] - top) / temperature) as f64).exp()).collect();
    let sum: f64 = p.iter().sum();
    let mut cum = 0.0;
    for x in p.iter_mut() {
        *x /= sum;
        let keep = cum < top_p as f64; // cumulative mass before this token
        cum += *x;
        if !keep {
            *x = 0.0;
        }
    }
    let total: f64 = p.iter().sum();
    let mut r = rng.next_f64() * total;
    for (j, &x) in p.iter().enumerate() {
        r -= x;
        if r < 0.0 && x > 0.0 {
            return idx[j];
        }
    }
    idx[0]
}

pub struct Chat {
    pub engine: Box<dyn Lm>,
    pub tok: Tok,
    pub gen: GenConfig,
    pub probe: Option<Probe>,
    pub governor: Governor,
    /// The reading core (Laya in Rust); None: every prompt is answered with the default settings.
    pub reader: Option<Laya>,
    /// Let the brain call the exact tools (tools.rs) outside idea mode.
    pub tools: bool,
    /// Answer a fact again when the first answer was marked unsure.
    pub second_look: bool,
    /// Where the brain's state after a shared prompt start is kept between runs (None: not kept).
    pub state_dir: Option<PathBuf>,
    /// Tap these layers and keep every sentence the probe looks at (probe training data); empty: off.
    record_taps: Vec<usize>,
    /// Where the probe's layer sits among the tapped ones, when more are tapped.
    probe_tap: Option<(usize, usize)>,
    cache_ids: Vec<u32>,
    /// The brain's header name (each brain names itself in its system prompt).
    pub name: String,
    /// A smaller, faster brain on the same GPUs: everyday prompts go to it (pick_brain).
    pub fast: Option<Box<Chat>>,
    /// Hand the fast brain's answers that its probe marked unsure to this brain.
    pub escalate: bool,
}

/// What Aegist tells itself before a second look. It goes in as a user turn (the chat format has
/// no other place), says it is not from the user, and is not kept in the conversation.
const SECOND_LOOK: &str = "(Aegist's own check, not a message from the user.) Your last answer was marked as likely \
wrong. Answer the question again from the start, briefly. If you do not actually know of the thing it asks about, or \
are not sure of the fact, say so plainly instead of guessing. Use a tool if one fits. Write only the new answer, \
for the user: do not mention this check, the mark or your last answer, and do not write marks such as [unsure] \
yourself.";

/// The tokens a saved state holds (from its header), or None if it is not one.
fn state_toks(path: &Path) -> Option<Vec<u32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut head = [0u8; 16];
    f.read_exact(&mut head).ok()?;
    if &head[..8] != b"AEGSTATE" {
        return None;
    }
    let mut json = vec![0u8; u64::from_le_bytes(head[8..].try_into().ok()?) as usize];
    f.read_exact(&mut json).ok()?;
    let v: Value = serde_json::from_slice(&json).ok()?;
    v["toks"].as_array()?.iter().map(|t| t.as_u64().map(|t| t as u32)).collect()
}

/// FNV-1a of a token list, for state file names.
fn fnv(ids: &[u32]) -> u64 {
    ids.iter().flat_map(|t| t.to_le_bytes()).fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// Tool calls one reply may make.
const MAX_TOOL_ROUNDS: usize = 4;
/// Most tokens of one tool call given back to the reply's max_new: without it, a few searches
/// could spend the whole budget, and the reply ended inside an unfinished call.
const TOOL_CALL_REFUND: usize = 128;
/// No new tool call this close to the end of the reply's budget: it could not finish, and the
/// reply would end with no answer.
const TOOL_CALL_ROOM: usize = 48;
/// Text shown before a reply's tool calls shorter than this is dropped from the saved answer
/// when an answer of at least a third of it follows.
const SHORT_BEFORE_TOOLS: usize = 120;

impl Chat {
    pub fn new(engine: Box<dyn Lm>, tok: Tok, gen: GenConfig) -> Chat {
        Chat { engine, tok, gen, probe: None, governor: Governor::default(), reader: None, tools: true, second_look: true,
               state_dir: None, record_taps: vec![], probe_tap: None, cache_ids: vec![],
               name: String::new(), fast: None, escalate: true }
    }

    /// Use a confidence probe to mark unsure answer sentences (None: stop marking).
    pub fn set_probe(&mut self, probe: Option<Probe>) -> Result<()> {
        let layer = probe.as_ref().and_then(|p| p.layer);
        let taps: Vec<usize> = match layer {
            Some(l) if !self.record_taps.is_empty() => {
                let i = self.record_taps.iter().position(|&t| t == l)
                    .ok_or_else(|| format!("the probe reads layer {l}, which is not among the recorded taps {:?}", self.record_taps))?;
                self.probe_tap = Some((i, self.record_taps.len()));
                self.record_taps.clone()
            }
            _ => {
                self.probe_tap = None;
                layer.into_iter().collect()
            }
        };
        self.engine.set_taps(&taps)?;
        self.probe = probe;
        Ok(())
    }

    /// Keep every sentence the probe looks at, with the hidden states of all `taps` (TurnStats::recorded).
    pub fn record_sentences(&mut self, taps: Vec<usize>) -> Result<()> {
        if self.probe.as_ref().and_then(|p| p.layer).is_none() {
            return Err("recording needs a probe that reads a layer".into());
        }
        self.record_taps = taps;
        let probe = self.probe.take();
        self.set_probe(probe)
    }

    pub fn max_context(&self) -> usize {
        self.engine.cap()
    }

    /// Feed `ids`, reusing the longest cached prefix the engine can go back to. Engines with
    /// recurrent state go back only to a snapshot, so snapshots are taken at `sys_at` (the end of
    /// the system message, shared by every new conversation) and at `snap_at` (the start of this
    /// reply, where the next turn's prompt will diverge). Returns the last logits and #reused tokens.
    fn prefill(&mut self, ids: &[u32], snap_at: usize, sys_at: usize) -> Result<(Vec<f32>, usize)> {
        let mut n = self.cache_ids.iter().zip(ids).take_while(|(a, b)| a == b).count();
        n = n.min(self.engine.length()).min(ids.len() - 1); // always run at least one token for fresh logits
        let mut kept = self.engine.crop(n);
        self.cache_ids.truncate(kept);
        if let Some(got) = self.load_saved(&ids[..ids.len() - 1], kept) {
            kept = got;
            self.cache_ids = ids[..got].to_vec();
        } else if self.engine.length() != kept {
            kept = 0; // a state failed to load and left the engine empty
            self.cache_ids.clear();
        }
        let mut points: Vec<usize> = [sys_at, snap_at].into_iter().filter(|&p| p > kept && p <= ids.len()).collect();
        points.sort_unstable();
        points.dedup();
        let (mut at, mut logits) = (kept, None);
        for p in points {
            logits = Some(self.engine.feed(&ids[at..p])?);
            self.engine.snapshot()?;
            at = p;
            if p == sys_at {
                self.save_state(&ids[..p]);
            }
        }
        let logits = match logits {
            Some(l) if at == ids.len() => l,
            _ => self.engine.feed(&ids[at..])?,
        };
        self.cache_ids.extend_from_slice(&ids[kept..]);
        Ok((logits, kept))
    }

    /// The saved states of this brain: (path, the tokens it holds).
    fn saved_states(&self) -> Vec<(PathBuf, Vec<u32>)> {
        let (Some(dir), Some(key)) = (&self.state_dir, self.engine.state_key()) else { return vec![] };
        let mut found = vec![];
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let p = e.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with(&format!("{key}-")) && name.ends_with(".state") {
                if let Some(toks) = state_toks(&p) {
                    found.push((p, toks));
                }
            }
        }
        found
    }

    /// Load the longest saved state that `ids` starts with, if it holds more than `kept` tokens.
    /// Returns its length.
    fn load_saved(&mut self, ids: &[u32], kept: usize) -> Option<usize> {
        let best = self.saved_states().into_iter().filter(|(_, t)| t.len() > kept && ids.starts_with(t)).max_by_key(|(_, t)| t.len())?;
        match self.engine.load_state(&best.0) {
            Ok(toks) => {
                // the last use, for pruning the oldest first
                let _ = std::fs::File::options().write(true).open(&best.0).and_then(|f| f.set_modified(std::time::SystemTime::now()));
                Some(toks.len())
            }
            Err(e) => {
                eprintln!("(a saved state could not be loaded, so it was removed: {e})");
                let _ = std::fs::remove_file(&best.0);
                self.cache_ids.clear();
                None
            }
        }
    }

    /// Save the engine's state after `ids` (its whole length) unless it is saved already, then
    /// remove the least recently used states past $AEGIST_STATE_GB (default 4).
    fn save_state(&self, ids: &[u32]) {
        let (Some(dir), Some(key)) = (&self.state_dir, self.engine.state_key()) else { return };
        if ids.len() < 64 || ids.len() != self.engine.length() {
            return;
        }
        let path = dir.join(format!("{key}-{:016x}.state", fnv(ids)));
        if path.exists() {
            return;
        }
        if let Err(e) = std::fs::create_dir_all(dir).map_err(|e| e.to_string()).and_then(|_| self.engine.save_state(&path)) {
            eprintln!("(the context state was not saved: {e})");
            return;
        }
        let limit = std::env::var("AEGIST_STATE_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(4.0) * 1e9;
        let mut all: Vec<(std::time::SystemTime, u64, PathBuf)> = std::fs::read_dir(dir).into_iter().flatten().flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "state"))
            .filter_map(|e| e.metadata().ok().map(|m| (m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len(), e.path())))
            .collect();
        all.sort();
        let mut total: u64 = all.iter().map(|a| a.1).sum();
        for (_, len, p) in all {
            if total as f64 <= limit || p == path {
                break;
            }
            if std::fs::remove_file(&p).is_ok() {
                total -= len;
            }
        }
    }

    fn step(&mut self, id: u32) -> Result<Vec<f32>> {
        self.cache_ids.push(id);
        self.engine.feed(&[id])
    }

    /// Forget the cached conversation (the next turn prefills from scratch).
    pub fn reset(&mut self) {
        self.cache_ids.clear();
        self.engine.crop(0);
    }

    /// The prompt for this turn, dropping the oldest turns until it fits with `reserve` tokens to
    /// spare. Also returns the open assistant header's position (no think markup), where the next
    /// turn's prompt will diverge from this one.
    fn fit(&self, system: &str, turns: &mut Vec<(String, String)>, think: bool, reserve: usize, tools: bool)
           -> Result<(Vec<u32>, usize, usize)> {
        let cap = self.max_context();
        let open = self.gen.open_think;
        let block = if tools { tools::system_block() } else { String::new() };
        let ids_for = |turns: &[(String, String)], think: bool, open: bool| self.tok.chat_ids_tools(&block, system, turns, think, open);
        let mut ids = ids_for(turns, think, open);
        if ids.len() + reserve > cap {
            // drop the oldest turns until the conversation fits
            while turns.len() > 1 && ids_for(turns, think, open).len() + reserve.max(512) > cap {
                turns.drain(..2.min(turns.len() - 1));
            }
            ids = ids_for(turns, think, open);
            if ids.len() + 8 > cap {
                return Err(format!("message too long: {} tokens, the context holds {cap}", ids.len()));
            }
        }
        // sys_at: where the system message's shared part ends (the tone and shield lines vary)
        let full = self.tok.system_ids(&block, system);
        let shared = self.tok.system_ids(&block, &system[..persona::shared_len(system)]);
        let sys_at = full.iter().zip(&shared).take_while(|(a, b)| a == b).count();
        Ok((ids, ids_for(turns, true, false).len(), sys_at))
    }

    /// One stretch of generation after `ids` (a snapshot is kept at `snap_at`), streamed to `emit`.
    fn run_pass(&mut self, ids: &[u32], snap_at: usize, sys_at: usize, pass: &Pass, rng: &mut Rng, emit: &mut dyn FnMut(Event))
                -> Result<PassOut> {
        let t0 = Instant::now();
        let (mut logits, reused) = self.prefill(ids, snap_at, sys_at)?;
        let mut res = PassOut { reused, prefill_seconds: t0.elapsed().as_secs_f64(), ..Default::default() };
        let max_new = pass.max_new.min(self.max_context().saturating_sub(ids.len() + 1));
        let use_probe = pass.mark_unsure && self.probe.is_some();
        let tapped = use_probe && self.probe.as_ref().is_some_and(|p| p.layer.is_some());
        let budget = pass.think_budget.unwrap_or(0);
        let mut temperature = pass.temperature;
        let mut thinking = pass.think_budget.is_some();
        // thinking runs light (the engine may skip more experts it would read from disk); the
        // answer, which the user reads and the probe judges, runs at full effort
        self.engine.set_light(thinking);
        let (mut seen, mut out, mut think_ids) = (HashSet::<u32>::new(), Vec::<u32>::new(), Vec::<u32>::new());
        let mut emitted = String::new();
        let mut tracker = Tracker::new(self.probe_tap, !self.record_taps.is_empty());
        let (mut lp_sum, mut lp_n) = (0f64, 0usize);
        let mut cut: Option<String> = None;
        // the answer text shown before each tool call (the call itself is not shown)
        let mut before_tools = String::new();
        let mut tool_rounds = 0;
        // continuation bytes the last token left owing (see Tok::guard_utf8)
        let mut need = 0u8;
        let mut i = 0;
        while i < max_new {
            i += 1;
            // the probe was trained on the model's own distribution, before any penalty
            let dist = ((use_probe || pass.measure) && !thinking).then(|| probe::logit_stats(&logits));
            if self.gen.presence_penalty != 0.0 {
                for &t in &seen {
                    logits[t as usize] -= self.gen.presence_penalty;
                }
            }
            self.tok.guard_utf8(need, &mut logits);
            if pass.tools && !thinking && (tool_rounds >= MAX_TOOL_ROUNDS || i + TOOL_CALL_ROOM > max_new) {
                // calls spent, or no room left to finish one: the reply has to answer with what it has
                if let Some(t) = self.tok.tool_open {
                    logits[t as usize] = f32::NEG_INFINITY;
                }
            }
            let nxt = sample(&logits, temperature, pass.top_p, pass.top_k, rng);
            if self.tok.eos.contains(&nxt) {
                break;
            }
            need = self.tok.utf8_after(need, nxt);
            let raw_logit = logits[nxt as usize] + if seen.contains(&nxt) { self.gen.presence_penalty } else { 0.0 };
            seen.insert(nxt);
            if thinking {
                think_ids.push(nxt);
                if nxt == self.tok.think_close {
                    thinking = false;
                    self.engine.set_light(false);
                    res.thinking_tokens = think_ids.len();
                    emitted.clear();
                    logits = self.step(nxt)?;
                    continue;
                }
                let text = self.tok.decode(&think_ids);
                if !text.ends_with('\u{FFFD}') {
                    let delta = text.get(emitted.len()..).unwrap_or("").replace("<think>", "");
                    emitted = text;
                    if !delta.is_empty() {
                        emit(Event::Think(&delta));
                    }
                }
                if think_ids.len() >= budget {
                    // budget spent: close the thought and make it answer
                    let forced = self.tok.encode_trusted(&format!("\n\nI have thought enough; time to answer.\n{THINK_CLOSE}\n\n"));
                    self.engine.set_light(false);
                    self.cache_ids.extend_from_slice(&forced);
                    logits = self.engine.feed(&forced)?;
                    need = 0;
                    thinking = false;
                    res.thinking_tokens = think_ids.len();
                    emitted.clear();
                    continue;
                }
                logits = self.step(nxt)?;
                continue;
            }
            out.push(nxt);
            if pass.tools {
                let full = self.tok.decode(&out);
                if let Some(k) = full.find("<tool_call>") {
                    // a tool call: hidden from the user and from the probe
                    if full[k..].contains("</tool_call>") {
                        if let Some(p) = self.probe.as_ref().filter(|_| use_probe).and_then(|p| tracker.break_here(p)) {
                            emit(Event::Unsure(p));
                        }
                        tool_rounds += 1;
                        let calls = tools::parse_calls(&full[k..]);
                        let results: Vec<String> = if tool_rounds > MAX_TOOL_ROUNDS {
                            vec!["error: too many tool calls in one reply; answer now with what you have".into()]
                        } else {
                            calls.iter().map(tools::run).collect()
                        };
                        for (c, r) in calls.iter().zip(&results) {
                            let line = tools::describe(c, r);
                            emit(Event::Tool(&line));
                            res.tools.push(line);
                        }
                        let before = full[..k].trim();
                        if !before.is_empty() {
                            before_tools.push_str(before);
                            before_tools.push_str("\n\n");
                        }
                        let cont = [&[nxt][..], &self.tok.tool_responses(&results)].concat();
                        self.cache_ids.extend_from_slice(&cont);
                        logits = self.engine.feed(&cont)?;
                        need = 0;
                        // the call's own tokens do not use up the answer's budget (the rounds are capped)
                        i = i.saturating_sub(out.len().min(TOOL_CALL_REFUND));
                        out.clear();
                        emitted.clear();
                        if tool_rounds > MAX_TOOL_ROUNDS + 1 {
                            break;
                        }
                    } else if i < max_new {
                        logits = self.step(nxt)?;
                    }
                    continue;
                }
            }
            let lp = dist.map(|(lse, _)| raw_logit as f64 - lse);
            if let Some(l) = lp {
                lp_sum += l;
                lp_n += 1;
            }
            let text = self.tok.decode(&out);
            let text = text.trim_start_matches('\n');
            if pass.stop_at_paragraph {
                if let Some(k) = text.find("\n\n").filter(|&k| !text[..k].trim().is_empty()) {
                    // one paragraph is enough: show up to the blank line and stop
                    emit(Event::Answer(text.get(emitted.len()..k).unwrap_or("")));
                    cut = Some(text[..k].to_string());
                    break;
                }
            }
            let mut delta = String::new();
            if !text.ends_with('\u{FFFD}') {
                delta = text.get(emitted.len()..).unwrap_or("").to_string();
                emitted = text.to_string();
            }
            let mark = match (self.probe.as_ref().filter(|_| use_probe), dist) {
                (Some(p), Some((_, ent))) => tracker.token(p, &delta, lp.unwrap_or(0.0), ent),
                _ => None,
            };
            let at = mark.map_or(delta.len(), |(at, _)| at);
            if at > 0 {
                emit(Event::Answer(&delta[..at]));
            }
            if let Some((_, p)) = mark {
                emit(Event::Unsure(p));
                if pass.cool_on_unsure && temperature > 0.0 {
                    // a fact sentence came out unsure: no more sampling noise in this answer
                    temperature = 0.0;
                    res.cooled = true;
                }
            }
            if at < delta.len() {
                emit(Event::Answer(&delta[at..]));
            }
            if i < max_new {
                logits = self.step(nxt)?;
                if tapped {
                    tracker.hidden(&self.engine.taps()?);
                }
            }
        }
        self.engine.set_light(false);
        if let Some(p) = self.probe.as_ref().filter(|_| use_probe).and_then(|p| tracker.finish(p)) {
            emit(Event::Unsure(p));
        }
        res.answer_tokens = out.len();
        res.mean_logprob = if lp_n > 0 { lp_sum / lp_n as f64 } else { 0.0 };
        res.scored = tracker.scored;
        res.unjudged = tracker.unjudged;
        res.marked = tracker.marked;
        res.highest = tracker.highest;
        res.sentences = tracker.record.take().unwrap_or_default();
        let mut last = cut.unwrap_or_else(|| self.tok.decode(&out));
        if let Some(k) = last.find("<tool_call>") {
            // the budget ran out inside a tool call: the call is never shown
            last.truncate(k);
        }
        let last = last.trim();
        // a short line before the lookups ("let me check", or a first guess the answer then
        // restates) is not kept when a real answer follows them
        let keep_before = before_tools.trim().len() >= SHORT_BEFORE_TOOLS || last.len() < SHORT_BEFORE_TOOLS / 3;
        res.text = if keep_before { format!("{before_tools}{last}") } else { last.to_string() }.trim().to_string();
        Ok(res)
    }

    /// Answer `user_text`: on the fast brain when one is loaded and the prompt is an everyday one,
    /// else on this one. A fast answer the probe marked unsure is answered again here.
    pub fn reply(&mut self, session: &mut Session, user_text: &str, seed: u64, mut emit: impl FnMut(Event)) -> Result<TurnStats> {
        let t0 = Instant::now();
        let Some(mut fast) = self.fast.take() else {
            return self.reply_here(session, user_text, seed, &mut emit);
        };
        let (big, why) = if session.mode == "idea" { (false, "idea mode".to_string()) } else { pick_brain(user_text, &session.brain) };
        if big {
            self.fast = Some(fast);
            let mut stats = self.reply_here(session, user_text, seed, &mut emit)?;
            stats.brain = format!("{} ({why})", self.name);
            return Ok(stats);
        }
        let before = session.turns.clone();
        // one reading core serves both brains; the big one's warm-up stays off the disk meanwhile
        fast.reader = self.reader.take();
        self.engine.hold_background(true);
        let r = fast.reply_here(session, user_text, seed, &mut emit);
        self.engine.hold_background(false);
        self.reader = fast.reader.take();
        let name = fast.name.clone();
        self.fast = Some(fast);
        let mut stats = r?;
        stats.brain = format!("{name} ({why})");
        // only facts go up: a marked line in a list of names or a poem is not worth the big brain
        if stats.final_marks.is_empty() || stats.kind != Some(Kind::Fact) || !self.escalate || session.brain == "fast" {
            return Ok(stats);
        }
        emit(Event::Phase("the fast brain was unsure of that answer, so the big brain answers it"));
        session.turns = before;
        let mut again = self.reply_here(session, user_text, seed, &mut emit)?;
        again.brain = format!("{} (the fast brain, {name}, was unsure)", self.name);
        again.seconds = t0.elapsed().as_secs_f64();
        Ok(again)
    }

    fn reply_here(&mut self, session: &mut Session, user_text: &str, seed: u64, mut emit: impl FnMut(Event)) -> Result<TurnStats> {
        let t0 = Instant::now();
        let mut stats = TurnStats::default();
        let report = shield::scan(user_text);
        if !report.findings.is_empty() {
            stats.shield = report.summary();
            emit(Event::Shield(&stats.shield));
        }
        let idea = session.mode == "idea";
        let tone = persona::read_tone(&report.clean);
        let plan = if idea {
            ThinkPlan { think: false, budget: 0, reason: "idea mode: wild ideas, then a review".into() }
        } else {
            plan_thinking(&report.clean, &session.think)
        };
        stats.think_plan = plan.reason.clone();
        let note = if report.bypass_attempt() || !report.hidden.is_empty() { stats.shield.as_str() } else { "" };
        let system = persona::system_prompt(&session.mode, &tone, note, &session.memory, &self.name);
        let mut turns = session.turns.clone();
        turns.push(("user".into(), report.clean.clone()));
        let mut rng = Rng::new(seed);
        if !idea {
            if let Some((k, p)) = self.read_prompt(&report.clean) {
                stats.kind = Some(k);
                stats.kind_p = p;
                let how = match k {
                    Kind::Fact => "a fact: answering carefully, unsure sentences marked with a percentage",
                    Kind::Creative => "something creative: free to invent, no unsure marks",
                    Kind::Other => "other help: default settings",
                };
                emit(Event::Read(k, p[k as usize], how));
            }
        }
        self.engine.reset_counts()?;
        let answer = if idea {
            self.idea_turn(&system, &mut turns, &mut rng, &mut stats, &mut emit)?
        } else {
            let use_tools = self.tools;
            let (ids, snap_at, sys_at) = self.fit(&system, &mut turns, plan.think, 72, use_tools)?;
            stats.prompt_tokens = ids.len();
            // a thinking reply samples with the thinking settings throughout, as the base model recommends
            let kind = stats.kind.unwrap_or(Kind::Other);
            let (temperature, top_p) = match (plan.think, kind) {
                (true, _) => (self.gen.think_temperature, self.gen.think_top_p),
                (false, Kind::Fact) => (self.governor.fact_temperature, self.gen.top_p),
                (false, Kind::Creative) => (self.governor.creative_temperature, self.governor.creative_top_p),
                (false, Kind::Other) => (self.governor.answer_temperature.unwrap_or(self.gen.temperature), self.gen.top_p),
            };
            let pass = Pass {
                temperature,
                top_p,
                top_k: self.gen.top_k,
                think_budget: plan.think.then_some(plan.budget),
                max_new: self.gen.max_new_tokens + plan.budget,
                // the probe learned that made-up things are made up: it would mark every invented line
                mark_unsure: kind != Kind::Creative,
                cool_on_unsure: kind == Kind::Fact,
                measure: false,
                stop_at_paragraph: false,
                tools: use_tools,
            };
            let mut first = self.run_pass(&ids, snap_at, sys_at, &pass, &mut rng, &mut emit)?;
            stats.add(&first);
            if !self.record_taps.is_empty() {
                stats.recorded.push((first.text.clone(), std::mem::take(&mut first.sentences)));
            }
            let mut out = first;
            if self.second_look && kind == Kind::Fact && !out.marked.is_empty() && self.probe.is_some() {
                // the probe thinks this fact answer is likely wrong: look again, greedily, and say
                // so if it does not actually know
                emit(Event::Phase("second look: that answer was marked unsure, so Aegist answers it again"));
                let mut again = turns.clone();
                again.push(("assistant".into(), out.text.clone()));
                again.push(("user".into(), SECOND_LOOK.into()));
                let (ids2, _, _) = self.fit(&system, &mut again, false, 72, use_tools)?;
                let pass2 = Pass { temperature: 0.0, think_budget: None, max_new: self.gen.max_new_tokens, ..pass };
                let mut second = self.run_pass(&ids2, snap_at, sys_at, &pass2, &mut rng, &mut emit)?;
                stats.add(&second);
                if !self.record_taps.is_empty() {
                    stats.recorded.push((second.text.clone(), std::mem::take(&mut second.sentences)));
                }
                stats.second_look = true;
                // the governor tracks how often first answers come out unsure
                self.observe(plan.think, kind, &out, &mut emit);
                out = second;
            } else {
                self.observe(plan.think, kind, &out, &mut emit);
            }
            stats.final_marks = out.marked.iter().map(|m| m.1).collect();
            if let Some(s) = unsure_summary(&out) {
                emit(Event::UnsureSummary(&s));
            }
            out.text
        };
        turns.push(("assistant".into(), answer));
        session.turns = turns;
        stats.cores_used = self.engine.core_counts()?.iter().filter(|&&c| c > 0).count();
        stats.seconds = t0.elapsed().as_secs_f64();
        Ok(stats)
    }

    fn observe(&mut self, think: bool, kind: Kind, out: &PassOut, emit: &mut dyn FnMut(Event)) {
        if think || self.probe.is_none() {
            return;
        }
        let n = match kind {
            Kind::Fact => self.governor.observe_fact(out.scored, out.marked.len()),
            Kind::Other => self.governor.observe_answer(out.scored, out.marked.len(), self.gen.temperature),
            Kind::Creative => None,
        };
        if let Some(n) = n {
            emit(Event::Note(&n));
        }
    }

    /// What the prompt asks for, from the reading core; None without one (or if it fails).
    fn read_prompt(&self, text: &str) -> Option<(Kind, [f32; 3])> {
        self.reader.as_ref().and_then(|r| r.read(text).ok())
    }

    /// Idea mode: a few wild ideas sampled hot (speculation, shown as such), then a sober review
    /// of them in the same reply. Every pass starts from the same prompt, so only the short part
    /// after the assistant header is fed again. No unsure marks here: the probe learned that
    /// sentences about made-up things are made up, so with made-up ideas in the context it
    /// flagged nearly every review sentence, right or wrong (14 of 15 in one measured reply).
    fn idea_turn(&mut self, system: &str, turns: &mut Vec<(String, String)>, rng: &mut Rng, stats: &mut TurnStats,
                 emit: &mut dyn FnMut(Event)) -> Result<String> {
        const IDEA_TOKENS: usize = 200;
        let n = self.governor.ideas;
        let (ids, snap_at, sys_at) = self.fit(system, turns, false, n * IDEA_TOKENS + 1024, false)?;
        stats.prompt_tokens = ids.len();
        let wild_ids = [ids.as_slice(), &self.tok.encode_untrusted("Wild idea (speculation): ")].concat();
        let mut ideas = vec![];
        for k in 0..n {
            emit(Event::Phase(&format!("wild idea {} of {n} (speculation, unchecked)", k + 1)));
            let pass = Pass {
                temperature: self.governor.idea_temperature,
                top_p: self.governor.idea_top_p,
                top_k: 200,
                think_budget: None,
                max_new: IDEA_TOKENS,
                mark_unsure: false,
                cool_on_unsure: false,
                measure: true,
                stop_at_paragraph: true,
                tools: false,
            };
            let out = self.run_pass(&wild_ids, snap_at, sys_at, &pass, rng, emit)?;
            stats.add(&out);
            ideas.push(IdeaStat { text: out.text, mean_logprob: out.mean_logprob });
        }
        let listing: String = ideas.iter().enumerate().map(|(i, s)| format!("{}. {}\n", i + 1, s.text)).collect();
        let head = format!("Wild ideas (speculation, unchecked):\n{listing}\nReview:");
        let review_ids = [ids.as_slice(), &self.tok.encode_untrusted(&head)].concat();
        emit(Event::Phase("review (sober; no unsure marks in idea mode)"));
        let pass = Pass {
            temperature: self.governor.answer_temperature.unwrap_or(self.gen.temperature),
            top_p: self.gen.top_p,
            top_k: self.gen.top_k,
            think_budget: None,
            max_new: self.gen.max_new_tokens,
            mark_unsure: false,
            cool_on_unsure: false,
            measure: false,
            stop_at_paragraph: false,
            tools: false,
        };
        let out = self.run_pass(&review_ids, snap_at, sys_at, &pass, rng, emit)?;
        stats.add(&out);
        if let Some(note) = self.governor.observe_ideas(&ideas) {
            emit(Event::Note(&note));
        }
        Ok(format!("{head}\n{}", out.text))
    }
}

/// How one stretch of generation samples.
#[derive(Clone, Copy)]
struct Pass {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    /// Some(budget): the prompt opened a think block, and thinking may run this many tokens.
    think_budget: Option<usize>,
    max_new: usize,
    mark_unsure: bool,
    /// Sample greedily from the first unsure sentence on (fact prompts).
    cool_on_unsure: bool,
    /// Track the mean log-prob of the answer tokens (the governor reads it).
    measure: bool,
    /// Stop at the first blank line after some text.
    stop_at_paragraph: bool,
    /// Run the brain's tool calls and feed their results back.
    tools: bool,
}

#[derive(Default)]
struct PassOut {
    text: String,
    answer_tokens: usize,
    thinking_tokens: usize,
    mean_logprob: f64,
    reused: usize,
    prefill_seconds: f64,
    scored: usize,
    unjudged: usize,
    marked: Vec<(String, f32)>,
    highest: f32,
    cooled: bool,
    tools: Vec<String>,
    /// The sentences the probe looked at, when recording.
    sentences: Vec<Sentence>,
}

/// What Aegist says after a reply with sentences marked unsure. The percentages are the
/// confidence probe's estimates of P(wrong), not checked facts.
fn unsure_summary(out: &PassOut) -> Option<String> {
    let n = out.marked.len();
    if n == 0 {
        return None;
    }
    let top = out.marked.iter().map(|m| m.1).fold(0.0f32, f32::max);
    let (which, it) = if n == 1 { ("1 sentence".to_string(), "it") } else { (format!("{n} sentences"), "them") };
    let mut s = format!(
        "I'm not sure about {which} above (of {} checked): the estimated chance of a mistake is up to {:.0}%. Check {it} before relying on {it}.",
        out.scored,
        100.0 * top
    );
    if out.cooled {
        s.push_str(" After the first one I stopped sampling and wrote the rest as plainly as I could.");
    }
    Some(s)
}

impl TurnStats {
    fn add(&mut self, p: &PassOut) {
        if self.prefill_seconds == 0.0 {
            self.reused_tokens = p.reused;
        }
        self.prefill_seconds += p.prefill_seconds;
        self.thinking_tokens += p.thinking_tokens;
        self.answer_tokens += p.answer_tokens;
        self.sentences_scored += p.scored;
        self.sentences_unsure += p.marked.len();
        self.sentences_unjudged += p.unjudged;
        self.unsure.extend(p.marked.iter().cloned());
        self.highest_unsure = self.highest_unsure.max(p.highest);
        self.cooled |= p.cooled;
        self.tools_used.extend(p.tools.iter().cloned());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_questions_skip_thinking() {
        assert!(!plan_thinking("How many legs does a spider have?", "auto").think);
        assert!(!plan_thinking("hey whats up", "auto").think);
        assert!(plan_thinking("If I have 3 apples and buy 2 bags of 4, how many? Solve it.", "auto").think);
        assert!(plan_thinking("What is 48213 * 27591?", "auto").think);
        assert!(plan_thinking("anything", "on").think);
        let p = plan_thinking(&"prove this and solve that ".repeat(40), "auto");
        assert!(p.think && p.budget <= THINK_MAX);
    }

    #[test]
    fn hard_prompts_go_to_the_big_brain() {
        assert!(!pick_brain("What is the capital of Australia?", "auto").0);
        assert!(pick_brain("Prove that the square root of 2 is irrational.", "auto").0);
        assert!(pick_brain(&"a".repeat(900), "auto").0);
        assert!(pick_brain("hi", "big").0);
        assert!(!pick_brain("Prove it", "fast").0);
    }
}
