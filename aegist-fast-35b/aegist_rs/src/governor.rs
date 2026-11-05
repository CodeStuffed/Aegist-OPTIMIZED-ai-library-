//! The hallucination governor, run-time half: it watches how much Aegist makes things up and
//! turns the sampling knobs so it does so at the right time and by the right amount.
//!
//! - Idea mode wants invention. If the wild ideas come out too alike (not enough), the wild
//!   temperature goes up; if they fall apart into unlikely or repetitive text (too much), it
//!   goes down.
//! - Normal and blunt modes want facts. If the confidence probe keeps marking answer sentences
//!   unsure, the answer temperature goes down (less sampling noise means fewer invented
//!   details); when that settles, it drifts back to the brain's default.
//! - The reading core (laya.rs) says what each prompt asks for. Fact prompts get their own,
//!   cooler track: they start at 0.3 and, when the probe keeps marking their sentences unsure,
//!   go down to greedy (no sampling at all), then drift back. Creative prompts are the one place
//!   invention is wanted outside idea mode: they sample hot (1.0) and get no unsure marks.
//!
//! Changes are small steps on smoothed measurements, within fixed bounds, saved to
//! governor.json in the Aegist folder. Changing weights is learning time's job (aegist/learn.py
//! and aegist/governor.py); this file only moves knobs.
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::PathBuf;

const SMOOTH: f32 = 0.3; // weight of the newest measurement
const STEP: f32 = 0.05;
const IDEA_TEMP: (f32, f32) = (0.8, 1.6);
/// Ideas sharing more than this share of content words with each other count as "too alike".
const MIN_DISTINCT: f32 = 0.55;
/// Over this share of broken ideas, the wild phase is too wild.
const MAX_BROKEN: f32 = 0.25;
/// An idea whose sampled tokens average below this log-prob (at temperature 1) reads as noise.
const BROKEN_LOGPROB: f64 = -4.0;
/// Over this share of answer sentences marked unsure, factual answers cool down.
const MAX_UNSURE: f32 = 0.3;
const MIN_ANSWER_TEMP: f32 = 0.3;
/// The answer temperature moves only after this many answer sentences have been scored: one
/// turn with a hard question in it is not a trend.
const MIN_EVIDENCE: u64 = 12;
/// Fact answers: temperature bounds (0 = greedy) and the start.
const FACT_TEMP: (f32, f32) = (0.0, 0.3);
const CREATIVE_TEMP: (f32, f32) = (0.8, 1.4);

pub struct Governor {
    pub idea_temperature: f32,
    pub idea_top_p: f32,
    pub ideas: usize,
    /// Answer temperature in normal/blunt mode; None = the brain's default.
    pub answer_temperature: Option<f32>,
    /// Answer temperature for prompts the reading core reads as asking for a fact.
    pub fact_temperature: f32,
    /// Sampling for prompts read as asking for something creative.
    pub creative_temperature: f32,
    pub creative_top_p: f32,
    fact_unsure: Option<f32>,
    fact_scored: u64,
    distinct: Option<f32>,
    broken: Option<f32>,
    unsure: Option<f32>,
    /// Answer sentences scored so far (all turns, saved).
    scored: u64,
    pub log: Vec<String>,
    path: Option<PathBuf>,
}

/// What one wild idea looked like.
pub struct IdeaStat {
    pub text: String,
    pub mean_logprob: f64,
}

fn words(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 3).map(str::to_lowercase).collect()
}

fn ema(old: Option<f32>, new: f32) -> Option<f32> {
    Some(old.map_or(new, |o| (1.0 - SMOOTH) * o + SMOOTH * new))
}

/// True when an idea reads as noise: very unlikely tokens, or the same words over and over.
pub fn broken(idea: &IdeaStat) -> bool {
    let all: Vec<&str> = idea.text.split_whitespace().collect();
    let unique: HashSet<&str> = all.iter().copied().collect();
    idea.mean_logprob < BROKEN_LOGPROB || (all.len() >= 20 && (unique.len() as f32) < 0.4 * all.len() as f32)
}

/// Mean over pairs of 1 - Jaccard(content words): 0 = the same idea, 1 = nothing shared.
pub fn distinctness(ideas: &[IdeaStat]) -> Option<f32> {
    let w: Vec<HashSet<String>> = ideas.iter().map(|i| words(&i.text)).collect();
    let mut pairs = vec![];
    for a in 0..w.len() {
        for b in a + 1..w.len() {
            let union = w[a].union(&w[b]).count();
            if union > 0 {
                pairs.push(1.0 - w[a].intersection(&w[b]).count() as f32 / union as f32);
            }
        }
    }
    (!pairs.is_empty()).then(|| pairs.iter().sum::<f32>() / pairs.len() as f32)
}

impl Default for Governor {
    fn default() -> Self {
        Governor { idea_temperature: 1.2, idea_top_p: 0.98, ideas: 3, answer_temperature: None, fact_temperature: FACT_TEMP.1,
                   creative_temperature: 1.0, creative_top_p: 0.95, fact_unsure: None, fact_scored: 0, distinct: None,
                   broken: None, unsure: None, scored: 0, log: vec![], path: None }
    }
}

impl Governor {
    /// governor.json in `home`, or the defaults.
    pub fn load(home: &std::path::Path) -> Governor {
        let path = home.join("governor.json");
        let mut g = Governor { path: Some(path.clone()), ..Default::default() };
        if let Some(v) = std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()) {
            let f = |k: &str| v[k].as_f64().map(|x| x as f32);
            g.idea_temperature = f("idea_temperature").unwrap_or(g.idea_temperature).clamp(IDEA_TEMP.0, IDEA_TEMP.1);
            g.idea_top_p = f("idea_top_p").unwrap_or(g.idea_top_p).clamp(0.5, 1.0);
            g.ideas = v["ideas"].as_u64().map_or(g.ideas, |n| (n as usize).clamp(1, 6));
            g.answer_temperature = f("answer_temperature");
            g.distinct = f("distinct");
            g.broken = f("broken");
            g.unsure = f("unsure");
            g.scored = v["scored"].as_u64().unwrap_or(0);
            g.fact_temperature = f("fact_temperature").unwrap_or(g.fact_temperature).clamp(FACT_TEMP.0, FACT_TEMP.1);
            g.creative_temperature = f("creative_temperature").unwrap_or(g.creative_temperature).clamp(CREATIVE_TEMP.0, CREATIVE_TEMP.1);
            g.creative_top_p = f("creative_top_p").unwrap_or(g.creative_top_p).clamp(0.5, 1.0);
            g.fact_unsure = f("fact_unsure");
            g.fact_scored = v["fact_scored"].as_u64().unwrap_or(0);
        }
        g
    }

    /// The same settings, never saved (the AQ test uses them without changing them).
    pub fn detached(mut self) -> Governor {
        self.path = None;
        self
    }

    fn save(&self) {
        if let Some(p) = &self.path {
            let v = json!({"idea_temperature": self.idea_temperature, "idea_top_p": self.idea_top_p, "ideas": self.ideas,
                           "answer_temperature": self.answer_temperature, "distinct": self.distinct,
                           "broken": self.broken, "unsure": self.unsure, "scored": self.scored,
                           "fact_temperature": self.fact_temperature, "creative_temperature": self.creative_temperature,
                           "creative_top_p": self.creative_top_p, "fact_unsure": self.fact_unsure, "fact_scored": self.fact_scored});
            let tmp = p.with_extension("json.tmp");
            if std::fs::write(&tmp, serde_json::to_string_pretty(&v).unwrap_or_default()).is_ok() {
                let _ = std::fs::rename(&tmp, p);
            }
        }
    }

    fn note(&mut self, s: String) -> Option<String> {
        self.log.push(s.clone());
        self.save();
        Some(s)
    }

    /// After an idea turn. Returns a note when a knob moved.
    pub fn observe_ideas(&mut self, ideas: &[IdeaStat]) -> Option<String> {
        if ideas.is_empty() {
            return None;
        }
        let broken_share = ideas.iter().filter(|i| broken(i)).count() as f32 / ideas.len() as f32;
        self.broken = ema(self.broken, broken_share);
        if let Some(d) = distinctness(ideas) {
            self.distinct = ema(self.distinct, d);
        }
        let t0 = self.idea_temperature;
        let why = if self.broken.unwrap_or(0.0) > MAX_BROKEN {
            self.idea_temperature = (t0 - STEP).max(IDEA_TEMP.0);
            "wild ideas were falling apart"
        } else if self.distinct.is_some_and(|d| d < MIN_DISTINCT) {
            self.idea_temperature = (t0 + STEP).min(IDEA_TEMP.1);
            "wild ideas were too alike"
        } else {
            ""
        };
        if self.idea_temperature != t0 {
            return self.note(format!("governor: idea temperature {t0:.2} -> {:.2} ({why})", self.idea_temperature));
        }
        self.save();
        None
    }

    /// After a factual answer: `scored` sentences checked, `flagged` marked unsure.
    /// `default` is the brain's answer temperature.
    pub fn observe_answer(&mut self, scored: usize, flagged: usize, default: f32) -> Option<String> {
        if scored == 0 {
            return None;
        }
        self.unsure = ema(self.unsure, flagged as f32 / scored as f32);
        self.scored += scored as u64;
        let t0 = self.answer_temperature.unwrap_or(default);
        let u = self.unsure.unwrap_or(0.0);
        let t = if self.scored < MIN_EVIDENCE {
            t0
        } else if u > MAX_UNSURE {
            (t0 - STEP).max(MIN_ANSWER_TEMP.min(default))
        } else if u < MAX_UNSURE / 2.0 {
            (t0 + STEP).min(default)
        } else {
            t0
        };
        self.answer_temperature = if (t - default).abs() < 1e-6 { None } else { Some(t) };
        if (t - t0).abs() > 1e-6 {
            let why = if t < t0 { "many answer sentences marked unsure" } else { "answers settled; back toward the default" };
            return self.note(format!("governor: answer temperature {t0:.2} -> {t:.2} ({why})"));
        }
        self.save();
        None
    }

    /// After an answer to a prompt read as asking for a fact: the fact track only.
    pub fn observe_fact(&mut self, scored: usize, flagged: usize) -> Option<String> {
        if scored == 0 {
            return None;
        }
        self.fact_unsure = ema(self.fact_unsure, flagged as f32 / scored as f32);
        self.fact_scored += scored as u64;
        let t0 = self.fact_temperature;
        let u = self.fact_unsure.unwrap_or(0.0);
        let t = if self.fact_scored < MIN_EVIDENCE {
            t0
        } else if u > MAX_UNSURE {
            (t0 - STEP).max(FACT_TEMP.0)
        } else if u < MAX_UNSURE / 2.0 {
            (t0 + STEP).min(FACT_TEMP.1)
        } else {
            t0
        };
        self.fact_temperature = t;
        if (t - t0).abs() > 1e-6 {
            let why = if t < t0 { "many fact sentences marked unsure" } else { "fact answers settled; back toward 0.30" };
            return self.note(format!("governor: fact temperature {t0:.2} -> {t:.2} ({why})"));
        }
        self.save();
        None
    }

    pub fn describe(&self, default_answer_temp: f32) -> String {
        let pct = |v: Option<f32>| v.map_or("-".to_string(), |x| format!("{:.0}%", x * 100.0));
        format!(
            "idea mode: {} wild ideas at temperature {:.2} (distinctness {}, broken {}); fact prompts: temperature {:.2} (unsure {}), cooled to greedy for the rest of an answer once a sentence is marked unsure; creative prompts: temperature {:.2}, top-p {:.2}, no unsure marks; other answers: temperature {:.2} (unsure {})",
            self.ideas, self.idea_temperature, pct(self.distinct), pct(self.broken), self.fact_temperature, pct(self.fact_unsure),
            self.creative_temperature, self.creative_top_p, self.answer_temperature.unwrap_or(default_answer_temp), pct(self.unsure)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idea(text: &str, lp: f64) -> IdeaStat {
        IdeaStat { text: text.into(), mean_logprob: lp }
    }

    #[test]
    fn alike_ideas_heat_up_and_broken_ideas_cool_down() {
        let mut g = Governor::default();
        let same = [idea("solar sails powered by moonlight reflectors", -1.5), idea("solar sails powered by moonlight mirrors", -1.5)];
        assert!(g.observe_ideas(&same).is_some());
        assert!(g.idea_temperature > 1.2);
        let mut g = Governor::default();
        let noise = [idea("zx qq vv", -6.0), idea("the the the the the the the the the the the the the the the the the the the the", -0.5),
                     idea("a bakery that prices bread by the weather", -1.8)];
        assert!(broken(&noise[0]) && broken(&noise[1]) && !broken(&noise[2]));
        assert!(g.observe_ideas(&noise).is_some());
        assert!(g.idea_temperature < 1.2);
        let mut g = Governor::default();
        let good = [idea("a bakery that prices bread by the weather", -1.8), idea("traffic lights that learn from birdsong", -2.0)];
        assert!(g.observe_ideas(&good).is_none());
        assert_eq!(g.idea_temperature, 1.2);
    }

    #[test]
    fn unsure_answers_cool_down_then_recover() {
        let mut g = Governor::default();
        assert!(g.observe_answer(2, 1, 0.7).is_none(), "one short turn is not enough to act on");
        assert_eq!(g.answer_temperature, None);
        for _ in 0..3 {
            g.observe_answer(4, 3, 0.7);
        }
        let low = g.answer_temperature.unwrap();
        assert!(low < 0.7 && low >= 0.3);
        for _ in 0..30 {
            g.observe_answer(4, 0, 0.7);
        }
        assert_eq!(g.answer_temperature, None, "back at the default");
    }

    #[test]
    fn unsure_facts_cool_to_greedy_then_recover() {
        let mut g = Governor::default();
        assert_eq!(g.fact_temperature, 0.3);
        for _ in 0..20 {
            g.observe_fact(4, 3);
        }
        assert_eq!(g.fact_temperature, 0.0, "greedy at the floor");
        assert_eq!(g.answer_temperature, None, "the other track did not move");
        for _ in 0..40 {
            g.observe_fact(4, 0);
        }
        assert!((g.fact_temperature - 0.3).abs() < 1e-6);
    }
}
