//! Hidden-text and bypass shield; a port of aegist/shield.py (see there for what is covered).
use base64::Engine as _;
use regex::{Regex, RegexBuilder};
use std::collections::BTreeMap;
use std::sync::OnceLock;
use unicode_general_category::{get_general_category, GeneralCategory as GC};
use unicode_normalization::UnicodeNormalization;
use unicode_script::{Script, UnicodeScript};

const ZERO_WIDTH: &[u32] = &[0x200B, 0x200C, 0x200D, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xFEFF, 0x00AD, 0x180E,
                             0x115F, 0x1160, 0x3164, 0xFFA0, 0x2800, 0x034F];

fn is_bidi(cp: u32) -> bool {
    (0x202A..0x202F).contains(&cp) || (0x2066..0x206A).contains(&cp) || matches!(cp, 0x200E | 0x200F | 0x061C)
}
fn is_vs(cp: u32) -> bool {
    (0xFE00..0xFE10).contains(&cp) || (0xE0100..0xE01F0).contains(&cp)
}

const OVERRIDE: &[&str] = &[
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any |the |your )?(previous|prior|above|system) (instructions|rules|prompt)",
    r"forget (all |your )?(previous |prior )?(instructions|rules)",
    r"you are (now )?(dan|in developer mode|jailbroken|unfiltered|unrestricted)",
    r"(enable|activate|enter) (developer|god|jailbreak|dan) mode",
    r"(new|updated|real) system prompt",
    r"pretend (that )?you (have|had) no (rules|restrictions|filters|guidelines)",
    r"(bypass|override|disable) (your )?(safety|filters?|restrictions|guardrails|rules)",
];
const CONTROL: &[&str] = &[r"<\|im_(start|end)\|>", r"<\|endoftext\|>", r"\[/?(SYSTEM|INST)\]", r"<</?SYS>>",
                           r"^\s*###\s*(system|instruction)s?\s*:", r"^\s*system\s*:\s"];

fn compiled(pats: &[&str], multiline: bool) -> Vec<Regex> {
    pats.iter().map(|p| RegexBuilder::new(p).case_insensitive(true).multi_line(multiline).build().unwrap()).collect()
}
fn override_re() -> &'static Vec<Regex> {
    static R: OnceLock<Vec<Regex>> = OnceLock::new();
    R.get_or_init(|| compiled(OVERRIDE, false))
}
fn control_re() -> &'static Vec<Regex> {
    static R: OnceLock<Vec<Regex>> = OnceLock::new();
    R.get_or_init(|| compiled(CONTROL, true))
}
fn matches(text: &str, res: &[Regex]) -> bool {
    let norm: String = text.nfkc().collect();
    res.iter().any(|r| r.is_match(&norm))
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // detail mirrors shield.py; the summary shows kinds only
pub struct Finding {
    pub kind: &'static str,
    pub detail: String,
}

#[derive(Debug, Default)]
pub struct Report {
    pub clean: String,
    pub findings: Vec<Finding>,
    pub hidden: Vec<String>,
}

impl Report {
    pub fn bypass_attempt(&self) -> bool {
        self.findings.iter().any(|f| matches!(f.kind, "override_phrase" | "forged_control" | "hidden_message" | "encoded_instruction"))
    }

    pub fn summary(&self) -> String {
        if self.findings.is_empty() {
            return String::new();
        }
        let mut order: Vec<&str> = Vec::new();
        let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
        for f in &self.findings {
            if !kinds.contains_key(f.kind) {
                order.push(f.kind);
            }
            *kinds.entry(f.kind).or_default() += 1;
        }
        let parts: Vec<String> = order.iter().map(|k| if kinds[k] > 1 { format!("{k} x{}", kinds[k]) } else { k.to_string() }).collect();
        let mut out = format!("shield: {}", parts.join(", "));
        for m in &self.hidden {
            out += &format!("\n  hidden text found: {m:?}");
        }
        out
    }
}

fn is_letter(c: char) -> bool {
    matches!(get_general_category(c), GC::UppercaseLetter | GC::LowercaseLetter | GC::TitlecaseLetter | GC::ModifierLetter | GC::OtherLetter)
}

fn printable_ratio(b: &[u8]) -> f64 {
    if b.is_empty() {
        return 0.0;
    }
    b.iter().filter(|&&c| (32..127).contains(&c) || matches!(c, 9 | 10 | 13)).count() as f64 / b.len() as f64
}

pub fn scan(text: &str) -> Report {
    let mut r = Report::default();
    let mut out = String::with_capacity(text.len());
    let mut tags = String::new();
    let mut vs: Vec<u32> = Vec::new();

    let flush_tags = |tags: &mut String, r: &mut Report| {
        if !tags.is_empty() {
            r.findings.push(Finding { kind: "hidden_message", detail: format!("{} chars in invisible tag characters", tags.chars().count()) });
            r.hidden.push(std::mem::take(tags));
        }
    };
    let flush_vs = |vs: &mut Vec<u32>, r: &mut Report, out: &mut String| {
        if vs.len() >= 4 {
            let raw: Vec<u8> = vs.iter().map(|&v| if v < 0xE0100 { (v - 0xFE00) as u8 } else { (v - 0xE0100 + 16) as u8 }).collect();
            r.hidden.push(String::from_utf8_lossy(&raw).into_owned());
            r.findings.push(Finding { kind: "hidden_message", detail: format!("{} bytes in variation selectors", vs.len()) });
        } else {
            for &v in vs.iter().filter(|&&v| v < 0xE0100) {
                out.push(char::from_u32(v).unwrap());
            }
        }
        vs.clear();
    };

    for ch in text.chars() {
        let cp = ch as u32;
        if (0xE0000..0xE0080).contains(&cp) {
            if (0xE0020..=0xE007E).contains(&cp) {
                tags.push(char::from_u32(cp - 0xE0000).unwrap());
            }
            continue;
        }
        flush_tags(&mut tags, &mut r);
        if is_vs(cp) {
            vs.push(cp);
            continue;
        }
        flush_vs(&mut vs, &mut r, &mut out);
        if ZERO_WIDTH.contains(&cp) {
            r.findings.push(Finding { kind: "invisible_char", detail: format!("U+{cp:04X}") });
            continue;
        }
        if is_bidi(cp) {
            r.findings.push(Finding { kind: "bidi_control", detail: format!("U+{cp:04X} (text may display in a different order)") });
            continue;
        }
        match get_general_category(ch) {
            GC::Control if !matches!(ch, '\n' | '\t' | '\r') => {
                r.findings.push(Finding { kind: "control_char", detail: format!("U+{cp:04X}") });
                continue;
            }
            GC::PrivateUse | GC::Unassigned => {
                r.findings.push(Finding { kind: "private_or_unassigned", detail: format!("U+{cp:04X}") });
                continue;
            }
            _ => {}
        }
        out.push(ch);
    }
    flush_tags(&mut tags, &mut r);
    flush_vs(&mut vs, &mut r, &mut out);

    static WORD: OnceLock<Regex> = OnceLock::new();
    static B64: OnceLock<Regex> = OnceLock::new();
    static HEX: OnceLock<Regex> = OnceLock::new();
    for w in WORD.get_or_init(|| Regex::new(r"\w{3,}").unwrap()).find_iter(&out) {
        let mut scripts: Vec<&str> = w
            .as_str()
            .chars()
            .filter(|&c| is_letter(c))
            .filter_map(|c| match c.script() {
                Script::Latin => Some("LATIN"),
                Script::Cyrillic => Some("CYRILLIC"),
                Script::Greek => Some("GREEK"),
                _ => None,
            })
            .collect();
        scripts.sort();
        scripts.dedup();
        if scripts.len() > 1 {
            r.findings.push(Finding { kind: "mixed_script", detail: format!("{:?} mixes {} letters", w.as_str(), scripts.join("/")) });
        }
    }
    for m in B64.get_or_init(|| Regex::new(r"[A-Za-z0-9+/]{24,}={0,2}").unwrap()).find_iter(&out) {
        let mut blob = m.as_str().to_string();
        while blob.len() % 4 != 0 {
            blob.push('=');
        }
        let Ok(raw) = base64::engine::general_purpose::STANDARD.decode(blob.as_bytes()) else { continue };
        if printable_ratio(&raw) > 0.95 {
            let dec = String::from_utf8_lossy(&raw).into_owned();
            r.findings.push(Finding { kind: "encoded_text", detail: format!("base64 decodes to {:?}", dec.chars().take(60).collect::<String>()) });
            if matches(&dec, override_re()) {
                r.findings.push(Finding { kind: "encoded_instruction", detail: "base64 hides an instruction-override".into() });
                r.hidden.push(dec);
            }
        }
    }
    for m in HEX.get_or_init(|| Regex::new(r"\b(?:[0-9a-fA-F]{2}){12,}\b").unwrap()).find_iter(&out) {
        let s = m.as_str();
        let raw: Vec<u8> = (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect();
        if printable_ratio(&raw) > 0.95 {
            let dec = String::from_utf8_lossy(&raw).into_owned();
            r.findings.push(Finding { kind: "encoded_text", detail: format!("hex decodes to {:?}", dec.chars().take(60).collect::<String>()) });
            if matches(&dec, override_re()) {
                r.findings.push(Finding { kind: "encoded_instruction", detail: "hex hides an instruction-override".into() });
                r.hidden.push(dec);
            }
        }
    }
    let hidden = r.hidden.clone();
    for h in &hidden {
        if matches(h, override_re()) {
            r.findings.push(Finding { kind: "override_phrase", detail: "inside hidden text".into() });
        }
    }
    if matches(&out, override_re()) {
        r.findings.push(Finding { kind: "override_phrase", detail: "instruction-override wording".into() });
    }
    if matches(&out, control_re()) {
        r.findings.push(Finding { kind: "forged_control", detail: "chat-control markup in input (treated as plain text)".into() });
    }
    r.clean = out;
    r
}
