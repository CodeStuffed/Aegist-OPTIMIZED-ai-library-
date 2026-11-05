//! Aegist's tokenizer (byte-level BPE, from the brain's tokenizer.json blob) and chat format.
//! Mirrors aegist/tokenizer.py: untrusted text can never forge control tokens.
use tokenizers::Tokenizer;

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";

pub struct Tok {
    tok: Tokenizer,
    specials: Vec<String>, // longest first
    pub eos: Vec<u32>,
    pub think_close: u32,
    /// The token that opens a tool call, if the vocabulary has one.
    pub tool_open: Option<u32>,
    /// Per token, the UTF-8 state after it from each state before it (the continuation bytes a
    /// character still owes: 0-3), or -1 where it would break a character: what keeps sampling
    /// from writing a broken character (U+FFFD). Added tokens leave no character open. Empty if
    /// the vocabulary is not byte-level.
    utf8: Vec<[i8; 4]>,
    /// Kimi's chat markup (<|im_user|>user<|im_middle|>...<|im_end|>) instead of ChatML.
    pub kimi: bool,
}

/// The UTF-8 state after `bytes`, starting owing `need` continuation bytes; -1: broken.
fn utf8_after(mut need: i8, bytes: &[u8]) -> i8 {
    for &b in bytes {
        if need > 0 {
            if b & 0xC0 != 0x80 {
                return -1;
            }
            need -= 1;
        } else {
            need = match b {
                0x00..=0x7F => 0,
                0xC2..=0xDF => 1,
                0xE0..=0xEF => 2,
                0xF0..=0xF4 => 3,
                _ => return -1,
            };
        }
    }
    need
}

/// Tok::utf8 for this vocabulary.
fn utf8_table(tok: &Tokenizer) -> Vec<[i8; 4]> {
    // the byte-level alphabet: printable bytes stand for themselves, the others for U+0100 on
    let mut byte_of = std::collections::HashMap::new();
    let mut n = 0u32;
    for b in 0..=255u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable {
            b
        } else {
            n += 1;
            255 + n
        };
        byte_of.insert(char::from_u32(c).unwrap(), b as u8);
    }
    let added = tok.get_added_tokens_decoder();
    let size = tok.get_vocab_size(true);
    let mut table = vec![[0i8; 4]; size];
    let mut unmapped = 0;
    for id in 0..size as u32 {
        if added.contains_key(&id) {
            continue;
        }
        let Some(s) = tok.id_to_token(id) else { continue };
        let Some(bytes) = s.chars().map(|c| byte_of.get(&c).copied()).collect::<Option<Vec<u8>>>() else {
            unmapped += 1;
            continue;
        };
        for k in 0..4 {
            table[id as usize][k] = utf8_after(k as i8, &bytes);
        }
    }
    if unmapped * 100 > size {
        return vec![];
    }
    table
}

impl Tok {
    pub fn from_json(json: &[u8]) -> Result<Tok, String> {
        let tok = Tokenizer::from_bytes(json).map_err(|e| format!("bad tokenizer: {e}"))?;
        // every added token (special or not: <think>, <tool_call>, ...) is control markup that
        // outside text must never produce
        let mut specials: Vec<String> = tok.get_added_tokens_decoder().values().map(|t| t.content.clone()).collect();
        specials.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        specials.dedup();
        let id = |s: &str| tok.token_to_id(s).ok_or(format!("tokenizer has no {s}"));
        let im_end = id(IM_END)?;
        let kimi = tok.token_to_id(IM_START).is_none() && tok.token_to_id("<|im_user|>").is_some();
        let eos = vec![im_end, id(if kimi { "[EOS]" } else { "<|endoftext|>" })?];
        let think_close = id(THINK_CLOSE)?;
        let tool_open = tok.token_to_id("<tool_call>");
        let utf8 = utf8_table(&tok);
        Ok(Tok { tok, specials, eos, think_close, tool_open, utf8, kimi })
    }

    /// Runtime-built text: control-token strings in it are real control tokens.
    pub fn encode_trusted(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }
        self.tok.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
    }

    /// Outside text: every control-token string is split so it stays plain text.
    pub fn encode_untrusted(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut last = 0;
        let mut i = 0;
        let b = text.as_bytes();
        while i < b.len() {
            if let Some(s) = self.specials.iter().find(|s| text[i..].starts_with(s.as_str())) {
                ids.extend(self.encode_trusted(&text[last..i]));
                let first = s.chars().next().unwrap().len_utf8();
                ids.extend(self.encode_trusted(&s[..first]));
                ids.extend(self.encode_trusted(&s[first..]));
                i += s.len();
                last = i;
            } else {
                i += text[i..].chars().next().unwrap().len_utf8();
            }
        }
        ids.extend(self.encode_trusted(&text[last..]));
        ids
    }

    /// Rule out the tokens that would break a character while `need` continuation bytes are owed.
    pub fn guard_utf8(&self, need: u8, logits: &mut [f32]) {
        for (l, t) in logits.iter_mut().zip(&self.utf8) {
            if t[need as usize] < 0 {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// The continuation bytes owed after token `id`.
    pub fn utf8_after(&self, need: u8, id: u32) -> u8 {
        self.utf8.get(id as usize).map_or(0, |t| t[need as usize].max(0) as u8)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids, false).unwrap_or_default()
    }

    /// System + turns, ending with an open assistant turn: free (think), already inside an open
    /// think block (think and open_think, as newer chat templates do), or with an empty think block.
    pub fn chat_ids(&self, system: &str, turns: &[(String, String)], think: bool, open_think: bool) -> Vec<u32> {
        self.chat_ids_tools("", system, turns, think, open_think)
    }

    /// The system message alone: the tools block (runtime-built, so its markup is real), then
    /// Aegist's system prompt. Every prompt with the same tools and system starts with these.
    pub fn system_ids(&self, tools: &str, system: &str) -> Vec<u32> {
        if self.kimi {
            let mut ids = self.encode_trusted(&format!("<|im_system|>system<|im_middle|>{tools}"));
            ids.extend(self.encode_untrusted(system));
            ids.extend(self.encode_trusted(IM_END));
            return ids;
        }
        let mut ids = self.encode_trusted(&format!("{IM_START}system\n{tools}"));
        ids.extend(self.encode_untrusted(system));
        ids.extend(self.encode_trusted(&format!("{IM_END}\n")));
        ids
    }

    /// One finished turn (an assistant one with its thinking dropped).
    fn turn_ids(&self, role: &str, text: &str) -> Vec<u32> {
        let text = if role == "assistant" { strip_thinking(text) } else { text.to_string() };
        let mut ids = if self.kimi {
            let tag = match role {
                "assistant" => "<|im_assistant|>",
                "user" => "<|im_user|>",
                _ => "<|im_system|>",
            };
            let think = if role == "assistant" { "<think></think>" } else { "" };
            self.encode_trusted(&format!("{tag}{role}<|im_middle|>{think}"))
        } else {
            self.encode_trusted(&format!("{IM_START}{role}\n"))
        };
        ids.extend(self.encode_untrusted(&text));
        if self.kimi {
            ids.extend(self.encode_trusted(IM_END));
        } else {
            ids.extend(self.encode_trusted(&format!("{IM_END}\n")));
        }
        ids
    }

    /// As chat_ids, with a tools block at the start of the system message.
    pub fn chat_ids_tools(&self, tools: &str, system: &str, turns: &[(String, String)], think: bool, open_think: bool) -> Vec<u32> {
        let mut ids = self.system_ids(tools, system);
        for (role, text) in turns {
            ids.extend(self.turn_ids(role, text));
        }
        ids.extend(self.assistant_open(think, open_think));
        ids
    }

    /// An open assistant turn, as at the end of chat_ids.
    pub fn assistant_open(&self, think: bool, open_think: bool) -> Vec<u32> {
        if self.kimi {
            // Kimi's template always opens the think block when thinking
            return self.encode_trusted(if think { "<|im_assistant|>assistant<|im_middle|><think>" } else { "<|im_assistant|>assistant<|im_middle|><think></think>" });
        }
        let mut ids = self.encode_trusted(&format!("{IM_START}assistant\n"));
        if !think {
            ids.extend(self.encode_trusted(&format!("{THINK_OPEN}\n\n{THINK_CLOSE}\n\n")));
        } else if open_think {
            ids.extend(self.encode_trusted(&format!("{THINK_OPEN}\n")));
        }
        ids
    }

    /// What follows a finished tool call: the tool responses in a user turn, as Qwen's chat
    /// template writes them, then a fresh assistant turn (no thinking). The results quote the
    /// brain's own arguments, so they are encoded as outside text.
    pub fn tool_responses(&self, results: &[String]) -> Vec<u32> {
        if self.kimi {
            let mut ids = self.encode_trusted(IM_END);
            ids.extend(self.turn_ids("user", &results.iter().map(|r| format!("## Tool result
{r}")).collect::<Vec<_>>().join("

")));
            ids.extend(self.assistant_open(false, false));
            return ids;
        }
        let mut ids = self.encode_trusted(&format!("{IM_END}\n{IM_START}user"));
        for r in results {
            ids.extend(self.encode_trusted("\n<tool_response>\n"));
            ids.extend(self.encode_untrusted(r));
            ids.extend(self.encode_trusted("\n</tool_response>"));
        }
        ids.extend(self.encode_trusted(&format!("{IM_END}\n")));
        ids.extend(self.assistant_open(false, false));
        ids
    }
}

pub fn strip_thinking(text: &str) -> String {
    let t = match text.find(THINK_CLOSE) {
        Some(i) => &text[i + THINK_CLOSE.len()..],
        None => text,
    };
    t.trim_start_matches('\n').to_string()
}

#[cfg(test)]
mod tests {
    use super::utf8_after;

    #[test]
    fn utf8_states() {
        let dash = "\u{2014}".as_bytes(); // E2 80 94
        assert_eq!(utf8_after(0, b"abc"), 0);
        assert_eq!(utf8_after(0, dash), 0);
        assert_eq!(utf8_after(0, &dash[..2]), 1);
        assert_eq!(utf8_after(1, &dash[2..]), 0);
        assert_eq!(utf8_after(0, &dash[1..]), -1); // starts mid-character
        assert_eq!(utf8_after(1, b" x"), -1); // a character left unfinished
        assert_eq!(utf8_after(0, &[0xF0, 0x9F]), 2);
        assert_eq!(utf8_after(0, &[0xC0]), -1);
    }
}
