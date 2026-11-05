"""Aegist tokenizer: 151k-entry byte-level BPE (HF `tokenizers`, Rust) + chat format.

Text that did not come from Aegist's own runtime (user messages, files, web
pages) is encoded so that control tokens can never be forged from it: a literal
"<|im_start|>" typed by a user becomes ordinary text tokens, not a turn marker.
"""
from __future__ import annotations

import json
import re

from tokenizers import Tokenizer

IM_START, IM_END, ENDOFTEXT = "<|im_start|>", "<|im_end|>", "<|endoftext|>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


class AegistTokenizer:
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        # every added token (special or not: <think>, <tool_call>, ...) is control markup that
        # outside text must never produce
        self.specials = sorted({t.content for t in tok.get_added_tokens_decoder().values()}, key=len, reverse=True)
        self._special_re = re.compile("|".join(re.escape(s) for s in self.specials)) if self.specials else None
        self.im_start = tok.token_to_id(IM_START)
        self.im_end = tok.token_to_id(IM_END)
        self.eos_ids = {self.im_end, tok.token_to_id(ENDOFTEXT)}
        self.think_close = tok.token_to_id(THINK_CLOSE)
        self.vocab_size = tok.get_vocab_size()

    @classmethod
    def from_file(cls, path: str) -> "AegistTokenizer":
        return cls(Tokenizer.from_file(path))

    @classmethod
    def from_json_str(cls, s: str) -> "AegistTokenizer":
        return cls(Tokenizer.from_str(s))

    def to_json_str(self) -> str:
        return self.tok.to_str()

    # --- encoding -------------------------------------------------------
    def encode_trusted(self, text: str) -> list[int]:
        """Runtime-built text: control tokens in it are real control tokens."""
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_untrusted(self, text: str) -> list[int]:
        """Outside text: any control-token string is split so it stays plain text."""
        if self._special_re is None:
            return self.encode_trusted(text)
        ids, last = [], 0
        for m in self._special_re.finditer(text):
            ids += self.encode_trusted(text[last : m.start()])
            s = m.group(0)
            ids += self.encode_trusted(s[:1]) + self.encode_trusted(s[1:])
            last = m.end()
        return ids + self.encode_trusted(text[last:])

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special)

    # --- chat format ----------------------------------------------------
    def chat_ids(self, system: str, turns: list[tuple[str, str]], think: bool, open_think: bool = False) -> list[int]:
        """turns: [(role, text)], role in user/assistant. Ends with an open assistant turn.

        Past assistant turns are stored without their thinking, as the base model was
        trained; the open turn either starts free (think), already inside an open think block
        (think and open_think, as newer chat templates do), or with an empty think block.
        """
        ids = self.encode_trusted(f"{IM_START}system\n") + self.encode_untrusted(system) + \
            self.encode_trusted(f"{IM_END}\n")
        for role, text in turns:
            if role == "assistant":
                text = strip_thinking(text)
            ids += self.encode_trusted(f"{IM_START}{role}\n") + self.encode_untrusted(text) + \
                self.encode_trusted(f"{IM_END}\n")
        ids += self.encode_trusted(f"{IM_START}assistant\n")
        if not think:
            ids += self.encode_trusted(f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n")
        elif open_think:
            ids += self.encode_trusted(f"{THINK_OPEN}\n")
        return ids


def strip_thinking(text: str) -> str:
    if THINK_CLOSE in text:
        text = text.split(THINK_CLOSE, 1)[1]
    return text.lstrip("\n")


def load_generation_defaults(path: str) -> dict:
    try:
        return json.load(open(path))
    except OSError:
        return {}
