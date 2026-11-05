"""Turn Kimi-K2.6's tiktoken vocabulary into a Hugging Face tokenizer.json (byte-level BPE), the format the
Rust runtime reads.

tiktoken stores only token ranks; BPE merges are recovered the standard way: a token of rank r is the merge
of the two pieces that byte-pair encoding with ranks < r leaves of its bytes. The pre-tokenizer is Moonshot's
split regex; specials are ids 163584.. from tokenizer_config.json ("<|reserved_token_N|>" for unnamed ones).

    python -m aegist.kimi_tokenizer [--check]
"""
from __future__ import annotations

import argparse
import base64
import json

from .fetch_k2 import HOME

PAT = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])
RESERVED = 256


def byte_alphabet() -> dict[int, str]:
    """GPT-2 bytes_to_unicode."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def load_ranks(path: str) -> dict[bytes, int]:
    out = {}
    for line in open(path, "rb"):
        if line.strip():
            tok, rank = line.split()
            out[base64.b64decode(tok)] = int(rank)
    return out


def bpe(ranks: dict[bytes, int], token: bytes, max_rank: int) -> list[bytes]:
    parts = [bytes([b]) for b in token]
    while True:
        best, at = None, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < max_rank and (best is None or r < best):
                best, at = r, i
        if best is None:
            return parts
        parts[at:at + 2] = [parts[at] + parts[at + 1]]


def specials(n_base: int) -> dict[int, str]:
    cfg = json.load(open(f"{HOME}/tokenizer_config.json", encoding="utf-8"))
    named = {int(i): v["content"] for i, v in cfg.get("added_tokens_decoder", {}).items()}
    return {i: named.get(i, f"<|reserved_token_{i}|>") for i in range(n_base, n_base + RESERVED)}


def build() -> dict:
    ranks = load_ranks(f"{HOME}/tiktoken.model")
    enc = byte_alphabet()
    s = lambda b: "".join(enc[x] for x in b)
    vocab = {s(t): r for t, r in ranks.items()}
    merges = []
    for t, r in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(t) < 2:
            continue
        p = bpe(ranks, t, r)
        assert len(p) == 2, (t, p)
        merges.append([s(p[0]), s(p[1])])
    sp = specials(len(ranks))
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True} for i, c in sp.items()],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": PAT}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": vocab, "merges": merges},
    }


def check(path: str):
    """Our BPE must give the same ids as plain tiktoken-style rank merging on varied text."""
    import regex
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(path)
    ranks = load_ranks(f"{HOME}/tiktoken.model")
    pat = regex.compile(PAT, flags=regex.V1)  # V1 reads && as class intersection, as tiktoken does
    texts = ["Hello, world! I'm here.", "  indented\n\n\tcode = f(x) + 12345;", "数学是科学的语言。", "naïve café — Ünïcödé",
             "<|im_user|>user<|im_middle|>Hi<|im_end|>", "1234567 apples", "don't STOP'S", "emoji 🎉 ok"]
    bad = 0
    for t in texts:
        ours = tk.encode(t, add_special_tokens=False).ids
        ref = []
        if "<|" in t:
            continue
        for piece in pat.findall(t):
            ref += [ranks[p] for p in bpe(ranks, piece.encode(), 1 << 30)]
        ok = ours == ref
        bad += not ok
        print(("ok  " if ok else "BAD ") + repr(t), ours[:12])
    sp = tk.encode("<|im_user|>user<|im_middle|>Hi<|im_end|>", add_special_tokens=False).ids
    print("chat specials:", sp)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = f"{HOME}/tokenizer.json"
    if not a.check:
        json.dump(build(), open(out, "w", encoding="utf-8"), ensure_ascii=False)
        print("wrote", out)
    raise SystemExit(1 if check(out) else 0)


if __name__ == "__main__":
    main()
