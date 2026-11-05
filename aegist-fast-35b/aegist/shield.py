"""Hidden-text and bypass shield.

Finds content a human reader cannot see or is not meant to notice, removes it
from what the model reads, and reports what it found, including any message
that was hidden. Detected bypass attempts are reported to the runtime, which
keeps Aegist's rules in force; the attempt text itself is never obeyed as an
instruction, because it reaches the model only as quoted user content.

Covered:
- invisible characters: zero-width, word joiners, BOM, soft hyphen, invisible
  operators, Hangul/Braille fillers
- Unicode "tag" characters (U+E0000 block), which can spell a hidden ASCII
  message; the message is decoded and shown
- variation-selector smuggling (bytes hidden as VS1-256 after a visible char)
- bidirectional overrides/isolates (text that displays differently than it reads)
- other control and private-use characters
- mixed-script words (Latin + Cyrillic/Greek look-alikes, e.g. "раypal")
- base64 / hex blobs that decode to readable text
- forged chat-control markup and common instruction-override phrasing
"""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field

ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xFEFF, 0x00AD, 0x180E,
              0x115F, 0x1160, 0x3164, 0xFFA0, 0x2800, 0x034F}
BIDI = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A)) | {0x200E, 0x200F, 0x061C}
TAGS = range(0xE0000, 0xE0080)
VS = set(range(0xFE00, 0xFE10)) | set(range(0xE0100, 0xE01F0))

OVERRIDE_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any |the |your )?(previous|prior|above|system) (instructions|rules|prompt)",
    r"forget (all |your )?(previous |prior )?(instructions|rules)",
    r"you are (now )?(dan|in developer mode|jailbroken|unfiltered|unrestricted)",
    r"(enable|activate|enter) (developer|god|jailbreak|dan) mode",
    r"(new|updated|real) system prompt",
    r"pretend (that )?you (have|had) no (rules|restrictions|filters|guidelines)",
    r"(bypass|override|disable) (your )?(safety|filters?|restrictions|guardrails|rules)",
]
CONTROL_MARKUP = [r"<\|im_(start|end)\|>", r"<\|endoftext\|>", r"\[/?(SYSTEM|INST)\]", r"<</?SYS>>",
                  r"^\s*###\s*(system|instruction)s?\s*:", r"^\s*system\s*:\s"]


@dataclass
class Finding:
    kind: str
    detail: str


@dataclass
class ShieldReport:
    clean: str
    findings: list[Finding] = field(default_factory=list)
    hidden_messages: list[str] = field(default_factory=list)

    @property
    def bypass_attempt(self) -> bool:
        return any(f.kind in ("override_phrase", "forged_control", "hidden_message", "encoded_instruction")
                   for f in self.findings)

    def summary(self) -> str:
        if not self.findings:
            return ""
        kinds: dict[str, int] = {}
        for f in self.findings:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        parts = [f"{k} x{n}" if n > 1 else k for k, n in kinds.items()]
        out = "shield: " + ", ".join(parts)
        for m in self.hidden_messages:
            out += f"\n  hidden text found: {m!r}"
        return out


def _script(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    for s in ("LATIN", "CYRILLIC", "GREEK"):
        if name.startswith(s):
            return s
    return "OTHER"


def _printable_ratio(b: bytes) -> float:
    if not b:
        return 0.0
    return sum(32 <= c < 127 or c in (9, 10, 13) for c in b) / len(b)


def scan(text: str) -> ShieldReport:
    findings: list[Finding] = []
    hidden: list[str] = []
    out = []
    tag_buf: list[str] = []
    vs_buf: list[int] = []

    def flush_tags():
        if tag_buf:
            msg = "".join(tag_buf)
            hidden.append(msg)
            findings.append(Finding("hidden_message", f"{len(msg)} chars in invisible tag characters"))
            tag_buf.clear()

    def flush_vs():
        if len(vs_buf) >= 4:  # a few selectors can be legitimate emoji presentation; runs are smuggling
            raw = bytes(v - 0xFE00 if v < 0xE0100 else v - 0xE0100 + 16 for v in vs_buf)
            msg = raw.decode("utf-8", errors="replace")
            hidden.append(msg)
            findings.append(Finding("hidden_message", f"{len(vs_buf)} bytes in variation selectors"))
        elif vs_buf:
            out.extend(chr(v) for v in vs_buf if v < 0xE0100)  # keep ordinary emoji presentation selectors
        vs_buf.clear()

    for ch in text:
        cp = ord(ch)
        if cp in TAGS:
            if 0xE0020 <= cp <= 0xE007E:
                tag_buf.append(chr(cp - 0xE0000))
            continue
        flush_tags()
        if cp in VS:
            vs_buf.append(cp)
            continue
        flush_vs()
        if cp in ZERO_WIDTH:
            findings.append(Finding("invisible_char", f"U+{cp:04X}"))
            continue
        if cp in BIDI:
            findings.append(Finding("bidi_control", f"U+{cp:04X} (text may display in a different order)"))
            continue
        cat = unicodedata.category(ch)
        if cat == "Cc" and ch not in "\n\t\r":
            findings.append(Finding("control_char", f"U+{cp:04X}"))
            continue
        if cat in ("Co", "Cn"):
            findings.append(Finding("private_or_unassigned", f"U+{cp:04X}"))
            continue
        out.append(ch)
    flush_tags()
    flush_vs()
    clean = "".join(out)

    for word in re.findall(r"\w{3,}", clean):
        scripts = {s for s in map(_script, word) if s in ("LATIN", "CYRILLIC", "GREEK")}
        if len(scripts) > 1:
            findings.append(Finding("mixed_script", f"{word!r} mixes {'/'.join(sorted(scripts))} letters"))

    for blob in re.findall(r"[A-Za-z0-9+/]{24,}={0,2}", clean):
        try:
            raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        if _printable_ratio(raw) > 0.95:
            dec = raw.decode("utf-8", errors="replace")
            findings.append(Finding("encoded_text", f"base64 decodes to {dec[:60]!r}"))
            if _matches(dec, OVERRIDE_PATTERNS):
                findings.append(Finding("encoded_instruction", "base64 hides an instruction-override"))
                hidden.append(dec)
    for blob in re.findall(r"\b(?:[0-9a-fA-F]{2}){12,}\b", clean):
        raw = bytes.fromhex(blob)
        if _printable_ratio(raw) > 0.95:
            dec = raw.decode("utf-8", errors="replace")
            findings.append(Finding("encoded_text", f"hex decodes to {dec[:60]!r}"))
            if _matches(dec, OVERRIDE_PATTERNS):
                findings.append(Finding("encoded_instruction", "hex hides an instruction-override"))
                hidden.append(dec)

    for hidden_msg in list(hidden):
        if _matches(hidden_msg, OVERRIDE_PATTERNS):
            findings.append(Finding("override_phrase", "inside hidden text"))
    if _matches(clean, OVERRIDE_PATTERNS):
        findings.append(Finding("override_phrase", "instruction-override wording"))
    if _matches(clean, CONTROL_MARKUP, flags=re.IGNORECASE | re.MULTILINE):
        findings.append(Finding("forged_control", "chat-control markup in input (treated as plain text)"))
    return ShieldReport(clean, findings, hidden)


def _matches(text: str, patterns, flags=re.IGNORECASE) -> bool:
    norm = unicodedata.normalize("NFKC", text)
    return any(re.search(p, norm, flags) for p in patterns)
