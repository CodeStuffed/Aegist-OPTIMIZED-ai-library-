"""A keyword rule for reading a prompt as fact, creative or other: a Python baseline only, used by
scripts/eval_prompt_reader.py to compare against Laya. The engine does not use it; Laya is the reader."""
from __future__ import annotations

import re

# help with code or the user's own text comes first: "write a function" is not creative writing
OTHER = re.compile(
    r"\b(code|function|bug|error|exception|traceback|regex|sql|query|bash|powershell|python|javascript|rust|json|yaml|"
    r"refactor|compile|translate|rephrase|rewrite|proofread|grammar|reword|shorter|more formal|friendlier|polite|"
    r"summari[sz]e (this|my)|my notes|these notes|this text|this sentence|this paragraph|this email|"
    r"should i|tips|advice|help me|opinion|what do you think|which .* should i|recommend|"
    r"hi|hello|hey|thanks|thank you|good night|see you|ok cool|what's up)\b", re.I)
CREATIVE = re.compile(
    r"\b(poem|poetry|haiku|sonnet|limerick|song|rap|lyrics|story|stories|tale|fable|novel|plot|dialogue|"
    r"invent|imagine|make up|made up|come up with|dream up|brainstorm|pitch|fictional|fantasy|"
    r"create (a|an|some|new)|design (a|an)|new (word|sport|species|holiday|flavour|flavor|recipe)|"
    r"names? for|slogans?|mascot|riddle|original|creative|wild ideas?)\b", re.I)
FACT = re.compile(
    r"^\s*(what|who|whom|when|where|which|why|how|is|are|was|were|does|do|did|can|explain|describe|list|"
    r"summari[sz]e|tell me about|define)\b|\b(true that|fact|history of|summary of)\b", re.I)


def keyword_kind(prompt: str) -> str:
    return keyword_cue(prompt) or "other"


def keyword_cue(prompt: str) -> str | None:
    """The kind a cue word gives, or None when no cue matched."""
    if CREATIVE.search(prompt) and not re.search(r"\b(code|function|regex|sql|bash|powershell|python|javascript)\b", prompt, re.I):
        return "creative"
    if OTHER.search(prompt):
        return "other"
    if FACT.search(prompt):
        return "fact"
    return None
