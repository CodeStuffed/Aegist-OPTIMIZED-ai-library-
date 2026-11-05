"""Aegist chat runtime: streaming generation, adaptive thinking, modes, shield.

Thinking is adaptive compute: the planner decides per message whether Aegist
thinks before answering and how many thinking tokens it may spend; when the
budget runs out the runtime closes the thought and makes it answer.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import torch

from . import persona
from .model import AegistLM, KVCache
from .shield import scan
from .tokenizer import THINK_CLOSE, AegistTokenizer

_REASONING = re.compile(
    r"(\d\s*[-+*/^=]\s*\d|prove|proof|solve|calculate|equation|integral|derivative|probability|how many|"
    r"step by step|algorithm|debug|bug|code|function|python|rust|sql|regex|why does|explain why|logic|puzzle|riddle|"
    r"compare|plan|design|optimi[sz]e|complexity|\bfix\b)", re.IGNORECASE)


@dataclass
class GenConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens: int = 2048
    think_temperature: float = 0.6
    think_top_p: float = 0.95
    presence_penalty: float = 0.0  # subtracted from the logit of every token already generated this reply
    open_think: bool = False  # when thinking, the prompt itself opens the think block (newer chat templates)

    @classmethod
    def for_arch(cls, arch: str) -> "GenConfig":
        """Built-in defaults per architecture ("hybrid-moe" uses its base model's recommended settings)."""
        if arch == "hybrid-moe":
            return cls(think_temperature=1.0, think_top_p=0.95, presence_penalty=1.5, open_think=True)
        return cls()

    @classmethod
    def from_header(cls, g: dict, arch: str) -> "GenConfig":
        """Settings from a brain header's "generation", falling back to the architecture's defaults.
        A hybrid header with only temperature / top_p (no think_* keys) predates the split; there those
        two are the thinking settings. Mirrors GenConfig::from_header in aegist_rs/src/chat.rs."""
        d = cls.for_arch(arch)
        split = "think_temperature" in g or arch != "hybrid-moe"
        return cls(
            temperature=float(g.get("temperature", d.temperature)) if split else d.temperature,
            top_p=float(g.get("top_p", d.top_p)) if split else d.top_p,
            top_k=max(1, int(g.get("top_k", d.top_k))),
            max_new_tokens=int(g.get("max_new_tokens", d.max_new_tokens)),
            think_temperature=float(g.get("think_temperature" if split else "temperature", d.think_temperature)),
            think_top_p=float(g.get("think_top_p" if split else "top_p", d.think_top_p)),
            presence_penalty=float(g.get("presence_penalty", d.presence_penalty)),
            open_think=bool(g.get("open_think", d.open_think)))


@dataclass
class ThinkPlan:
    think: bool
    budget: int
    reason: str


def plan_thinking(text: str, setting: str) -> ThinkPlan:
    """setting: 'off' | 'on' | 'auto'."""
    if setting == "off":
        return ThinkPlan(False, 0, "thinking off")
    if setting == "on":
        return ThinkPlan(True, 4096, "thinking on")
    hits = len(_REASONING.findall(text))
    if hits == 0 and len(text) < 200:
        return ThinkPlan(False, 0, "auto: conversational")
    budget = min(4096, 512 + 512 * hits + len(text) // 2)
    return ThinkPlan(True, budget, f"auto: reasoning cues x{hits}")


@dataclass
class TurnStats:
    prompt_tokens: int = 0
    reused_tokens: int = 0
    thinking_tokens: int = 0
    answer_tokens: int = 0
    seconds: float = 0.0
    think_plan: str = ""
    shield: str = ""
    cores_used: int = 0

    @property
    def tok_per_s(self) -> float:
        n = self.thinking_tokens + self.answer_tokens
        return n / self.seconds if self.seconds else 0.0


@dataclass
class Session:
    mode: str = "normal"
    think: str = "auto"
    turns: list[tuple[str, str]] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)


class Aegist:
    def __init__(self, model: AegistLM, tok: AegistTokenizer, device="cuda", gen: GenConfig | None = None,
                 max_context: int = 8192):
        self.model, self.tok, self.device = model.eval(), tok, device
        self.gen = gen or GenConfig()
        self.max_context = max_context
        self.cache: KVCache | None = None
        self.cache_ids: list[int] = []
        self.dtype = next(model.parameters()).dtype

    # --- low level --------------------------------------------------------
    def _ensure_cache(self):
        if self.cache is None:
            if hasattr(self.model, "new_cache"):  # hybrid: KV cache plus recurrent state, with snapshots
                self.cache = self.model.new_cache(1, self.max_context, self.device, self.dtype)
            else:
                self.cache = KVCache(self.model.cfg, 1, self.max_context, self.device, self.dtype)

    def _feed(self, ids: list[int]) -> torch.Tensor:
        logits = None
        for i in range(0, len(ids), 1024):  # chunked so long prompts do not spike memory
            chunk = torch.tensor([ids[i : i + 1024]], device=self.device)
            logits = self.model(chunk, cache=self.cache, last_only=True)["logits"][:, -1]
        return logits

    @torch.no_grad()
    def _prefill(self, ids: list[int], snap_at: int | None = None) -> tuple[torch.Tensor, int]:
        """Feed `ids`, reusing the longest cached prefix the cache can go back to. A recurrent
        state goes back only to a snapshot, so one is taken at `snap_at` (the start of this reply),
        where the next turn's prompt will diverge. Returns last logits and #reused tokens."""
        self._ensure_cache()
        n = 0
        for a, b in zip(self.cache_ids, ids):
            if a != b:
                break
            n += 1
        n = min(n, self.cache.length, len(ids) - 1)  # always run at least one token to get fresh logits
        kept = self.cache.crop(n)
        kept = n if kept is None else kept
        self.cache_ids = self.cache_ids[:kept]
        if hasattr(self.cache, "snapshot") and snap_at is not None and kept < snap_at <= len(ids):
            logits = self._feed(ids[kept:snap_at])
            self.cache.snapshot(keep=2)
            if snap_at < len(ids):
                logits = self._feed(ids[snap_at:])
        else:
            logits = self._feed(ids[kept:])
        self.cache_ids += ids[kept:]
        return logits, kept

    @torch.no_grad()
    def _step(self, tok_id: int) -> torch.Tensor:
        x = torch.tensor([[tok_id]], device=self.device)
        self.cache_ids.append(tok_id)
        return self.model(x, cache=self.cache, last_only=True)["logits"][:, -1]

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int, gen: torch.Generator) -> int:
        logits = logits.float().squeeze(0)
        if temperature <= 0:
            return int(logits.argmax())
        v, i = logits.topk(top_k)
        p = (v / temperature).softmax(-1)
        keep = p.cumsum(-1) - p < top_p
        p = p * keep
        return int(i[torch.multinomial(p / p.sum(), 1, generator=gen)])

    # --- chat -------------------------------------------------------------
    def reply(self, session: Session, user_text: str, seed: int | None = None):
        """Generator of (kind, text) events: kind in shield/think/answer/done(stats)."""
        t0 = time.perf_counter()
        stats = TurnStats()
        report = scan(user_text)
        if report.findings:
            stats.shield = report.summary()
            yield "shield", stats.shield
        tone = persona.read_tone(report.clean)
        plan = plan_thinking(report.clean, session.think)
        stats.think_plan = plan.reason
        system = persona.system_prompt(session.mode, tone, stats.shield if report.bypass_attempt or report.hidden_messages
                                       else "", session.memory)
        turns = session.turns + [("user", report.clean)]
        open_think = self.gen.open_think
        ids = self.tok.chat_ids(system, turns, think=plan.think, open_think=open_think)
        budget_left = self.max_context - len(ids) - 8
        if budget_left < 64:  # drop the oldest turns until the conversation fits
            while len(turns) > 1 and len(self.tok.chat_ids(system, turns, plan.think, open_think)) > self.max_context - 512:
                turns = turns[2:]
            ids = self.tok.chat_ids(system, turns, think=plan.think, open_think=open_think)
        # the open assistant header with no think markup: the next turn's prompt matches up to here
        snap_at = len(self.tok.chat_ids(system, turns, think=True, open_think=False))
        stats.prompt_tokens = len(ids)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed if seed is not None else int(time.time() * 1000) % 2**31)
        # a thinking reply samples with the thinking settings throughout, as the base model recommends
        temperature, top_p = ((self.gen.think_temperature, self.gen.think_top_p) if plan.think
                              else (self.gen.temperature, self.gen.top_p))
        seen: set[int] = set()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            logits, stats.reused_tokens = self._prefill(ids, snap_at)
            out: list[int] = []
            thinking = plan.think
            think_ids: list[int] = []
            emitted = ""
            usage0 = self.model.core_usage()
            max_new = min(self.gen.max_new_tokens + plan.budget, self.max_context - len(ids) - 1)
            for _ in range(max_new):
                if self.gen.presence_penalty and seen:
                    logits = logits.float().clone()
                    logits[..., list(seen)] -= self.gen.presence_penalty
                nxt = self._sample(logits, temperature, top_p, self.gen.top_k, gen)
                if nxt in self.tok.eos_ids:
                    break
                seen.add(nxt)
                if thinking:
                    think_ids.append(nxt)
                    if nxt == self.tok.think_close:
                        thinking = False
                        stats.thinking_tokens = len(think_ids)
                        logits = self._step(nxt)
                        continue
                    text = self.tok.decode(think_ids)
                    if not text.endswith("�"):
                        delta = text[len(emitted):]
                        emitted = text
                        if delta:
                            yield "think", delta.replace("<think>", "")
                    if len(think_ids) >= plan.budget:  # budget spent: close the thought, make it answer
                        forced = self.tok.encode_trusted("\n\nI have thought enough; time to answer.\n" + THINK_CLOSE + "\n\n")
                        for t in forced[:-1]:
                            self._step(t)
                        logits = self._step(forced[-1])
                        thinking = False
                        stats.thinking_tokens = len(think_ids)
                        emitted = ""
                        continue
                    logits = self._step(nxt)
                    continue
                out.append(nxt)
                text = self.tok.decode(out).lstrip("\n")
                if not text.endswith("�"):
                    delta = text[len(emitted):]
                    emitted = text
                    if delta:
                        yield "answer", delta
                logits = self._step(nxt)
            stats.answer_tokens = len(out)
        answer = self.tok.decode(out).strip()
        session.turns = turns + [("assistant", answer)]
        stats.cores_used = int(((self.model.core_usage() - usage0) > 0).sum())
        stats.seconds = time.perf_counter() - t0
        yield "done", stats
