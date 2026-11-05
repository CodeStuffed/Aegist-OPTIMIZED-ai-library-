"""Learning time for the 177B brain: cores in the top layers, versions as cores files.

Learning time's usual way does not fit the 177B brain. Every kept version would be a 185 GB copy,
and one training step through all 48 layers reads about 70 GB of experts. So:

- Only the top --top-layers layers learn (default 3: layers 45, 46, 47). The layers below them do
  not change, so the Rust engine runs every example once, up to the first learning layer, and saves
  the hidden lanes going into it (`aegist lanes`). Training runs only the top layers on those
  lanes, with all of their experts on the GPU (2.5 GB a layer), so no expert is read twice.
- A kept version is a cores file in brains/history: the learned cores of every layer, the
  tokenizer, licence and lineage, and the path and SHA-256 of the base brain they were learned on
  (tens of MB, not 185 GB). The Rust engine opens it by opening the base and putting the cores in
  place (aegist_rs/src/nx.rs); so does this module.

The material, the early stopping on drift and the gate are aegist/learn.py's, with two checks on
the Rust engine, since that is what answers the user: its loss on the held-out material must agree
with Python's (the cores do the same there), and it must still say it is Aegist. The governor's
check (held-out questions answered by the new version) also runs on the engine.

What this cannot do yet: change the layers below the top ones, and the drift reference ("the base")
is the current version with its top-layer cores switched off.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from . import confidence as C
from . import governor as gov
from . import hc as HC
from . import hybrid as H
from . import lessons, persona
from .nx import DTYPES, NxFile, save as nx_save


def log(msg: str):
    print(msg, flush=True)


# --- cores files ---------------------------------------------------------------------

def base_of(path: str) -> tuple[str, dict]:
    """(the base brain's path, the header of `path`): a cores file names its base."""
    with NxFile(path) as nx:
        h = nx.header
    return ((h.get("base") or {}).get("path") or path), h


def stored_cores(path: str) -> dict[str, torch.Tensor]:
    """The cores a cores file holds (copies); {} for a base brain."""
    nx = NxFile(path)
    try:
        if "base" not in nx.header:
            return {}
        return {n: nx.tensor(n).clone() for n in nx.header["tensors"]}
    finally:
        nx.close()


def save_cores(out: str, cores: dict[str, torch.Tensor], base: str, base_header: dict, tok_json: str,
               license_text: str, lineage: list) -> dict:
    return nx_save(out, cores, base_header["config"], tok_json, notice=base_header.get("notice", ""),
                   license_text=license_text, persona=base_header.get("persona"),
                   generation=base_header.get("generation"), lineage=lineage,
                   extra={"name": base_header.get("name", "Aegist"),
                          "base": {"path": base.replace("\\", "/"), "sha256": base_header["sha256"]["base"]},
                          "dtype": base_header.get("dtype", "bf16"), "kind": "cores"})


# --- the top layers -------------------------------------------------------------------

class TopLM(nn.Module):
    """Layers start..n_layers-1 of a lanes brain, the output lanes' mix and the head. Input: the
    hidden lanes going into layer `start` [B, T, hc*d] and the token ids [B, T]."""

    def __init__(self, cfg: H.HybridConfig, start: int):
        super().__init__()
        self.cfg, self.start = cfg, start
        self.blocks = nn.ModuleList(H.HybridBlock(cfg, i) for i in range(start, cfg.n_layers))
        counters = {"attn": 0, "delta": 0}
        for i, kind in enumerate(cfg.layer_types):
            if i >= start:
                self.blocks[i - start].slot = counters[kind]
            counters[kind] += 1
        self.hc_out = HC.HyperConnection(cfg.d_model, cfg.hc_count, cfg.hc_lowrank, cfg.norm_eps, combine=False)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._rope = None

    def set_cores(self, enabled: bool):
        for b in self.blocks:
            b.cores_enabled = enabled

    def core_usage(self) -> torch.Tensor:
        dev = self.lm_head.weight.device
        n = self.cfg.n_cores
        return torch.stack([(b.core_hits if b.core_hits is not None and b.core_hits.shape[0] == n
                             else torch.zeros(n, dtype=torch.long, device=dev)).to(dev) for b in self.blocks])

    def hidden(self, x, ids, checkpoint: bool = False):
        T = x.shape[1]
        if self._rope is None or self._rope[0].shape[0] < T or self._rope[0].device != x.device:
            self._rope = H.rope_tables(self.cfg, max(T, 1024), x.device)
        cos, sin = self._rope
        for b in self.blocks:
            if checkpoint and torch.is_grad_enabled():
                x, _ = torch.utils.checkpoint.checkpoint(b, x, cos, sin, None, ids, use_reentrant=False)
            else:
                x, _ = b(x, cos, sin, None, ids)
        return self.hc_out(x)

    def logits(self, h):
        return self.lm_head(h)

    def _ce_sum(self, h, t):
        return F.cross_entropy(self.logits(h).float(), t, reduction="sum")

    def ce(self, x, ids, y, checkpoint: bool = False):
        h = self.hidden(x, ids, checkpoint)
        keep = y != -100
        hs, ts = h[keep], y[keep]
        if hs.shape[0] == 0:
            return h.sum() * 0.0
        if torch.is_grad_enabled():
            return sum(torch.utils.checkpoint.checkpoint(self._ce_sum, a, b, use_reentrant=False)
                       for a, b in zip(hs.split(256), ts.split(256))) / hs.shape[0]
        return self._ce_sum(hs, ts) / hs.shape[0]


def load_top(brain: str, start: int, device="cuda", log=log) -> tuple[TopLM, NxFile]:
    """The top layers of `brain` (a base brain or a cores file over one), every expert on `device`."""
    base, _ = base_of(brain)
    nx = NxFile(base)
    h = nx.header
    cfg = H.HybridConfig.from_dict(h["config"])
    if cfg.hc_count <= 1 or not 0 < start < cfg.n_layers:
        nx.close()
        raise ValueError(f"{brain}: top-layer learning is for lanes brains, start 1..{cfg.n_layers - 1}")
    dtype = DTYPES[h.get("dtype", "bf16")]
    with torch.device("meta"):
        top = TopLM(cfg, start).to(dtype)
        for b in top.blocks:
            for n in ("gate_up", "down"):
                b.moe.experts._parameters.pop(n, None)
            b.moe.experts.dense = False
    top = top.to_empty(device=device)
    key = "experts.fp8" if cfg.expert_format == "fp8" else "experts.q4"

    def ours(n: str) -> str | None:
        if n.startswith("blocks."):
            i = int(n.split(".")[1])
            return f"blocks.{i - start}.{n.split('.', 2)[2]}" if i >= start else None
        return n if n.startswith(("hc_out.", "lm_head.")) else None

    sd = {}
    for n, m in h["tensors"].items():
        o = ours(n)
        if o is None or n.endswith("." + key) or n.endswith(".ple.table_scale") or \
                (n.endswith(".ple.table.weight") and m["dtype"] == "f8e4m3"):
            continue
        sd[o] = nx.tensor(n)
    for n, t in stored_cores(brain).items():
        o = ours(n)
        if o is not None:
            sd[o] = t
    missing, unexpected = top.load_state_dict(sd, strict=False)
    if missing or unexpected:
        nx.close()
        raise ValueError(f"{brain}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    t0 = time.time()
    for j, b in enumerate(top.blocks):
        i = start + j
        if b.ple is not None and f"blocks.{i}.ple.table.weight" in h["tensors"]:
            b.ple.use_file_table(nx.tensor(f"blocks.{i}.ple.table.weight"), nx.tensor(f"blocks.{i}.ple.table_scale"))
        ex = b.moe.experts.page(nx.tensor(f"blocks.{i}.moe.{key}"), cfg.n_experts, device)
        for e in range(cfg.n_experts):  # all of them, once: nothing is paged while learning
            ex._row(e)
        ex.refetch = True
    torch.cuda.synchronize()
    log(f"top layers {start}-{cfg.n_layers - 1} of {os.path.basename(base)} on the GPU, every expert resident "
        f"({torch.cuda.memory_allocated() / 1e9:.1f} GB, {time.time() - t0:.0f}s)")
    top._nx = nx
    return top, nx


def pick_device(want: str = "auto") -> str:
    """'auto': the GPU with the most free memory, since a game or another app may be using one."""
    if want != "auto":
        return want
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("learning time on this brain needs an NVIDIA GPU")
    free = [torch.cuda.mem_get_info(i)[0] for i in range(n)]
    return f"cuda:{max(range(n), key=free.__getitem__)}"


# --- the engine's side ----------------------------------------------------------------

def run_engine(args: list[str], what: str):
    r = subprocess.run([C.exe_path(), *args])  # progress goes to our stderr
    if r.returncode != 0:
        raise RuntimeError(f"aegist {what} exited with {r.returncode}")


class Lanes:
    """The engine's hidden lanes going into layer `start` for each example (bf16 on disk).

    Reading them is the slow part, so they are kept in
    `cache`, keyed by the brain's hashes (not its path), the layer and the token ids; the newest `keep` are kept."""

    def __init__(self, brain: str, examples: list, start: int, width: int, work: str, cache: str, keep: int = 3):
        os.makedirs(work, exist_ok=True)
        os.makedirs(cache, exist_ok=True)
        prompts = [e[0][:-1] for e in examples]
        with NxFile(brain) as nx:
            sha = nx.header.get("sha256")
        key = hashlib.sha256(json.dumps({"sha256": sha, "layer": start, "prompts": prompts}).encode()).hexdigest()[:24]
        out = os.path.join(cache, f"lanes-{key}.bin")
        n_tokens = sum(len(p) for p in prompts)
        if os.path.exists(out):
            log(f"the hidden lanes of these {len(examples)} examples ({n_tokens} tokens) are cached")
            os.utime(out)
        else:
            inp = os.path.join(work, "lanes.json")
            json.dump({"prompts": prompts}, open(inp, "w", encoding="utf-8"))
            t0 = time.time()
            log(f"the engine reads {len(examples)} examples up to layer {start} ({n_tokens} tokens)")
            run_engine(["lanes", brain, inp, out + ".part", "--layer", str(start)], "lanes")
            os.replace(out + ".part", out)
            log(f"done in {time.time() - t0:.0f}s")
            old = sorted((f for f in os.listdir(cache) if f.startswith("lanes-") and f.endswith(".bin")),
                         key=lambda f: os.path.getmtime(os.path.join(cache, f)))
            for f in old[:-keep]:
                os.remove(os.path.join(cache, f))
        self.width = width
        self.raw = np.memmap(out, dtype=np.int16, mode="r")
        lens = [len(e[0]) - 1 for e in examples]
        self.offsets = np.concatenate([[0], np.cumsum(lens)]) * width
        if self.offsets[-1] != self.raw.shape[0]:
            raise RuntimeError(f"lanes file has {self.raw.shape[0]} values, expected {self.offsets[-1]}")

    def get(self, i: int, device) -> torch.Tensor:
        a = np.array(self.raw[self.offsets[i]:self.offsets[i + 1]])
        return torch.from_numpy(a).view(torch.bfloat16).view(1, -1, self.width).to(device)


def engine_scores(brain: str, examples: list, work: str) -> list[float]:
    """Mean loss on each example's scored tokens, on the Rust engine."""
    os.makedirs(work, exist_ok=True)
    inp, out = os.path.join(work, "score.json"), os.path.join(work, "score-out.json")
    json.dump({"items": [{"ids": e[0], "start": e[1].index(True)} for e in examples]}, open(inp, "w", encoding="utf-8"))
    run_engine(["score", brain, inp, out], "score")
    return [r["nll"] for r in json.load(open(out, encoding="utf-8"))]


def engine_answers(brain: str, prompts: list[str], work: str, max_new: int = 80) -> list[str]:
    os.makedirs(work, exist_ok=True)
    inp, out = os.path.join(work, "ask.json"), os.path.join(work, "ask-out.json")
    json.dump({"prompts": prompts, "think": "off", "max_new": max_new}, open(inp, "w", encoding="utf-8"))
    run_engine(["exam", inp, out, "--brain", brain, "--no-second-look"], "exam")
    return [r["answer"] for r in json.load(open(out, encoding="utf-8"))]


# --- learning time ----------------------------------------------------------------------

def learn_top(a, brain: str) -> int:
    from . import learn as L

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    device = pick_device(getattr(a, "device", "auto"))
    torch.cuda.set_device(device)
    base, head = base_of(brain)
    with NxFile(base) as nx:
        base_header = dict(nx.header)
        cfg = H.HybridConfig.from_dict(nx.header["config"])
        tok_json = nx.blob("tokenizer.json").decode("utf-8")
        license_text = nx.blob("LICENSE").decode("utf-8")
    persona.MODEL_NAME = base_header.get("name")  # the system prompt names the model, as in chat
    from .tokenizer import AegistTokenizer
    tok = AegistTokenizer.from_json_str(tok_json)
    start = cfg.n_layers - a.top_layers
    stem = os.path.splitext(os.path.basename(base))[0]
    probes_dir = os.path.join(a.probes, stem) if os.path.isdir(os.path.join(a.probes, stem)) else a.probes

    inbox_dir = os.path.join(a.learn_dir, "inbox")
    fb_path = os.path.join(a.learn_dir, "feedback.jsonl")
    lessons_path = os.path.join(a.learn_dir, "lessons.jsonl")
    docs = L.inbox_records(inbox_dir)
    fbs, fb_skipped = L.feedback_records(fb_path)
    lsn = L.lesson_records(lessons_path)
    cal_dirs, cal = [], []
    if a.calibrate:
        cal_dirs = gov.collections_for(probes_dir, base)
        if cal_dirs:
            cal, why = gov.training_records(cal_dirs)
            log(f"governor: {len(cal)} records from {len(cal_dirs)} question collections ({why})")
        else:
            log(f"governor: no question collection answered by {os.path.basename(base)} in {probes_dir}")
    new = docs + fbs + lsn + cal
    if not new:
        log(f"Nothing to learn: put files in {inbox_dir} or rate replies in the launcher first.")
        return 1
    log(f"brain: {brain}" + (f" (cores over {base})" if base != brain else ""))
    log(f"learning layers {start}-{cfg.n_layers - 1} of {cfg.n_layers} on {torch.cuda.get_device_name(device)} ({device})")
    log(f"new material: {len(docs)} files, {len(fbs)} feedback items ({fb_skipped} bad ratings without a "
        f"correction skipped), {len(lsn)} prepared lessons, {len(cal)} governor records")

    new_ex, long_new = L.examples_for(tok, new, a.seq_len)
    ranked = sorted(new_ex, key=lambda e: __import__("hashlib").sha256((e[2] + str(e[0][:32])).encode()).digest())
    n_held = max(1, len(ranked) // 10) if len(ranked) >= 5 else 0
    held, train_new = ranked[:n_held], ranked[n_held:]
    if not held:
        held, train_new, held_note = new_ex, new_ex, "no held-out split (too little material); measured on training data"
    else:
        held_note = f"{len(held)} held-out examples"
    replay = L.examples_for(tok, L.replay_records(a.brains, max(len(train_new), 8), rng), a.seq_len)[0]
    ident = L.examples_for(tok, L.identity_records(base_header.get("name")), a.seq_len)[0]
    train = train_new + replay + ident
    probes = L.examples_for(tok, [{"kind": "chat", "mode": m, "turns": [["user", q]], "target": r, "source": "probe"}
                                  for m, q, r in lessons.PROBES], a.seq_len)[0]
    n_tok = sum(len(e[0]) for e in train)
    log(f"examples: {len(train_new)} new + {len(replay)} replay + {len(ident)} identity = {len(train)} "
        f"({n_tok} tokens); {held_note}; {long_new} chats longer than {a.seq_len} tokens skipped")
    if a.dry_run:
        for e in train_new[:10]:
            log(f"  {e[2]}: {len(e[0])} tokens")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = os.path.join(a.learn_dir, "work", f"{stem}-{stamp}")
    everything = train + held + probes
    at = {id(e): i for i, e in enumerate(everything)}
    t_all = time.time()
    lanes = Lanes(brain, everything, start, cfg.hc_count * cfg.d_model, work,
                  cache=os.path.join(a.learn_dir, "work", "lanes-cache"))
    top, nx = load_top(brain, start, device)

    def batch(e):
        ids, scored, _ = e
        x = lanes.get(at[id(e)], device)
        idx = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([[t if s else -100 for t, s in zip(ids[1:], scored[1:])]], device=device)
        return x, idx, y

    @torch.no_grad()
    def mean_ce(exs):
        top.eval()
        tot, n = 0.0, 0
        for e in exs:
            x, idx, y = batch(e)
            k = int((y != -100).sum())
            if k:
                tot += float(top.ce(x, idx, y)) * k
                n += k
        return tot / max(n, 1)

    @torch.no_grad()
    def references(k: int = 32):
        top.eval()
        top.set_cores(False)
        refs = []
        for e in probes:
            x, idx, y = batch(e)
            hh = top.hidden(x, idx)
            v, i = top.logits(hh[0, (y[0] != -100).nonzero().flatten()]).float().log_softmax(-1).topk(k, -1)
            refs.append((i, v))
        top.set_cores(True)
        return refs

    def probe_kl(e, ref, checkpoint=False):
        x, idx, y = batch(e)
        hh = top.hidden(x, idx, checkpoint)
        logq = top.logits(hh[0, (y[0] != -100).nonzero().flatten()]).float().log_softmax(-1)
        i, v = ref
        p = v.exp()
        p = p / p.sum(-1, keepdim=True)
        return (p * (p.log() - logq.gather(-1, i))).sum(-1).mean()

    @torch.no_grad()
    def mean_drift(refs, keep_mode=False):
        if not keep_mode:
            top.eval()
        return sum(float(probe_kl(e, r)) for e, r in zip(probes, refs)) / max(len(probes), 1)

    t0 = time.time()
    refs = references()
    kl_before = mean_drift(refs)
    ce_before = mean_ce(held)
    log(f"before: held-out loss {ce_before:.4f}, probe drift {kl_before:.4f} nats/token ({time.time() - t0:.0f}s)")

    # the working cores: the ones the core engine switches on most for the new material
    top.eval()
    before = top.core_usage().clone()
    with torch.no_grad():
        for e in train_new[:256]:
            x, idx, _ = batch(e)
            top.hidden(x, idx)
    counts = (top.core_usage() - before).cpu()
    work_cores = [[int(c) for c in row.topk(a.working_cores).indices if row[c] > 0] for row in counts]
    covered = sum(int(counts[i, w].sum()) for i, w in enumerate(work_cores)) / max(int(counts.sum()), 1)
    for p in top.parameters():
        p.requires_grad_(False)
    params = []
    for b, w in zip(top.blocks, work_cores):
        params += [p for p in b.cores.begin_learning(w, device) if p.numel()]
    n_params = sum(p.numel() for p in params)
    log(f"working set: {sum(map(len, work_cores))} cores in {len(top.blocks)} layers ({n_params / 1e6:.1f}M params) "
        f"covering {covered:.0%} of the core switch-ons for the new material")

    accum = a.accum
    steps = min(a.max_steps, max(1, math.ceil(len(train) * a.epochs / accum)))
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    warm = max(1, steps // 20)
    order: list = []
    top.train()
    curve = []
    target = a.max_kl * a.drift_margin
    best = {"step": 0, "drift": round(kl_before, 4), "weights": [p.detach().to("cpu", copy=True) for p in params]}
    drift_curve = [[0, round(kl_before, 4)]]
    stopped = None
    t0 = time.time()
    for step in range(steps):
        lr = a.lr * (step + 1) / warm if step < warm else \
            a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
        for g in opt.param_groups:
            g["lr"] = lr
        ce_sum = 0.0
        for _ in range(accum):
            if not order:
                order = train[:]
                rng.shuffle(order)
            x, idx, y = batch(order.pop())
            ce = top.ce(x, idx, y, checkpoint=True)
            (ce / accum).backward()
            ce_sum += float(ce.detach()) / accum
        if a.kl_weight > 0 and probes:
            j = step % len(probes)
            (a.kl_weight * probe_kl(probes[j], refs[j], checkpoint=True)).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        curve.append(round(ce_sum, 4))
        if probes and ((step + 1) % a.check_every == 0 or step == steps - 1):
            drift = mean_drift(refs, keep_mode=True)
            drift_curve.append([step + 1, round(drift, 4)])
            if drift <= target:
                best = {"step": step + 1, "drift": round(drift, 4),
                        "weights": [p.detach().to("cpu", copy=True) for p in params]}
            log(f"step {step + 1}/{steps}: loss {ce_sum:.4f}, drift on all probes {drift:.4f} "
                f"({'kept' if drift <= target else 'over target ' + format(target, '.3f')}), lr {lr:.2e}, "
                f"{time.time() - t0:.0f}s")
            if drift > target:
                stopped = step + 1
                break
        elif step % 10 == 0:
            log(f"step {step + 1}/{steps}: loss {ce_sum:.4f}, lr {lr:.2e}, {time.time() - t0:.0f}s")
    del opt
    kept = best["step"] if probes else steps
    if probes and kept != len(curve):
        with torch.no_grad():
            for p, w in zip(params, best["weights"]):
                p.copy_(w)
        log(f"{'stopped at step ' + str(stopped) + ': drift over target; ' if stopped else ''}"
            f"going back to the weights from step {kept} (drift {best['drift']:.4f})")
    del best
    for b in top.blocks:
        b.cores.end_learning(keep=True)
    top.eval()
    ce_after = mean_ce(held)
    kl_after = mean_drift(refs)
    learned = ce_after < ce_before * (1 - a.min_gain)
    stays_close = kl_after <= a.max_kl
    log(f"after:  held-out loss {ce_after:.4f} (was {ce_before:.4f}), probe drift {kl_after:.4f} (limit {a.max_kl})")

    # the version: earlier sessions' cores below the learning layers, this session's above
    cores = {n: t for n, t in stored_cores(brain).items() if int(n.split(".")[1]) < start}
    for j, b in enumerate(top.blocks):
        if b.cores.ids.numel():
            for k in ("ids", "gate", "up", "down"):
                cores[f"blocks.{start + j}.cores.{k}"] = getattr(b.cores, k).detach().cpu().contiguous()
    n_live = sum(t.shape[0] for n, t in cores.items() if n.endswith(".ids"))
    del top, params, refs, lanes
    nx.close()
    torch.cuda.empty_cache()  # the engine needs the GPU now

    entry = {"event": "learned", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "from": os.path.basename(brain),
             "layers": [start, cfg.n_layers - 1], "live_cores": n_live,
             "material": {"files": [r["source"] for r in docs], "feedback": len(fbs), "governor": len(cal),
                          "lessons": dict(sorted(Counter(r.get("source", "") for r in lsn).items())),
                          "replay": len(replay), "identity": len(ident), "tokens": n_tok},
             "working_cores": sum(map(len, work_cores)), "working_coverage": round(covered, 3), "steps": steps,
             "steps_run": len(curve), "kept_step": kept, "lr": a.lr,
             "held_out_loss": [round(ce_before, 4), round(ce_after, 4)],
             "probe_kl": [round(kl_before, 4), round(kl_after, 4)], "loss_curve": curve, "drift_curve": drift_curve}
    if not (learned and stays_close):
        failed = ", ".join(k for k, ok in (("learned", learned), ("stays_close_to_base", stays_close)) if not ok)
        log(f"REJECTED ({failed}). The current brain stays as it was; the material is left in place.")
        return 2

    out = os.path.join(a.brains, "history", f"{stem}-{stamp}.nx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_cores(out, cores, base, base_header, tok_json, license_text, head.get("lineage", []) + [entry])
    log(f"wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB, {n_live} live cores)")

    # the engine: same loss on the held-out material (the cores work there), still Aegist
    eng = engine_scores(out, held, work)
    k = [sum(e[1][1:]) for e in held]
    ce_engine = sum(x * n for x, n in zip(eng, k)) / max(sum(k), 1)
    agrees = abs(ce_engine - ce_after) <= max(0.05, 0.05 * ce_after)
    log(f"engine: held-out loss {ce_engine:.4f} with the new cores (Python {ce_after:.4f}): "
        f"{'agrees' if agrees else 'DISAGREES'}")
    reply = engine_answers(out, ["Who are you?"], work)[0]
    ident_ok = "aegist" in reply.lower()
    log(f"identity: {'ok' if ident_ok else 'FAILED'}: {reply.strip()[:200]!r}")
    gate = {"learned": learned, "stays_close_to_base": stays_close, "engine_agrees": agrees, "still_aegist": ident_ok}
    entry.update(gate=gate, engine_held_out_loss=round(ce_engine, 4))
    if not all(gate.values()):
        failed = ", ".join(k for k, v in gate.items() if not v)
        log(f"REJECTED ({failed}). The current brain stays as it was. The rejected version is {out}.")
        os.replace(out, out + ".rejected")
        return 2
    if cal_dirs:
        probe = gov.probe_file(probes_dir)
        before = gov.report(cal_dirs, probe, held_out=True)
        after = gov.check_version(out, cal_dirs, probe, os.path.join(probes_dir, f"check-{stamp}"), log=log)
        checks = gov.gate(before, after)
        made_up = lambda r: r["real_wrong_rate"] + r["fake_fabricated_rate"]
        log(f"governor: made up {made_up(before):.3f} -> {made_up(after):.3f}, over-cautious "
            f"{before['real_over_cautious_rate']:.3f} -> {after['real_over_cautious_rate']:.3f}, probe AUROC "
            f"{before.get('probe_auroc', float('nan')):.3f} -> {after.get('probe_auroc', float('nan')):.3f}")
        with open(out + ".governor.json", "w", encoding="utf-8") as f:
            json.dump({"collections": cal_dirs, "probe": probe, "before": before, "after": after, "gate": checks}, f,
                      indent=1)
        if not all(checks.values()):
            failed = ", ".join(k for k, v in checks.items() if not v)
            log(f"REJECTED by the governor ({failed}). The current brain stays as it was.")
            os.replace(out, out + ".rejected")
            return 2
    # the lineage entry with the gate goes into the kept file
    save_cores(out, cores, base, base_header, tok_json, license_text, head.get("lineage", []) + [entry])
    L.set_current(a.brains, out)
    L.after_accept(a, new, docs, fb_path, lessons_path, lsn, stamp)
    log(f"ACCEPTED: {out} is now the live brain ({time.time() - t_all:.0f}s in all).")
    return 0
