"""Learning time: Aegist changes its own weights.

    python -m aegist.learn              # learn from learn/inbox and learn/feedback.jsonl
    python -m aegist.learn --dry-run    # show what would be learned
    python -m aegist.learn --status     # versions and what each one learned
    python -m aegist.learn --rollback   # go back one version

What changes: a working set of switchable cores, namely the ones the core engine
switches on for the new material. The always-on core and the core engine stay
frozen. Frozen routing keeps old lessons addressable across sessions; fixed
routing is in the spirit of hash layers (Roller et al. 2021).

What it learns from:
- files in learn/inbox (text, markdown, code), trained with language-model loss;
- conversation feedback in learn/feedback.jsonl, one JSON object per line:
  {"mode": "normal", "turns": [["user", "..."], ...], "reply": "...",
   "rating": "good" | "bad", "correction": "..."}. A good reply is reinforced. A
  bad reply with a correction trains toward the correction. A bad reply without
  one is skipped;
- prepared lessons in learn/lessons.jsonl (the curriculum's drills, aegist/curriculum.py),
  one chat record per line: {"kind": "chat", "mode": "normal", "turns": [["user", "..."]],
  "target": "...", "source": "..."};
- identity examples (aegist/lessons.py), so it stays Aegist;
- a replay buffer (brains/replay.jsonl) of earlier lessons, so new ones don't
  erase them;
- the hallucination governor (aegist/governor.py): from the question collections the
  current brain answered (probes/data-*), the training split's mistakes become records:
  decline the made-up things it invented answers for, answer the real questions it got
  wrong or declined. Off with --no-calibrate.

Early stopping. Every --check-every steps the drift on all probe conversations is
measured. The weights from the latest check within max-kl * --drift-margin are kept,
and learning stops at the first check over it (the weights go back to that check).

The gate. A new version is kept only if all three hold:
1. its loss on held-out new material dropped;
2. its drift from the frozen base on the probe conversations (mean KL per token)
   stays under --max-kl;
3. it still says it is Aegist.
When governor records were used, the new version also answers the held-out questions
(the Rust engine) and is kept only if it makes things up no more often, is no more
over-cautious, improved on one of the two, and the confidence probe still works on it.

Kept versions are written to brains/history/, and brains/current.txt points at
the live one. Nothing is overwritten, so rollback is just a pointer change.

The 177B brain (hyper-connection lanes) learns in its top layers only and keeps each version as a
cores file over the base brain: see aegist/learn_top.py.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter

import torch

from . import governor as gov
from . import lessons, persona
from .nx import NxFile, load_model, save_model

TEXT_EXT = {".txt", ".md", ".rst", ".py", ".rs", ".js", ".ts", ".c", ".cc", ".cpp", ".h", ".hpp", ".java", ".go",
            ".cs", ".json", ".csv", ".html", ".css", ".toml", ".yaml", ".yml", ".sql", ".sh", ".ps1"}


# --- versions -------------------------------------------------------------------

def current_brain(brains: str) -> str:
    p = os.path.join(brains, "current.txt")
    if os.path.exists(p):
        path = os.path.join(brains, open(p, encoding="utf-8").read().strip())
        if os.path.exists(path):
            return path
    return default_brain(brains)


def default_brain(brains: str) -> str:
    """The 35B brain: aegist-nx-35b.nx, or aegist.nx from before the rename."""
    for name in ("aegist-nx-35b.nx", "aegist.nx"):
        if os.path.exists(os.path.join(brains, name)):
            return os.path.join(brains, name)
    return os.path.join(brains, "aegist-nx-35b.nx")


def set_current(brains: str, path: str):
    rel = os.path.relpath(path, brains).replace("\\", "/")
    tmp = os.path.join(brains, "current.txt.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(rel)
    os.replace(tmp, os.path.join(brains, "current.txt"))


def versions(brains: str) -> list[str]:
    hist = os.path.join(brains, "history")
    out = [default_brain(brains)]
    if os.path.isdir(hist):
        out += sorted(os.path.join(hist, f) for f in os.listdir(hist) if f.endswith(".nx"))
    return [p for p in out if os.path.exists(p)]


# --- material -------------------------------------------------------------------

def inbox_records(inbox: str) -> list[dict]:
    recs = []
    if not os.path.isdir(inbox):
        return recs
    for root, _, files in os.walk(inbox):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in TEXT_EXT:
                continue
            path = os.path.join(root, f)
            text = open(path, encoding="utf-8", errors="replace").read().strip()
            if text:
                recs.append({"kind": "doc", "text": text, "source": os.path.relpath(path, inbox)})
    return recs


def feedback_records(path: str) -> tuple[list[dict], int]:
    recs, skipped = [], 0
    if not os.path.exists(path):
        return recs, skipped
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        fb = json.loads(line)
        target = fb.get("reply") if fb.get("rating") == "good" else fb.get("correction")
        if not target:
            skipped += 1
            continue
        recs.append({"kind": "chat", "mode": fb.get("mode", "normal"), "turns": fb["turns"], "target": target,
                     "source": "feedback"})
    return recs, skipped


def lesson_records(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def identity_records(model_name: str | None = None) -> list[dict]:
    """The identity lessons; with the brain's model name (such as reid1), also what model it is."""
    recs = [{"kind": "chat", "mode": m, "turns": [["user", q]], "target": a, "source": "identity"}
            for m, q, a in lessons.IDENTITY]
    if persona.model_line(model_name):
        for m, q, a in lessons.model_identity(model_name):
            recs.append({"kind": "chat", "mode": m, "turns": [["user", q]], "target": a, "source": "identity"})
    return recs


def replay_records(brains: str, n: int, rng: random.Random) -> list[dict]:
    path = os.path.join(brains, "replay.jsonl")
    if not os.path.exists(path) or n <= 0:
        return []
    recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rng.shuffle(recs)
    return [dict(r, source="replay:" + r.get("source", "")) for r in recs[:n]]


# --- tokens ---------------------------------------------------------------------

def chat_example(tok, rec: dict, seq_len: int):
    turns = [tuple(t) for t in rec["turns"]]
    last_user = next((text for role, text in reversed(turns) if role == "user"), "")
    system = persona.system_prompt(rec.get("mode", "normal"), persona.read_tone(last_user))
    prompt = tok.chat_ids(system, turns, think=False)
    answer = tok.encode_untrusted(rec["target"]) + [tok.im_end]
    ids = prompt + answer
    if len(ids) > seq_len:
        return None
    return ids, [False] * len(prompt) + [True] * len(answer)


def doc_examples(tok, rec: dict, seq_len: int):
    ids = tok.encode_untrusted(rec["text"]) + [tok.tok.token_to_id("<|endoftext|>")]
    return [(ids[i : i + seq_len], [False] + [True] * (len(ids[i : i + seq_len]) - 1))
            for i in range(0, len(ids) - 1, seq_len) if len(ids[i : i + seq_len]) > 1]


def examples_for(tok, recs: list[dict], seq_len: int) -> tuple[list, int]:
    out, too_long = [], 0
    for r in recs:
        if r["kind"] == "doc":
            out += [(ids, sc, r["source"]) for ids, sc in doc_examples(tok, r, seq_len)]
        else:
            ex = chat_example(tok, r, seq_len)
            if ex is None:
                too_long += 1
            else:
                out.append((*ex, r["source"]))
    return out, too_long


def to_tensors(ex, device):
    ids, scored, _ = ex
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([[t if s else -100 for t, s in zip(ids[1:], scored[1:])]], device=device)
    return x, y


# --- measurements ----------------------------------------------------------------

@torch.no_grad()
def mean_ce(model, exs, device) -> float:
    total, n = 0.0, 0
    for ex in exs:
        x, y = to_tensors(ex, device)
        k = int((y != -100).sum())
        if k:
            total += float(model(x, targets=y)["ce"]) * k
            n += k
    return total / max(n, 1)


def _answer_positions(y):
    return (y[0] != -100).nonzero().flatten()


@torch.no_grad()
def probe_references(model, probes, device, k: int = 32):
    """The frozen base's top-k next-token log-probs on each probe reply (cores off)."""
    model.eval()
    model.set_cores(False)
    refs = []
    for ex in probes:
        x, y = to_tensors(ex, device)
        h, _ = model.hidden(x)
        logp = model.logits(h[0, _answer_positions(y)]).float().log_softmax(-1)
        v, i = logp.topk(k, -1)
        refs.append((i, v))
    model.set_cores(True)
    return refs


def probe_kl(model, ex, ref, device, checkpoint=False):
    """KL(base || Aegist) per token on a probe reply, with the base truncated to its top-k
    and renormalised (so the value is never negative)."""
    x, y = to_tensors(ex, device)
    h, _ = model.hidden(x, checkpoint=checkpoint)
    logq = model.logits(h[0, _answer_positions(y)]).float().log_softmax(-1)
    i, v = ref
    p = v.exp()
    p = p / p.sum(-1, keepdim=True)
    return (p * (p.log() - logq.gather(-1, i))).sum(-1).mean()


@torch.no_grad()
def mean_probe_kl(model, probes, refs, device, keep_mode: bool = False) -> float:
    """Mean drift over all probes. keep_mode measures in the current mode; use it while learning,
    where eval mode would skip banks whose silent flag is stale."""
    if not keep_mode:
        model.eval()
    return sum(float(probe_kl(model, ex, r, device)) for ex, r in zip(probes, refs)) / max(len(probes), 1)


def identity_check(model, tok, device) -> tuple[bool, str]:
    from .chat import Aegist, GenConfig, Session

    a = Aegist(model, tok, device=device, gen=GenConfig(temperature=0.0, max_new_tokens=32), max_context=1024)
    s = Session(mode="normal", think="off")
    reply = "".join(x for kind, x in a.reply(s, "Who are you?", seed=0) if kind == "answer")
    return "aegist" in reply.lower(), reply.strip()


@torch.no_grad()
def route_counts(model, exs, device) -> torch.Tensor:
    model.eval()
    before = model.core_usage().clone()
    for ex in exs:
        x, _ = to_tensors(ex, device)
        model(x, last_only=True)
    return (model.core_usage() - before).cpu()


# --- the session ------------------------------------------------------------------

def log(msg: str):
    print(msg, flush=True)


def learn(a) -> int:
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inbox_dir = os.path.join(a.learn_dir, "inbox")
    fb_path = os.path.join(a.learn_dir, "feedback.jsonl")
    lessons_path = os.path.join(a.learn_dir, "lessons.jsonl")

    docs = inbox_records(inbox_dir)
    fbs, fb_skipped = feedback_records(fb_path)
    lsn = lesson_records(lessons_path)
    brain = current_brain(a.brains)
    with NxFile(brain) as nx:
        lanes = (nx.header.get("config") or {}).get("hc_count", 1) > 1
    if lanes:
        from .learn_top import learn_top
        return learn_top(a, brain)
    cal_dirs, cal = [], []
    if a.calibrate:
        cal_dirs = gov.collections_for(a.probes, brain)
        if cal_dirs:
            cal, why = gov.training_records(cal_dirs)
            log(f"governor: {len(cal)} records from {len(cal_dirs)} question collections ({why})")
        else:
            log("governor: no question collection answered by this brain yet "
                "(python -m aegist.confidence collect --bank all); nothing to calibrate from")
    new = docs + fbs + lsn + cal
    if not new:
        log(f"Nothing to learn: put files in {inbox_dir} or rate replies in the launcher first.")
        return 1
    log(f"brain: {brain}")
    log(f"new material: {len(docs)} files, {len(fbs)} feedback items ({fb_skipped} bad ratings without a "
        f"correction skipped), {len(lsn)} prepared lessons, {len(cal)} governor records")

    model, tok, nx = load_model(brain, device=device, slots=a.slots, cores_in_ram=True)
    hdr = dict(nx.header)
    persona.MODEL_NAME = hdr.get("name")  # the system prompt names the model, as in chat
    tok_json = nx.blob("tokenizer.json").decode("utf-8")
    license_text = nx.blob("LICENSE").decode("utf-8")

    new_ex, long_new = examples_for(tok, new, a.seq_len)
    ranked = sorted(new_ex, key=lambda e: hashlib.sha256((e[2] + str(e[0][:32])).encode()).digest())
    n_held = max(1, len(ranked) // 10) if len(ranked) >= 5 else 0
    held, train_new = ranked[:n_held], ranked[n_held:]
    if not held:  # too little material to hold any back
        held, train_new, held_note = new_ex, new_ex, "no held-out split (too little material); measured on training data"
    else:
        held_note = f"{len(held)} held-out examples"
    replay = examples_for(tok, replay_records(a.brains, max(len(train_new), 8), rng), a.seq_len)[0]
    ident = examples_for(tok, identity_records(hdr.get("name")), a.seq_len)[0]
    train = train_new + replay + ident
    probes = examples_for(tok, [{"kind": "chat", "mode": m, "turns": [["user", q]], "target": r, "source": "probe"}
                                for m, q, r in lessons.PROBES], a.seq_len)[0]
    n_tok = sum(len(e[0]) for e in train)
    log(f"examples: {len(train_new)} new + {len(replay)} replay + {len(ident)} identity = {len(train)} "
        f"({n_tok} tokens); {held_note}; {long_new} chats longer than {a.seq_len} tokens skipped")
    if a.dry_run:
        for e in train_new[:10]:
            log(f"  {e[2]}: {len(e[0])} tokens")
        return 0

    t0 = time.time()
    refs = probe_references(model, probes, device)
    kl_before = mean_probe_kl(model, probes, refs, device)
    ce_before = mean_ce(model, held, device)
    log(f"before: held-out loss {ce_before:.4f}, probe drift {kl_before:.4f} nats/token ({time.time() - t0:.0f}s)")

    counts = route_counts(model, train_new[:256], device)
    work = [[int(c) for c in row.topk(a.working_cores).indices if row[c] > 0] for row in counts]
    covered = sum(int(counts[i, w].sum()) for i, w in enumerate(work)) / max(int(counts.sum()), 1)
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for b, w in zip(model.blocks, work):  # every bank enters learning mode, even with nothing to train
        params += [p for p in b.cores.begin_learning(w, device) if p.numel()]
    n_params = sum(p.numel() for p in params)
    log(f"working set: {sum(map(len, work))} cores ({n_params / 1e6:.1f}M params) covering {covered:.0%} of the "
        f"core switch-ons for the new material")

    accum = a.accum
    steps = min(a.max_steps, max(1, math.ceil(len(train) * a.epochs / accum)))
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    warm = max(1, steps // 20)
    order: list = []
    model.train()
    curve = []
    # Early stopping on drift: every --check-every steps measure drift on all probes; keep the
    # latest weights still within the drift target and stop once a check goes over it.
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
            x, y = to_tensors(order.pop(), device)
            ce = model(x, targets=y, checkpoint=True)["ce"]
            (ce / accum).backward()
            ce_sum += float(ce.detach()) / accum
        kl = torch.zeros(())
        if a.kl_weight > 0 and probes:
            j = step % len(probes)
            kl = probe_kl(model, probes[j], refs[j], device, checkpoint=True)
            (a.kl_weight * kl).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        curve.append(round(ce_sum, 4))
        check = probes and ((step + 1) % a.check_every == 0 or step == steps - 1)
        if check:
            drift = mean_probe_kl(model, probes, refs, device, keep_mode=True)
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
    for b in model.blocks:
        b.cores.end_learning(keep=True)
    model.eval()

    ce_after = mean_ce(model, held, device)
    kl_after = mean_probe_kl(model, probes, refs, device)
    ident_ok, ident_reply = identity_check(model, tok, device)
    learned = ce_after < ce_before * (1 - a.min_gain)
    stays_close = kl_after <= a.max_kl
    log(f"after:  held-out loss {ce_after:.4f} (was {ce_before:.4f}), probe drift {kl_after:.4f} (limit {a.max_kl})")
    log(f"identity: {'ok' if ident_ok else 'FAILED'}: {ident_reply!r}")
    gate = {"learned": learned, "stays_close_to_base": stays_close, "still_aegist": ident_ok}
    entry = {"event": "learned", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "from": os.path.basename(brain),
             "material": {"files": [r["source"] for r in docs], "feedback": len(fbs), "governor": len(cal),
                          "lessons": dict(sorted(Counter(r.get("source", "") for r in lsn).items())),
                          "replay": len(replay),
                          "identity": len(ident), "tokens": n_tok},
             "working_cores": sum(map(len, work)), "working_coverage": round(covered, 3), "steps": steps,
             "steps_run": len(curve), "kept_step": kept, "lr": a.lr,
             "held_out_loss": [round(ce_before, 4), round(ce_after, 4)],
             "probe_kl": [round(kl_before, 4), round(kl_after, 4)], "gate": gate, "loss_curve": curve,
             "drift_curve": drift_curve}
    if not all(gate.values()):
        failed = ", ".join(k for k, v in gate.items() if not v)
        log(f"REJECTED ({failed}). The current brain stays as it was; the material is left in place.")
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = hdr.get("name", "Aegist")
    out = os.path.join(a.brains, "history", f"{name.lower().replace(' ', '-')}-{stamp}.nx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_model(out, model, tok_json, notice=hdr.get("notice", ""), license_text=license_text,
               persona=hdr.get("persona"), generation=hdr.get("generation"),
               lineage=hdr.get("lineage", []) + [entry], extra={"version": hdr.get("version", 1), "name": name})
    with NxFile(out) as check:
        if not all(check.verify().values()):
            log("written file failed its checksums; keeping the current brain")
            return 3
    if cal_dirs:
        # the Rust engine needs the GPU memory the Python model holds
        del model, params, b, refs
        gc.collect()
        torch.cuda.empty_cache()
        probe = gov.probe_file(a.probes)
        before = gov.report(cal_dirs, probe, held_out=True)
        after = gov.check_version(out, cal_dirs, probe, os.path.join(a.probes, f"check-{stamp}"), log=log)
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
            os.remove(out)
            return 2
    set_current(a.brains, out)
    after_accept(a, new, docs, fb_path, lessons_path, lsn, stamp)
    prune(a.brains, a.keep)
    log(f"ACCEPTED: {out} is now the live brain ({time.time() - t0:.0f}s of learning).")
    return 0


def after_accept(a, new, docs, fb_path, lessons_path, lsn, stamp):
    """The material of an accepted session joins the replay buffer and moves to learn/learned/."""
    inbox_dir = os.path.join(a.learn_dir, "inbox")
    with open(os.path.join(a.brains, "replay.jsonl"), "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    done = os.path.join(a.learn_dir, "learned", stamp)
    os.makedirs(done, exist_ok=True)
    for r in docs:
        dst = os.path.join(done, r["source"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(os.path.join(inbox_dir, r["source"]), dst)
    if os.path.exists(fb_path):
        shutil.move(fb_path, os.path.join(done, "feedback.jsonl"))
    if lsn:
        shutil.move(lessons_path, os.path.join(done, "lessons.jsonl"))


def is_cores_file(path: str) -> bool:
    with NxFile(path) as nx:
        return "base" in nx.header


def prune(brains: str, keep: int):
    """Old full copies (35B versions) beyond `keep`; cores files are small and are all kept."""
    cur = os.path.abspath(current_brain(brains))
    hist = [v for v in versions(brains)
            if os.path.dirname(os.path.abspath(v)).endswith("history") and not is_cores_file(v)]
    for old in hist[:-keep] if keep > 0 else []:
        if os.path.abspath(old) != cur:
            try:
                os.remove(old)
            except OSError:
                pass  # in use by a running Aegist; removed next time


def status(a) -> int:
    cur = os.path.abspath(current_brain(a.brains))
    for v in versions(a.brains):
        with NxFile(v) as nx:
            last = nx.header.get("lineage", [{}])[-1]
        mark = "*" if os.path.abspath(v) == cur else " "
        what = last.get("event", "?")
        if what == "learned":
            m = last["material"]
            what += (f": {len(m['files'])} files, {m['feedback']} feedback; held-out loss "
                     f"{last['held_out_loss'][0]} -> {last['held_out_loss'][1]}, drift {last['probe_kl'][1]}")
        log(f"{mark} {os.path.relpath(v, a.brains)}  {os.path.getsize(v) / 1e9:.2f} GB  {what}")
    return 0


def rollback(a) -> int:
    cur = os.path.abspath(current_brain(a.brains))
    with NxFile(cur) as nx:
        base = (nx.header.get("base") or {}).get("path")
    if base:  # a cores file: back to the one before it over the same base, or to the base
        same = []
        for v in versions(a.brains):
            with NxFile(v) as nx:
                if (nx.header.get("base") or {}).get("path") == base:
                    same.append(os.path.abspath(v))
        i = same.index(cur) if cur in same else len(same)
        to = same[i - 1] if i > 0 else base
        set_current(a.brains, to)
        log(f"Rolled back to {to}.")
        return 0
    vs = [os.path.abspath(v) for v in versions(a.brains) if not is_cores_file(v)]
    i = vs.index(cur) if cur in vs else 0
    if i == 0:
        log("Already on the original brain; nothing to roll back to.")
        return 1
    set_current(a.brains, vs[i - 1])
    log(f"Rolled back to {os.path.relpath(vs[i - 1], a.brains)}.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aegist.learn", description="Aegist learning time")
    ap.add_argument("--brains", default="brains")
    ap.add_argument("--learn-dir", default="learn")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq-len", type=int, default=768)
    ap.add_argument("--working-cores", type=int, default=8, help="cores per layer trained this session")
    ap.add_argument("--top-layers", type=int, default=3, help="177B brain: how many top layers learn")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--kl-weight", type=float, default=4.0)
    ap.add_argument("--max-kl", type=float, default=0.1, help="gate: max probe drift from the base, nats/token")
    ap.add_argument("--check-every", type=int, default=2, help="steps between drift checks on all probes")
    ap.add_argument("--drift-margin", type=float, default=0.9,
                    help="early stopping keeps weights with drift <= max-kl * this (headroom for the gate)")
    ap.add_argument("--min-gain", type=float, default=0.01, help="gate: required relative drop in held-out loss")
    ap.add_argument("--keep", type=int, default=5, help="learned versions kept in brains/history")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="177B brain: the GPU that trains (cuda:0, cuda:1; auto: most free memory)")
    ap.add_argument("--probes", default="probes", help="question collections and the confidence probe")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                    help="leave out the hallucination governor's records and gate")
    a = ap.parse_args(argv)
    if a.status:
        return status(a)
    if a.rollback:
        return rollback(a)
    return learn(a)


if __name__ == "__main__":
    sys.exit(main())
