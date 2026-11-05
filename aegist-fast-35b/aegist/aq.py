"""Aegist's AQ test (artificial qualification): the live brain answers questions whose answers
can be checked, and gets a score out of 100.

AQ is Aegist's own measure, made from the confidence probe's question sets (aegist/confidence.py).
It is not an IQ: it is not compared with people or with other AIs, only with earlier runs. Six
sections, each the mean accuracy over its kinds of question (so a quick run and a full run
measure the same thing), and AQ is 100 x the mean of the sections:

  knowledge       capitals, atomic numbers, authors, years
  maths           multiplication up to 6 x 5 digits, digit sums
  letters         letter counts, spelling backwards
  honesty         questions about made-up things: right only when it says they do not exist
  judgement       true facts, myths and invented mechanisms: yes or no
  self-knowledge  do the unsure marks land on the wrong answers? (needs the brain's probe)

Self-knowledge is balanced accuracy: (wrong answers marked + right answers left unmarked) / 2,
counting answers the probe does not judge as unmarked. The fact lists (capitals, elements,
authors, years, claims) are the ones the probe was trained on, so self-knowledge is optimistic
there; numbers, letters and made-up names change with --seed.

By default the questions go through the whole chat pipeline (`aegist exam`), as a user would ask
them: the reading core, the tools (exact arithmetic and spelling), the unsure marks and the second
look at unsure facts. An answer is marked unsure when its final reply carries a mark. --no-tools,
--no-web (the Wikipedia search tool only) and --no-second-look switch those parts off, to see what each is worth; --raw answers straight from the
brain (no tools, no reading core) and marks with the probe, as the first AQ runs did. Each run is
compared with the last run of the same setup only.

Run:  aegist aq [--full] [--per-kind N] [--seed S] [--brain FILE] [--no-tools] [--no-web] [--no-second-look] [--raw]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time

import torch

from . import confidence as C
from .nx import NxFile

SECTIONS = {
    "knowledge": ("capital", "element", "author", "year"),
    "maths": ("multiply", "digit sum"),
    "letters": ("letters", "reverse"),
    "honesty": ("fake",),
    "judgement": ("claim",),
}


def section_of(kind: str) -> str:
    for name, prefixes in SECTIONS.items():
        if any(kind == p or kind.startswith(p + " ") for p in prefixes):
            return name
    raise ValueError(f"no section for {kind!r}")


def live_brain(brains: str) -> str:
    p = os.path.join(brains, "current.txt")
    if os.path.exists(p):
        path = os.path.join(brains, open(p, encoding="utf-8").read().strip())
        if os.path.exists(path):
            return path
    from .learn import default_brain
    return default_brain(brains)


def find_probe(probes: str, brain: str) -> str | None:
    """As aegist_rs/src/probe.rs finds it: the brain's own probe first (a cores file uses its base
    brain's), then the shared one."""
    stems = [os.path.splitext(os.path.basename(brain))[0]]
    try:
        with NxFile(brain) as nx:
            base = (nx.header.get("base") or {}).get("path")
        if base:
            stems.append(os.path.splitext(os.path.basename(base))[0])
    except Exception:
        pass
    for d in [os.path.join(probes, s) for s in stems] + [probes]:
        p = os.path.join(d, "current.txt")
        if os.path.exists(p):
            path = os.path.join(d, open(p, encoding="utf-8").read().strip())
            if os.path.exists(path):
                return path
    return None


def questions(seed: int, per_kind: int | None) -> list[dict]:
    items = C.question_bank(seed) + C.hard_bank(seed) + C.claim_bank(seed)
    if per_kind:
        by_kind: dict[str, list[dict]] = {}
        for it in items:
            by_kind.setdefault(it["kind"], []).append(it)
        rng = random.Random(seed)
        items = [it for k in sorted(by_kind) for it in rng.sample(by_kind[k], min(per_kind, len(by_kind[k])))]
    return items


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def score(rows: list[dict], marks: bool) -> tuple[dict, dict, dict | None]:
    kinds: dict[str, list[bool]] = {}
    for r in rows:
        kinds.setdefault(r["kind"], []).append(r["right"])
    sections = {s: mean(mean(v) for k, v in kinds.items() if section_of(k) == s) for s in SECTIONS
                if any(section_of(k) == s for k in kinds)}
    unsure = None
    if marks:
        wrong = [r for r in rows if not r["right"] and "marked_unsure" in r]
        right = [r for r in rows if r["right"] and "marked_unsure" in r]
        caught = mean(r["marked_unsure"] for r in wrong) if wrong else None
        spared = mean(not r["marked_unsure"] for r in right) if right else None
        sections["self-knowledge"] = mean(v for v in (caught, spared) if v is not None)
        unsure = {"wrong_answers_marked": f"{sum(r['marked_unsure'] for r in wrong)}/{len(wrong)}",
                  "right_answers_marked": f"{sum(r['marked_unsure'] for r in right)}/{len(right)}"}
    return kinds, sections, unsure


def run_chat(brain: str, probes: str, out_dir: str, *, seed: int, per_kind: int | None, tools: bool = True,
             second_look: bool = True, web: bool = True, log=print) -> dict:
    """The questions through `aegist exam`: each in a new conversation, as the user would ask it."""
    items = questions(seed, per_kind)
    probe_path = find_probe(probes, brain)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = os.path.join(out_dir, f"aq-{stamp}")
    os.makedirs(work, exist_ok=True)
    setup = "chat" + ("" if tools else ", no tools") + ("" if web or not tools else ", no web") + ("" if second_look else ", no second look")
    log(f"AQ test: {len(items)} questions for {os.path.basename(brain)} through chat ({setup}; "
        f"{'all of them' if not per_kind else f'up to {per_kind} of each kind'}, seed {seed})")
    inp, outp = os.path.join(work, "exam.json"), os.path.join(work, "answers.json")
    json.dump({"prompts": [it["q"] + C.ASK for it in items], "think": "off", "seed": seed, "max_new": 200},
              open(inp, "w", encoding="utf-8"))
    cmd = [C.exe_path(), "exam", inp, outp, "--brain", brain]
    cmd += [] if tools else ["--no-tools"]
    cmd += [] if second_look else ["--no-second-look"]
    t0 = time.time()
    r = subprocess.run(cmd, env=dict(os.environ, **({} if web else {"AEGIST_WEB": "off"})))  # its progress goes to our stderr
    if r.returncode != 0:
        raise RuntimeError(f"aegist exam exited with {r.returncode}")
    log(f"done in {time.time() - t0:.0f}s")
    got = json.load(open(outp, encoding="utf-8"))
    marks = probe_path is not None
    rows = []
    for it, g in zip(items, got):
        row = {"kind": it["kind"], "section": section_of(it["kind"]), "q": it["q"], "answer": g["answer"],
               "right": C.is_correct(g["answer"], it), "tools": g["tools"], "second_look": g["second_look"],
               "reading": g["kind"]}
        if marks:
            row["marked_unsure"] = bool(g["final_marks"])
        rows.append(row)
    kinds, sections, unsure = score(rows, marks)
    res = {"aq": round(100 * mean(sections.values()), 1), "brain": os.path.abspath(brain),
           "probe": probe_path if marks else None, "setup": setup, "seed": seed, "per_kind": per_kind,
           "questions": len(rows), "right": sum(r["right"] for r in rows),
           "sections": {k: round(100 * v, 1) for k, v in sections.items()},
           "kinds": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(kinds.items())},
           "tool_use": sum(bool(r["tools"]) for r in rows), "second_looks": sum(r["second_look"] for r in rows),
           "seconds": round(time.time() - t0), "when": stamp, "rows": rows}
    if unsure:
        res["unsure_marks"] = unsure
    return res


def run(brain: str, probes: str, out_dir: str, *, seed: int, per_kind: int | None, log=print) -> dict:
    """The first AQ runs: answers straight from the brain (`aegist probe-data`), marked by the probe."""
    items = questions(seed, per_kind)
    probe_path = find_probe(probes, brain)
    probe = C.load_probe(probe_path) if probe_path else None
    layer = probe[0]["layer"] if probe else None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = os.path.join(out_dir, f"aq-{stamp}")
    log(f"AQ test: {len(items)} questions for {os.path.basename(brain)} "
        f"({'all of them' if not per_kind else f'up to {per_kind} of each kind'}, seed {seed})")
    C.answer_items(brain, items, work, taps=[layer] if layer is not None else [C.DEFAULT_TAPS[0]],
                   samples=0, max_new=40, seed=seed, log=log)
    got, answers, pooled, stats, y, _, _ = C.load_collection(work)
    by_group = {it["group"]: (a, float(lab), i) for i, (it, a, lab) in enumerate(zip(got, answers, y.tolist()))}

    flags = None
    if probe is not None:
        cfg, m = probe
        x = C.features(pooled, stats, 0 if layer is not None else None)
        if m[0].numel() != x.shape[1]:
            log(f"the probe {probe_path} was made for another brain; self-knowledge is not scored")
            probe = None
        else:
            p = C.predict(m, x)
            judged = C.ood_score(m, x) <= cfg["ood_max"] if cfg.get("ood_max") is not None else torch.ones(len(p), dtype=torch.bool)
            flags = ((p >= cfg["threshold"]) & judged).tolist()
            p_wrong = p.tolist()

    rows = []
    for i, it in enumerate(items):
        a, lab, j = by_group.get(f"{work}#{i}", ("", 1.0, None))  # an empty answer counts as wrong
        row = {"kind": it["kind"], "section": section_of(it["kind"]), "q": it["q"], "answer": a, "right": lab == 0.0}
        if flags is not None and j is not None:
            row["marked_unsure"], row["p_wrong"] = flags[j], round(p_wrong[j], 3)
        rows.append(row)

    kinds, sections, unsure = score(rows, flags is not None)
    aq = 100 * mean(sections.values())
    res = {"aq": round(aq, 1), "brain": os.path.abspath(brain), "probe": probe_path if flags is not None else None,
           "setup": "raw", "seed": seed, "per_kind": per_kind, "questions": len(rows), "right": sum(r["right"] for r in rows),
           "sections": {k: round(100 * v, 1) for k, v in sections.items()},
           "kinds": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(kinds.items())},
           "when": stamp, "rows": rows}
    if unsure:
        res["unsure_marks"] = unsure
    return res


def report(res: dict, previous: dict | None) -> str:
    lines = [f"AQ {res['aq']:.0f} / 100   ({res['right']} of {res['questions']} answers right, "
             f"{os.path.basename(res['brain'])}, {res['setup']}, seed {res['seed']})"]
    if previous:
        lines[0] += f"   last run like this: {previous['aq']:.0f} ({previous['when']})"
    lines.append("")
    for s, v in res["sections"].items():
        bar = "#" * round(v / 5)
        lines.append(f"  {s:15} {v:5.1f}  {bar}")
    if "self-knowledge" not in res["sections"]:
        lines.append(f"  {'self-knowledge':15}   -    (this brain has no confidence probe yet)")
    if "unsure_marks" in res:
        u = res["unsure_marks"]
        lines.append(f"\n  unsure marks: {u['wrong_answers_marked']} wrong answers marked, "
                     f"{u['right_answers_marked']} right answers marked")
    if "tool_use" in res:
        lines.append(f"  tools used on {res['tool_use']} questions, a second look at {res['second_looks']}")
    lines.append("\n  by kind: " + ", ".join(f"{k} {v}" for k, v in res["kinds"].items()))
    wrong = [r for r in res["rows"] if not r["right"]]
    if wrong:
        lines.append("\n  some answers it got wrong:")
        for r in wrong[:8]:
            lines.append(f"    {r['q'][:70]}\n      -> {r['answer'][:110]}")
    lines.append("\nAQ is Aegist's own score on checkable questions. It is not an IQ and is not compared with people "
                 "or other AIs, only with earlier runs. Full answers: " + res["file"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aegist aq", description="Aegist's AQ test (artificial qualification).")
    ap.add_argument("--brain", help="the brain to test (default: the live one)")
    ap.add_argument("--brains", default="brains")
    ap.add_argument("--probes", default="probes")
    ap.add_argument("--out", default="aq")
    ap.add_argument("--full", action="store_true", help="every question (about 720), not a sample")
    ap.add_argument("--per-kind", type=int, default=6, help="questions per kind in a quick run (default 6)")
    ap.add_argument("--seed", type=int, default=1, help="changes the numbers, words and made-up names")
    ap.add_argument("--no-tools", action="store_true", help="no exact arithmetic or spelling tools")
    ap.add_argument("--no-web", action="store_true", help="no Wikipedia search tool (the others stay)")
    ap.add_argument("--no-second-look", action="store_true", help="no second look at unsure facts")
    ap.add_argument("--raw", action="store_true", help="answers straight from the brain, as the first AQ runs")
    a = ap.parse_args(argv)
    brain = a.brain or live_brain(a.brains)
    os.makedirs(a.out, exist_ok=True)
    hist = os.path.join(a.out, "history.jsonl")
    per_kind = None if a.full else a.per_kind
    if a.raw:
        res = run(brain, a.probes, a.out, seed=a.seed, per_kind=per_kind)
    else:
        res = run_chat(brain, a.probes, a.out, seed=a.seed, per_kind=per_kind, tools=not a.no_tools,
                       second_look=not a.no_second_look, web=not a.no_web)
    previous = None
    if os.path.exists(hist):
        same = [h for h in (json.loads(l) for l in open(hist, encoding="utf-8") if l.strip())
                if (h.get("setup", "raw"), h["brain"], h["seed"], h["per_kind"])
                == (res["setup"], res["brain"], res["seed"], res["per_kind"])]
        previous = same[-1] if same else None
    res["file"] = os.path.abspath(os.path.join(a.out, f"aq-{res['when']}.json"))
    json.dump(res, open(res["file"], "w", encoding="utf-8"), indent=1)
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in res.items() if k != "rows"}) + "\n")
    text = report(res, previous)
    open(res["file"][:-5] + ".txt", "w", encoding="utf-8").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
