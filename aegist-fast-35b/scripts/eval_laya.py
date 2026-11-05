"""How good a judge is Laya? Checkable questions only.

    python -I scripts/eval_laya.py --laya D:/aegist/models/laya --out D:/aegist/laya/eval.json

1. claims: the confidence probe's 80 yes/no claims (40 true, 20 myths, 20 invented mechanisms),
   asked three ways; AUROC needs no threshold, accuracy uses P(true) >= 0.5.
2. capitals: the capital of each of the probe's 60 countries among 4 cities (chance 25%).
3. reading: made-up villages, so only the passage can give the answer. Each passage describes two
   villages; a question asks about one, and the other village's value is among the 4 options
   (chance 25%).
Writes every answer and the summary to --out. --aegist-brain also asks a brain the reading and claim
questions through the Rust engine, for comparison (one process; greedy, thinking off).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aegist.confidence import CAPITALS, FALSE_CLAIMS, TRUE_CLAIMS  # noqa: E402

EXE = os.path.join(ROOT, "aegist_rs", "target", "release", "aegist.exe" if os.name == "nt" else "aegist")

SYL = ["var", "ol", "sk", "brin", "tam", "esk", "lu", "dor", "fen", "ka", "mir", "zo", "pel", "ruth", "an", "gel"]
PEOPLE = ["Ada Morrow", "Tomas Venn", "Ilse Quarry", "Rafi Okon", "Mara Lindqvist", "Jun Arakawa", "Odile Fasse",
          "Pieter Hal", "Nia Brookes", "Elan Sorell", "Greta Moll", "Kofi Ansah"]
RIVERS = ["Olm", "Sarre", "Tavin", "Brisk", "Uldra", "Mennet", "Coral", "Fenwick"]
MONTHS = ["January", "March", "April", "June", "August", "September", "October", "December"]


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def village(rng, used):
    while True:
        n = (rng.choice(SYL) + rng.choice(SYL)).capitalize()
        if n not in used:
            used.add(n)
            return n


def reading_items(n_passages=10, seed=0):
    rng = random.Random(seed)
    used, items = set(), []
    facts = {  # field: (question about village V, value pool)
        "year": ("In which year did the bakery in {v} open?", [str(y) for y in range(1890, 1990)]),
        "baker": ("Who runs the bakery in {v}?", PEOPLE),
        "river": ("Which river does {v} lie on?", RIVERS),
        "bridges": ("How many bridges does {v} have?", [str(i) for i in range(2, 15)]),
        "month": ("In which month is the festival of {v} held?", MONTHS),
    }
    for _ in range(n_passages):
        vs = [village(rng, used), village(rng, used)]
        val = {v: {f: rng.choice(pool) for f, (_, pool) in facts.items()} for v in vs}
        for f, (_, pool) in facts.items():  # the two villages differ on every fact
            while val[vs[1]][f] == val[vs[0]][f]:
                val[vs[1]][f] = rng.choice(pool)
        a, b = vs
        text = (f"The village of {a} lies on the {val[a]['river']} river, and its neighbour {b} lies on the "
                f"{val[b]['river']}. The bakery in {a} opened in {val[a]['year']} and is run by {val[a]['baker']}; "
                f"the one in {b} opened in {val[b]['year']} and is run by {val[b]['baker']}. {a} has "
                f"{val[a]['bridges']} bridges and holds its festival in {val[a]['month']}, while {b} has "
                f"{val[b]['bridges']} bridges and celebrates in {val[b]['month']}.")
        for f, (q, pool) in facts.items():
            v = rng.choice(vs)
            other = vs[1] if v == vs[0] else vs[0]
            gold, lure = val[v][f], val[other][f]
            opts = [gold, lure] + rng.sample([x for x in pool if x not in (gold, lure)], 2)
            rng.shuffle(opts)
            items.append({"passage": text, "q": q.format(v=v), "options": opts, "gold": gold, "lure": lure})
    return items


def capital_items(seed=0):
    rng = random.Random(seed + 1)
    cities = sorted(set(CAPITALS.values()))
    out = []
    for country, city in CAPITALS.items():
        opts = [city] + rng.sample([c for c in cities if c != city], 3)
        rng.shuffle(opts)
        out.append({"q": f"Which city is the capital of {country}?", "options": opts, "gold": city})
    return out


def run_laya(laya_dir, out):
    from aegist.laya import Laya

    t = time.time()
    j = Laya(laya_dir)
    res = {"load_seconds": round(time.time() - t, 1), "device": j.device}
    claims = [(c, True) for c in TRUE_CLAIMS] + [(c, False) for c in FALSE_CLAIMS]
    labels = [y for _, y in claims]
    ways = {
        "noul, claim as context": [("noul", "Is this statement true?", None, c) for c, _ in claims],
        "noul, claim as the question": [("noul", c, None, "") for c, _ in claims],
        "choice true/false": [("choice", "Is this statement true or false?", {"true": None, "false": None}, c)
                              for c, _ in claims],
    }
    res["claims"] = {}
    for name, items in ways.items():
        t = time.time()
        r = j.run(items)
        dt = time.time() - t
        p_true = [p[1] if it[0] == "noul" else p[0] for (p, _), it in zip(r, items)]
        acc = sum((p >= 0.5) == y for p, y in zip(p_true, labels)) / len(labels)
        res["claims"][name] = {"auroc": round(auroc(p_true, labels), 3), "accuracy": round(acc, 3),
                               "mean_p_true": round(sum(p_true) / len(p_true), 3),
                               "ms_per_item": round(1000 * dt / len(items), 1)}
        out.setdefault("claim_answers", {})[name] = [{"claim": c, "true": y, "p_true": round(p, 4)}
                                                     for (c, y), p in zip(claims, p_true)]
    for name, items in (("capitals", capital_items()), ("reading", reading_items())):
        t = time.time()
        r = j.run([("choice", it["q"], {o: None for o in it["options"]}, it.get("passage", "")) for it in items])
        dt = time.time() - t
        picks = [it["options"][max(range(len(p)), key=p.__getitem__)] for (p, _), it in zip(r, items)]
        right = [pk == it["gold"] for pk, it in zip(picks, items)]
        res[name] = {"accuracy": round(sum(right) / len(items), 3), "n": len(items), "chance": 0.25,
                     "mean_confidence": round(sum(max(p) for p, _ in r) / len(r), 3),
                     "ms_per_item": round(1000 * dt / len(items), 1)}
        if name == "reading":
            res[name]["picked_the_other_village"] = sum(pk == it["lure"] for pk, it in zip(picks, items))
        out[f"{name}_answers"] = [{"q": it["q"], "options": it["options"], "gold": it["gold"], "laya": pk,
                                   "p": [round(x, 4) for x in p]} for it, pk, (p, _) in zip(items, picks, r)]
    return res


def run_aegist(brain, out, context=4096):
    """The same reading and claim questions, answered by an Aegist brain (greedy, thinking off)."""
    from aegist import persona
    from aegist.nx import NxFile
    from aegist.tokenizer import AegistTokenizer

    with NxFile(brain) as nx:
        tok = AegistTokenizer.from_json_str(nx.blob("tokenizer.json").decode("utf-8"))
    reading = reading_items()
    claims = [(c, True) for c in TRUE_CLAIMS] + [(c, False) for c in FALSE_CLAIMS]
    qs = [f"{it['passage']}\n\n{it['q']} Choose one of: {', '.join(it['options'])}. Reply with the answer only."
          for it in reading]
    qs += [f"Is it true that {c}? Start your answer with Yes or No." for c, _ in claims]
    system = persona.system_prompt("normal")
    prompts = [tok.chat_ids(system, [("user", q)], think=False) for q in qs]
    tmp = tempfile.mkdtemp()
    inp, o = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
    with open(inp, "w", encoding="utf-8") as f:
        json.dump({"prompts": prompts, "max_new": 24, "samples": 0}, f)
    t = time.time()
    r = subprocess.run([EXE, "probe-data", brain, inp, o, "--context", str(context)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"aegist probe-data failed: {r.stderr[-2000:]}")
    dt = time.time() - t
    answers = [tok.decode(a["greedy"], skip_special=True).strip() for a in json.load(open(o, encoding="utf-8"))]
    rd = answers[:len(reading)]
    right = [a.lower().startswith(it["gold"].lower()) or a.lower().rstrip(".") == it["gold"].lower()
             for a, it in zip(rd, reading)]
    cl = answers[len(reading):]
    said_yes = [a.lower().startswith("yes") for a in cl]
    res = {"brain": os.path.basename(brain), "seconds": round(dt),
           "reading": {"accuracy": round(sum(right) / len(reading), 3), "n": len(reading)},
           "claims": {"accuracy": round(sum(s == y for s, (_, y) in zip(said_yes, claims)) / len(claims), 3)}}
    out["aegist_answers"] = {"reading": rd, "claims": cl}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya", default="D:/aegist/models/laya")
    ap.add_argument("--aegist-brain", default="")
    ap.add_argument("--out", default="D:/aegist/laya/eval.json")
    a = ap.parse_args()
    out = {}
    summary = {"laya": run_laya(a.laya, out)}
    print(json.dumps(summary, indent=1), flush=True)
    if a.aegist_brain:
        import torch

        torch.cuda.empty_cache()  # the engine wants the GPU to itself
        summary["aegist"] = run_aegist(a.aegist_brain, out)
        print(json.dumps(summary["aegist"], indent=1), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, **out}, f, indent=1)


if __name__ == "__main__":
    main()
