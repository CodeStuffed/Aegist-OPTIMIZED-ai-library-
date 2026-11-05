"""Can Laya tell fact prompts from creative ones? Compared with a plain keyword rule.

    python -I scripts/eval_prompt_reader.py --laya D:/aegist/models/laya --out D:/aegist/laya/prompt_reader.json

Every Laya framing is scored on DEV (aegist/prompt_kinds.py); the best one on DEV is then measured
once on TEST, so the TEST number is not tuned on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aegist.prompt_kinds import DEV, FRESH, KINDS, TEST  # noqa: E402

DESC = {
    "fact": "a fact or an explanation about the real world, with a correct answer",
    "creative": "something new and made up, like a story, poem, name, invention or idea",
    "other": "help with code or the user's own text, advice, an opinion, or small talk",
}


def rows(split):
    return [(p, k) for k in KINDS for p in split[k]]


def framings():
    opts = {k: DESC[k] for k in KINDS}
    bare = {k: None for k in KINDS}
    return {
        "request as context, described options": lambda p: ("choice", "What does this request ask for?", opts, p),
        "request as the question, described options": lambda p: ("choice", f"What does this request ask for? {p}", opts, ""),
        "request as context, bare options": lambda p: ("choice", "Is this request asking for a fact, something creative, or other help?", bare, p),
    }


def score(picks, gold):
    acc = sum(a == b for a, b in zip(picks, gold)) / len(gold)
    per = {k: round(sum(a == b == k for a, b in zip(picks, gold)) / sum(g == k for g in gold), 3) for k in KINDS}
    # the costly mistakes: a fact prompt sampled hot, or a creative one held to the facts
    fact_hot = sum(g == "fact" and a == "creative" for a, g in zip(picks, gold))
    creative_cold = sum(g == "creative" and a == "fact" for a, g in zip(picks, gold))
    return {"accuracy": round(acc, 3), "recall": per, "fact_read_as_creative": fact_hot, "creative_read_as_fact": creative_cold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya", default="D:/aegist/models/laya")
    ap.add_argument("--out", default="D:/aegist/laya/prompt_reader.json")
    a = ap.parse_args()
    from aegist.laya import Laya
    from aegist.prompt_reader import keyword_cue, keyword_kind

    j = Laya(a.laya)
    res = {"dev": {}, "keyword rule": {}}
    best, best_acc = None, -1
    for name, make in framings().items():
        r = j.run([make(p) for p, _ in rows(DEV)])
        picks = [KINDS[max(range(3), key=pr.__getitem__)] for pr, _ in r]
        s = score(picks, [k for _, k in rows(DEV)])
        res["dev"][name] = s
        if s["accuracy"] > best_acc:
            best, best_acc = name, s["accuracy"]
    t = time.time()
    r = j.run([framings()[best](p) for p, _ in rows(TEST)])
    ms = 1000 * (time.time() - t) / len(rows(TEST))
    picks = [KINDS[max(range(3), key=pr.__getitem__)] for pr, _ in r]
    res["laya test"] = {"framing": best, **score(picks, [k for _, k in rows(TEST)]), "ms_per_prompt": round(ms, 1),
                        "answers": [{"prompt": p, "gold": k, "laya": pk, "p": [round(x, 3) for x in pr]}
                                    for (p, k), pk, (pr, _) in zip(rows(TEST), picks, r)]}
    for split, data in (("dev", DEV), ("test", TEST)):
        res["keyword rule"][split] = score([keyword_kind(p) for p, _ in rows(data)], [k for _, k in rows(data)])
    # FRESH: written after both readers were fixed; measured once
    fr = rows(FRESH)
    gold = [k for _, k in fr]
    r = j.run([framings()[best](p) for p, _ in fr])
    laya = [KINDS[max(range(3), key=pr.__getitem__)] for pr, _ in r]
    cues = [keyword_cue(p) for p, _ in fr]
    res["fresh"] = {
        "laya": score(laya, gold),
        "keyword rule": score([c or "other" for c in cues], gold),
        "keyword rule, Laya when no cue": score([c or l for c, l in zip(cues, laya)], gold),
        "no cue matched": sum(c is None for c in cues),
        "answers": [{"prompt": p, "gold": k, "laya": l, "keyword": c} for (p, k), l, c in zip(fr, laya, cues)],
    }
    show = {k: v for k, v in res["laya test"].items() if k != "answers"}
    fresh = {k: v for k, v in res["fresh"].items() if k != "answers"}
    print(json.dumps({"laya dev": res["dev"], "laya test": show, "keyword rule": res["keyword rule"], "fresh": fresh}, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
