"""The unsure-mark probe trained on chat answers: what the user actually gets, with the tools, the
second look and up to 200 tokens, rather than the short raw answers of aegist/confidence.py.

Why: the first probe was trained on raw answers (no tools, one short sentence). In chat it marked
right answers as unsure, above all honest refusals about made-up things ("I can't find any
country called ..."): in raw answers a made-up name nearly always came with an invented answer,
so the probe learned "made-up name in the question = wrong".

collect  every question goes through the chat pipeline (`aegist exam` with "taps"), which keeps
         each sentence the probe looks at: its hidden-state means at the tapped layers, its four
         logit statistics, and the old probe's P(wrong). Both answer passes are kept (the first
         answer and, when there was one, the second look's). A stopped run goes on where it left
         off. The AQ test's own questions (seed 1, 3 per kind) are left out, so AQ can still
         measure the new probe.
train    each answer pass is graded with confidence.is_correct. The probe marks sentences, and an
         answer counts as marked when any sentence is, so it learns from whole answers: P(answer
         wrong) = 1 - prod(1 - P(sentence wrong)) (noisy-OR). Paragraph answers are graded per
         sentence (confidence.label_sentences). The raw collections can be mixed in, and they
         matter: chat answers are seldom wrong, so most of the wrong examples are raw rows.
         Every question, chat or raw, is split train / validation / test (by question, within
         each kind). The layer, L2 and raw-data weight are chosen on validation by a balanced
         score: wrong answers caught (chat and raw), against right chat answers and right raw
         rows left alone (each half of the "right" side). The test questions are judged once,
         by the new probe and by the old one (its live marks on chat answers; raw rows scored as
         in chat). The old probe was trained on the raw collections, so its raw test rows are
         ones it has seen: that favours the old probe.

Run:  python -m aegist.chat_probe collect --out DIR [--brain FILE] [--seed 7]
      python -m aegist.chat_probe status --data DIR
      python -m aegist.chat_probe train --data DIR... [--raw DIR...] [--out probe.nx]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time

import torch

from . import confidence as C

TAPS = [12, 18, 24, 30, 36, 42]  # the reid1 raw collections' taps (the old probe reads 18)
MAX_NEW = 200  # as the AQ test asks


def per_kind(kind: str) -> int:
    """How many questions of each kind: most where chat answers go wrong or get wrongly marked."""
    if kind.startswith("para"):
        return 5
    if kind.startswith("fake"):
        return 10
    if kind in ("capital", "element", "author", "year"):
        return 14
    if kind.startswith("claim"):
        return 12
    if kind.startswith("letters") or kind == "reverse":
        return 4
    return 3  # arithmetic: the calculator answers it


def plan(seed: int = 7) -> list[dict]:
    """The questions, interleaved by kind (so any first part of the run covers every kind), minus
    the AQ test's (seed 1, 3 per kind)."""
    from .aq import questions
    held = {it["q"] for it in questions(1, 3)}
    items = C.question_bank(seed) + C.hard_bank(seed) + C.claim_bank(seed) + C.para_bank(seed)
    rng = random.Random(seed)
    by_kind: dict[str, list[dict]] = {}
    for it in items:
        if it["q"] not in held:
            by_kind.setdefault(it["kind"], []).append(it)
    picked = {k: rng.sample(v, min(per_kind(k), len(v))) for k, v in sorted(by_kind.items())}
    out = []
    for i in range(max(map(len, picked.values()))):
        round_ = [v[i] for v in picked.values() if i < len(v)]
        rng.shuffle(round_)
        out += round_
    return out


def prompt(it: dict) -> str:
    if it.get("para"):
        # "Write three sentences ..." reads as a creative request, and creative replies carry no
        # unsure marks, so the probe would see nothing: ask for the same thing as facts
        return it["q"].replace("Write three short sentences about", "Tell me three short facts about")
    return it["q"] + C.ASK


def collect(out_dir: str, brain: str | None = None, *, seed: int = 7, home: str = "D:/aegist", limit: int | None = None,
            log=print) -> str:
    from .aq import live_brain
    brain = brain or live_brain(os.path.join(home, "brains"))
    os.makedirs(out_dir, exist_ok=True)
    items = plan(seed)[:limit]
    meta_path = os.path.join(out_dir, "items.json")
    if os.path.exists(meta_path):
        old = json.load(open(meta_path, encoding="utf-8"))
        if [it["q"] for it in old["items"]] != [it["q"] for it in items] or old["brain"] != os.path.abspath(brain):
            raise RuntimeError(f"{out_dir} holds a different collection; use a new directory")
    from .aq import find_probe
    old = find_probe(os.path.join(home, "probes"), brain)
    if old is None:
        raise RuntimeError("recording needs the brain's probe (the old probe marks the answers as it would in chat)")
    old_thr = C.load_probe(old)[0]["threshold"]
    json.dump({"items": items, "taps": TAPS, "brain": os.path.abspath(brain), "seed": seed, "chat": True,
               "old_probe": old, "old_threshold": old_thr}, open(meta_path, "w", encoding="utf-8"), indent=1)
    inp, outp = os.path.join(out_dir, "exam.json"), os.path.join(out_dir, "answers.json")
    json.dump({"prompts": [prompt(it) for it in items], "think": "off", "seed": seed, "max_new": MAX_NEW, "taps": TAPS},
              open(inp, "w", encoding="utf-8"))
    log(f"{len(items)} questions through chat with {os.path.basename(brain)}, keeping layers {TAPS}")
    t0 = time.time()
    r = subprocess.run([C.exe_path(), "exam", inp, outp, "--brain", brain, "--resume"], cwd=home)
    if r.returncode != 0:
        raise RuntimeError(f"aegist exam exited with {r.returncode}")
    log(f"done in {time.time() - t0:.0f}s")
    return out_dir


# --- loading -------------------------------------------------------------------------------------

def load(data_dir: str):
    """-> (bags, x [rows, taps, d], stats [rows, 4], old_p [rows] (NaN: not judged), taps).
    A bag is one answer pass (label: is it wrong), or for paragraph questions one checkable
    sentence; bag["rows"] are its sentences. Answers still being collected are read from
    answers.json.partial."""
    meta = json.load(open(os.path.join(data_dir, "items.json"), encoding="utf-8"))
    path = os.path.join(data_dir, "answers.json")
    res = json.load(open(path if os.path.exists(path) else path + ".partial", encoding="utf-8"))
    taps_path = os.path.join(data_dir, "answers.json.taps")
    end = res[-1]["taps_end"] if res else 0
    raw = torch.from_file(taps_path, dtype=torch.float32, size=end // 4) if end else torch.zeros(0)
    n_taps = len(meta["taps"])
    bags, feats, stats, old_p = [], [], [], []
    at = 0
    for i, (it, r) in enumerate(zip(meta["items"], res)):
        for k, ps in enumerate(r["passes"]):
            rows = []
            for s in ps["sentences"]:
                w = s["width"]
                feats.append(raw[at // 4: at // 4 + w].view(n_taps, w // n_taps))
                at += 4 * w
                stats.append(torch.tensor(s["stats"], dtype=torch.float32))
                old_p.append(float("nan") if s["p"] is None else s["p"])
                rows.append(len(feats) - 1)
            base = {"group": it["q"], "kind": it["kind"], "item": i, "pass": k, "final": k == len(r["passes"]) - 1,
                    "text": ps["text"]}
            if it.get("para"):
                texts = [s["text"] for s in ps["sentences"]]
                for row, lab in zip(rows, C.label_sentences(texts, it)):
                    if lab is not None:
                        bags.append(dict(base, rows=[row], label=lab, sentence=True))
            elif rows:
                bags.append(dict(base, rows=rows, label=0.0 if C.is_correct(ps["text"], it) else 1.0, sentence=False))
    assert at == end, f"{data_dir}: read {at} bytes of hidden states, the answers say {end}"
    x = torch.stack(feats) if feats else torch.zeros(0, n_taps, 1)
    st = torch.stack(stats) if stats else torch.zeros(0, 4)
    return bags, x, st, torch.tensor(old_p), meta["taps"]


def load_many(dirs: list[str]):
    parts = [load(d) for d in dirs]
    assert all(p[4] == parts[0][4] for p in parts), "collections with different taps"
    bags, off = [], 0
    for p in parts:
        bags += [dict(b, rows=[r + off for r in b["rows"]]) for b in p[0]]
        off += len(p[1])
    return bags, torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]), torch.cat([p[3] for p in parts]), parts[0][4]


# --- training ------------------------------------------------------------------------------------

def balanced(y: torch.Tensor, total: float) -> torch.Tensor:
    """Weights that give each class half of `total`."""
    w = torch.zeros(len(y), dtype=torch.float64)
    for c in (0.0, 1.0):
        n = int((y == c).sum())
        if n:
            w[y == c] = total / (2 * n) if int((y != c).sum()) else total / n
    return w


def fit(xb, bag_of, yb, xr, yr, *, l2: float, raw_w: float, steps: int = 200):
    """L2 logistic regression on sentences, trained on whole answers by noisy-OR (and on the raw
    rows directly, weighted raw_w in all). Standardised first. -> (mean, std, w, b)."""
    allx = torch.cat([xb, xr]) if raw_w > 0 else xb
    mean, std = allx.mean(0), allx.std(0).clamp(min=1e-6)
    zb = ((xb - mean) / std).double()
    zr = ((xr - mean) / std).double() if raw_w > 0 else None
    wb, wr = balanced(yb, 1.0), balanced(yr, raw_w) if raw_w > 0 else None
    yb64, yr64 = yb.double(), yr.double()
    n_bags = len(yb)
    w = torch.zeros(zb.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = torch.zeros(n_bags, dtype=torch.float64).index_add(0, bag_of, -torch.nn.functional.softplus(zb @ w + b))
        s = s.clamp(max=-1e-9)  # log P(every sentence right)
        nll = torch.where(yb64 == 1, -torch.log(-torch.expm1(s)), -s)
        loss = (wb * nll).sum() + l2 * (w * w).sum() / 2
        if zr is not None:
            loss = loss + (wr * torch.nn.functional.binary_cross_entropy_with_logits(zr @ w + b, yr64, reduction="none")).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return mean.float(), std.float(), w.detach().float(), b.detach().float()


def bag_scores(p: torch.Tensor, judged: torch.Tensor, bags) -> torch.Tensor:
    """An answer's score: its highest judged sentence (0 when none is judged)."""
    out = torch.zeros(len(bags))
    for i, bg in enumerate(bags):
        v = [float(p[r]) for r in bg["rows"] if judged[r]]
        out[i] = max(v) if v else 0.0
    return out


def bal_acc(marked: torch.Tensor, y: torch.Tensor) -> float:
    parts = [float(marked[y == 1].float().mean()) if (y == 1).any() else None,
             float((~marked[y == 0]).float().mean()) if (y == 0).any() else None]
    parts = [v for v in parts if v is not None]
    return sum(parts) / len(parts) if parts else float("nan")


SPLITS = ("train", "val", "test")


def split_questions(bags, raw_items, seed: int, frac=(0.6, 0.2), test_qs: set[str] | None = None):
    """Every question, chat or raw, -> "train" / "val" / "test", within each kind. A question in
    both kinds of collection gets one split, so neither leaks into the other's test. test_qs: the
    test questions are these (the old probe's), the rest split between train and validation."""
    rng = random.Random(seed)
    by: dict[str, set[str]] = {}
    for q, kind in [(b["group"], b["kind"]) for b in bags] + [(it["q"], it["kind"]) for it in raw_items]:
        by.setdefault(kind, set()).add(q)
    part = {}
    if test_qs is not None:
        frac = (frac[0] / (frac[0] + frac[1]), 1.0)
    for kind in sorted(by):
        qs = sorted(by[kind])
        if test_qs is not None:
            part.update({q: "test" for q in qs if q in test_qs})
            qs = [q for q in qs if q not in test_qs]
        rng.shuffle(qs)
        a, c = round(len(qs) * frac[0]), round(len(qs) * (frac[0] + frac[1]))
        part.update({q: "train" for q in qs[:a]} | {q: "val" for q in qs[a:c]} | {q: "test" for q in qs[c:]})
    return part


def old_test_rows(old_probe: str | None, raw_dirs: list[str], counts: list[int], items: list[dict]) -> torch.Tensor | None:
    """The raw rows (numbered as raw_dirs joined) that the old probe held out as its test split,
    from its `.split.json`, or None: {"test_questions": [...]} as train writes it, or {"data": its
    collections in order, "seed": ...} for a probe from confidence.train, whose split is rebuilt. Judging both probes only on these rows keeps the old
    one from being judged on rows it was trained on."""
    side = old_probe and os.path.splitext(old_probe)[0] + ".split.json"
    if not side or not os.path.exists(side):
        return None
    spec = json.load(open(side, encoding="utf-8"))
    if "test_questions" in spec:
        qs = set(spec["test_questions"])
        return torch.tensor([it["q"] in qs for it in items], dtype=torch.bool)
    norm = lambda d: os.path.normcase(os.path.abspath(d))
    at = {norm(d): sum(counts[:i]) for i, d in enumerate(raw_dirs)}
    if not all(norm(d) in at for d in spec["data"]):
        return None
    items = []
    for d in spec["data"]:
        it = C.load_collection(d)[0]
        items += [(norm(d), j, x) for j, x in enumerate(it)]
    _, _, te = C.split([x for _, _, x in items], spec.get("seed", 0))
    rows = torch.zeros(sum(counts), dtype=torch.bool)
    rows[[at[items[i][0]] + items[i][1] for i in te]] = True
    return rows


def marks_summary(sc, yc, sr, yr, thr: float) -> dict:
    """Marks at `thr` on chat answers (sc, yc) and raw rows (sr, yr): the wrong ones caught (chat
    and raw together), the right ones left alone (chat and raw apart), and their balanced mean:
    (caught + mean of the two left-alone rates) / 2. Chat answers count as much as raw ones on the
    right side, so a probe cannot win by marking only the raw collections' kind of text."""
    mc, mr = sc >= thr, sr >= thr
    wrong = torch.cat([mc[yc == 1], mr[yr == 1]])
    caught = float(wrong.float().mean()) if len(wrong) else float("nan")
    calm = [float((~m[y == 0]).float().mean()) for m, y in ((mc, yc), (mr, yr)) if (y == 0).any()]
    calm_mean = sum(calm) / len(calm) if calm else float("nan")
    parts = [v for v in (caught, calm_mean) if math.isfinite(v)]
    return {"score": sum(parts) / len(parts) if parts else float("nan"),
            "wrong_marked": f"{int(mc[yc == 1].sum()) + int(mr[yr == 1].sum())}/{len(wrong)}",
            "chat_right_marked": f"{int(mc[yc == 0].sum())}/{int((yc == 0).sum())}",
            "raw_right_marked": f"{int(mr[yr == 0].sum())}/{int((yr == 0).sum())}"}


def best_mark_threshold(sc, yc, sr, yr) -> tuple[float, float]:
    best = (0.5, -1.0)
    for t in torch.linspace(0.05, 0.95, 91).tolist():
        s = marks_summary(sc, yc, sr, yr, t)["score"]
        if math.isfinite(s) and s > best[1] + 1e-9:
            best = (t, s)
    return best


def train(data_dirs: list[str], raw_dirs: list[str], out: str | None = None, *, seed: int = 0, old_probe: str | None = None,
          log=print) -> dict:
    """old_probe: the probe to beat (default: the one live when the chat answers were collected)."""
    bags, X, S, old_p, taps = load_many(data_dirs)
    meta = json.load(open(os.path.join(data_dirs[0], "items.json"), encoding="utf-8"))
    old_probe = old_probe or meta.get("old_probe")
    if raw_dirs:
        parts = [C.load_collection(d) for d in raw_dirs]
        items = sum((p[0] for p in parts), [])
        pooled, rstats, ry = (torch.cat([p[k] for p in parts]) for k in (2, 3, 4))
        assert all(p[6] == taps for p in parts), f"raw collections tap {parts[0][6]}, chat {taps}"
        old_te = old_test_rows(old_probe, raw_dirs, [len(p[0]) for p in parts], items)
    else:
        items, pooled, rstats, ry = [], torch.zeros(0, len(taps), X.shape[-1]), torch.zeros(0, 4), torch.zeros(0)
        old_te = None
    test_qs = None if old_te is None else {items[i]["q"] for i in old_te.nonzero().flatten().tolist()}
    part = split_questions(bags, items, seed, test_qs=test_qs)
    sets = {k: [b for b in bags if part[b["group"]] == k] for k in SPLITS}
    # with the old probe's split: its test rows are the raw test; other rows of those questions,
    # which it trained on, are left out of everything
    in_set = lambda i, k: part[items[i]["q"]] == k and (old_te is None or k != "test" or bool(old_te[i]))
    rsets = {k: torch.tensor([i for i in range(len(items)) if in_set(i, k)], dtype=torch.long) for k in SPLITS}
    if old_te is not None:
        log(f"test: the old probe's own held-out rows ({int(old_te.sum())}) and the chat answers to those questions")
    for k in SPLITS:
        y = [b["label"] for b in sets[k]]
        log(f"{k}: chat {len(sets[k])} answers/sentences from {len({b['group'] for b in sets[k]})} questions, "
            f"{int(sum(y))} wrong; raw {len(rsets[k])} rows, {int(ry[rsets[k]].sum())} wrong")

    def design(li, v):
        """-> features of the bags' sentences, each sentence's bag, the labels, and the bags with
        their rows numbered as in the features."""
        rows = [r for b in v for r in b["rows"]]
        bag_of = torch.tensor([i for i, b in enumerate(v) for _ in b["rows"]], dtype=torch.long)
        x = torch.cat([X[rows, li], S[rows]], 1) if li is not None else S[rows]
        at = iter(range(len(rows)))
        local = [dict(b, rows=[next(at) for _ in b["rows"]]) for b in v]
        return x, bag_of, torch.tensor([b["label"] for b in v]), local

    def raw_x(li, idx):
        return torch.cat([pooled[idx, li], rstats[idx]], 1) if li is not None else rstats[idx]

    def ood_of(m, x):
        return float(torch.quantile(C.ood_score(m, x).double(), 0.99))

    def fit_on(li, v, idx, l2, raw_w):
        """-> the probe fitted on chat bags v and raw rows idx, and its out-of-range bound."""
        x, bag_of, y, _ = design(li, v)
        m = fit(x, bag_of, y, raw_x(li, idx), ry[idx], l2=l2, raw_w=raw_w)
        return m, ood_of(m, torch.cat([x, raw_x(li, idx)]) if raw_w > 0 else x)

    def scores(m, ood, li, v, idx):
        """-> chat answers' scores and raw rows' scores; a sentence out of range is not judged (0)."""
        x, _, _, local = design(li, v)
        sc = bag_scores(C.predict(m, x), C.ood_score(m, x) <= ood, local)
        xr = raw_x(li, idx)
        sr = torch.where(C.ood_score(m, xr) <= ood, C.predict(m, xr), torch.zeros(len(idx)))
        return sc, sr

    lab = lambda v: torch.tensor([b["label"] for b in v])
    yv, yrv = lab(sets["val"]), ry[rsets["val"]]
    best = None
    for li in [None] + list(range(len(taps))):
        for raw_w in ([0.0, 0.5, 1.0, 2.0] if len(ry) else [0.0]):
            for l2 in [1e-3, 1e-2, 1e-1, 1.0]:
                m, ood = fit_on(li, sets["train"], rsets["train"], l2, raw_w)
                sc, sr = scores(m, ood, li, sets["val"], rsets["val"])
                score = best_mark_threshold(sc, yv, sr, yrv)[1]
                a = C.auroc(torch.cat([sc, sr]), torch.cat([yv, yrv]))
                name = ("logit stats only" if li is None else f"layer {taps[li]} + stats") + f", l2 {l2}, raw weight {raw_w}"
                key = (round(score, 3), a if math.isfinite(a) else -1)
                if best is None or key > best["key"]:
                    best = {"key": key, "li": li, "l2": l2, "raw_w": raw_w, "name": name, "val_score": score, "val_auroc": a}
    log(f"chosen on validation: {best['name']} (val balanced score {best['val_score']:.3f}, AUROC {best['val_auroc']:.3f})")
    li, l2, raw_w = best["li"], best["l2"], best["raw_w"]
    # the threshold from cross-fitted scores on train + validation (5 folds of questions)
    trva, trva_r = sets["train"] + sets["val"], torch.cat([rsets["train"], rsets["val"]])
    qs = sorted({q for q, s in part.items() if s != "test"})
    random.Random(seed).shuffle(qs)
    fold_of = {q: i % 5 for i, q in enumerate(qs)}
    cross, cross_r = torch.zeros(len(trva)), torch.zeros(len(trva_r))
    for f in range(5):
        out_i = [i for i, b in enumerate(trva) if fold_of[b["group"]] == f]
        out_j = [j for j, r in enumerate(trva_r.tolist()) if fold_of[items[r]["q"]] == f]
        in_r = torch.tensor([r for r in trva_r.tolist() if fold_of[items[r]["q"]] != f], dtype=torch.long)
        m, ood = fit_on(li, [b for b in trva if fold_of[b["group"]] != f], in_r, l2, raw_w)
        sc, sr = scores(m, ood, li, [trva[i] for i in out_i], trva_r[out_j])
        cross[out_i], cross_r[out_j] = sc, sr
    thr, cross_score = best_mark_threshold(cross, lab(trva), cross_r, ry[trva_r])
    # the probe: refit on train + validation; the test questions judged once
    m, ood = fit_on(li, trva, trva_r, l2, raw_w)
    yt, yrt = lab(sets["test"]), ry[rsets["test"]]
    s_new, r_new = scores(m, ood, li, sets["test"], rsets["test"])
    # the old probe on the same test questions, scored from the same recorded features as the new
    # one (the live marks in the collection may come from more than one probe)
    old_thr = float(meta.get("old_threshold", 0.4))
    s_old = bag_scores(torch.nan_to_num(old_p), torch.isfinite(old_p), sets["test"])
    r_old = torch.zeros(len(rsets["test"]))
    if old_probe:
        ocfg, om = C.load_probe(old_probe)
        old_thr = float(ocfg["threshold"])
        oli = None if ocfg["layer"] is None else taps.index(ocfg["layer"])
        s_old, r_old = scores(om, ocfg["ood_max"] if ocfg.get("ood_max") is not None else math.inf, oli, sets["test"], rsets["test"])
    new, old = marks_summary(s_new, yt, r_new, yrt, thr), marks_summary(s_old, yt, r_old, yrt, old_thr)
    res = {
        "choice": best["name"], "threshold": round(thr, 3), "cross_fitted_score": cross_score, "ood_max": ood,
        "test_chat": len(yt), "test_chat_wrong": int(yt.sum()), "test_raw": len(yrt), "test_raw_wrong": int(yrt.sum()),
        "test_score": new["score"], "old_test_score": old["score"],
        "test_auroc_raw": C.auroc(r_new, yrt), "old_test_auroc_raw": C.auroc(r_old, yrt),
        "test_auroc_chat": C.auroc(s_new, yt), "old_test_auroc_chat": C.auroc(s_old, yt),
        **{f"test_{k}": v for k, v in new.items() if k != "score"},
        **{f"old_test_{k}": v for k, v in old.items() if k != "score"},
        "beats_old": bool(math.isfinite(new["score"]) and new["score"] > old["score"]),
        "raw_rows": len(ry), "chat_collections": [os.path.abspath(d) for d in data_dirs],
    }
    by_kind = {}
    for b, n, o in zip(sets["test"], (s_new >= thr).tolist(), (s_old >= old_thr).tolist()):
        k = by_kind.setdefault(b["kind"], {"n": 0, "wrong": 0, "new_marks": 0, "old_marks": 0})
        k["n"] += 1
        k["wrong"] += int(b["label"])
        k["new_marks"] += n
        k["old_marks"] += o
    res["test_chat_by_kind"] = by_kind
    for who, s in (("new", new), ("old", old)):
        log(f"test, {who} probe: balanced score {s['score']:.3f}; wrong marked {s['wrong_marked']}, right chat answers marked "
            f"{s['chat_right_marked']}, right raw rows marked {s['raw_right_marked']}")
    log(f"test AUROC on raw rows: new {res['test_auroc_raw']:.3f}, old {res['old_test_auroc_raw']:.3f} "
        f"{'(held out by both probes)' if old_te is not None else '(the old probe was trained on the raw collections: in-sample for it)'}; "
        f"{'the new probe is better' if res['beats_old'] else 'the new probe is NOT better: keep the old one'}")
    if out:
        C.save_probe(out, m, layer=None if li is None else taps[li], threshold=thr, metrics=res,
                     choice=best["name"] + " (chat answers, noisy-OR, plus raw rows)", ood_max=ood)
        # the next retrain judges this probe on its own held-out questions
        json.dump({"test_questions": sorted(q for q, k in part.items() if k == "test"), "seed": seed,
                   "chat": [os.path.abspath(d) for d in data_dirs], "raw": [os.path.abspath(d) for d in raw_dirs]},
                  open(os.path.splitext(out)[0] + ".split.json", "w", encoding="utf-8"), indent=1)
        log(f"saved {out}")
    return res


def status(data_dir: str, log=print) -> dict:
    meta = json.load(open(os.path.join(data_dir, "items.json"), encoding="utf-8"))
    bags, X, _, old_p, _ = load(data_dir)
    answers = [b for b in bags if not b["sentence"]]
    y = torch.tensor([b["label"] for b in answers]) if answers else torch.zeros(0)
    s_old = bag_scores(torch.nan_to_num(old_p), torch.isfinite(old_p), answers)
    marked = s_old >= meta.get("old_threshold", 0.4)
    done = len({b["item"] for b in bags})  # questions that gave the probe something
    res = {"questions_with_sentences": done, "questions": len(meta["items"]), "sentences": len(X),
           "answer_passes": len(answers), "wrong": int(y.sum()), "para_sentences_labeled": len(bags) - len(answers),
           "old_probe_bal_acc": bal_acc(marked, y), "old_wrong_marked": f"{int(marked[y == 1].sum())}/{int((y == 1).sum())}",
           "old_right_marked": f"{int(marked[y == 0].sum())}/{int((y == 0).sum())}"}
    log(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aegist.chat_probe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--brain")
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--home", default="D:/aegist")
    c.add_argument("--limit", type=int, help="only the first N questions (a quick check)")
    s = sub.add_parser("status")
    s.add_argument("--data", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, nargs="+", help="chat collections")
    t.add_argument("--raw", nargs="*", default=[], help="raw collections (aegist.confidence collect) to mix in")
    t.add_argument("--out")
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.out, a.brain, seed=a.seed, home=a.home, limit=a.limit)
    elif a.cmd == "status":
        status(a.data)
    else:
        print(json.dumps(train(a.data, a.raw, a.out, seed=a.seed), indent=1))


if __name__ == "__main__":
    main()
