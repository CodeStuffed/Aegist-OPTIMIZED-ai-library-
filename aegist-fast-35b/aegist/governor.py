"""The hallucination governor, learning-time half. (The run-time half, which moves sampling knobs,
is aegist_rs/src/governor.rs.)

It measures when Aegist makes things up, on questions with known answers (aegist.confidence):
  - wrong:          a real question answered with the wrong fact;
  - over-cautious:  a real question it declined to answer (too little: it should have known);
  - fabricated:     a question about something that does not exist, answered with invented
                    specifics (too much, at the wrong time);
and how well the confidence probe separates right from wrong answers (AUROC, calibration error).

From the same measurements it builds training records for learning time: decline the made-up
things it invented answers for, and answer the real questions it got wrong or declined. Only
questions from the training split become records; the gate judges a new version on the held-out
split, and a version is kept only if it does not make things up more often, does not become
more cautious on real questions, and the probe stays as good.

    python -m aegist.governor report --data DIR [--probe probe.nx]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import torch

from . import confidence as C

ABSTAIN_TEMPLATES = {
    "fake book": "I'm not aware of a novel called {name}, so I can't say who wrote it. It may not exist, or it may be "
                 "too obscure for me; if you have a source, I can look at it.",
    "fake country": "I don't know of a Republic of {name}. It may not exist under that name; there is no capital I "
                    "can give you.",
    "fake person": "I'm not aware of a physicist named {name} who won a Nobel Prize. I can't give a year without "
                   "making one up.",
    "fake element": "There is no element called {name} that I know of, so it has no atomic number I can give.",
    "fake treaty": "I don't know of a Treaty of {name}. It may not exist, so I can't give a year for it.",
    "fake film": "I'm not aware of a film called {name}, so I can't say who directed it. It may not exist.",
    "fake theorem": "I don't know of a {name} theorem. It may not exist, so I can't say who proved it.",
    "fake battle": "I don't know of a Battle of {name}. It may not exist, so I can't give a year for it.",
}
ANSWER_TEMPLATES = {
    "capital": "The capital of {subject} is {gold}.",
    "element": "The atomic number of {subject} is {gold}.",
    "author": "'{subject}' was written by {gold_full}.",
    "year": "That happened in {gold}.",
    "letters": "The letter appears {gold} times.",
}


def outcome(answer: str, item: dict) -> str:
    """"right", "wrong", "over-cautious" or "fabricated"."""
    declined = bool(C.ABSTAIN.search(answer))
    if item["gold"] is None:
        return "right" if C.is_correct(answer, item) else "fabricated"
    if C.is_correct(answer, item):
        return "right"
    return "over-cautious" if declined else "wrong"


def ece(p: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    """Expected calibration error of P(wrong) against the wrong labels."""
    total = 0.0
    for b in range(bins):
        m = (p >= b / bins) & ((p < (b + 1) / bins) if b < bins - 1 else (p <= 1))
        if m.any():
            total += float(m.float().mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return total


def report(data_dir: str, probe_path: str | None = None, held_out: bool = False, seed: int = 0) -> dict:
    """held_out: only the test split (the questions no training or choice has seen)."""
    items, answers, pooled, stats, y, agree, taps = C.load_collection(data_dir)
    idx = C.split(items, seed)[2] if held_out else list(range(len(items)))
    idx = [i for i in idx if not items[i].get("para")]  # outcomes are per question
    counts = {"right": 0, "wrong": 0, "over-cautious": 0, "fabricated": 0}
    for i in idx:
        counts[outcome(answers[i], items[i])] += 1
    real = sum(1 for i in idx if items[i]["gold"] is not None)
    fake = len(idx) - real
    res = {
        "questions": len(idx),
        "real_wrong_rate": counts["wrong"] / max(real, 1),
        "real_over_cautious_rate": counts["over-cautious"] / max(real, 1),
        "fake_fabricated_rate": counts["fabricated"] / max(fake, 1),
        "counts": counts,
    }
    if probe_path:
        cfg, m = C.load_probe(probe_path)
        li = None if cfg["layer"] is None else taps.index(cfg["layer"])
        sel = torch.tensor(idx)
        p = C.predict(m, C.features(pooled, stats, li)[sel])
        res["probe_auroc"] = C.auroc(p, y[sel])
        res["probe_ece"] = ece(p, y[sel])
    return res


def training_records(data_dir: str, seed: int = 0) -> tuple[list[dict], dict]:
    """Learning-time records from the training split's mistakes, plus a summary.
    Each record is a chat example in aegist.learn's format."""
    items, answers, *_ = C.load_collection(data_dir)
    tr, _, _ = C.split(items, seed)
    recs, why = [], {"fabricated": 0, "wrong": 0, "over-cautious": 0, "skipped": 0}
    for i in tr:
        it, kind = items[i], items[i]["kind"]
        if it.get("para"):
            continue
        o = outcome(answers[i], it)
        if o == "right":
            continue
        target = None
        if o == "fabricated" and kind in ABSTAIN_TEMPLATES:
            target = ABSTAIN_TEMPLATES[kind].format(name=subject_of(it))
        elif o in ("wrong", "over-cautious"):
            target = answer_for(it)
        if target is None:
            why["skipped"] += 1
            continue
        why[o] += 1
        recs.append({"kind": "chat", "mode": "normal", "turns": [["user", it["q"] + C.ASK]], "target": target,
                     "source": f"governor:{o}:{kind}"})
    return recs, why


def collections_for(probes_dir: str, brain: str) -> list[str]:
    """The question collections (probes/data-*) that `brain` answered and that have single
    questions in them: the governor's measurements of that brain."""
    out = []
    for d in sorted(glob.glob(os.path.join(probes_dir, "data-*"))):
        try:
            meta = json.load(open(os.path.join(d, "items.json"), encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (os.path.exists(os.path.join(d, "answers.json")) and os.path.exists(os.path.join(d, "answers.json.taps"))
                and os.path.normcase(os.path.abspath(meta.get("brain", ""))) == os.path.normcase(os.path.abspath(brain))
                and any(not it.get("para") for it in meta["items"])):
            out.append(d)
    return out


def probe_file(probes_dir: str) -> str | None:
    """The live probe: probes/current.txt names it, else probes/confidence.nx (as the Rust engine)."""
    try:
        p = os.path.join(probes_dir, open(os.path.join(probes_dir, "current.txt"), encoding="utf-8").read().strip())
        if os.path.exists(p):
            return p
    except OSError:
        pass
    p = os.path.join(probes_dir, "confidence.nx")
    return p if os.path.exists(p) else None


def held_out_items(data_dir, seed: int = 0) -> list[dict]:
    """The questions report(..., held_out=True) is measured on."""
    items = C.load_collection(data_dir)[0]
    return [items[i] for i in C.split(items, seed)[2] if not items[i].get("para")]


def check_version(brain: str, data_dir, probe: str | None, work_dir: str, seed: int = 0, log=print) -> dict:
    """Have `brain` (a candidate version) answer the held-out questions of `data_dir` with the
    Rust engine, and report on its answers like report(data_dir, held_out=True) does on the
    current brain's, so the two can go through gate()."""
    dirs = data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]
    first = json.load(open(os.path.join(dirs[0], "spec.json"), encoding="utf-8"))
    items = [{k: v for k, v in it.items() if k != "group"} for it in held_out_items(dirs, seed)]
    log(f"governor: the new version answers {len(items)} held-out questions")
    C.answer_items(brain, items, work_dir, taps=first["taps"], max_new=first.get("max_new", 40), seed=seed, log=log)
    return report(work_dir, probe)


def subject_of(item: dict) -> str:
    q = item["q"]
    for pat in (r"'([^']+)'", r"the ([\w-]+) theorem", r"Battle of (\w+)", r"Republic of (\w+)", r"physicist ([\w ]+?) win", r"element (\w+)", r"Treaty of (\w+)",
                r"capital of ([\w ,.'-]+?)\?", r"number of ([\w ]+?)\?", r"did (.+) happen\?"):
        m = re.search(pat, q)
        if m:
            name = m.group(1)
            return f"'{name}'" if pat.startswith("'") else name
    return "that"


def answer_for(item: dict) -> str | None:
    kind, gold = item["kind"], item["gold"][0]
    if kind.startswith("claim"):
        return f"Yes, it is true that {item['claim']}." if gold == "yes" else f"No, it is not true that {item['claim']}."
    if kind.startswith("multiply"):
        a, b = re.findall(r"\d+", item["q"])
        return f"{a} times {b} is {int(gold):,}."
    if kind == "author":
        book = subject_of(item).strip("'")
        return ANSWER_TEMPLATES[kind].format(subject=book, gold_full=AUTHOR_FULL.get(book, gold))
    if kind == "reverse":
        return f"'{subject_of(item).strip(chr(39))}' spelled backwards is '{gold}'."
    if kind == "digit sum":
        return f"The digits of {re.findall(r'[0-9]+', item['q'])[-1]} add up to {gold}."
    if kind.startswith("letters"):
        ch, word = re.findall(r"'([^']+)'", item["q"])
        n = int(gold)
        return f"The letter '{ch}' appears {n} time{'s' if n != 1 else ''} in '{word}'."
    if kind in ANSWER_TEMPLATES:
        return ANSWER_TEMPLATES[kind].format(subject=subject_of(item), gold=gold)
    return None


# full names for the author answers (the bank grades on surnames)
AUTHOR_FULL = {
    "Pride and Prejudice": "Jane Austen", "Middlemarch": "George Eliot", "Moby-Dick": "Herman Melville",
    "War and Peace": "Leo Tolstoy", "The Master and Margarita": "Mikhail Bulgakov", "Things Fall Apart": "Chinua Achebe",
    "One Hundred Years of Solitude": "Gabriel Garcia Marquez", "The Tale of Genji": "Murasaki Shikibu",
    "Dead Souls": "Nikolai Gogol", "The Leopard": "Giuseppe Tomasi di Lampedusa", "Pedro Paramo": "Juan Rulfo",
    "The Good Soldier Svejk": "Jaroslav Hasek", "Kristin Lavransdatter": "Sigrid Undset", "Hunger": "Knut Hamsun",
    "The Man Without Qualities": "Robert Musil", "Independent People": "Halldor Laxness",
    "The Radetzky March": "Joseph Roth", "Season of Migration to the North": "Tayeb Salih",
    "The Book of Disquiet": "Fernando Pessoa", "Sinuhe the Egyptian": "Mika Waltari", "Max Havelaar": "Multatuli",
    "Oblomov": "Ivan Goncharov", "The Death of Virgil": "Hermann Broch", "Embers": "Sandor Marai",
    "The Glass Bead Game": "Hermann Hesse", "Frankenstein": "Mary Shelley", "Don Quixote": "Miguel de Cervantes",
    "The Trial": "Franz Kafka", "Beloved": "Toni Morrison", "The Name of the Rose": "Umberto Eco",
    "Blindness": "Jose Saramago", "The Tin Drum": "Gunter Grass", "Snow Country": "Yasunari Kawabata",
    "The Makioka Sisters": "Junichiro Tanizaki", "Doctor Zhivago": "Boris Pasternak", "The Magic Mountain": "Thomas Mann",
    "Nostromo": "Joseph Conrad", "Ficciones": "Jorge Luis Borges", "Hopscotch": "Julio Cortazar",
    "The Cairo Trilogy": "Naguib Mahfouz", "Wide Sargasso Sea": "Jean Rhys", "Rebecca": "Daphne du Maurier",
    "Germinal": "Emile Zola", "Fathers and Sons": "Ivan Turgenev", "The Betrothed": "Alessandro Manzoni",
    "Quo Vadis": "Henryk Sienkiewicz", "Effi Briest": "Theodor Fontane", "The Story of the Stone": "Cao Xueqin",
    "Journey to the West": "Wu Cheng'en", "Gargantua and Pantagruel": "Francois Rabelais",
}


def gate(before: dict, after: dict, slack: float = 0.02) -> dict:
    """A new version passes when, on the held-out split, it makes things up no more often, is no
    more over-cautious, and its probe is no worse (each within `slack`), and at least one of the
    first two improved."""
    made_up = lambda r: r["real_wrong_rate"] + r["fake_fabricated_rate"]
    checks = {
        "makes_up_less_or_same": made_up(after) <= made_up(before) + slack,
        "not_more_cautious": after["real_over_cautious_rate"] <= before["real_over_cautious_rate"] + slack,
        "improved_somewhere": made_up(after) < made_up(before) - 1e-9 or
                              after["real_over_cautious_rate"] < before["real_over_cautious_rate"] - 1e-9,
    }
    if "probe_auroc" in before and "probe_auroc" in after:
        a, b = after["probe_auroc"], before["probe_auroc"]
        checks["probe_no_worse"] = not (math.isfinite(a) and math.isfinite(b)) or a >= b - slack
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aegist.governor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--data", required=True)
    r.add_argument("--probe")
    r.add_argument("--held-out", action="store_true", help="only the test split")
    t = sub.add_parser("records")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True, help="jsonl for learn/inbox-style review")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        print(json.dumps(report(a.data, a.probe, a.held_out), indent=1))
    else:
        recs, why = training_records(a.data)
        with open(a.out, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(why))


if __name__ == "__main__":
    main()
