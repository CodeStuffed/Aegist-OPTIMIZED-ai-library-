"""The curriculum: Aegist studies English from letters up to research abstracts, one stage at a
time, and moves up only when a stage is done.

    python -m aegist.curriculum status        # the stages, where Aegist is, exam scores
    python -m aegist.curriculum lesson        # put the current stage's next lesson in learn/
    python -m aegist.curriculum exam          # the current stage's exam, on the live brain
    python -m aegist.curriculum run           # lesson, learning time, exam; repeat with --lessons N

Stages. The names give the reading level of the material, not a measured level of Aegist.
  letters        spelling drills: spell a word, count a letter, spell it backwards, find the
                 n-th letter, put words in alphabetical order. Generated, so every answer is
                 right by construction, and each answer shows its method (spell first, then
                 count). This is where the model is weakest: it reads tokens, not letters.
  early reader   Simple English Wikipedia, the easiest articles (readability grade under 7)
  middle school  Simple English Wikipedia, grade 7 to 10
  high school    English Wikipedia, grade 10 to 14
  university     English Wikipedia, grade 14 and up
  research       arXiv abstracts (mathematics, physics, computer science, biology, statistics,
                 economics)

A stage is done when `quota` of its lessons were accepted by learning time (counted from the
live brain's lineage, so a rollback also rolls progress back) and, for letters, its exam
passed: questions about words the drills never use, answered by the Rust engine and graded
exactly. Reading stages have no exam of their own: learning time's gates decide each lesson
(held-out loss on articles it did not train on, drift, identity, and the governor).

Sources and licences. Wikipedia text is CC BY-SA 4.0: every saved article starts with its
title, address, revision and licence, and learn/curriculum/sources.jsonl lists them all. arXiv
metadata (titles and abstracts) is CC0. Requests go one at a time with a pause between them
(1 s for Wikipedia, 3 s for arXiv, as its API terms ask) and a User-Agent that names the tool.
Wikimedia asks for a contact in it: put "contact" in aegist.json to add yours. Downloaded text
is only ever read as text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from . import confidence as C

STAGES = [
    {"name": "letters", "kind": "drills", "quota": 3, "pass": 0.8},
    {"name": "early reader", "kind": "wiki", "host": "simple.wikipedia.org", "grade": (0, 7), "quota": 3},
    {"name": "middle school", "kind": "wiki", "host": "simple.wikipedia.org", "grade": (7, 10), "quota": 3},
    {"name": "high school", "kind": "wiki", "host": "en.wikipedia.org", "grade": (10, 14), "quota": 3},
    {"name": "university", "kind": "wiki", "host": "en.wikipedia.org", "grade": (14, 99), "quota": 3},
    {"name": "research", "kind": "arxiv", "quota": 3,
     "cats": ["math.CO", "math.PR", "math.NT", "physics.optics", "cond-mat.stat-mech", "astro-ph.EP", "quant-ph",
              "cs.CL", "cs.LG", "cs.DS", "q-bio.NC", "q-bio.PE", "stat.ME", "econ.TH"]},
]
WIKI_LICENSE = "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)"
ARXIV_LICENSE = "arXiv metadata, CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/)"
END_SECTIONS = {"references", "other websites", "related pages", "see also", "external links", "notes",
                "further reading", "sources", "bibliography", "citations", "footnotes"}
MAX_WORDS = 1500


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80] or "x"


# --- readability ------------------------------------------------------------------------------

def syllables(word: str) -> int:
    w = word.lower()
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def grade(text: str) -> float:
    """Flesch-Kincaid grade level (Kincaid et al. 1975): school grade a reader needs."""
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0
    sents = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))
    return 0.39 * len(words) / sents + 11.8 * sum(map(syllables, words)) / len(words) - 15.59


# --- letters: drills and their exam ------------------------------------------------------------

# the probe's own test words stay out of the drills, so its measurements stay honest; number
# words too, since grading reads "seven" as 7
PROBE_WORDS = set(C.WORDS) | set(C.LONG_WORDS) | set(C.REVERSE_WORDS) | set(C._WORD_NUM)
WORD_RE = re.compile(r"(?<![\w'*-])[a-z]{4,14}(?![\w'*-])")
DRILLS = ["spell", "count", "reverse", "nth", "alpha"]


def word_pool(texts) -> list[str]:
    words = set()
    for t in texts:
        words.update(WORD_RE.findall(t))
    return sorted(words - PROBE_WORDS)


def is_exam_word(w: str) -> bool:
    """One word in five, fixed by its hash, is kept for the exam and never drilled."""
    return hashlib.sha256(w.encode()).digest()[0] % 5 == 0


def ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def drill(kind: str, words: list[str], rng: random.Random) -> dict:
    """One question with its gold answers and a target answer that shows the method."""
    w = rng.choice(words)
    d = "-".join(w)
    if kind == "spell":
        return {"q": f"Spell the word '{w}' letter by letter.", "gold": [" ".join(w)],
                "target": f"'{w}' is spelled {d}."}
    if kind == "count":
        absent = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in w]
        ch = rng.choice(w) if rng.random() < 0.8 or not absent else rng.choice(absent)
        n = w.count(ch)
        return {"q": f"How many times does the letter '{ch}' appear in the word '{w}'?", "gold": [str(n)],
                "target": f"'{w}' is spelled {d}, so the letter '{ch}' appears {n} time{'' if n == 1 else 's'}."}
    if kind == "reverse":
        return {"q": f"Spell the word '{w}' backwards.", "gold": [w[::-1], " ".join(w[::-1])],
                "target": f"'{w}' is spelled {d}, so backwards it is {'-'.join(w[::-1])}: '{w[::-1]}'."}
    if kind == "nth":
        n = rng.randint(1, len(w))
        return {"q": f"What is the {ordinal(n)} letter of the word '{w}'?", "gold": [w[n - 1]],
                "target": f"'{w}' is spelled {d}, so its {ordinal(n)} letter is '{w[n - 1]}'."}
    if kind == "alpha":
        ws = rng.sample(sorted(set(words)), rng.randint(3, 5))
        return {"q": f"Put these words in alphabetical order: {', '.join(ws)}.", "gold": sorted(ws),
                "target": f"In alphabetical order: {', '.join(sorted(ws))}."}
    raise ValueError(kind)


def drills(pool: list[str], n: int, seed: int, exam: bool = False) -> list[dict]:
    words = [w for w in pool if is_exam_word(w) == exam]
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = DRILLS[i % len(DRILLS)]
        out.append(dict(drill(kind, words, rng), kind=kind))
    return out


def grade_drill(answer: str, item: dict) -> bool:
    a = f" {C.norm(answer)} "
    kind, gold = item["kind"], item["gold"]
    if kind in ("spell", "reverse"):
        return any(f" {g} " in a for g in gold)
    if kind == "count":
        nums = re.findall(r"\b\d+\b", a)
        return bool(nums) and nums[-1] == gold[0]
    if kind == "nth":
        singles = [t for t in a.split() if len(t) == 1 and t.isalpha()]
        return bool(singles) and singles[-1] == gold[0]
    if kind == "alpha":
        at = [a.rfind(f" {w} ") for w in gold]
        return all(p >= 0 for p in at) and at == sorted(at)
    raise ValueError(kind)


# --- fetching ---------------------------------------------------------------------------------

def user_agent(home: str = ".") -> str:
    ua = "AegistCurriculum/0.1 (local, personal study tool; one request at a time)"
    try:
        contact = json.load(open(os.path.join(home, "aegist.json"), encoding="utf-8")).get("contact")
    except (OSError, ValueError):
        contact = None
    return f"{ua} {contact}" if contact else ua


def http_get(url: str, ua: str, pause: float, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read(5_000_000)
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == tries - 1:
                raise
            time.sleep(max(pause, float(e.headers.get("Retry-After", 5 * 2 ** k))))
        except urllib.error.URLError:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)
    raise RuntimeError("unreachable")


def clean_extract(text: str) -> str:
    """Plain article text, cut at the reference sections and at MAX_WORDS (on a paragraph)."""
    out, n = [], 0
    for para in re.split(r"\n\s*\n|\n", text):
        p = para.strip()
        if not p:
            continue
        if p.lower().strip("= ") in END_SECTIONS:
            break
        n += len(p.split())
        if n > MAX_WORDS and out:
            break
        out.append(p)
    return "\n\n".join(out)


def wiki_articles(host: str, band: tuple, n: int, get, seen: set, tries: int | None = None, log=print) -> list[dict]:
    """n random articles from `host` whose readability grade is in [band[0], band[1])."""
    q = {"action": "query", "format": "json", "formatversion": 2, "generator": "random", "grnnamespace": 0,
         "grnlimit": 1, "prop": "extracts|info", "explaintext": 1, "exsectionformat": "plain", "inprop": "url",
         "maxlag": 5}
    url = f"https://{host}/w/api.php?" + urllib.parse.urlencode(q)
    out = []
    for _ in range(tries or n * 12):
        if len(out) >= n:
            break
        pages = json.loads(get(url)).get("query", {}).get("pages", [])
        if not pages or pages[0]["title"] in seen:
            continue
        p = pages[0]
        text = clean_extract(p.get("extract", ""))
        g = grade(text)
        if len(text.split()) < 150 or not band[0] <= g < band[1]:
            continue
        seen.add(p["title"])
        out.append({"title": p["title"], "url": p.get("fullurl", f"https://{host}/wiki/{p['title']}"),
                    "revision": p.get("lastrevid"), "license": WIKI_LICENSE, "grade": round(g, 1), "text": text})
        log(f"  {p['title']} (grade {g:.1f}, {len(text.split())} words)")
    return out


def arxiv_abstracts(cats: list[str], n: int, get, seen: set, rng: random.Random, log=print) -> list[dict]:
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    out = []
    for _ in range(n * 3):
        if len(out) >= n:
            break
        cat = rng.choice(cats)
        q = {"search_query": f"cat:{cat}", "start": rng.randint(0, 1000), "max_results": 5,
             "sortBy": "submittedDate", "sortOrder": "descending"}
        root = ET.fromstring(get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(q)))
        for e in root.findall("a:entry", ns):
            aid = (e.findtext("a:id", "", ns) or "").rsplit("/abs/", 1)[-1]
            title = " ".join((e.findtext("a:title", "", ns) or "").split())
            abstract = " ".join((e.findtext("a:summary", "", ns) or "").split())
            if not aid or aid in seen or len(abstract.split()) < 60 or len(out) >= n:
                continue
            seen.add(aid)
            authors = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]
            prim = e.find("arxiv:primary_category", ns)
            out.append({"title": title, "url": f"https://arxiv.org/abs/{aid}", "revision": aid, "license": ARXIV_LICENSE,
                        "authors": authors, "category": prim.get("term") if prim is not None else cat,
                        "published": e.findtext("a:published", "", ns), "text": abstract})
            log(f"  arXiv:{aid} {title[:70]}")
    return out


def article_file(a: dict) -> str:
    head = f"# {a['title']}\n\nSource: {a['url']} (revision {a['revision']}). Licence: {a['license']}.\n"
    if a.get("authors"):
        head += f"Authors: {', '.join(a['authors'][:12])}{' et al.' if len(a['authors']) > 12 else ''}.\n"
    return f"{head}\n{a['text']}\n"


# --- progress -----------------------------------------------------------------------------------

class Curriculum:
    def __init__(self, home: str = ".", brains: str = "brains", learn_dir: str = "learn", probes: str = "probes"):
        self.home, self.brains, self.learn_dir, self.probes = home, brains, learn_dir, probes
        self.dir = os.path.join(learn_dir, "curriculum")
        os.makedirs(self.dir, exist_ok=True)

    def path(self, *p) -> str:
        return os.path.join(self.dir, *p)

    def lineage(self) -> list[dict]:
        from .learn import current_brain
        from .nx import NxFile
        with NxFile(current_brain(self.brains)) as nx:
            return nx.header.get("lineage", [])

    def lessons_done(self) -> dict[str, int]:
        done = {s["name"]: 0 for s in STAGES}
        for e in self.lineage():
            m = e.get("material", {})
            files = [f.replace("\\", "/") for f in m.get("files", [])]
            tags = {f.split("/")[1] for f in files if f.startswith("curriculum/") and f.count("/") >= 2}
            tags |= {k.split("/", 1)[1] for k in m.get("lessons", {}) if k.startswith("curriculum/")}
            for s in STAGES:
                if slug(s["name"]) in tags:
                    done[s["name"]] += 1
        return done

    def exams(self) -> list[dict]:
        p = self.path("exams.jsonl")
        return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()] if os.path.exists(p) else []

    def passed(self, stage: dict) -> bool | None:
        """None: the stage has no exam. Else whether the live brain passed it."""
        if "pass" not in stage:
            return None
        from .learn import current_brain
        brain = os.path.basename(current_brain(self.brains))
        mine = [e for e in self.exams() if e["stage"] == stage["name"] and e["brain"] == brain]
        return bool(mine) and mine[-1]["passed"]

    def stage(self) -> dict | None:
        done = self.lessons_done()
        for s in STAGES:
            if done[s["name"]] < s["quota"] or self.passed(s) is False:
                return s
        return None

    def seen(self) -> set:
        """Titles and arXiv ids already fetched, so no text is read twice."""
        p, out = self.path("sources.jsonl"), set()
        if os.path.exists(p):
            for x in open(p, encoding="utf-8"):
                if x.strip():
                    r = json.loads(x)
                    out |= {r["title"], str(r["revision"])}
        return out

    def texts(self):
        """English text on hand for the word pool: saved articles, and the brain's own answers."""
        for root, _, files in os.walk(self.path("texts")):
            for f in files:
                yield open(os.path.join(root, f), encoding="utf-8").read()
        for d in sorted(os.listdir(self.probes)) if os.path.isdir(self.probes) else []:
            ans = os.path.join(self.probes, d, "answers.json")
            if d.startswith("data-") and os.path.exists(ans):
                meta = json.load(open(os.path.join(self.probes, d, "items.json"), encoding="utf-8"))
                tok = _tokenizer(meta["brain"])
                for r in json.load(open(ans, encoding="utf-8")):
                    yield tok.decode(r["greedy"])


_TOKS = {}


def _tokenizer(brain: str):
    if brain not in _TOKS:
        from .nx import NxFile
        from .tokenizer import AegistTokenizer
        with NxFile(brain) as nx:
            _TOKS[brain] = AegistTokenizer.from_json_str(nx.blob("tokenizer.json").decode("utf-8"))
    return _TOKS[brain]


# --- commands -----------------------------------------------------------------------------------

def make_lesson(cur: Curriculum, stage: dict, *, n: int | None = None, seed: int = 0, get=None, log=print) -> int:
    """Put one lesson of `stage` where learning time reads it. Returns how many items."""
    tag = slug(stage["name"])
    if stage["kind"] == "drills":
        pool = word_pool(cur.texts())
        log(f"word pool: {len(pool)} words ({sum(map(is_exam_word, pool))} kept for the exam)")
        rng = random.Random(seed)
        recs = []
        for it in drills(pool, n or 80, seed=seed + 7919 * (cur.lessons_done()[stage["name"]] + 1)):
            q = it["q"] + (C.ASK if rng.random() < 0.5 else "")
            recs.append({"kind": "chat", "mode": "normal", "turns": [["user", q]], "target": it["target"],
                         "source": f"curriculum/{tag}"})
        with open(os.path.join(cur.learn_dir, "lessons.jsonl"), "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"{len(recs)} drills added to {os.path.join(cur.learn_dir, 'lessons.jsonl')}")
        return len(recs)
    ua = user_agent(cur.home)
    seen = cur.seen()
    if stage["kind"] == "wiki":
        get = get or (lambda u: http_get(u, ua, 1.0))
        log(f"reading {stage['host']}, readability grade {stage['grade'][0]} to {stage['grade'][1]}:")
        arts = wiki_articles(stage["host"], stage["grade"], n or 12, get, seen, log=log)
    else:
        get = get or (lambda u: http_get(u, ua, 3.0))
        log("reading arXiv abstracts:")
        arts = arxiv_abstracts(stage["cats"], n or 20, get, seen, random.Random(seed + int(time.time())), log=log)
    inbox = os.path.join(cur.learn_dir, "inbox", "curriculum", tag)
    os.makedirs(inbox, exist_ok=True)
    os.makedirs(cur.path("texts", tag), exist_ok=True)
    with open(cur.path("sources.jsonl"), "a", encoding="utf-8") as src:
        for a in arts:
            name = f"{slug(a['title'])}-{hashlib.sha256(a['url'].encode()).hexdigest()[:8]}.md"
            body = article_file(a)
            for d in (inbox, cur.path("texts", tag)):
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    f.write(body)
            src.write(json.dumps({k: v for k, v in a.items() if k != "text"} | {"stage": stage["name"], "file": name,
                                 "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False) + "\n")
    log(f"{len(arts)} texts added to {inbox}")
    return len(arts)


def exam(cur: Curriculum, stage: dict, *, n: int = 100, seed: int = 0, log=print) -> dict | None:
    """The letters exam on the live brain: n questions about words the drills never use."""
    if stage["kind"] != "drills":
        log(f"{stage['name']} has no exam of its own; learning time's gates decide each lesson")
        return None
    from .learn import current_brain
    brain = current_brain(cur.brains)
    items = drills(word_pool(cur.texts()), n, seed=seed + 1, exam=True)
    out_dir = cur.path("exams", f"{os.path.splitext(os.path.basename(brain))[0]}-{slug(stage['name'])}")
    C.answer_items(brain, items, out_dir, taps=[20], max_new=96, seed=seed, log=log)
    tok = _tokenizer(brain)
    res = json.load(open(os.path.join(out_dir, "answers.json"), encoding="utf-8"))
    by = {}
    for it, r in zip(items, res):
        by.setdefault(it["kind"], []).append(grade_drill(tok.decode(r["greedy"]), it))
    score = sum(sum(v) for v in by.values()) / len(items)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "brain": os.path.basename(brain), "stage": stage["name"],
           "n": len(items), "score": round(score, 3), "by_kind": {k: round(sum(v) / len(v), 3) for k, v in by.items()},
           "passed": score >= stage["pass"]}
    with open(cur.path("exams.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"exam ({stage['name']}): {score:.0%} right (pass at {stage['pass']:.0%}); by kind: " +
        ", ".join(f"{k} {v:.0%}" for k, v in rec["by_kind"].items()))
    return rec


def status(cur: Curriculum, log=print) -> int:
    done, now = cur.lessons_done(), cur.stage()
    for s in STAGES:
        p = cur.passed(s)
        exam_note = "" if p is None else f", exam {'passed' if p else 'not passed'}"
        log(f"{'->' if s is now else '  '} {s['name']:14s} {done[s['name']]}/{s['quota']} lessons{exam_note}")
    for e in cur.exams()[-5:]:
        log(f"   exam {e['time']} on {e['brain']}: {e['stage']} {e['score']:.0%}")
    if now is None:
        log("every stage is done")
    return 0


def run(cur: Curriculum, a, log=print) -> int:
    from . import learn as L

    for k in range(a.lessons):
        stage = cur.stage()
        if stage is None:
            log("every stage is done")
            return 0
        log(f"--- lesson {k + 1}/{a.lessons}: {stage['name']} ---")
        if stage.get("pass") is not None and not any(e["stage"] == stage["name"] for e in cur.exams()):
            exam(cur, stage, seed=a.seed, log=log)  # a baseline before the first lesson
        if not make_lesson(cur, stage, n=a.n, seed=a.seed, log=log):
            log("nothing could be fetched for this lesson; stopping")
            return 1
        code = L.main(["--brains", cur.brains, "--learn-dir", cur.learn_dir, "--probes", cur.probes] + a.learn_args)
        if code != 0:
            put_aside(cur, log)
            log(f"learning time did not accept the lesson (exit {code}); stopping")
            return code
        if stage.get("pass") is not None:
            exam(cur, stage, seed=a.seed, log=log)
        if a.refresh_governor:
            n = max([int(d[5:]) for d in os.listdir(cur.probes) if re.fullmatch(r"data-\d+", d)], default=0) + 1
            log("governor: the new version answers the question bank, for the next lesson's records and gate")
            C.collect(L.current_brain(cur.brains), os.path.join(cur.probes, f"data-{n}"), bank="single", samples=0,
                      log=log)
    return status(cur, log)


def put_aside(cur: Curriculum, log=print):
    """A rejected lesson's curriculum material is moved out of learning time's way."""
    import shutil
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = cur.path("rejected", stamp)
    inbox = os.path.join(cur.learn_dir, "inbox", "curriculum")
    lessons = os.path.join(cur.learn_dir, "lessons.jsonl")
    if os.path.isdir(inbox):
        os.makedirs(dst, exist_ok=True)
        shutil.move(inbox, os.path.join(dst, "inbox"))
    if os.path.exists(lessons):
        os.makedirs(dst, exist_ok=True)
        shutil.move(lessons, os.path.join(dst, "lessons.jsonl"))
    log(f"lesson material moved to {dst}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aegist.curriculum", description="Aegist's curriculum")
    ap.add_argument("cmd", choices=["status", "lesson", "exam", "run"])
    ap.add_argument("--brains", default="brains")
    ap.add_argument("--learn-dir", default="learn")
    ap.add_argument("--probes", default="probes")
    ap.add_argument("--lessons", type=int, default=1, help="run: lessons in a row")
    ap.add_argument("--n", type=int, default=None, help="items per lesson (drills 80, articles 12, abstracts 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-governor-refresh", dest="refresh_governor", action="store_false",
                    help="run: skip answering the question bank with each new version")
    a, rest = ap.parse_known_args(argv)
    a.learn_args = rest  # anything else goes to learning time, e.g. --lr 2e-5
    cur = Curriculum(".", a.brains, a.learn_dir, a.probes)
    if a.cmd == "status":
        return status(cur)
    stage = cur.stage()
    if stage is None:
        print("every stage is done")
        return 0
    if a.cmd == "lesson":
        return 0 if make_lesson(cur, stage, n=a.n, seed=a.seed) else 1
    if a.cmd == "exam":
        exam(cur, stage, seed=a.seed)
        return 0
    return run(cur, a)


if __name__ == "__main__":
    sys.exit(main())
