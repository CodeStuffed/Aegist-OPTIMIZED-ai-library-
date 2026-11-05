"""The tricky test: 50 questions built to catch the mistakes a quick reader makes, graded by code.

Five sections of 10, each question asked in a new conversation through the whole chat pipeline
(`aegist exam`), as a user would ask it, with thinking left on "auto":

  riddles       famous riddles with a twist (the surgeon who IS the father), so a memorised answer is wrong
  made-up       books, treaties, elements and events that do not exist, and false premises: right only
                when the reply says so instead of inventing details
  instructions  exact constraints (10 words all starting with S, no letter e, only a number), checked
                on the whole reply, so a preamble or an extra word fails
  logic         trap arithmetic and reasoning (9.11 vs 9.9, Sally's sisters, the lily pads)
  look-ups      events from 2023-2025 that need the web search tool (Wikipedia)

Score = right answers out of 50. --no-web switches the search tool off, to see what it is worth.
Results go to tricky/ in the Aegist folder: every answer as JSON, a text report and history.jsonl.

Run:  aegist tricky [--brain FILE] [--no-web] [--no-tools] [--think auto|on|off] [--seed S]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time

from . import confidence as C
from .aq import live_brain

# --- graders --------------------------------------------------------------------------------

DENY = (r"(doesn't|does not|didn't|did not|don't|do not) (seem to )?exist|no such|not a real|isn't a real|is not a real|"
        r"there (is|was|are|were) no\b|fictional|made[- ]up|not aware of|no record|no information|couldn't find|"
        r"could not find|can't find|cannot find|unable to find|not familiar|never (existed|written|signed|made|released|"
        r"published)|no (known|official|evidence)|not (an? )?(actual|recognized|recognised|known)|invented|"
        r"doesn't appear|does not appear|not appear to exist|no (book|film|movie|treaty|element|country|play)|"
        r"(don't|do not) know (of )?(any|a|such|this)|isn't part of|is not part of|not one of")


def clean(a: str) -> str:
    a = re.sub(r"\s*\[unsure[^\]]*\]", "", a)
    a = re.sub(r"```\w*", "", a).replace("**", "").replace("__", "")
    return a.strip().strip('"“”\'`').strip()


def words(a: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", a)


def has(*pats):
    return lambda a: all(re.search(p, a, re.I) for p in pats)


def lacks(*pats):
    return lambda a: not any(re.search(p, a, re.I) for p in pats)


def both(*fs):
    return lambda a: all(f(a) for f in fs)


def either(*fs):
    return lambda a: any(f(a) for f in fs)


NUMBER_WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def number(n, traps=()):
    """the first number in the reply is the right one, or the right number appears and no trap number
    does (number words count: "Nine words.")"""
    def f(a):
        t = re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", lambda m: NUMBER_WORDS[m.group(1).lower()], a, flags=re.I)
        got = re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", t.replace(",", ""))
        return bool(got) and (got[0] == str(n) or (str(n) in got and not any(str(t) in got for t in traps)))
    return f


def denies(extra=None):
    return either(has(DENY), has(extra)) if extra else has(DENY)


def exact(s):
    return lambda a: clean(a).rstrip(".").strip().lower() == s.lower()


def ten_s(a):
    w = words(clean(a))
    return len(w) == 10 and all(x[0].lower() == "s" for x in w)


def no_e(a):
    c = clean(a)
    return len(words(c)) >= 8 and "e" not in c.lower()


def three_sentences(a):
    c = " ".join(clean(a).split())
    return len(re.findall(r"[^.!?]+[.!?]+", c)) == 3 and c[-1:] in ".!?"


def four_fruits(a):
    c = clean(a).rstrip(".")
    parts = [p.strip() for p in c.split(",")]
    return "\n" not in c and len(parts) == 4 and all(p and len(p.split()) <= 3 for p in parts)


def five_caps(a):
    c = clean(a)
    w = words(c)
    return len(w) == 5 and all(x.upper() == x for x in w) and any(ch.isalpha() for ch in c)


def acrostic_cat(a):
    lines = [l.strip().lstrip("*-#>0123456789. ").strip() for l in clean(a).splitlines() if l.strip()]
    return len(lines) == 3 and [l[:1].upper() for l in lines] == ["C", "A", "T"]


def four_letters(a):
    w = words(clean(a))
    return len(w) >= 4 and all(len(x) == 4 for x in w)


def twelve_no_a(a):
    c = clean(a)
    return len(words(c)) == 12 and "a" not in c.lower()


# --- the questions -----------------------------------------------------------------------------

Q = [
    # riddles with a twist
    ("riddles", "A father and his son are in a car crash. The father is unhurt, and the son is rushed to hospital. "
                "The surgeon, who is the boy's father, says: \"I can't operate on this boy, he's my son!\" Who is the surgeon?",
     both(has(r"father|dad"), lacks(r"^\W*(the surgeon is |it's |it is )?(the boy's |his |the )?mother\b"))),
    ("riddles", "A rooster lays an egg on the very top of a sloped barn roof. Which side does the egg roll down?",
     has(r"roosters? (don't|do not|can't|cannot|doesn't|does not|never)|roosters are male|male chickens?|hens lay|no egg|wouldn't lay|won't lay")),
    ("riddles", "How many animals of each kind did Moses take on the ark?", has(r"noah")),
    ("riddles", "A plane crashes exactly on the border between the United States and Canada. Where do they bury the survivors?",
     has(r"(don't|do not|wouldn't|would not|not|never) bury|survivors are (alive|living|not dead)|they're alive|still alive|alive")),
    ("riddles", "A farmer has 17 sheep. All but 9 run away. How many sheep does he have left?", number(9, traps=[8])),
    ("riddles", "Which weighs more: a kilogram of feathers or a kilogram of steel?", has(r"same|neither|equal|both weigh|weigh the same")),
    ("riddles", "You're running a race and you overtake the person in second place. What place are you in now?",
     both(has(r"second|2nd"), lacks(r"(you are|you're|you'd be|now) (in )?first"))),
    ("riddles", "A doctor gives you three pills and tells you to take one every half hour. How many minutes until you have taken all of them?",
     number(60, traps=[90])),
    ("riddles", "Mary's father has five daughters: Nana, Nene, Nini and Nono. What is the name of the fifth daughter?",
     both(has(r"\bmary\b"), lacks(r"\bnunu\b(?!.*mary)"))),
    ("riddles", "I have two coins that add up to 30 cents, and one of them is not a nickel. What are the two coins?",
     has(r"quarter", r"nickel")),

    # things that do not exist, and false premises
    ("made-up", "Give me a short summary of the plot of 'Harry Potter and the Crystal of Shadows'.", denies()),
    ("made-up", "Who won the Nobel Prize in Mathematics in 2019?",
     denies(r"no nobel|isn't a nobel|is not a nobel|not a nobel|nobel (prize )?(doesn't|does not|has no|isn't)|abel|fields")),
    ("made-up", "Summarise the Treaty of Velmora, signed between Portugal and Japan in 1823.", denies()),
    ("made-up", "What is the melting point of the element zorbium?", denies(r"not an? (element|real)|isn't an? (element|real)")),
    ("made-up", "Who was the fourth person to walk on Mars?",
     has(r"no one|nobody|no human|no person|no people|not yet|hasn't|has not|have not|haven't|never|no astronaut")),
    ("made-up", "In what year did Shakespeare write his play 'The Merchant of Copenhagen'?",
     denies(r"merchant of venice|not (one of )?(his|a shakespeare)")),
    ("made-up", "Explain how Python's built-in list.flatten() method works.",
     either(denies(), has(r"no (built-in )?(flatten|such)|doesn't have|does not have|isn't a|is not a|not a built-in|lists? (don't|do not) have|has no"))),
    ("made-up", "Why was the Great Wall of China built in 1950?",
     has(r"wasn't|was not|not built in 1950|isn't|centuries|\bbc\b|\bbce\b|ming|qin|ancient|false|incorrect|mistaken|didn't|did not")),
    ("made-up", "Tell me about the 1998 film 'The Glass Orchard', directed by Steven Spielberg.", denies(r"not (a |one of )?(his|spielberg)")),
    ("made-up", "What is the capital of the country Valdoria?", denies(r"not an? (real|actual) country|isn't an? (real|actual) country")),

    # instructions checked exactly
    ("instructions", "Write exactly 10 words, every word starting with the letter S. Reply with only those words.", ten_s),
    ("instructions", "Write one sentence of at least 8 words that does not contain the letter 'e' anywhere. Reply with only the sentence.", no_e),
    ("instructions", "Reply with exactly three sentences about the sea, and nothing else.", three_sentences),
    ("instructions", "Write the word 'strawberry' backwards, and nothing else.", exact("yrrebwarts")),
    ("instructions", "List exactly four fruits, separated by commas, with no other text.", four_fruits),
    ("instructions", "Write a sentence of exactly five words, all in capital letters. Reply with only the sentence.", five_caps),
    ("instructions", "Write an acrostic poem for the word CAT: exactly three lines whose first letters spell C, A, T. Reply with only the poem.", acrostic_cat),
    ("instructions", "Answer with only a number: how many days are in a leap year?", exact("366")),
    ("instructions", "Write one sentence in which every word has exactly four letters. Reply with only the sentence.", four_letters),
    ("instructions", "Write exactly 12 words about dogs without using the letter 'a'. Reply with only the words.", twelve_no_a),

    # trap logic and arithmetic
    ("logic", "Which number is larger, 9.11 or 9.9?", both(has(r"9\.9\b"), lacks(r"9\.11 is (the )?(larger|bigger|greater)"))),
    ("logic", "How many times does the letter 'r' appear in the word 'strawberry'?", number(3, traps=[2])),
    ("logic", "If 5 machines take 5 minutes to make 5 widgets, how many minutes would 100 machines take to make 100 widgets?",
     number(5, traps=[100, 20])),
    ("logic", "A patch of lily pads doubles in size every day. It takes 48 days to cover the whole lake. How many days does it take to cover half the lake?",
     number(47, traps=[24])),
    ("logic", "Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have?",
     both(either(number(1, traps=[6]), has(r"\bone\b")), lacks(r"sally has (two|2|six|6) sisters"))),
    ("logic", "A clock takes 5 seconds to strike six o'clock (six strikes). How many seconds does it take to strike twelve o'clock?",
     number(11, traps=[10])),
    ("logic", "If today is Wednesday, what day of the week was it 100 days ago?", both(has(r"monday"), lacks(r"\b(was|it's|it is) (a )?(tuesday|sunday|friday)"))),
    ("logic", "Did Albert Einstein win his 1921 Nobel Prize for the theory of relativity?", has(r"photoelectric")),
    ("logic", "How many words are in this sentence: 'The quick brown fox jumps over the lazy dog'?", number(9, traps=[8, 10])),
    ("logic", "I'm facing north. I turn 90 degrees to my right, then turn 180 degrees, then turn 90 degrees to my left. Which way am I facing now?",
     both(has(r"south"), lacks(r"facing (north|east|west)\b(?!.*south)"))),

    # look-ups that need the web
    ("look-ups", "Who won the 2025 Nobel Prize in Literature?", has(r"krasznahorkai")),
    ("look-ups", "Who was elected pope in May 2025, and what papal name did he take?", has(r"leo\s*(xiv|14)")),
    ("look-ups", "Which team won Super Bowl LIX in February 2025?", has(r"eagles|philadelphia")),
    ("look-ups", "Who won the men's Ballon d'Or in 2024?", has(r"rodri")),
    ("look-ups", "Which country won the UEFA Euro 2024 football championship?", has(r"spain")),
    ("look-ups", "Who were the winners of the 2024 Nobel Prize in Physics?", has(r"hopfield", r"hinton")),
    ("look-ups", "What is the title of the 2017 paper that introduced the Transformer architecture?", has(r"attention is all you need")),
    ("look-ups", "Which country won the 2023 Rugby World Cup?", has(r"south africa")),
    ("look-ups", "Which city hosted the 2024 Summer Olympics?", has(r"paris")),
    ("look-ups", "Who won the 2024 Nobel Prize in Chemistry?", has(r"baker", r"hassabis", r"jumper")),
]
assert len(Q) == 50 and all(sum(s == n for s, _, _ in Q) == 10 for n in ("riddles", "made-up", "instructions", "logic", "look-ups"))
SECTIONS = ["riddles", "made-up", "instructions", "logic", "look-ups"]


def run(brain: str, out_dir: str, *, web: bool, tools: bool, think: str, seed: int) -> dict:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = os.path.join(out_dir, f"tricky-{stamp}")
    os.makedirs(work, exist_ok=True)
    setup = "chat, think " + think + ("" if tools else ", no tools") + ("" if web or not tools else ", no web")
    print(f"tricky test: 50 questions for {os.path.basename(brain)} through chat ({setup})", flush=True)
    inp, outp = os.path.join(work, "exam.json"), os.path.join(work, "answers.json")
    json.dump({"prompts": [q for _, q, _ in Q], "think": think, "seed": seed, "max_new": 600}, open(inp, "w", encoding="utf-8"))
    cmd = [C.exe_path(), "exam", inp, outp, "--brain", brain] + ([] if tools else ["--no-tools"])
    env = dict(os.environ, **({} if web else {"AEGIST_WEB": "off"}))
    t0 = time.time()
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"aegist exam exited with {r.returncode}")
    got = json.load(open(outp, encoding="utf-8"))
    rows = []
    for (sec, q, grade), g in zip(Q, got):
        a = g["answer"]
        rows.append({"section": sec, "q": q, "answer": a, "right": bool(grade(a)), "tools": g["tools"],
                     "seconds": round(g.get("seconds", 0), 1)})
    secs = {s: sum(r["right"] for r in rows if r["section"] == s) for s in SECTIONS}
    return {"score": sum(secs.values()), "of": 50, "sections": secs, "brain": os.path.abspath(brain), "setup": setup,
            "seed": seed, "web_searches": sum(str(t).startswith("web search") for r in rows for t in r["tools"]),
            "seconds": round(time.time() - t0), "when": stamp, "rows": rows}


def report(res: dict, previous: dict | None) -> str:
    lines = [f"tricky test: {res['score']} / 50   ({os.path.basename(res['brain'])}, {res['setup']})"]
    if previous:
        lines[0] += f"   last run like this: {previous['score']} ({previous['when']})"
    lines.append("")
    for s, v in res["sections"].items():
        lines.append(f"  {s:13} {v:2d}/10  {'#' * v}")
    lines.append(f"\n  web searches: {res['web_searches']}; {res['seconds']} s")
    wrong = [r for r in res["rows"] if not r["right"]]
    if wrong:
        lines.append("\n  wrong:")
        for r in wrong:
            a = " ".join(r["answer"].split())
            lines.append(f"    [{r['section']}] {r['q'][:90]}\n      -> {a[:160]}")
    lines.append("\nFull answers: " + res["file"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aegist tricky", description="50 tricky questions, graded by code.")
    ap.add_argument("--brain", help="the brain to test (default: the live one)")
    ap.add_argument("--brains", default="brains")
    ap.add_argument("--out", default="tricky")
    ap.add_argument("--no-web", action="store_true", help="switch the web search tool off")
    ap.add_argument("--no-tools", action="store_true", help="no tools at all")
    ap.add_argument("--think", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--regrade", metavar="FILE", help="grade a saved run again with the current graders (no brain)")
    a = ap.parse_args(argv)
    if a.regrade:
        res = json.load(open(a.regrade, encoding="utf-8"))
        grade = {q: g for _, q, g in Q}
        for r in res["rows"]:
            r["right"] = bool(grade[r["q"]](r["answer"]))
        res["sections"] = {s: sum(r["right"] for r in res["rows"] if r["section"] == s) for s in SECTIONS}
        res["score"] = sum(res["sections"].values())
        res["regraded"] = time.strftime("%Y%m%d-%H%M%S")
        json.dump(res, open(a.regrade, "w", encoding="utf-8"), indent=1)
        text = report(res, None)
        open(a.regrade[:-5] + ".txt", "w", encoding="utf-8").write(text + "\n")
        print(text)
        return
    brain = a.brain or live_brain(a.brains)
    os.makedirs(a.out, exist_ok=True)
    res = run(brain, a.out, web=not a.no_web, tools=not a.no_tools, think=a.think, seed=a.seed)
    hist = os.path.join(a.out, "history.jsonl")
    previous = None
    if os.path.exists(hist):
        same = [h for h in (json.loads(l) for l in open(hist, encoding="utf-8") if l.strip())
                if (h["setup"], h["brain"]) == (res["setup"], res["brain"])]
        previous = same[-1] if same else None
    res["file"] = os.path.abspath(os.path.join(a.out, f"tricky-{res['when']}.json"))
    json.dump(res, open(res["file"], "w", encoding="utf-8"), indent=1)
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in res.items() if k != "rows"}) + "\n")
    text = report(res, previous)
    open(res["file"][:-5] + ".txt", "w", encoding="utf-8").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
