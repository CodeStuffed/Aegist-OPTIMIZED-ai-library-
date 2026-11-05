"""The language workbench: Aegist designs its own languages, and the workbench makes them exact.

    python -m aegist.conlang design                 # Aegist proposes a language; checked, completed, saved
    python -m aegist.conlang new --name Velani      # the same without Aegist: choices from flags and a seed
    python -m aegist.conlang show NAME              # the grammar sheet
    python -m aegist.conlang translate NAME "The big dogs saw the fish."
    python -m aegist.conlang check NAME "kato mirata sulin"
    python -m aegist.conlang lessons NAME           # drills into learn/lessons.jsonl for learning time
    python -m aegist.conlang exam NAME              # Aegist translates new sentences; graded exactly

A language is a JSON file in langs/: its sounds (consonants and vowels, in a plain-ASCII
spelling), the syllable shapes it allows, word order, where adjectives go, how plurals,
objects, tenses and negation are marked, and a lexicon of 70 core words. Every word and
every inflected form is checked against the sound rules, and no two meanings share a form.

Aegist's part: `design` asks the live brain (the Rust engine) for the design choices as
JSON. Each choice is checked; a choice that is missing or not allowed is filled from the seed,
and the sheet says which choices are Aegist's and which were filled. Words are generated
from the chosen sounds, so they are new; the meanings are a fixed core vocabulary.

Sentences follow one template, so translation in both directions is exact:
  The (adjective) noun(s) [do/does/did not | will (not)] verb (the (adjective) noun(s)).
Learning: `lessons` writes translation and vocabulary drills (in both directions);
`exam` asks for translations of sentences the drills never contained, closed-book by
default (the brain must have learned the language) or --open-book (the grammar sheet is in
the question).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys

CONSONANTS = ["p", "t", "k", "b", "d", "g", "m", "n", "s", "z", "f", "v", "h", "l", "r", "w", "y",
              "sh", "ch", "kh", "ts", "ng"]
VOWELS = ["a", "e", "i", "o", "u", "aa", "ee", "ii", "oo", "uu"]
SYLLABLES = ["CV", "CVC", "V", "VC", "CVN", "CCV"]
ORDERS = ["SVO", "SOV", "VSO", "VOS", "OVS", "OSV"]
NASALS = {"m", "n", "ng"}
GLIDES = {"l", "r", "w", "y"}

NOUNS = {"dog": "dogs", "cat": "cats", "bird": "birds", "fish": "fish", "tree": "trees", "stone": "stones",
         "river": "rivers", "mountain": "mountains", "house": "houses", "child": "children", "woman": "women",
         "man": "men", "friend": "friends", "king": "kings", "star": "stars", "moon": "moons", "flower": "flowers",
         "boat": "boats", "horse": "horses", "snake": "snakes", "wolf": "wolves", "apple": "apples", "book": "books",
         "song": "songs", "door": "doors", "road": "roads", "cloud": "clouds", "fire": "fires", "egg": "eggs",
         "bear": "bears"}
ADJECTIVES = ["big", "small", "old", "new", "red", "black", "white", "good", "bad", "fast", "slow", "happy", "cold",
              "warm", "tall"]
# base: (third person singular, past)
TRANSITIVE = {"see": ("sees", "saw"), "eat": ("eats", "ate"), "find": ("finds", "found"), "hear": ("hears", "heard"),
              "follow": ("follows", "followed"), "carry": ("carries", "carried"), "love": ("loves", "loved"),
              "help": ("helps", "helped"), "build": ("builds", "built"), "catch": ("catches", "caught"),
              "hold": ("holds", "held"), "want": ("wants", "wanted"), "know": ("knows", "knew"),
              "call": ("calls", "called"), "push": ("pushes", "pushed")}
INTRANSITIVE = {"sleep": ("sleeps", "slept"), "run": ("runs", "ran"), "swim": ("swims", "swam"), "fly": ("flies", "flew"),
                "sing": ("sings", "sang"), "fall": ("falls", "fell"), "laugh": ("laughs", "laughed"),
                "wait": ("waits", "waited"), "walk": ("walks", "walked"), "sit": ("sits", "sat")}
VERBS = TRANSITIVE | INTRANSITIVE
TENSES = ["present", "past", "future"]


# --- sounds -------------------------------------------------------------------------------------

def segment(word: str, spec: dict) -> list[str] | None:
    """The word's sounds, longest spelling first; None if a letter is not in the language."""
    inv = sorted(spec["consonants"] + spec["vowels"], key=len, reverse=True)
    out, i = [], 0
    while i < len(word):
        s = next((p for p in inv if word.startswith(p, i)), None)
        if s is None:
            return None
        out.append(s)
        i += len(s)
    return out


def _classes(pattern: str) -> list[str]:
    return ["C", "L", "V"] if pattern == "CCV" else list(pattern)


def _fits(cls: str, s: str, spec: dict) -> bool:
    if cls == "V":
        return s in spec["vowels"]
    if cls == "N":
        return s in NASALS and s in spec["consonants"]
    if cls == "L":
        return s in GLIDES and s in spec["consonants"]
    return s in spec["consonants"]


def is_word(word: str, spec: dict) -> bool:
    """A possible word: its sounds split into the language's syllable shapes."""
    seg = segment(word, spec)
    if not seg:
        return False
    pats = [_classes(p) for p in spec["syllables"]]
    ok = [True] + [False] * len(seg)
    for i in range(len(seg)):
        if ok[i]:
            for p in pats:
                if i + len(p) <= len(seg) and all(_fits(c, s, spec) for c, s in zip(p, seg[i:])):
                    ok[i + len(p)] = True
    return ok[-1]


def syllable(spec: dict, rng: random.Random, pattern: str | None = None) -> str:
    pattern = pattern or rng.choice(spec["syllables"])
    out = []
    for c in _classes(pattern):
        pool = [s for s in spec["consonants"] + spec["vowels"] if _fits(c, s, spec)]
        if c == "C" and pattern == "CCV":  # a cluster starts with a non-glide
            pool = [s for s in pool if s not in GLIDES] or pool
        out.append(rng.choice(pool))
    return "".join(out)


def new_word(spec: dict, rng: random.Random, n_syll: int) -> str:
    for _ in range(200):
        w = "".join(syllable(spec, rng) for _ in range(n_syll))
        if 2 <= len(w) <= 12 and is_word(w, spec):
            return w
    raise ValueError("could not make a word from these sounds")


# --- a language -----------------------------------------------------------------------------------

def complete(choices: dict, seed: int) -> tuple[dict, dict]:
    """A full, valid design from `choices` (Aegist's or the user's): each choice is checked and
    kept if allowed, else filled from the seed. Returns (spec, where each choice came from)."""
    rng = random.Random(seed)
    spec, origin = {}, {}

    def take(key, ok, fill):
        v = choices.get(key)
        if v is not None and ok(v):
            spec[key], origin[key] = v, "chosen"
        else:
            spec[key], origin[key] = fill(), ("filled" if v is None else f"filled (not allowed: {v!r})")

    name_ok = lambda v: isinstance(v, str) and re.fullmatch(r"[A-Za-z]{3,16}", v)
    take("name", name_ok, lambda: None)
    take("consonants", lambda v: isinstance(v, list) and 6 <= len(set(v)) <= 18 and set(v) <= set(CONSONANTS)
         and bool(set(v) - GLIDES - NASALS), lambda: rng.sample(CONSONANTS, rng.randint(9, 14)))
    spec["consonants"] = sorted(set(spec["consonants"]), key=CONSONANTS.index)
    take("vowels", lambda v: isinstance(v, list) and 3 <= len(set(v)) <= 8 and set(v) <= set(VOWELS),
         lambda: ["a", "i", "u"] + rng.sample(["e", "o", "aa", "ii", "uu"], rng.randint(0, 3)))
    spec["vowels"] = sorted(set(spec["vowels"]), key=VOWELS.index)
    take("syllables", lambda v: isinstance(v, list) and v and set(v) <= set(SYLLABLES),
         lambda: ["CV"] + rng.sample(["CVC", "V", "CVN", "CCV"], rng.randint(1, 2)))
    if "CV" not in spec["syllables"]:  # affixes are CV syllables, so every language has them
        spec["syllables"] = ["CV"] + list(spec["syllables"])
        origin["syllables"] += " (CV added)"
    if "CVN" in spec["syllables"] and not NASALS & set(spec["consonants"]):
        spec["syllables"] = [s for s in spec["syllables"] if s != "CVN"]
        origin["syllables"] += " (CVN dropped: no nasals)"
    if "CCV" in spec["syllables"] and not GLIDES & set(spec["consonants"]):
        spec["syllables"] = [s for s in spec["syllables"] if s != "CCV"]
        origin["syllables"] += " (CCV dropped: no l, r, w or y)"
    take("word_order", lambda v: v in ORDERS, lambda: rng.choice(["SOV", "SVO", "SVO", "VSO", "SOV"]))
    take("adjective", lambda v: v in ("before", "after"), lambda: rng.choice(["before", "after"]))
    take("plural", lambda v: v in ("suffix", "prefix"), lambda: rng.choice(["suffix", "suffix", "prefix"]))
    take("object_marker", lambda v: isinstance(v, bool), lambda: spec["word_order"] not in ("SVO", "VSO"))
    take("negation", lambda v: v in ("before", "after"), lambda: rng.choice(["before", "after"]))
    take("flavor", lambda v: isinstance(v, str) and 0 < len(v) <= 200, lambda: "")
    spec["seed"] = seed
    make_words(spec, rng)
    if spec["name"] is None:  # a name in its own sounds
        spec["name"] = new_word(spec, rng, 2)
    spec["name"] = spec["name"].capitalize()
    return spec, origin


def make_words(spec: dict, rng: random.Random):
    """Affixes, particles and the lexicon; every surface form is a possible word and no two
    meanings share one."""
    for _ in range(100):
        aff = {k: syllable(spec, rng, "CV") for k in ("plural", "object", "past", "future")}
        neg = new_word(spec, rng, 1)
        if len(set(aff.values()) | {neg}) < 5:
            continue
        spec["affixes"], spec["negation_word"] = aff, neg
        lex, used = {}, {neg}
        ok = True
        for gloss in list(NOUNS) + ADJECTIVES + list(VERBS):
            for _ in range(60):
                w = new_word(spec, rng, rng.choice([1, 2, 2, 2, 3]))
                forms = set(all_forms(spec, gloss, w))
                if not forms & used and all(is_word(f, spec) for f in forms):
                    lex[gloss], used = w, used | forms
                    break
            else:
                ok = False
                break
        if ok:
            spec["lexicon"] = lex
            return
    raise ValueError("these sounds are too few for a lexicon; choose more consonants or vowels")


def all_forms(spec: dict, gloss: str, stem: str) -> list[str]:
    if gloss in NOUNS:
        return [noun(spec, stem, pl, obj) for pl in (False, True) for obj in (False, True)]
    if gloss in VERBS:
        return [verb(spec, stem, t) for t in TENSES]
    return [stem]


def noun(spec: dict, stem: str, plural: bool, obj: bool) -> str:
    a = spec["affixes"]
    w = (stem + a["plural"] if spec["plural"] == "suffix" else a["plural"] + stem) if plural else stem
    return w + a["object"] if obj and spec["object_marker"] else w


def verb(spec: dict, stem: str, tense: str) -> str:
    return stem + spec["affixes"][tense] if tense != "present" else stem


# --- sentences --------------------------------------------------------------------------------------

def random_sentence(rng: random.Random) -> dict:
    v = rng.choice(list(VERBS))
    np = lambda: {"noun": rng.choice(list(NOUNS)), "plural": rng.random() < 0.4,
                  "adj": rng.choice(ADJECTIVES) if rng.random() < 0.5 else None}
    return {"subj": np(), "verb": v, "tense": rng.choice(TENSES), "neg": rng.random() < 0.25,
            "obj": np() if v in TRANSITIVE else None}


def english(s: dict) -> str:
    def np(x):
        n = NOUNS[x["noun"]] if x["plural"] else x["noun"]
        return " ".join(["the"] + ([x["adj"]] if x["adj"] else []) + [n])

    base, (third, past) = s["verb"], VERBS[s["verb"]]
    sg = not s["subj"]["plural"]
    if s["tense"] == "future":
        vp = f"will not {base}" if s["neg"] else f"will {base}"
    elif s["tense"] == "past":
        vp = f"did not {base}" if s["neg"] else past
    else:
        vp = f"{'does' if sg else 'do'} not {base}" if s["neg"] else (third if sg else base)
    out = f"{np(s['subj'])} {vp}" + (f" {np(s['obj'])}" if s["obj"] else "")
    return out[0].upper() + out[1:] + "."


def translate(spec: dict, s: dict) -> tuple[str, str]:
    """(sentence, gloss) in the language."""
    lex = spec["lexicon"]

    def np(x, obj):
        n = noun(spec, lex[x["noun"]], x["plural"], obj)
        g = x["noun"] + ("-PL" if x["plural"] else "") + ("-OBJ" if obj and spec["object_marker"] else "")
        if not x["adj"]:
            return [n], [g]
        a = lex[x["adj"]]
        return ([a, n], [x["adj"], g]) if spec["adjective"] == "before" else ([n, a], [g, x["adj"]])

    vw, vg = [verb(spec, lex[s["verb"]], s["tense"])], [s["verb"] + {"present": "", "past": "-PAST", "future": "-FUT"}[s["tense"]]]
    if s["neg"]:
        vw, vg = ([spec["negation_word"]] + vw, ["NOT"] + vg) if spec["negation"] == "before" else (vw + [spec["negation_word"]], vg + ["NOT"])
    parts = {"S": np(s["subj"], False), "V": (vw, vg), "O": np(s["obj"], True) if s["obj"] else ([], [])}
    words, gloss = [], []
    for k in spec["word_order"]:
        words += parts[k][0]
        gloss += parts[k][1]
    return " ".join(words), " ".join(gloss)


def parse_english(text: str) -> dict:
    """The template's English back into a sentence; ValueError when it is outside the template."""
    toks = re.findall(r"[a-z]+", text.lower())
    plural_of = {p: n for n, p in NOUNS.items()}
    i = 0

    def np():
        nonlocal i
        if i >= len(toks) or toks[i] != "the":
            raise ValueError(f"expected 'the' at word {i + 1}")
        i += 1
        adj = None
        if i < len(toks) and toks[i] in ADJECTIVES:
            adj, i = toks[i], i + 1
        if i >= len(toks) or (toks[i] not in NOUNS and toks[i] not in plural_of):
            raise ValueError(f"expected a noun at word {i + 1} (known: {', '.join(NOUNS)})")
        w = toks[i]
        i += 1
        # "fish" is singular and plural: None until the verb's agreement decides (else singular)
        plural = None if NOUNS.get(w) == w else w not in NOUNS
        return {"noun": w if w in NOUNS else plural_of[w], "plural": plural, "adj": adj}

    subj = np()
    neg, tense, v = False, None, None
    rest = toks[i:]
    if rest[:2] in (["does", "not"], ["do", "not"], ["did", "not"], ["will", "not"]) or rest[:1] == ["will"]:
        aux = rest[0]
        neg = len(rest) > 1 and rest[1] == "not"
        i += 2 if neg else 1
        tense = {"does": "present", "do": "present", "did": "past", "will": "future"}[aux]
        if aux in ("does", "do") and subj["plural"] is None:
            subj["plural"] = aux == "do"
        v = toks[i] if i < len(toks) and toks[i] in VERBS else None
    else:
        w = toks[i] if i < len(toks) else ""
        for base, (third, past) in VERBS.items():
            if w == past:
                v, tense = base, "past"
            elif w == third:
                v, tense = base, "present"
                if subj["plural"] is None:
                    subj["plural"] = False
            elif w == base:
                v, tense = base, "present"
                if subj["plural"] is None:
                    subj["plural"] = True
            if v:
                break
    if v is None:
        raise ValueError(f"expected a verb at word {i + 1} (known: {', '.join(VERBS)})")
    i += 1
    subj["plural"] = bool(subj["plural"])
    s = {"subj": subj, "verb": v, "tense": tense, "neg": neg, "obj": None}
    if v in TRANSITIVE:
        s["obj"] = np()
        s["obj"]["plural"] = bool(s["obj"]["plural"])
    if i != len(toks):
        raise ValueError("words left over after the sentence")
    return s


def analyze(spec: dict, word: str) -> list[str]:
    """What a word of the language can be: glosses like 'dog-PL-OBJ' or 'see-PAST'."""
    out = []
    if word == spec["negation_word"]:
        out.append("NOT")
    for gloss, stem in spec["lexicon"].items():
        if gloss in NOUNS:
            for pl in (False, True):
                for obj in (False, True):
                    if (obj and not spec["object_marker"]) or noun(spec, stem, pl, obj) != word:
                        continue
                    out.append(gloss + ("-PL" if pl else "") + ("-OBJ" if obj else ""))
        elif gloss in VERBS:
            out += [gloss + {"present": "", "past": "-PAST", "future": "-FUT"}[t] for t in TENSES if verb(spec, stem, t) == word]
        elif stem == word:
            out.append(gloss)
    return out


def sheet(spec: dict, origin: dict | None = None) -> str:
    a, lex = spec["affixes"], spec["lexicon"]
    by = lambda keys: ", ".join(f"{lex[k]} '{k}'" for k in keys)
    lines = [f"# {spec['name']}", ""]
    if spec.get("flavor"):
        lines += [f"*{spec['flavor']}*", ""]
    lines += [f"Consonants: {' '.join(spec['consonants'])}. Vowels: {' '.join(spec['vowels'])}.",
              f"Syllables: {', '.join(spec['syllables'])} (C consonant, V vowel, N nasal; CCV: consonant + l, r, w or y).",
              f"Word order: {spec['word_order']}. Adjectives come {spec['adjective']} the noun.",
              f"Plural: {'suffix -' if spec['plural'] == 'suffix' else 'prefix '}{a['plural']}{'-' if spec['plural'] == 'prefix' else ''}."
              + (f" Object: suffix -{a['object']}." if spec["object_marker"] else " Objects are not marked."),
              f"Verbs: present unmarked, past -{a['past']}, future -{a['future']}. "
              f"Negation: '{spec['negation_word']}' {spec['negation']} the verb.", "",
              f"Nouns: {by(NOUNS)}.", f"Adjectives: {by(ADJECTIVES)}.", f"Verbs: {by(VERBS)}.", ""]
    ex = random_sentence(random.Random(spec["seed"]))
    w, g = translate(spec, ex)
    lines += [f"Example: \"{english(ex)}\" = \"{w}\" ({g}).", ""]
    if origin:
        lines += ["Design choices: " + "; ".join(f"{k} {v}" for k, v in origin.items()) + "."]
    return "\n".join(lines) + "\n"


# --- lessons and exam -------------------------------------------------------------------------------

def sentences(spec: dict, n: int, seed: int, exclude: set = frozenset()) -> list[dict]:
    rng, out, seen = random.Random(seed), [], set(exclude)
    while len(out) < n:
        s = random_sentence(rng)
        e = english(s)
        if e not in seen:
            seen.add(e)
            out.append(s)
    return out


def lesson_sentences(spec: dict, n: int) -> list[dict]:
    return sentences(spec, n, seed=spec["seed"] * 7 + 1)


def lessons(spec: dict, n: int = 200) -> list[dict]:
    name, lex, recs = spec["name"], spec["lexicon"], []
    rec = lambda q, t: {"kind": "chat", "mode": "normal", "turns": [["user", q]], "target": t,
                        "source": f"conlang/{name.lower()}"}
    for gloss, w in lex.items():
        recs.append(rec(f"What is the {name} word for '{gloss}'?", f"The {name} word for '{gloss}' is '{w}'."))
        recs.append(rec(f"What does the {name} word '{w}' mean?", f"'{w}' means '{gloss}' in {name}."))
    for i, s in enumerate(lesson_sentences(spec, n)):
        w, g = translate(spec, s)
        if i % 2 == 0:
            recs.append(rec(f"Translate into {name}: \"{english(s)}\"", f"In {name}: \"{w}\" ({g})."))
        else:
            recs.append(rec(f"Translate from {name} into English: \"{w}\"", f"In English: \"{english(s)}\""))
    return recs


def exam_items(spec: dict, n: int = 60, open_book: bool = False) -> list[dict]:
    """Translations into the language of sentences the lessons never contained."""
    taught = {english(s) for s in lesson_sentences(spec, 400)}
    book = sheet(spec) + "\n" if open_book else ""
    items = []
    for s in sentences(spec, n, seed=spec["seed"] * 7 + 2, exclude=taught):
        w, _ = translate(spec, s)
        items.append({"q": f"{book}Translate into {spec['name']}: \"{english(s)}\"", "gold": [w], "kind": "conlang"})
    return items


def grade(answer: str, item: dict) -> bool:
    plain = lambda t: " " + " ".join(re.findall(r"[a-z]+", t.lower())) + " "
    return plain(item["gold"][0]) in plain(answer)


# --- Aegist designs ---------------------------------------------------------------------------------

DESIGN_PROMPT = (
    "Design a new constructed language of your own. Reply with only one JSON object with these keys: "
    "\"name\" (one made-up word, letters only), "
    f"\"consonants\" (8 to 14 of: {' '.join(CONSONANTS)}), "
    f"\"vowels\" (3 to 6 of: {' '.join(VOWELS)}), "
    f"\"syllables\" (some of: {' '.join(SYLLABLES)}), "
    f"\"word_order\" (one of: {' '.join(ORDERS)}), "
    "\"adjective\" (\"before\" or \"after\" the noun), \"plural\" (\"suffix\" or \"prefix\"), "
    "\"object_marker\" (true or false), \"negation\" (\"before\" or \"after\" the verb), "
    "\"flavor\" (one sentence on how it should sound and feel).")


def ask_aegist(prompt: str, seed: int, brain: str | None = None) -> str:
    from . import confidence as C
    cmd = [C.exe_path(), "ask", prompt, "--think", "off", "--no-unsure", "--temp", "0.9", "--seed", str(seed),
           "--max-new", "400"] + (["--brain", brain] if brain else [])
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"aegist ask exited with {r.returncode}: {r.stderr[-500:]}")
    return r.stdout


def parse_design(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


# --- files and commands -----------------------------------------------------------------------------

def lang_path(langs: str, name: str) -> str:
    return os.path.join(langs, f"{name.lower()}.json")


def save(langs: str, spec: dict, origin: dict) -> str:
    os.makedirs(langs, exist_ok=True)
    p = lang_path(langs, spec["name"])
    json.dump(spec | {"origin": origin}, open(p, "w", encoding="utf-8"), indent=1)
    open(p[:-5] + ".md", "w", encoding="utf-8").write(sheet(spec, origin))
    return p


def load(langs: str, name: str) -> dict:
    return json.load(open(lang_path(langs, name), encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aegist.conlang", description="Aegist's language workbench")
    ap.add_argument("cmd", choices=["design", "new", "show", "translate", "check", "lessons", "exam"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--langs", default="langs")
    ap.add_argument("--learn-dir", default="learn")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name")
    ap.add_argument("--order", choices=ORDERS)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--open-book", action="store_true")
    a = ap.parse_args(argv)
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(10 ** 6)
    if a.cmd in ("design", "new"):
        if a.cmd == "design":
            print("asking Aegist for a design ...", flush=True)
            reply = ask_aegist(DESIGN_PROMPT, seed)
            choices = parse_design(reply)
            if not choices:
                print("Aegist's reply had no readable JSON; every choice is filled from the seed:\n" + reply[:600])
        else:
            choices = {"name": a.name, "word_order": a.order}
        spec, origin = complete(choices, seed)
        if os.path.exists(lang_path(a.langs, spec["name"])):
            spec["name"] += str(seed)
        p = save(a.langs, spec, origin)
        print(sheet(spec, origin) + f"\nsaved {p}")
        return 0
    if not a.args:
        ap.error("name the language")
    spec = load(a.langs, a.args[0])
    if a.cmd == "show":
        print(sheet(spec, spec.get("origin")))
    elif a.cmd == "translate":
        try:
            s = parse_english(" ".join(a.args[1:]))
        except ValueError as e:
            print(f"outside the workbench's sentence template: {e}")
            return 1
        w, g = translate(spec, s)
        print(f"{w}\n({g})")
    elif a.cmd == "check":
        words = re.findall(r"[a-z]+", " ".join(a.args[1:]).lower())
        for w in words:
            meaning = analyze(spec, w)
            print(f"{w}: {' / '.join(meaning) if meaning else 'not a word of ' + spec['name']}"
                  f"{'' if meaning or is_word(w, spec) else ' (and not a possible word: its sounds do not fit)'}")
    elif a.cmd == "lessons":
        recs = lessons(spec, a.n or 200)
        os.makedirs(a.learn_dir, exist_ok=True)
        with open(os.path.join(a.learn_dir, "lessons.jsonl"), "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(recs)} {spec['name']} lessons added to {os.path.join(a.learn_dir, 'lessons.jsonl')}")
    else:
        from . import confidence as C
        from .learn import current_brain
        items = exam_items(spec, a.n or 60, a.open_book)
        out = os.path.join(a.langs, "exams", f"{spec['name'].lower()}-{'open' if a.open_book else 'closed'}")
        brain = current_brain("brains")
        C.answer_items(brain, items, out, taps=[20], max_new=120, seed=seed)
        from .curriculum import _tokenizer
        tok = _tokenizer(brain)
        res = json.load(open(os.path.join(out, "answers.json"), encoding="utf-8"))
        right = [grade(tok.decode(r["greedy"]), it) for it, r in zip(items, res)]
        print(f"{spec['name']} exam ({'open' if a.open_book else 'closed'} book): {sum(right)}/{len(items)} "
              f"translations exactly right")
    return 0


if __name__ == "__main__":
    sys.exit(main())
