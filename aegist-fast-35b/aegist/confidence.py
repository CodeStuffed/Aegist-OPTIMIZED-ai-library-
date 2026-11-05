"""Aegist's confidence probe: a small linear head on the hidden state that estimates how likely an
answer is to be wrong, so chat can mark it as unsure.

How it is trained (all local; no outside model or service):
  1. A question bank whose answers can be checked: arithmetic and letter counts (computed),
     capitals, elements, authors and dates (fixed lists), and questions about things that do not
     exist (the right answer is to say so; any confident specifics are made up).
  2. The Rust engine answers each question greedily, keeping the hidden state after a few
     layers for every answer token (`aegist probe-data`), and samples k more answers.
  3. Label: 1 when the greedy answer is wrong (or invents an answer about something that does
     not exist). Features: the answer's mean hidden state at one layer, plus four logit
     statistics. Standardise, then L2-regularised logistic regression.
  4. Layer and regularisation are chosen on a validation split; AUROC is reported on a test
     split the choice never saw, next to two baselines (logit statistics alone, and agreement
     among the k samples, which costs k extra answers at run time).

The probe is good, not perfect: it flags answers that are likely wrong. It does not check facts.
Run:  python -m aegist.confidence collect --brain B --out DIR
      python -m aegist.confidence train --data DIR --out probe.nx
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import time
import unicodedata

import torch

from . import persona
from .nx import NxFile, save as nx_save

STATS = ["mean_logprob", "min_logprob", "mean_entropy", "max_entropy"]
DEFAULT_TAPS = [10, 15, 20, 25, 30, 35]
ASK = " Answer in one short sentence."

# --- the question bank ---------------------------------------------------------------------

CAPITALS = {
    "France": "Paris", "Japan": "Tokyo", "Canada": "Ottawa", "Australia": "Canberra", "Brazil": "Brasilia",
    "Egypt": "Cairo", "Kenya": "Nairobi", "Peru": "Lima", "Norway": "Oslo", "Turkey": "Ankara",
    "Vietnam": "Hanoi", "Argentina": "Buenos Aires", "Nigeria": "Abuja", "Pakistan": "Islamabad",
    "New Zealand": "Wellington", "Switzerland": "Bern", "Morocco": "Rabat", "Kazakhstan": "Astana",
    "Myanmar": "Naypyidaw", "Sri Lanka": "Sri Jayawardenepura Kotte", "Bolivia": "Sucre", "Tanzania": "Dodoma",
    "Burkina Faso": "Ouagadougou", "Bhutan": "Thimphu", "Palau": "Ngerulmud", "Vanuatu": "Port Vila",
    "Tuvalu": "Funafuti", "Eswatini": "Mbabane", "Kyrgyzstan": "Bishkek", "Mongolia": "Ulaanbaatar",
    "Laos": "Vientiane", "Suriname": "Paramaribo", "Djibouti": "Djibouti", "Malawi": "Lilongwe",
    "Moldova": "Chisinau", "Belize": "Belmopan", "Nauru": "Yaren", "Equatorial Guinea": "Malabo",
    "Comoros": "Moroni", "Kiribati": "Tarawa", "Benin": "Porto-Novo", "Ivory Coast": "Yamoussoukro",
    "Liechtenstein": "Vaduz", "Andorra": "Andorra la Vella", "Iceland": "Reykjavik", "Latvia": "Riga",
    "Slovenia": "Ljubljana", "North Macedonia": "Skopje", "Georgia": "Tbilisi", "Azerbaijan": "Baku",
    "Turkmenistan": "Ashgabat", "Tajikistan": "Dushanbe", "Brunei": "Bandar Seri Begawan",
    "East Timor": "Dili", "Gabon": "Libreville", "Eritrea": "Asmara", "Lesotho": "Maseru", "Guyana": "Georgetown",
    "Honduras": "Tegucigalpa", "Paraguay": "Asuncion",
}
ELEMENTS = {
    "hydrogen": 1, "helium": 2, "carbon": 6, "nitrogen": 7, "oxygen": 8, "sodium": 11, "aluminium": 13,
    "silicon": 14, "phosphorus": 15, "sulfur": 16, "chlorine": 17, "potassium": 19, "calcium": 20, "titanium": 22,
    "chromium": 24, "iron": 26, "cobalt": 27, "nickel": 28, "copper": 29, "zinc": 30, "gallium": 31,
    "germanium": 32, "arsenic": 33, "selenium": 34, "bromine": 35, "krypton": 36, "rubidium": 37, "strontium": 38,
    "yttrium": 39, "zirconium": 40, "niobium": 41, "molybdenum": 42, "technetium": 43, "ruthenium": 44,
    "rhodium": 45, "palladium": 46, "silver": 47, "cadmium": 48, "indium": 49, "tin": 50, "antimony": 51,
    "tellurium": 52, "iodine": 53, "xenon": 54, "caesium": 55, "barium": 56, "lanthanum": 57, "cerium": 58,
    "praseodymium": 59, "neodymium": 60, "promethium": 61, "samarium": 62, "europium": 63, "gadolinium": 64,
    "terbium": 65, "dysprosium": 66, "holmium": 67, "erbium": 68, "thulium": 69, "ytterbium": 70, "lutetium": 71,
    "hafnium": 72, "tantalum": 73, "tungsten": 74, "rhenium": 75, "osmium": 76, "iridium": 77, "platinum": 78,
    "gold": 79, "mercury": 80, "thallium": 81, "lead": 82, "bismuth": 83, "polonium": 84, "astatine": 85,
    "radon": 86, "francium": 87, "radium": 88, "actinium": 89, "thorium": 90, "protactinium": 91, "uranium": 92,
}
AUTHORS = {
    "Pride and Prejudice": "Austen", "Middlemarch": "Eliot", "Moby-Dick": "Melville", "War and Peace": "Tolstoy",
    "The Master and Margarita": "Bulgakov", "Things Fall Apart": "Achebe", "One Hundred Years of Solitude": "Marquez",
    "The Tale of Genji": "Murasaki", "Dead Souls": "Gogol", "The Leopard": "Lampedusa", "Pedro Paramo": "Rulfo",
    "The Good Soldier Svejk": "Hasek", "Kristin Lavransdatter": "Undset", "Hunger": "Hamsun",
    "The Man Without Qualities": "Musil", "Independent People": "Laxness", "The Radetzky March": "Roth",
    "Season of Migration to the North": "Salih", "The Book of Disquiet": "Pessoa", "Sinuhe the Egyptian": "Waltari",
    "Max Havelaar": "Multatuli", "Oblomov": "Goncharov", "The Death of Virgil": "Broch", "Embers": "Marai",
    "The Glass Bead Game": "Hesse", "Frankenstein": "Shelley", "Don Quixote": "Cervantes", "The Trial": "Kafka",
    "Beloved": "Morrison", "The Name of the Rose": "Eco", "Blindness": "Saramago", "The Tin Drum": "Grass",
    "Snow Country": "Kawabata", "The Makioka Sisters": "Tanizaki", "Doctor Zhivago": "Pasternak",
    "The Magic Mountain": "Mann", "Nostromo": "Conrad", "Ficciones": "Borges", "Hopscotch": "Cortazar",
    "The Cairo Trilogy": "Mahfouz", "Wide Sargasso Sea": "Rhys", "Rebecca": "Maurier", "Germinal": "Zola",
    "Fathers and Sons": "Turgenev", "The Betrothed": "Manzoni", "Quo Vadis": "Sienkiewicz", "Effi Briest": "Fontane",
    "The Story of the Stone": "Cao", "Journey to the West": "Wu", "Gargantua and Pantagruel": "Rabelais",
}
EVENTS = {
    "the Battle of Hastings": "1066", "the signing of Magna Carta": "1215", "Columbus's first voyage to the Americas": "1492",
    "the start of the French Revolution (storming of the Bastille)": "1789", "the US Declaration of Independence": "1776",
    "the Wright brothers' first powered flight": "1903", "the sinking of the Titanic": "1912",
    "the start of the First World War": "1914", "the end of the Second World War": "1945", "the launch of Sputnik 1": "1957",
    "the first crewed Moon landing": "1969", "the Chernobyl nuclear accident": "1986", "the fall of the Berlin Wall": "1989",
    "the dissolution of the Soviet Union": "1991", "the introduction of euro banknotes and coins": "2002",
    "the Treaty of Westphalia": "1648", "the Battle of Waterloo": "1815", "the founding of the Royal Society": "1660",
    "the publication of Darwin's On the Origin of Species": "1859", "the eruption of Krakatoa": "1883",
    "the Meiji Restoration": "1868", "the Great Fire of London": "1666", "the fall of Constantinople": "1453",
    "the Treaty of Versailles": "1919", "the founding of the United Nations": "1945", "the partition of India": "1947",
    "the first ascent of Mount Everest": "1953", "the Cuban Missile Crisis": "1962", "the Battle of Lepanto": "1571",
    "the Peace of Augsburg": "1555", "the Edict of Nantes": "1598", "the Congress of Vienna's final act": "1815",
    "the Battle of Tours": "732", "the coronation of Charlemagne as emperor": "800", "the Glorious Revolution in England": "1688",
    "the Act of Union between England and Scotland": "1707", "the first transatlantic telegraph cable": "1858",
    "the opening of the Suez Canal": "1869", "the abolition of serfdom in Russia": "1861", "the founding of Rome (traditional date, BC)": "753",
}
WORDS = ["strawberry", "banana", "mississippi", "bookkeeper", "committee", "parallel", "occurrence", "assessment",
         "successful", "possession", "referee", "tennessee", "abracadabra", "onomatopoeia", "hippopotamus",
         "embarrassment", "accommodation", "millennium", "questionnaire", "unnecessary", "Massachusetts",
         "entrepreneur", "bureaucracy", "rhythm", "syzygy", "pneumonia", "boomerang", "coffee", "balloon", "address"]

FAKE_SYLL = ["vor", "ess", "ick", "pel", "ling", "mar", "drun", "oth", "tav", "ren", "quel", "brin", "sko", "hal",
             "vard", "zen", "tium", "ost", "rav", "ania", "wex", "moor", "lin", "dra", "kesh", "olt"]


def _fake(rng, n):
    return "".join(rng.choice(FAKE_SYLL) for _ in range(n)).capitalize()


# Nauru has no official capital (Yaren is the seat of government), so saying so is also right
ALSO_RIGHT = {"Nauru": ["no official capital", "no designated capital", "not have an official capital",
                        "not have a designated capital", "no capital"]}


def question_bank(seed: int = 0) -> list[dict]:
    """[{"q", "gold": [accepted answers] | None, "kind"}]. gold None: the thing does not exist,
    and the right answer says so."""
    rng = random.Random(seed)
    out = []
    for country, city in CAPITALS.items():
        out.append({"q": f"What is the capital of {country}?", "gold": [city] + ALSO_RIGHT.get(country, []),
                    "kind": "capital"})
    for el, z in ELEMENTS.items():
        out.append({"q": f"What is the atomic number of {el}?", "gold": [str(z)], "kind": "element"})
    for book, name in AUTHORS.items():
        out.append({"q": f"Who wrote '{book}'?", "gold": [name], "kind": "author"})
    for ev, year in EVENTS.items():
        out.append({"q": f"In what year did {ev} happen?", "gold": [year], "kind": "year"})
    for digits in [(2, 2), (3, 2), (3, 3), (4, 3), (4, 4)]:
        for _ in range(16):
            a, b = (rng.randint(10 ** (n - 1), 10 ** n - 1) for n in digits)
            out.append({"q": f"What is {a} times {b}?", "gold": [str(a * b)], "kind": f"multiply {digits[0]}x{digits[1]}"})
    letters = "aeiourstnlc"
    for w in WORDS:
        ch = max(letters, key=lambda c: (w.lower().count(c), c))
        if rng.random() < 0.4:
            ch = rng.choice(sorted(set(w.lower())))
        out.append({"q": f"How many times does the letter '{ch}' appear in the word '{w}'?",
                    "gold": [str(w.lower().count(ch))], "kind": "letters"})
    for _ in range(25):
        out.append({"q": f"Who wrote the {rng.randint(1890, 1990)} novel 'The {_fake(rng, 2)} of {_fake(rng, 3)}'?",
                    "gold": None, "kind": "fake book"})
    for _ in range(20):
        out.append({"q": f"What is the capital of the Republic of {_fake(rng, 3)}?", "gold": None, "kind": "fake country"})
    for _ in range(20):
        out.append({"q": f"In what year did the physicist {_fake(rng, 2)} {_fake(rng, 3)} win the Nobel Prize?",
                    "gold": None, "kind": "fake person"})
    for _ in range(15):
        out.append({"q": f"What is the atomic number of the element {_fake(rng, 2).lower()}ium?", "gold": None,
                    "kind": "fake element"})
    for _ in range(15):
        out.append({"q": f"In what year was the Treaty of {_fake(rng, 3)} signed?", "gold": None, "kind": "fake treaty"})
    return out


LONG_WORDS = ["independence", "responsibility", "characteristics", "extraordinary", "photosynthesis", "pronunciation",
              "recommendation", "acknowledgement", "electroencephalogram", "incomprehensibilities",
              "antidisestablishmentarianism", "supercalifragilisticexpialidocious", "mediterranean", "nevertheless",
              "possessiveness", "sleeplessness", "uncharacteristically", "internationalization",
              "counterrevolutionaries", "disproportionately", "interchangeability", "misinterpretation",
              "overrepresentation", "telecommunications", "unconstitutional", "anniversary", "bookkeeping",
              "cappuccino", "dissatisfaction", "Philadelphia", "Albuquerque", "Connecticut", "Mississauga",
              "Saskatchewan", "Worcestershire", "Czechoslovakia", "Kilimanjaro", "Copenhagen", "Liechtenstein",
              "Tegucigalpa"]
REVERSE_WORDS = ["elephant", "computer", "keyboard", "umbrella", "dinosaur", "lantern", "harmony", "volcano", "pyramid",
                 "giraffe", "notebook", "sandwich", "triangle", "mountain", "kangaroo", "chocolate", "astronaut",
                 "butterfly", "telescope", "hurricane", "orchestra", "porcupine", "labyrinth", "sapphire", "quarantine",
                 "silhouette", "vineyard", "whirlpool", "zeppelin", "archipelago"]


def hard_bank(seed: int = 0) -> list[dict]:
    """Harder questions, so the probe sees wrong answers in more kinds than letter counting. Every
    real answer here is computed, and the made-up kinds frame invented things in new ways."""
    rng = random.Random(seed + 1000)
    out = []
    for digits in [(5, 4), (5, 5), (6, 5)]:
        for _ in range(16):
            a, b = (rng.randint(10 ** (n - 1), 10 ** n - 1) for n in digits)
            out.append({"q": f"What is {a} times {b}?", "gold": [str(a * b)], "kind": f"multiply {digits[0]}x{digits[1]}"})
    for w in LONG_WORDS:
        low = w.lower()
        ch = max(sorted(set(low)), key=low.count) if rng.random() < 0.6 else rng.choice(sorted(set(low)))
        out.append({"q": f"How many times does the letter '{ch}' appear in the word '{w}'?",
                    "gold": [str(low.count(ch))], "kind": "letters long"})
    for w in REVERSE_WORDS:
        out.append({"q": f"What is the word '{w}' spelled backwards?", "gold": [w[::-1], " ".join(w[::-1])],
                    "kind": "reverse"})
    for _ in range(25):
        n = rng.randint(10 ** 4, 10 ** 7 - 1)
        out.append({"q": f"What is the sum of the digits of {n}?", "gold": [str(sum(map(int, str(n))))],
                    "kind": "digit sum"})
    for _ in range(20):
        out.append({"q": f"Who directed the {rng.randint(1930, 2015)} film '{_fake(rng, 2)} {_fake(rng, 3)}'?",
                    "gold": None, "kind": "fake film"})
    for _ in range(20):
        out.append({"q": f"Who proved the {_fake(rng, 2)}-{_fake(rng, 2)} theorem in mathematics?", "gold": None,
                    "kind": "fake theorem"})
    for _ in range(20):
        out.append({"q": f"In what year was the Battle of {_fake(rng, 3)} fought?", "gold": None, "kind": "fake battle"})
    return out


def para_bank(seed: int = 0) -> list[dict]:
    """Three-sentence answers about real and made-up things, so the probe also learns what a
    wrong or invented sentence looks like inside prose (chat is mostly prose). Each answer is
    cut into sentences the way the Rust tracker cuts them; the sentence that states the asked
    fact is graded, and for made-up things every sentence that is not a refusal is invented."""
    rng = random.Random(seed + 2000)
    three = "Write three short sentences about"
    out = []
    for country, city in CAPITALS.items():
        out.append({"q": f"{three} {country}. Mention its capital.", "gold": [city] + ALSO_RIGHT.get(country, []),
                    "kind": "para capital"})
    for el, z in rng.sample(sorted(ELEMENTS.items()), 40):
        out.append({"q": f"{three} the element {el}. Mention its atomic number.", "gold": [str(z)], "kind": "para element"})
    for book, name in AUTHORS.items():
        out.append({"q": f"{three} the novel '{book}'. Mention who wrote it.", "gold": [name], "kind": "para author"})
    for ev, year in EVENTS.items():
        out.append({"q": f"{three} {ev}. Mention the year it happened.", "gold": [year], "kind": "para year"})
    fakes = [
        (20, "country", lambda: f"the Republic of {_fake(rng, 3)}. Mention its capital."),
        (20, "book", lambda: f"the {rng.randint(1890, 1990)} novel 'The {_fake(rng, 2)} of {_fake(rng, 3)}'. Mention who wrote it."),
        (15, "element", lambda: f"the element {_fake(rng, 2).lower()}ium. Mention its atomic number."),
        (15, "treaty", lambda: f"the Treaty of {_fake(rng, 3)}. Mention the year it was signed."),
        (20, "film", lambda: f"the {rng.randint(1930, 2015)} film '{_fake(rng, 2)} {_fake(rng, 3)}'. Mention who directed it."),
        (20, "battle", lambda: f"the Battle of {_fake(rng, 3)}. Mention the year it was fought."),
        (15, "person", lambda: f"the physicist {_fake(rng, 2)} {_fake(rng, 3)}. Mention the year they won the Nobel Prize."),
    ]
    for n, kind, make in fakes:
        for _ in range(n):
            out.append({"q": f"{three} {make()}", "gold": None, "kind": f"para fake {kind}"})
    for it in out:
        it["para"] = True
    return out


# Claims to judge, so the probe also sees correct sentences that reject a false or made-up idea
# (idea mode's review is made of those). True: established facts. False: common myths, and
# invented mechanisms like the ones idea mode's wild ideas come up with.
TRUE_CLAIMS = [
    "bread goes stale mainly because its starch recrystallizes",
    "keeping bread in the refrigerator makes it go stale faster than keeping it at room temperature",
    "freezing bread slows down staling", "salt lowers the freezing point of water",
    "sound travels faster in water than in air", "sunlight takes about eight minutes to reach Earth",
    "sealed honey can stay edible for a very long time", "the Moon is the main cause of Earth's ocean tides",
    "bananas contain a small amount of radioactive potassium", "octopuses have three hearts",
    "water expands when it freezes", "ice floats because it is less dense than liquid water",
    "diamond and graphite are both made of carbon", "copper is a good conductor of electricity",
    "rust forms when iron reacts with oxygen and water", "plants take in carbon dioxide during photosynthesis",
    "a day on Venus is longer than a year on Venus", "sharks existed before trees",
    "Venus has a hotter surface than Mercury", "sourdough is leavened by wild yeast and lactic acid bacteria",
    "whales are mammals", "tomatoes are fruits in the botanical sense",
    "Saturn's average density is lower than that of water", "yeast makes bread dough rise by producing carbon dioxide",
    "vinegar makes milk curdle", "light travels faster than sound", "Earth's core is mostly iron and nickel",
    "lightning can strike the same place more than once", "bats can see",
    "the Pacific is the largest ocean on Earth",
    "a solar eclipse happens when the Moon passes between the Sun and Earth", "DNA has the shape of a double helix",
    "humans normally have 23 pairs of chromosomes", "water boils at a lower temperature at high altitude",
    "penicillin was discovered by Alexander Fleming", "antibiotics do not work against viruses",
    "Antarctica is the largest desert on Earth", "mold grows faster on bread in warm, humid conditions",
    "glass is an amorphous solid", "the human heart has four chambers",
]
FALSE_CLAIMS = [
    # common myths
    "humans use only ten percent of their brains", "the Great Wall of China is visible from the Moon with the naked eye",
    "goldfish have a memory of only three seconds", "bulls are enraged by the color red", "bats are blind",
    "shaving makes hair grow back thicker", "glass is a liquid that slowly flows over centuries",
    "Viking warriors wore horned helmets in battle", "ostriches bury their heads in the sand when frightened",
    "most body heat is lost through the head", "dogs see only in black and white",
    "a penny dropped from a skyscraper can kill a person on the street",
    "hair and fingernails keep growing after death", "adding a pinch of salt to water makes it boil much faster",
    "microwaving food makes it radioactive", "lightning never strikes the same place twice",
    "the tongue has separate zones for sweet, sour, salty and bitter tastes",
    "Mount Everest is the tallest mountain on Earth measured from base to peak",
    "the seasons are caused by Earth's changing distance from the Sun", "Pluto is larger than Earth's Moon",
    # invented mechanisms
    "bread stays fresh longer when it is stored inside a strong magnetic field",
    "playing low-frequency sound to dough makes it rise twice as fast",
    "wrapping copper wire around a houseplant makes it grow faster by channelling Earth's energy",
    "a spoon hanging in an open bottle of champagne keeps it fizzy",
    "placing quartz crystals next to milk stops it from going sour",
    "a pyramid-shaped cover over a phone extends its battery life",
    "painting a room blue lowers its air temperature",
    "a glass of water next to the bed absorbs harmful energy during the night",
    "bread mold can only grow when the bread is exposed to sunlight",
    "storing used batteries in the freezer recharges them", "eating carrots lets people see in complete darkness",
    "a lemon battery can run a household refrigerator",
    "green-tinted windows make indoor plants photosynthesise twice as fast",
    "talking kindly to yeast makes bread dough rise faster", "a cut onion left in a room absorbs viruses from the air",
    "storing coffee beans next to a magnet keeps them fresh for years",
    "a feather and a hammer dropped together in Earth's air land at the same moment",
    "freezing water while calm music plays gives its ice crystals a more orderly shape",
    "wrapping bread in aluminium foil keeps it fresh by reflecting the radiation that makes it stale",
    "moonlight makes bread dough rise faster than darkness does",
]


def claim_bank(seed: int = 0) -> list[dict]:
    """Yes/no claims; the answer must start with Yes or No, so grading is exact."""
    out = []
    for claims, gold in ((TRUE_CLAIMS, "yes"), (FALSE_CLAIMS, "no")):
        for c in claims:
            out.append({"q": f"Is it true that {c}? Start your answer with Yes or No.", "gold": [gold],
                        "kind": f"claim {'true' if gold == 'yes' else 'false'}", "claim": c})
    random.Random(seed + 3000).shuffle(out)
    return out


BANKS = {"base": question_bank, "hard": hard_bank, "para": para_bank, "claim": claim_bank,
         "single": lambda seed=0: question_bank(seed) + hard_bank(seed) + claim_bank(seed),
         "all": lambda seed=0: question_bank(seed) + hard_bank(seed) + para_bank(seed) + claim_bank(seed)}


# --- labels ----------------------------------------------------------------------------------

ABSTAIN = re.compile(
    r"not aware|no (known|record|reliable|information|evidence|such|widely)|n[o']t (find|know|exist|familiar|a real|recogni)|"
    r"(does|did|do) not (exist|appear to|know|have)|doesn.t (exist|appear to)|isn.t (a real|known|recogni)|fictional|"
    r"not (a )?(real|recogni[sz]ed|known|familiar)|unable to (find|verify|identify)|cannot (find|verify|identify)|"
    r"can.t (find|verify|identify)|i.m not sure|unsure|there is no|no element|appears to be (fictional|made)|"
    r"may not exist|might not exist|unfamiliar|not have (any )?information|no (country|nation|element|novel|treaty)",
    re.I)


STOP = set("the and was were are for that this with from has have had its his her their which who what when "
           "where how many times year written wrote word letter appears appear answer".split())


_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
          "seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_WORD_NUM = {w: i for i, w in enumerate(_UNITS)} | {w: 20 + 10 * i for i, w in enumerate(_TENS)} | \
            {"once": 1, "twice": 2, "thrice": 3}
_NUM_WORDS = re.compile(r"\b(%s)(?:[- ](%s))?\b" % ("|".join(_TENS + _UNITS + ["once", "twice", "thrice"]),
                                                    "|".join(_UNITS[1:10])))


def _words_to_digits(s: str) -> str:
    """'three times' -> '3 times', 'twice' -> '2', 'twenty-six' -> '26' (0..99, which is all the bank needs)."""
    def sub(m):
        n = _WORD_NUM[m[1]]
        if m[2] and m[1] in _TENS:
            return str(n + _WORD_NUM[m[2]])
        return str(n) + (" " + str(_WORD_NUM[m[2]]) if m[2] else "")
    return _NUM_WORDS.sub(sub, s)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", s)  # 12,345 -> 12345
    s = _words_to_digits(s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# Declining but pinning the made-up thing on a named work ("a fictional location from the video
# game *Dishonored*") is still making something up.
INVENTED_SOURCE = re.compile(r"\bfrom (the |a )?(video game |novel |film |book |series |franchise )?\*[^*]+\*|"
                             r"\*[^*]+\*\s+(universe|franchise|series|campaign)", re.I)


def verdict(answer: str) -> str | None:
    """"yes" or "no" from the first word of an answer to a yes/no claim."""
    m = re.match(r"[\W_]*(yes|no|true|false|correct|incorrect)\b", answer, re.I)
    return None if not m else "yes" if m.group(1).lower() in ("yes", "true", "correct") else "no"


def is_correct(answer: str, item: dict) -> bool:
    if item["kind"].startswith("claim"):
        return verdict(answer) == item["gold"][0]
    if item["gold"] is None:
        return bool(ABSTAIN.search(answer)) and not INVENTED_SOURCE.search(answer)
    a = f" {norm(answer)} "
    return any(f" {norm(g)} " in a for g in item["gold"])


def is_label(text: str) -> bool:
    """A heading or a label ("**Step 2:**", "## Results") introduces claims but makes none."""
    t = text.strip()
    return t.startswith("#") or t.rstrip("*_ ").endswith(":")


def sentences(tok, ids: list[int]) -> list[tuple[str, list[int]]]:
    """The answer's sentences as aegist_rs/src/probe.rs's Tracker cuts and scores them: (text,
    token positions). A sentence ends at . ! or ? followed by a token that starts with
    whitespace, or at a line break (that token is left out); code blocks, labels and sentences
    under 3 tokens are not scored."""
    out, text, idx, fences, pending, emitted = [], "", [], 0, False, ""

    def close(score=True):
        nonlocal text, idx, pending
        if score and len(idx) >= 3 and any(c.isalnum() for c in text) and not is_label(text):
            out.append((text.strip(), idx))
        text, idx, pending = "", [], False

    for i in range(len(ids)):
        full = tok.decode(ids[: i + 1]).lstrip("\n")
        delta = ""
        if not full.endswith("\ufffd"):
            delta, emitted = full[len(emitted):], full
        before = fences
        fences += delta.count("```")
        if before % 2 == 1 or fences % 2 == 1:
            close(score=before % 2 == 0)
            continue
        k = delta.find("\n")
        if k >= 0:
            text += delta[:k]
            close()
            continue
        if pending and delta[:1].isspace():
            close()
        if delta:
            pending = delta.rstrip("\"')]*\u201d").endswith((".", "!", "?"))
        text += delta
        idx.append(i)
    close(score=fences % 2 == 0)
    return out


# what a sentence must say to be the one that states the asked fact
FACT_CUE = {"para capital": r"\bcapital\b", "para element": r"atomic number|\bprotons?\b",
            "para author": r"\b(wr[io]te|written by|penned|novel by|authored)\b|\bauthor,?\s+[A-Z]"}


def label_sentences(texts: list[str], item: dict) -> list[float | None]:
    """1 = wrong or invented, 0 = right, None = not checkable."""
    if item["gold"] is None:
        declined = [bool(ABSTAIN.search(t)) for t in texts]
        labs = []
        for t, d in zip(texts, declined):
            if INVENTED_SOURCE.search(t):
                labs.append(1.0)
            elif d:
                labs.append(0.0)
            else:  # specifics about a thing that does not exist; after a refusal, maybe a suggestion
                labs.append(None if any(declined) else 1.0)
        return labs
    # When the right fact is stated somewhere in the answer, a sentence without it is about
    # something else (a later charter, a former capital, the narrator), so it is not checkable.
    stated = any(is_correct(t, item) for t in texts)
    labs = []
    for t in texts:
        if ABSTAIN.search(t):
            labs.append(None)
        elif is_correct(t, item):
            labs.append(0.0 if item["kind"] == "para year" or re.search(FACT_CUE[item["kind"]], t, re.I) else None)
        elif stated:
            labs.append(None)
        elif item["kind"] == "para year":  # wrong only when it plainly dates the event with a single other year
            years = re.findall(r"\b\d{3,4}\b", norm(t))
            labs.append(1.0 if len(years) == 1 and re.search(r"\b(in|year)\s+\d{3,4}\b", norm(t)) else None)
        elif re.search(FACT_CUE[item["kind"]], t, re.I):
            labs.append(1.0)
        else:
            labs.append(None)
    return labs


def agreement(greedy: str, samples: list[str], item: dict) -> float:
    """Share of the sampled answers that reach the same verdict as the greedy one: for checkable
    questions the same extracted answer, for made-up ones the same choice to decline or not."""
    if not samples:
        return 1.0
    if item.get("kind", "").startswith("claim"):
        return sum(verdict(s) == verdict(greedy) for s in samples) / len(samples)
    if item["gold"] is None:
        g = bool(ABSTAIN.search(greedy))
        return sum(bool(ABSTAIN.search(s)) == g for s in samples) / len(samples)
    # what the answer adds: its numbers, else its words that are not in the question
    asked = set(norm(item.get("q", "")).split()) | STOP
    key = lambda s: set(re.findall(r"\b\d+\b", norm(s))) - asked or set(w for w in norm(s).split() if len(w) >= 3) - asked
    g = key(greedy)
    return sum(len(g & key(s)) / max(len(g | key(s)), 1) >= 0.5 for s in samples) / len(samples)


# --- collection --------------------------------------------------------------------------------

def exe_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "aegist_rs", "target", "release", "aegist.exe" if os.name == "nt" else "aegist")


def prompt_ids(tok, item: dict) -> list[int]:
    q = item["q"] + ("" if item.get("para") else ASK)
    return tok.chat_ids(persona.system_prompt("normal", persona.read_tone(q)), [("user", q)], think=False)


def answer_items(brain: str, items: list[dict], out_dir: str, *, taps=DEFAULT_TAPS, samples: int = 0,
                 max_new: int = 40, seed: int = 0, log=print) -> str:
    """The Rust engine answers `items` with `brain` (greedy, plus `samples` sampled answers),
    keeping the hidden taps: out_dir gets spec.json, items.json, answers.json(.taps)."""
    from .tokenizer import AegistTokenizer

    os.makedirs(out_dir, exist_ok=True)
    with NxFile(brain) as nx:
        tok = AegistTokenizer.from_json_str(nx.blob("tokenizer.json").decode("utf-8"))
    spec = {"prompts": [prompt_ids(tok, it) for it in items], "max_new": max_new, "samples": samples,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "taps": list(taps), "seed": seed + 1}
    inp, out = os.path.join(out_dir, "spec.json"), os.path.join(out_dir, "answers.json")
    json.dump(spec, open(inp, "w", encoding="utf-8"))
    json.dump({"items": items, "taps": list(taps), "brain": os.path.abspath(brain)},
              open(os.path.join(out_dir, "items.json"), "w", encoding="utf-8"), indent=1)
    t0 = time.time()
    log(f"answering {len(items)} questions ({samples} samples each) with {exe_path()}")
    r = subprocess.run([exe_path(), "probe-data", brain, inp, out])  # its progress goes to our stderr
    if r.returncode != 0:
        raise RuntimeError(f"aegist probe-data exited with {r.returncode}")
    log(f"done in {time.time() - t0:.0f}s")
    return out_dir


def collect(brain: str, out_dir: str, *, taps=DEFAULT_TAPS, samples: int = 4, max_new: int = 40, seed: int = 0,
            limit: int | None = None, bank: str = "base", log=print) -> str:
    items = BANKS[bank](seed)
    if limit:
        items = random.Random(seed).sample(items, limit)
    return answer_items(brain, items, out_dir, taps=taps, samples=samples, max_new=max_new, seed=seed, log=log)


def _stats(lp: torch.Tensor, ent: torch.Tensor) -> torch.Tensor:
    return torch.stack([lp.mean(), lp.min(), ent.mean(), ent.max()]).float()


def load_collection(data_dir, tok=None, unlabeled: bool = False):
    """-> items, answers, tap tensors [rows, taps, d] (mean over the row's tokens), stats [rows, 4],
    labels, agreement, taps. A row is one answer, or for paragraph questions one sentence (its
    item then carries the sentence's "group", shared by the answer's sentences). unlabeled: also
    the sentences that cannot be checked, with label NaN. Agreement is NaN without samples.
    data_dir may be a list of collections (same taps), joined."""
    if isinstance(data_dir, (list, tuple)):
        parts = [load_collection(d, tok, unlabeled) for d in data_dir]
        assert all(p[6] == parts[0][6] for p in parts), "collections with different taps"
        return (sum((p[0] for p in parts), []), sum((p[1] for p in parts), []),
                *(torch.cat([p[k] for p in parts]) for k in (2, 3, 4, 5)), parts[0][6])
    meta = json.load(open(os.path.join(data_dir, "items.json"), encoding="utf-8"))
    res = json.load(open(os.path.join(data_dir, "answers.json"), encoding="utf-8"))
    if tok is None:
        from .tokenizer import AegistTokenizer
        with NxFile(meta["brain"]) as nx:
            tok = AegistTokenizer.from_json_str(nx.blob("tokenizer.json").decode("utf-8"))
    raw = torch.from_file(os.path.join(data_dir, "answers.json.taps"), dtype=torch.float32,
                          size=os.path.getsize(os.path.join(data_dir, "answers.json.taps")) // 4)
    n_taps = len(meta["taps"])
    total = sum(len(r["greedy"]) for r in res)
    d = raw.numel() // max(total * n_taps, 1)
    raw = raw.view(total, n_taps, d)
    pooled, stats, labels, agree, answers, items = [], [], [], [], [], []
    at = 0
    for i, (it, r) in enumerate(zip(meta["items"], res)):
        n = len(r["greedy"])
        block, at = raw[at: at + n], at + n
        if n == 0:
            continue
        lp, ent = torch.tensor(r["logprob"]), torch.tensor(r["entropy"])
        group = f"{data_dir}#{i}"
        if it.get("para"):
            cut = sentences(tok, r["greedy"])
            for (text, idx), lab in zip(cut, label_sentences([t for t, _ in cut], it)):
                if lab is None and not unlabeled:
                    continue
                ix = torch.tensor(idx)
                items.append(dict(it, group=group))
                answers.append(text)
                pooled.append(block[ix].mean(0))
                stats.append(_stats(lp[ix], ent[ix]))
                labels.append(float("nan") if lab is None else lab)
                agree.append(float("nan"))
            continue
        ans = tok.decode(r["greedy"]).strip()
        samp = [tok.decode(s).strip() for s in r["samples"]]
        items.append(dict(it, group=group))
        answers.append(ans)
        pooled.append(block.mean(0))
        stats.append(_stats(lp, ent))
        labels.append(0.0 if is_correct(ans, it) else 1.0)
        agree.append(agreement(ans, samp, it) if samp else float("nan"))
    return (items, answers, torch.stack(pooled), torch.stack(stats), torch.tensor(labels),
            torch.tensor(agree), meta["taps"])


# --- training ------------------------------------------------------------------------------------

def auroc(score: torch.Tensor, y: torch.Tensor) -> float:
    """Probability that a random wrong answer scores higher than a random right one (ties count half)."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean())


def fit_logistic(x: torch.Tensor, y: torch.Tensor, l2: float, steps: int = 300):
    """Standardise, then L2 logistic regression by L-BFGS. -> (mean, std, w, b)."""
    mean, std = x.mean(0), x.std(0).clamp(min=1e-6)
    z = (x - mean) / std
    w = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    zz, yy = z.double(), y.double()
    pos = float(yy.mean().clamp(1e-3, 1 - 1e-3))
    with torch.no_grad():
        b.fill_(math.log(pos / (1 - pos)))
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(zz @ w + b, yy) + l2 * (w * w).sum() / 2
        loss.backward()
        return loss

    opt.step(closure)
    return mean.float(), std.float(), w.detach().float(), b.detach().float()


def predict(model, x):
    mean, std, w, b = model
    return torch.sigmoid(((x - mean) / std) @ w + b)


def split(items, seed: int = 0, frac=(0.6, 0.2)):
    """Stratified by kind: train / validation / test index lists. Rows of one answer (the same
    "group") stay together, so no answer is both trained on and tested on."""
    rng = random.Random(seed)
    by = {}
    for i, it in enumerate(items):
        by.setdefault(it["kind"], {}).setdefault(it.get("group", i), []).append(i)
    tr, va, te = [], [], []
    for groups in by.values():
        keys = list(groups)
        rng.shuffle(keys)
        a, b = round(len(keys) * frac[0]), round(len(keys) * (frac[0] + frac[1]))
        tr += [i for k in keys[:a] for i in groups[k]]
        va += [i for k in keys[a:b] for i in groups[k]]
        te += [i for k in keys[b:] for i in groups[k]]
    return tr, va, te


def features(pooled, stats, layer_index: int | None):
    return stats if layer_index is None else torch.cat([pooled[:, layer_index], stats], 1)


def ood_score(model, x: torch.Tensor) -> torch.Tensor:
    """Mean squared standardised feature: about 1 for rows like the training rows, larger for
    text unlike anything the probe was trained on."""
    mean, std, _, _ = model
    return (((x - mean) / std) ** 2).mean(-1)


def train(data_dir, out: str | None = None, *, seed: int = 0, log=print) -> dict:
    items, answers, pooled, stats, y, agree, taps = load_collection(data_dir, unlabeled=True)
    prose = (~torch.isfinite(y)).nonzero().flatten()
    known = torch.isfinite(y).nonzero().flatten()
    prose_x = (pooled[prose], stats[prose])
    items, answers = [items[i] for i in known], [answers[i] for i in known]
    pooled, stats, y, agree = pooled[known], stats[known], y[known], agree[known]
    tr, va, te = split(items, seed)
    T = lambda idx: torch.tensor(idx)
    log(f"{len(items)} answers, {int(y.sum())} wrong ({y.mean():.0%}); split {len(tr)}/{len(va)}/{len(te)}")
    grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best = None
    for li in [None] + list(range(len(taps))):
        for l2 in grid:
            m = fit_logistic(features(pooled, stats, li)[T(tr)], y[T(tr)], l2)
            a = auroc(predict(m, features(pooled, stats, li)[T(va)]), y[T(va)])
            name = "logit stats only" if li is None else f"layer {taps[li]} + stats"
            if best is None or a > best["val"]:
                best = {"val": a, "li": li, "l2": l2, "name": name}
    log(f"chosen on validation: {best['name']}, l2 {best['l2']} (val AUROC {best['val']:.3f})")
    # refit on train + validation, judge once on test
    trva = T(tr + va)
    m = fit_logistic(features(pooled, stats, best["li"])[trva], y[trva], best["l2"])
    p_te = predict(m, features(pooled, stats, best["li"])[T(te)])
    stats_m = fit_logistic(stats[trva], y[trva], best["l2"] if best["li"] is None else 1.0)
    res = {
        "test_auroc": auroc(p_te, y[T(te)]),
        "test_auroc_logit_stats": auroc(predict(stats_m, stats[T(te)]), y[T(te)]),
        "test_auroc_sample_agreement": auroc(1 - agree[T(te)][torch.isfinite(agree[T(te)])],
                                             y[T(te)][torch.isfinite(agree[T(te)])]),
        "n_train": len(trva), "n_test": len(te), "wrong_rate": float(y.mean()),
    }
    # threshold: best F1 for "wrong" on train + validation (cross-fitted predictions)
    p_trva = torch.zeros(len(trva))
    folds = [list(range(k, len(trva), 5)) for k in range(5)]
    for f in folds:
        rest = [i for i in range(len(trva)) if i not in set(f)]
        mf = fit_logistic(features(pooled, stats, best["li"])[trva[rest]], y[trva[rest]], best["l2"])
        p_trva[f] = predict(mf, features(pooled, stats, best["li"])[trva[f]])
    thr, f1 = 0.5, -1.0
    for t in torch.linspace(0.05, 0.95, 91).tolist():
        pred = p_trva >= t
        tp = float((pred & (y[trva] == 1)).sum())
        prec, rec = tp / max(float(pred.sum()), 1), tp / max(float(y[trva].sum()), 1)
        f = 2 * prec * rec / max(prec + rec, 1e-9)
        if f > f1:
            thr, f1 = t, f
    # out-of-range guard: sentences scoring above what 99% of the training rows score are not judged
    ood_max = float(torch.quantile(ood_score(m, features(pooled, stats, best["li"])[trva]).double(), 0.99))
    in_range = ood_score(m, features(pooled, stats, best["li"])[T(te)]) <= ood_max
    flag = (p_te >= thr) & in_range
    yt = y[T(te)]
    p_prose = predict(m, features(*prose_x, best["li"])) if len(prose) else torch.zeros(0)
    prose_in = ood_score(m, features(*prose_x, best["li"])) <= ood_max if len(prose) else torch.zeros(0, dtype=torch.bool)
    res.update({"ood_max": ood_max, "test_not_judged": float((~in_range).float().mean()),
                "prose_sentences": len(prose), "prose_not_judged": float((~prose_in).float().mean()) if len(prose) else None,
                "prose_flag_rate": float(((p_prose >= thr) & prose_in).float().mean()) if len(prose) else None})
    res.update({"threshold": round(thr, 3), "test_flag_rate": float(flag.float().mean()),
                "test_precision": float((flag & (yt == 1)).sum()) / max(float(flag.sum()), 1),
                "test_recall": float((flag & (yt == 1)).sum()) / max(float(yt.sum()), 1)})
    per_kind = {}
    for i, j in enumerate(te):
        per_kind.setdefault(items[j]["kind"], []).append((float(p_te[i]), float(yt[i])))
    res["test_by_kind"] = {k: {"n": len(v), "wrong": sum(b for _, b in v), "mean_p_wrong": round(sum(a for a, _ in v) / len(v), 3)}
                           for k, v in sorted(per_kind.items())}
    # a probe could score well just by knowing which kinds of question are hard; this asks whether
    # it also tells right from wrong among questions of the same kind (kinds with both outcomes only)
    within = [(auroc(torch.tensor([a for a, _ in v]), torch.tensor([b for _, b in v])), len(v))
              for v in per_kind.values() if 0 < sum(b for _, b in v) < len(v)]
    res["test_auroc_within_kind"] = sum(a * n for a, n in within) / sum(n for _, n in within) if within else float("nan")
    log(f"test AUROC: probe {res['test_auroc']:.3f} | logit stats alone {res['test_auroc_logit_stats']:.3f} | "
        f"agreement of {n_samples(data_dir)} samples "
        f"{res['test_auroc_sample_agreement']:.3f} | probe within one kind {res['test_auroc_within_kind']:.3f}")
    log(f"threshold {thr:.2f}: flags {res['test_flag_rate']:.0%} of test answers, precision {res['test_precision']:.2f}, "
        f"recall {res['test_recall']:.2f}; {res['test_not_judged']:.0%} out of range (not judged)")
    if len(prose):
        log(f"uncheckable prose sentences ({len(prose)}, from answers about real and made-up things): flagged "
            f"{res['prose_flag_rate']:.0%}, out of range {res['prose_not_judged']:.0%}")
    if out:
        save_probe(out, m, layer=None if best["li"] is None else taps[best["li"]], threshold=thr,
                   metrics=res, choice=best["name"], ood_max=ood_max)
        log(f"saved {out}")
    return res


def n_samples(data_dir) -> int:
    first = data_dir[0] if isinstance(data_dir, (list, tuple)) else data_dir
    return len(json.load(open(os.path.join(first, "answers.json")))[0]["samples"])


def save_probe(path: str, model, *, layer: int | None, threshold: float, metrics: dict, choice: str,
               ood_max: float | None = None):
    mean, std, w, b = model
    state = {"probe.mean": mean.contiguous(), "probe.std": std.contiguous(), "probe.w": w.contiguous(),
             "probe.b": b.reshape(1).contiguous()}
    config = {"kind": "confidence-probe", "layer": layer, "stats": STATS, "threshold": threshold,
              "features": "mean hidden state over the sentence after `layer` (if any), then stats",
              "choice": choice, "ood_max": ood_max}
    clean = lambda v: None if isinstance(v, float) and not math.isfinite(v) else v  # the Rust JSON reader rejects NaN
    metrics = {k: ({a: clean(b) for a, b in v.items()} if isinstance(v, dict) else clean(v)) for k, v in metrics.items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nx_save(path, state, config, "", extra={"metrics": metrics})


def load_probe(path: str):
    with NxFile(path) as nx:
        cfg = nx.config
        m = tuple(nx.tensor(k).clone() for k in ("probe.mean", "probe.std", "probe.w", "probe.b"))
    return cfg, m


def score(probe, hidden_mean: torch.Tensor | None, stats: torch.Tensor) -> float | None:
    """P(wrong) for one sentence: hidden_mean [d] (mean over its tokens) or None, stats [4].
    None when the sentence is out of the probe's range (not judged)."""
    cfg, m = probe
    x = stats if cfg["layer"] is None else torch.cat([hidden_mean, stats])
    if cfg.get("ood_max") is not None and float(ood_score(m, x[None])[0]) > cfg["ood_max"]:
        return None
    return float(predict(m, x[None])[0])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m aegist.confidence")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--brain", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--samples", type=int, default=4)
    c.add_argument("--max-new", type=int, default=40)
    c.add_argument("--limit", type=int)
    c.add_argument("--bank", choices=sorted(BANKS), default="base")
    c.add_argument("--taps", type=int, nargs="+", default=DEFAULT_TAPS,
                   help="hidden layers to keep (the default suits 40 layers; spread them over the brain's own)")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, nargs="+", help="one or more collections")
    t.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.brain, a.out, taps=a.taps, samples=a.samples, max_new=a.max_new, limit=a.limit, bank=a.bank)
    else:
        print(json.dumps(train(a.data, a.out), indent=1))


if __name__ == "__main__":
    main()
