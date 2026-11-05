"""The confidence probe's pieces (aegist/confidence.py): question bank, grading, AUROC, the
logistic fit, and the probe file. No GPU needed; the engine side is tested in Rust (probe.rs)."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from aegist import confidence as C  # noqa: E402


def test_question_bank_golds_are_right():
    bank = C.question_bank()
    assert len({it["q"] for it in bank}) == len(bank), "duplicate questions"
    for it in bank:
        if it["kind"].startswith("multiply"):
            a, b = map(int, re.findall(r"\d+", it["q"]))
            assert it["gold"] == [str(a * b)]
        if it["kind"] == "letters":
            ch, word = re.findall(r"'([^']+)'", it["q"])
            assert it["gold"] == [str(word.lower().count(ch))]
    fake = [it for it in bank if it["gold"] is None]
    assert 0.15 < len(fake) / len(bank) < 0.35
    # the invented names must not be real entries of the fixed lists
    for it in fake:
        assert not any(k.lower() in it["q"].lower() for k in list(C.CAPITALS) + list(C.ELEMENTS) if len(k) > 4)


def test_grading():
    cap = {"q": "What is the capital of Bolivia?", "gold": ["Sucre"], "kind": "capital"}
    assert C.is_correct("The capital of Bolivia is Sucre.", cap)
    assert not C.is_correct("The capital of Bolivia is La Paz.", cap)
    mul = {"q": "What is 1234 times 5678?", "gold": ["7006652"], "kind": "multiply 4x4"}
    assert C.is_correct("1234 times 5678 is 7,006,652.", mul)
    assert not C.is_correct("It is 7,006,682.", mul)
    assert not C.is_correct("70066520", mul), "a longer number that contains the answer is wrong"
    let = {"q": "How many times does the letter 'e' appear in the word 'bookkeeper'?", "gold": ["3"], "kind": "letters"}
    assert C.is_correct("The letter 'e' appears three times in the word 'bookkeeper'.", let)
    assert not C.is_correct("The letter 'e' appears twice in the word 'bookkeeper'.", let)
    assert not C.is_correct("The letter 'e' appears 33 times.", let)
    assert C.is_correct("It appears once.", dict(let, gold=["1"]))
    assert C.is_correct("Iron has atomic number twenty-six.", dict(let, gold=["26"]))
    assert not C.is_correct("The letter 'u' does not appear in 'occurrence'.", dict(let, gold=["1"]))
    fake = {"q": "Who wrote 'The Voress of Pelling'?", "gold": None, "kind": "fake book"}
    assert C.is_correct("I'm not aware of a novel by that title.", fake)
    assert C.is_correct("There is no record of such a book; it may not exist.", fake)
    assert not C.is_correct("'The Voress of Pelling' was written by Henrietta Marsh in 1923.", fake)
    battle = {"q": "In what year was the Battle of Keshosthal fought?", "gold": None, "kind": "fake battle"}
    assert C.is_correct("The Battle of Keshosthal did not occur, as it is not a recognized historical event.", battle)
    assert not C.is_correct("The Battle of Keshosthal did not occur, as Keshosthal is a fictional location from the "
                            "video game *Dishonored*.", battle), "declining while inventing a source is made up"
    assert not C.is_correct("It is a fictional location from the *Star Wars* universe.", battle)


def test_claims():
    bank = C.claim_bank()
    assert len({it["claim"] for it in bank}) == len(bank) == len(C.TRUE_CLAIMS) + len(C.FALSE_CLAIMS)
    yes = next(it for it in bank if it["gold"] == ["yes"])
    no = next(it for it in bank if it["gold"] == ["no"])
    assert C.is_correct("Yes, that is right.", yes) and C.is_correct("**No.** It does not work.", no)
    assert not C.is_correct("No, it is a myth.", yes) and not C.is_correct("Yes.", no)
    assert not C.is_correct("Nobody knows.", no), "a word that starts with no is not a verdict"
    assert C.agreement("No, it is false.", ["No.", "Yes, it is.", "False."], no) == 2 / 3


def test_agreement():
    cap = {"q": "What is the capital of Bolivia?", "gold": ["Sucre"]}
    assert C.agreement("The capital of Bolivia is Sucre.", ["Sucre is the capital.", "It is Sucre."], cap) == 1.0
    assert C.agreement("The capital is Sucre.", ["The capital of Bolivia is La Paz.", "Sucre."], cap) == 0.5
    mul = {"q": "What is 12 times 34?", "gold": ["408"]}
    assert C.agreement("12 times 34 is 408.", ["12 times 34 is 418.", "It is 408."], mul) == 0.5
    fake = {"gold": None}
    assert C.agreement("No such country exists.", ["It is Varn.", "I am not aware of it."], fake) == 0.5


def test_auroc():
    y = torch.tensor([0, 0, 1, 1.0])
    assert C.auroc(torch.tensor([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert C.auroc(torch.tensor([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert C.auroc(torch.tensor([0.5, 0.5, 0.5, 0.5]), y) == 0.5


def test_fit_and_probe_file_round_trip(tmp_path):
    g = torch.Generator().manual_seed(0)
    n, d = 400, 16
    x = torch.randn(n, d, generator=g) * torch.linspace(0.5, 5, d)
    y = ((x[:, 3] / 2 - x[:, 7] / 4 + 0.3 * torch.randn(n, generator=g)) > 0).float()
    m = C.fit_logistic(x[:300], y[:300], l2=1e-2)
    assert C.auroc(C.predict(m, x[300:]), y[300:]) > 0.9
    # a probe over (hidden [12], stats [4]) written and read back scores identically
    path = str(tmp_path / "p.nx")
    C.save_probe(path, m, layer=20, threshold=0.4, metrics={"test_auroc": 0.9, "nan": float("nan")}, choice="test")
    cfg, m2 = C.load_probe(path)
    assert cfg["layer"] == 20 and cfg["threshold"] == 0.4 and cfg["kind"] == "confidence-probe"
    for a, b in zip(m, m2):
        assert torch.equal(a.reshape(-1), b.reshape(-1))
    h, s = x[0, :12], x[0, 12:]
    want = float(torch.sigmoid(((x[0] - m[0]) / m[1]) @ m[2] + m[3]))
    assert abs(C.score((cfg, m2), h, s) - want) < 1e-6
    from aegist.nx import NxFile
    with NxFile(path) as nx:
        assert nx.header["metrics"]["nan"] is None, "NaN must not reach the JSON header"


class _Tok:
    """Each id decodes to one piece of text."""

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def test_sentences_match_the_rust_tracker():
    pieces = ["**", "Facts", ":**", "\n", "The", " sky", " is", " blue", ".", " Pi", " is", " 3", ".", "14", ".\n",
              "```", "py", "\n", "x", " =", " 1", "\n", "```", "\n", "Done", " and", " dusted"]
    cut = C.sentences(_Tok(pieces), list(range(len(pieces))))
    assert [t for t, _ in cut] == ["The sky is blue.", "Pi is 3.14.", "Done and dusted"], cut
    assert cut[0][1] == [4, 5, 6, 7, 8]
    assert cut[1][1] == [9, 10, 11, 12, 13], "the token that carries the line break is left out"


def test_paragraph_labels():
    cap = {"q": "Write three short sentences about Bolivia. Mention its capital.", "gold": ["Sucre"],
           "kind": "para capital", "para": True}
    assert C.label_sentences(["Bolivia is in South America.", "Its capital is Sucre.", "La Paz is its capital."], cap) \
        == [None, 0.0, None], "the right capital is stated; the other sentence is about the seat of government"
    assert C.label_sentences(["Bolivia is in South America.", "Its capital is Lima."], cap) == [None, 1.0]
    yr = {"q": "...", "gold": ["1919"], "kind": "para year", "para": True}
    assert C.label_sentences(["It was signed in 1920.", "The war ran from 1914 to 1918."], yr) == [1.0, None]
    assert C.label_sentences(["It was signed in 1919.", "It came into force in 1920."], yr) == [0.0, None], \
        "with the right year stated, another year is about something else"
    book = {"q": "...", "gold": ["Hamsun"], "kind": "para author", "para": True}
    assert C.label_sentences(["Hunger is a novel by Knut Hamsun.", "Its unnamed writer starves in Oslo."], book) \
        == [0.0, None]
    assert C.label_sentences(["Hunger was written by Henrik Ibsen.", "The writer starves."], book) == [1.0, None]
    fake = {"q": "...", "gold": None, "kind": "para fake country", "para": True}
    assert C.label_sentences(["Voress is a small nation.", "Its capital is Vorstad."], fake) == [1.0, 1.0]
    assert C.label_sentences(["I don't know of a Republic of Voress.", "Perhaps you mean Belarus?"], fake) == [0.0, None]
    assert C.label_sentences(["It does not exist; it is from the video game *Dishonored*."], fake) == [1.0]


def test_split_is_stratified_and_disjoint():
    bank = C.question_bank()
    tr, va, te = C.split(bank)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(bank)
    assert {bank[i]["kind"] for i in te} == {it["kind"] for it in bank}
