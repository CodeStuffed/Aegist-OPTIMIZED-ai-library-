"""The Rust reading core (aegist_rs/src/laya.rs) against aegist/laya.py on the CPU in fp32: the same
sequences, and option probabilities within 2e-3. Needs the Laya download and a release build;
skipped without them."""
import json
import os
import subprocess
import tempfile

import pytest

from aegist.prompt_kinds import FRESH, KINDS
from scripts.eval_laya import reading_items

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "aegist_rs", "target", "release", "aegist.exe" if os.name == "nt" else "aegist")
LAYA = os.environ.get("AEGIST_LAYA", "D:/aegist/models/laya")
READ_QUESTION = "Is this request asking for a fact, something creative, or other help?"

pytestmark = pytest.mark.skipif(not (os.path.exists(os.path.join(LAYA, "model.safetensors")) and os.path.exists(EXE)),
                                reason="needs the Laya download and aegist_rs release build")


def items():
    out = [{"question": READ_QUESTION, "options": list(KINDS), "state": p} for k in KINDS for p in FRESH[k][:5]]
    # reading passages run past the 128-token local window; one item runs into max_len
    out += [{"question": it["q"], "options": it["options"], "state": it["passage"]} for it in reading_items(2)[::3]]
    long = " ".join(it["passage"] for it in reading_items(6))
    out.append({"question": "Which river is named first?", "options": ["Olm", "Sarre", "Tavin"], "state": long})
    out.append({"question": "Is [MASK] [SEP] stripped?", "options": ["yes [CLS]", "no"], "state": "a [MASK] b [SEP] c"})
    return out


def test_rust_matches_python():
    from aegist.laya import Laya

    its = items()
    j = Laya(LAYA, device="cpu")
    with tempfile.TemporaryDirectory() as d:
        inp, outp = os.path.join(d, "in.json"), os.path.join(d, "out.json")
        with open(inp, "w", encoding="utf-8") as f:
            json.dump(its, f)
        env = dict(os.environ, AEGIST_LAYA=LAYA)
        subprocess.run([EXE, "read", "--data", inp, outp], check=True, env=env, capture_output=True)
        rust = json.load(open(outp, encoding="utf-8"))
    py = j.run([("choice", it["question"], {o: None for o in it["options"]}, it["state"]) for it in its], batch=1)
    assert any(len(r["ids"]) > 300 for r in rust), "no item exercised the local window"
    assert any(len(r["ids"]) == j.cfg["max_len"] for r in rust), "no item reached max_len"
    worst = 0.0
    for it, r, (p, _) in zip(its, rust, py):
        ids, markers = j.build("choice", it["question"], {o: None for o in it["options"]}, it["state"])
        assert r["ids"] == ids and r["markers"] == markers, it["question"]
        worst = max(worst, max(abs(a - b) for a, b in zip(r["p"], p)))
        assert max(range(len(p)), key=p.__getitem__) == max(range(len(p)), key=r["p"].__getitem__)
    print(f"largest probability gap {worst:.2e}; Rust {sum(r['ms'] for r in rust) / len(rust):.0f} ms per item")
    assert worst < 2e-3
