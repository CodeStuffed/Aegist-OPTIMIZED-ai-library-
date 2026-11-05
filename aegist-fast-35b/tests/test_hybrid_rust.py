"""The Rust hybrid-MoE engine (aegist.exe) must match the Python reference (aegist/hybrid.py).

Opt-in, because it loads the 21 GB brain on the GPU twice:
    set AEGIST_GPU_TESTS=1 and run pytest tests/test_hybrid_rust.py -s
Brain: $AEGIST_BRAIN, else D:/aegist/brains/aegist.nx.

Covers every kernel path: one token (decode), 2-8 tokens (per-pick expert kernels), 9-64 tokens
(tiled expert kernels), over 64 (expert streaming), over 1024 (several prefill chunks), and the
snapshot / crop rewind that chat uses between turns.

Long prompts are judged on every position, not one: this model is chaotic at some positions (a
1e-7 change, or storing keys/values in bf16 instead of fp32, flips a few next-token choices in the
Python reference itself), so one final row can differ a lot while the engine is right. The long
test reads every row's hidden state ($AEGIST_DUMP_FINAL) and checks long-range copying, overall
agreement, and that only a handful of rows disagree.
"""
import gc
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

EXE = os.path.join(ROOT, "aegist_rs", "target", "release", "aegist.exe" if os.name == "nt" else "aegist")
BRAIN = os.environ.get("AEGIST_BRAIN") or "D:/aegist/brains/aegist.nx"

pytestmark = pytest.mark.skipif(
    not (os.environ.get("AEGIST_GPU_TESTS") and os.path.exists(EXE) and os.path.exists(BRAIN)
         and torch.cuda.is_available()),
    reason="set AEGIST_GPU_TESTS=1, build aegist_rs (cargo build --release) and have the hybrid brain")

PARAGRAPHS = [
    "The water cycle describes how water moves between the oceans, the air and the land. Sunlight warms the sea, "
    "some of the water evaporates, and the vapour rises, cools and condenses into clouds.",
    "def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n"
    "        if items[mid] == target:\n            return mid\n        lo, hi = (mid + 1, hi) if items[mid] < target "
    "else (lo, mid - 1)\n    return -1\n",
    "To differentiate f(x) = x^3 - 4x + 2, apply the power rule term by term: f'(x) = 3x^2 - 4, which is zero at "
    "x = 2/sqrt(3) and x = -2/sqrt(3).",
    "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen. It happens in chloroplasts, where "
    "chlorophyll absorbs mostly red and blue light and reflects green.",
    "In 1969 the Apollo 11 mission landed two astronauts on the Moon. They collected rock samples, set up "
    "experiments and returned safely to Earth after eight days.",
]


def tokenizer():
    from aegist.nx import NxFile
    from aegist.tokenizer import AegistTokenizer

    with NxFile(BRAIN) as nx:
        return AegistTokenizer.from_json_str(nx.blob("tokenizer.json").decode("utf-8"))


COLOURS = ["green", "amber", "violet", "crimson", "teal", "ochre", "silver", "indigo", "coral", "olive"]
THINGS = ["lighthouse door", "garden gate", "ferry", "library roof", "market stall", "bicycle", "kite", "tram",
          "windmill", "clock tower", "rowing boat", "post box"]
TOWNS = ["Harwick", "Lindale", "Port Esk", "Marrow", "Kettering Vale", "Saltmere", "Brindle", "Ashby Cross",
         "Fennick", "Grayhollow", "Wexmoor", "Dunmore"]
PEOPLE = ["Mara", "Tobias", "Ines", "Kwame", "Liesel", "Arjun", "Sofia", "Bram", "Yuki", "Odette", "Ravi", "Nell"]
VERBS = ["painted", "repaired", "photographed", "sketched", "measured", "cleaned", "inspected", "described"]


def recall_ids(tok, n_text, seed=7):
    """A varied list of about n_text tokens, an instruction to repeat it, then the repeat: copying
    across a long gap that is not a degenerate loop. Returns (ids, index where the repeat starts)."""
    import random

    rng = random.Random(seed)
    lines = []
    while len(tok.encode_trusted("\n".join(lines))) < n_text:
        lines.append(f"{len(lines) + 1}. In {rng.choice(TOWNS)}, {rng.choice(PEOPLE)} {rng.choice(VERBS)} the "
                     f"{rng.choice(COLOURS)} {rng.choice(THINGS)} on day {rng.randint(2, 97)}.")
    text = "\n".join(lines)
    ask = "\n\nRepeat the list above word for word.\n\n"
    head = tok.encode_trusted(text + ask)
    ids = tok.encode_trusted(text + ask + text)
    assert ids[: len(head)] == head
    return ids, len(head)


def long_ids(tok, n):
    ids, i = [], 0
    while len(ids) < n:
        ids += tok.encode_trusted(f"\n\nSection {i + 1}. " + PARAGRAPHS[i % len(PARAGRAPHS)])
        i += 1
    return ids[:n]


@torch.no_grad()
def python_feeds(feeds_list, greedy_steps):
    """fp32 reference. For each case: logits after each feed in order (through one cache), then
    `greedy_steps[i]` greedy single-token steps. Returns [(rows [n, vocab], steps)]."""
    from aegist.nx import load_model

    model, _, nx = load_model(BRAIN, device="cuda", slots=16)
    model.float()  # fp32 activations like the Rust engine; int8/4-bit weights dequantize to fp32
    out = []
    for feeds, n_steps in zip(feeds_list, greedy_steps):
        cache = model.new_cache(1, sum(map(len, feeds)) + n_steps + 8, dtype=torch.float32)
        rows = []
        for f in feeds:
            rows.append(model(torch.tensor([f], device="cuda"), cache=cache, last_only=True)["logits"][0, -1]
                        .float().cpu())
        steps = []
        for _ in range(n_steps):
            steps.append(int(rows[-1].argmax()))
            rows.append(model(torch.tensor([[steps[-1]]], device="cuda"), cache=cache, last_only=True)["logits"][0, -1]
                        .float().cpu())
        out.append((torch.stack(rows), steps))
        del cache
    del model
    nx.close()
    gc.collect()
    torch.cuda.empty_cache()
    return out


def rust_ops(tmp_path, ops, name="case", env=None):
    inp, out = tmp_path / f"{name}.json", tmp_path / f"{name}.bin"
    inp.write_text(json.dumps({"ops": ops}), encoding="utf-8")
    r = subprocess.run([EXE, "debug-logits", BRAIN, str(inp), str(out), "--context", "4096"], capture_output=True,
                       text=True, env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stderr
    n = sum(1 for o in ops if "feed" in o)
    rows = torch.frombuffer(bytearray(out.read_bytes()), dtype=torch.float32).view(n, -1)
    return rows, r.stderr


def compare(label, py, rs):
    assert rs.shape == py.shape and torch.isfinite(rs).all()
    logp_py, logp_rs = py.log_softmax(-1), rs.log_softmax(-1)
    kl = (logp_py.exp() * (logp_py - logp_rs)).sum(-1)
    top1 = (py.argmax(-1) == rs.argmax(-1)).float()
    margin = py.topk(2, -1).values.diff(dim=-1).abs().squeeze(-1)
    print(f"{label}: {len(py)} rows, top-1 agreement {top1.mean():.1%}, mean KL {kl.mean():.2e}, max KL {kl.max():.2e}")
    assert kl.max() < 2e-2, f"{label}: Rust distribution drifted from the Python reference"
    assert bool(((top1 == 1) | (margin < 0.05)).all()), f"{label}: top-1 differs where it is not a near-tie"


@pytest.fixture(scope="module")
def cases():
    from aegist import persona

    tok = tokenizer()
    q = "If a train leaves at 3pm going 60 km/h, how far has it gone by 5:30pm? Answer briefly."
    chat = tok.chat_ids(persona.system_prompt("normal", persona.read_tone(q)), [("user", q)], think=False)
    mixed = long_ids(tok, 5 + 30 + 100)
    feeds = {
        "chat prompt + greedy decode": ([chat], 24),
        "feeds of 5, 30, 100 tokens": ([mixed[:5], mixed[5:35], mixed[35:]], 4),
    }
    ref = python_feeds([f for f, _ in feeds.values()], [n for _, n in feeds.values()])
    return {k: (f, r) for (k, (f, _)), r in zip(feeds.items(), ref)}, tok


def test_logits_match_python(cases, tmp_path):
    cs, tok = cases
    for i, (label, (feeds, (py, steps))) in enumerate(cs.items()):
        ops = [{"feed": f} for f in feeds] + [{"feed": [s]} for s in steps]
        rs, err = rust_ops(tmp_path, ops, f"case{i}")
        compare(label, py, rs)
        if steps:
            print("  greedy text:", repr(tok.decode(steps)))


@torch.no_grad()
def test_long_prompt_every_position(tmp_path):
    """About 1450 tokens: two prefill chunks, the second crossing position 1024."""
    from aegist.nx import load_model

    tok = tokenizer()
    ids, start = recall_ids(tok, 700)
    n = len(ids) - 1
    hidden = {}
    for name, env in {"tiled experts": {}, "per-pick experts": {"AEGIST_GEMV_TOKENS": "100000"}}.items():
        dump = tmp_path / f"final_{len(hidden)}.f32"
        last, _ = rust_ops(tmp_path, [{"feed": ids[:n]}], f"long{len(hidden)}", {**env, "AEGIST_DUMP_FINAL": str(dump)})
        x = torch.frombuffer(bytearray(dump.read_bytes()), dtype=torch.float32)
        hidden[name] = (x.view(n, -1), last[0])

    model, _, nx = load_model(BRAIN, device="cuda", slots=16)
    model.float()
    cache = model.new_cache(1, n + 8, dtype=torch.float32)
    ref = model.hidden(torch.tensor([ids[:n]], device="cuda"), cache)[0][0]
    del cache
    target = torch.tensor(ids[1:], device="cuda")
    rep = slice(start - 1, n)  # rows whose next token is part of the repeat

    def scores(h):
        nll, top, lps = [], [], []
        for a in range(0, n, 128):
            lp = model.logits(h[a: a + 128]).float().log_softmax(-1)
            nll.append(-lp.gather(-1, target[a: a + 128, None])[:, 0])
            top.append(lp.argmax(-1))
            lps.append(lp)
        return torch.cat(nll), torch.cat(top), lps

    py_nll, py_top, py_lps = scores(ref)
    print(f"python: repeat loss {py_nll[rep].mean():.4f}, repeat top-1 {(py_top == target)[rep].float().mean():.2%}")
    assert py_nll[rep].mean() < 0.05, "the reference itself no longer copies; the test text is wrong"
    for name, (x, last) in hidden.items():
        h = model.norm(x.cuda())
        nll, top, lps = scores(h)
        kl = torch.cat([(p.exp() * (p - r)).sum(-1) for p, r in zip(py_lps, lps)])
        agree = (top == py_top).float().mean()
        rep_top = (top == target)[rep].float().mean()
        head_kl = float((lps[-1][-1].exp() * (lps[-1][-1] - last.cuda().log_softmax(-1))).sum())
        print(f"{name}: repeat loss {nll[rep].mean():.4f}, repeat top-1 {rep_top:.2%}, top-1 agreement {agree:.2%}, "
              f"median KL {kl.median():.2e}, rows with KL > 0.5: {int((kl > 0.5).sum())}, own head vs dump {head_kl:.1e}")
        assert torch.isfinite(x).all()
        assert head_kl < 1e-3, f"{name}: the engine's own output head disagrees with its dumped hidden state"
        assert nll[rep].mean() < py_nll[rep].mean() + 0.01, f"{name}: copies worse than the reference"
        assert rep_top >= (py_top == target)[rep].float().mean() - 0.005, f"{name}: copies worse than the reference"
        assert agree > 0.97 and kl.median() < 1e-3, f"{name}: drifted from the reference"
        assert (kl > 0.5).float().mean() < 0.01, f"{name}: too many positions disagree with the reference"
    del model
    nx.close()
    gc.collect()
    torch.cuda.empty_cache()


def test_snapshot_crop_rewinds(tmp_path):
    tok = tokenizer()
    ids = long_ids(tok, 300)
    a, b, c = ids[:120], ids[120:200], ids[200:260]
    fresh, _ = rust_ops(tmp_path, [{"feed": a + c}], "fresh")
    ops = [{"feed": a[:50]}, {"snapshot": True}, {"feed": a[50:]}, {"snapshot": True}, {"feed": b},
           {"crop": len(a) + 10}, {"feed": c},  # back to the snapshot at len(a), then C
           {"crop": 20}, {"feed": [1, 2, 3]}]  # before every snapshot: starts over
    rows, err = rust_ops(tmp_path, ops, "rewind")
    kept = [int(m) for m in re.findall(r"crop \d+ -> (\d+)", err)]
    assert kept == [len(a), 0], err
    compare("rewound to a snapshot vs fed fresh", fresh[0:1], rows[3:4])
    fresh2, _ = rust_ops(tmp_path, [{"feed": [1, 2, 3]}], "fresh2")
    compare("crop before every snapshot vs fed fresh", fresh2, rows[4:5])
