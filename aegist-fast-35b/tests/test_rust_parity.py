"""The Rust encoder must be a drop-in replacement: identical pieces and ids."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

nexus_rs = pytest.importorskip("nexus_rs")

from nexus.tokenizer import BPETokenizer, pretokenize  # noqa: E402

TOK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "data", "tokenizer.json")


def random_strings(n=3000, seed=0):
    rng = random.Random(seed)
    pools = [(0x0, 0x7F), (0x80, 0x24F), (0x300, 0x36F), (0x370, 0x3FF), (0x600, 0x6FF), (0x900, 0x97F),
             (0x1C, 0x20), (0x2000, 0x206F), (0x3000, 0x30FF), (0x4E00, 0x4E50), (0xFF00, 0xFFEF),
             (0x1F600, 0x1F64F), (0x10330, 0x1034A), (0x2160, 0x2188), (0xB2, 0xBE)]
    for _ in range(n):
        out = []
        for _ in range(rng.randint(0, 40)):
            lo, hi = rng.choice(pools)
            c = rng.randint(lo, hi)
            if 0xD800 <= c <= 0xDFFF:
                continue
            out.append(chr(c) if rng.random() > 0.3 else rng.choice(" _\n\t  1a"))
        yield "".join(out)


def test_pretokenize_parity_random_unicode():
    for s in random_strings():
        assert nexus_rs.pretokenize(s) == pretokenize(s), repr(s)


@pytest.mark.skipif(not os.path.exists(TOK_PATH), reason="run scripts/prepare_data.py first")
def test_encode_parity_trained_tokenizer():
    py = BPETokenizer.load(TOK_PATH)
    py._rs = None  # force the pure-Python path for the reference
    if not os.environ.get("NEXUS_NO_RUST"):
        assert BPETokenizer.load(TOK_PATH).backend == "rust"
    rs = nexus_rs.RsBpe(py.merges)
    samples = list(random_strings(1000, seed=1)) + ["Q: 98765+43210=?\nA: 141975\n", "def f(x):\n    return x\n"]
    for s in samples:
        assert rs.encode(s) == py.encode(s), repr(s)
    assert rs.encode_batch(samples) == [py.encode(s) for s in samples]
