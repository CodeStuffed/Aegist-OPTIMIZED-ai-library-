"""Build the hybrid-MoE Aegist brain from Qwen3.6-35B-A3B (Apache-2.0) in one streaming pass.

    python -I scripts/make_aegist_moe.py --base D:/aegist/models/qwen3.6-35b-a3b --out D:/aegist/brains/aegist-nx-35b.nx

The result is one .nx file (about 22 GB): 4-bit experts, int8 big linears, bf16 embedding and
small tensors, an empty sparse core bank (grid --grid rows x cols, grows to --rows-max rows),
tokenizer, license notice, persona, generation defaults and lineage.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from aegist.convert_moe import build_brain  # noqa: E402
from aegist.nx import NxFile  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="D:/aegist/models/qwen3.6-35b-a3b")
    ap.add_argument("--out", default="D:/aegist/brains/aegist-nx-35b.nx")
    ap.add_argument("--grid", type=int, nargs=2, default=[8, 64], metavar=("ROWS", "COLS"))
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--active", type=int, default=2)
    ap.add_argument("--rows-max", type=int, default=85)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.time()
    header = build_brain(a.base, a.out, core_grid=a.grid, core_rank=a.rank, cores_active=a.active,
                         core_rows_max=a.rows_max, expert_group=a.group, seed=a.seed, device=device,
                         log=lambda m: print(m, flush=True))
    errs = header["expert_error"]
    print(f"expert 4-bit relative weight error: mean {sum(errs.values()) / len(errs):.4f}, "
          f"worst {max(errs.values()):.4f} (block {max(errs, key=errs.get)})", flush=True)
    print(f"params: {header['params']}", flush=True)
    t2 = time.time()
    with NxFile(a.out) as nx:
        ok = nx.verify()
    print(f"checksums {ok} ({time.time() - t2:.0f}s); total {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
