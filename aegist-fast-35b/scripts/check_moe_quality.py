"""How much does the Aegist brain's quantisation change the base model? Measured, not assumed.

Reference: the original bf16 checkpoint, run one block at a time on the GPU straight from its
safetensors shards (the whole bf16 model does not fit in RAM or VRAM). Candidate: the .nx brain
through the Python reference (4-bit experts, int8 linears). Both run the same Aegist code, which
tests/test_hybrid.py checks against transformers' implementation.

Reports per text: next-token loss for both, KL(reference || brain) per token, top-1 agreement;
then a short greedy chat reply from the brain as a sanity check.

    python -I scripts/check_moe_quality.py --base D:/aegist/models/qwen3.6-35b-a3b --brain D:/aegist/brains/aegist-nx-35b.nx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from aegist.convert_moe import Shards, adapt, config_from_hf  # noqa: E402
from aegist.hybrid import HybridBlock, ZNorm, rope_tables  # noqa: E402
from aegist.nx import load_model  # noqa: E402
from aegist.tokenizer import AegistTokenizer  # noqa: E402

TEXTS = {
    "encyclopedic": (
        "The water cycle describes how water moves between the oceans, the air and the land. Sunlight warms the "
        "surface of the sea, and some of the water evaporates into vapour. As the vapour rises it cools, condenses "
        "into tiny droplets and forms clouds. When the droplets grow heavy enough they fall as rain or snow. Some of "
        "that water soaks into the ground and feeds rivers and lakes, and eventually much of it flows back to the sea, "
        "where the cycle begins again."),
    "code": (
        "def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n"
        "        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
        "    return -1\n\n\nprint(binary_search([1, 3, 5, 7, 9, 11], 7))\n"),
    "math": (
        "To find the derivative of f(x) = x^3 - 4x + 2, apply the power rule to each term. The derivative of x^3 is "
        "3x^2, the derivative of -4x is -4, and the derivative of a constant is zero. So f'(x) = 3x^2 - 4. Setting "
        "f'(x) = 0 gives x^2 = 4/3, so the critical points are x = 2/sqrt(3) and x = -2/sqrt(3)."),
    "casual": (
        "ngl the new update is kinda mid, the menu moved again and now i cant find the settings button lol. "
        "anyway are we still going to the game on saturday or nah? my cousin said he can drive if we need a ride."),
}


@torch.no_grad()
def reference_logprobs(base: str, ids_list: list[list[int]], device) -> list[torch.Tensor]:
    cfg = config_from_hf(json.load(open(os.path.join(base, "config.json"), encoding="utf-8")))
    shards = Shards(base)
    have = shards.aegist_names()
    emb = shards.get(have["embed.weight"])
    xs = [emb[torch.tensor(ids)].unsqueeze(0).to(device) for ids in ids_list]
    del emb
    T = max(len(i) for i in ids_list)
    cos, sin = rope_tables(cfg, T, device)
    for i in range(cfg.n_layers):
        with torch.device("meta"):
            blk = HybridBlock(cfg, i).to(torch.bfloat16)
        blk = blk.to_empty(device=device)
        blk.cores_enabled = False
        sd = blk.state_dict()
        for name, t in sd.items():
            full = f"blocks.{i}.{name}"
            if full in have:
                t.copy_(adapt(full, shards.get(have[full])))
            elif ".engine." not in full and ".cores." not in full:
                raise KeyError(full)
        xs = [blk(x, cos, sin, None)[0] for x in xs]
        del blk
        torch.cuda.empty_cache()
        print(f"  reference block {i + 1}/{cfg.n_layers}", end="\r", flush=True)
    print()
    norm = ZNorm(cfg.d_model, cfg.norm_eps).to(device, torch.bfloat16)
    norm.weight.copy_(shards.get(have["norm.weight"]))
    head = shards.get(have["lm_head.weight"]).to(device)
    out = [F.linear(norm(x), head)[0].float().log_softmax(-1).cpu() for x in xs]
    del head
    torch.cuda.empty_cache()
    return out


def report(name, ids, ref, got):
    tgt = torch.tensor(ids[1:])
    ce_ref = float(-ref[:-1].gather(1, tgt[:, None]).mean())
    ce_got = float(-got[:-1].gather(1, tgt[:, None]).mean())
    kl = (ref.exp() * (ref - got)).sum(-1)
    agree = float((ref.argmax(-1) == got.argmax(-1)).float().mean())
    print(f"{name:13s} {len(ids):4d} tokens | loss bf16 {ce_ref:.4f}  brain {ce_got:.4f} | KL mean {float(kl.mean()):.4f}"
          f" max {float(kl.max()):.3f} | top-1 agreement {agree:.1%}", flush=True)
    return float(kl.mean()), agree, ce_ref, ce_got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="D:/aegist/models/qwen3.6-35b-a3b")
    ap.add_argument("--brain", default="D:/aegist/brains/aegist-nx-35b.nx")
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    device = "cuda"
    tok = AegistTokenizer.from_json_str(open(os.path.join(a.base, "tokenizer.json"), encoding="utf-8").read())
    ids_list = [tok.encode_trusted(t) for t in TEXTS.values()]

    t = time.time()
    refs = reference_logprobs(a.base, ids_list, device)
    print(f"bf16 reference: {time.time() - t:.0f}s", flush=True)

    t = time.time()
    model, tok2, nx = load_model(a.brain, device=device, slots=a.slots)
    print(f"brain loaded in {time.time() - t:.0f}s; GPU memory {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    rows = {}
    with torch.no_grad():
        for (name, _), ids, ref in zip(TEXTS.items(), ids_list, refs):
            t = time.time()
            got = model(torch.tensor([ids], device=device))["logits"][0].float().log_softmax(-1).cpu()
            rows[name] = report(name, ids, ref, got)
    kls = [r[0] for r in rows.values()]
    print(f"mean KL over texts {sum(kls) / len(kls):.4f}", flush=True)

    # sanity: a short greedy chat reply, with the cache (prefill + one token at a time)
    q = "What is the capital of France? Answer in one sentence."
    ids = tok2.chat_ids("You are a helpful assistant.", [("user", q)], think=False)
    cache = model.new_cache(1, 4096)
    t = time.time()
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device), cache=cache, last_only=True)["logits"][0, -1]
        out = []
        for _ in range(40):
            nxt = int(logits.argmax())
            if nxt in tok2.eos_ids:
                break
            out.append(nxt)
            logits = model(torch.tensor([[nxt]], device=device), cache=cache, last_only=True)["logits"][0, -1]
    dt = time.time() - t
    reply = tok2.decode(out)
    print(f"greedy reply ({len(out)} tokens, {len(out) / dt:.2f} tok/s in the Python reference): {reply!r}", flush=True)
    if a.out:
        json.dump({"texts": rows, "reply": reply}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
