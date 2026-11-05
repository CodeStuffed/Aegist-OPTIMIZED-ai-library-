"""Download moonshotai/Kimi-K2.6 one shard at a time and lay its experts out for one-read-per-expert loading.

K2.6 is published already quantised (int4, quantisation-aware trained), so nothing is requantised here:
the bytes are copied. Shards 1-61 each hold one decoder layer; 62-64 hold embeddings, head and the
vision tower (skipped: this brain is text only).

Expert format "q4g32s" (Moonshot's compressed-tensors "pack-quantized", symmetric, group 32), one record
per expert: gate then up then down, each [codes, scales]:
    codes   int32 [out, in/8]; 8 values per word, value i in bits 4i..4i+3, stored as v+8 (0..15)
    scales  bf16 [out, in/32]; w = scale * (nibble - 8)
gate/up are [2048, 7168], down is [7168, 2048]. One file per layer on E:, 384 records of EXPERT_BYTES.
Everything else (MLA attention, norms, router + correction bias, shared experts, the dense layer 0,
embeddings, head) is kept byte-exact in dense/ on D:.

    python -m aegist.fetch_k2 [--shards N]
Resumable: progress.json lists finished shards; a layer file is written to .tmp and renamed when complete.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .fetch_max import low_priority

REPO = "moonshotai/Kimi-K2.6"
REV = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
HOME = "D:/aegist/models/kimi-k2.6"
STAGE = HOME + "/stage"
DENSE = HOME + "/dense"
EXPERTS = "E:/aegist/brains/kimi/experts"  # complete copy; a hot subset goes to D: after profiling
HIDDEN, INTER, N_EXPERTS, GROUP = 7168, 2048, 384, 32
SHAPES = (("gate", INTER, HIDDEN), ("up", INTER, HIDDEN), ("down", HIDDEN, INTER))
PREFIX = "language_model.model."
EXPERT = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight_(packed|scale|shape)$")


def mat_bytes(out: int, inn: int) -> tuple[int, int]:
    return out * inn // 2, out * inn // GROUP * 2


EXPERT_BYTES = sum(sum(mat_bytes(o, i)) for _, o, i in SHAPES)


def layer_file(layer: int) -> str:
    return f"{EXPERTS}/L{layer:02d}.q4"


def raw(t: torch.Tensor) -> bytes:
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.contiguous().numpy().tobytes()


def nibbles(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    return ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(packed.shape[0], -1)


def download(fn: str) -> str:
    from huggingface_hub import hf_hub_download
    for attempt in range(20):
        try:
            return hf_hub_download(REPO, fn, revision=REV, local_dir=STAGE)
        except Exception as e:  # network hiccups: wait and retry
            print(f"  download {fn} failed ({e}); retry {attempt + 1}", flush=True)
            time.sleep(min(600, 30 * (attempt + 1)))
    raise RuntimeError(f"could not download {fn}")


def convert(path: str, fn: str) -> dict:
    st = {"experts": 0, "dense": 0}
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        experts = {}
        dense = []
        for k in keys:
            m = EXPERT.search(k)
            if m:
                experts.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
            elif k.startswith("language_model."):
                dense.append(k)
        for layer, es in experts.items():
            assert es == set(range(N_EXPERTS)), f"{fn}: layer {layer} has {len(es)} experts"
            out, tmp = layer_file(layer), layer_file(layer) + ".tmp"
            hist = torch.zeros(16, dtype=torch.long)
            with open(tmp, "wb", buffering=64 << 20) as h:
                for e in range(N_EXPERTS):
                    base = f"{PREFIX}layers.{layer}.mlp.experts.{e}."
                    rec = []
                    for p, o, i in SHAPES:
                        q = f.get_tensor(base + f"{p}_proj.weight_packed")
                        s = f.get_tensor(base + f"{p}_proj.weight_scale")
                        shape = f.get_tensor(base + f"{p}_proj.weight_shape").tolist()
                        assert shape == [o, i] and tuple(q.shape) == (o, i // 8) and q.dtype == torch.int32, (base, p, shape, q.shape)
                        assert tuple(s.shape) == (o, i // GROUP) and s.dtype == torch.bfloat16, (base, p, s.shape, s.dtype)
                        if e == 0:
                            hist += torch.bincount(nibbles(q).flatten(), minlength=16)
                        rec += [raw(q), raw(s)]
                    blob = b"".join(rec)
                    assert len(blob) == EXPERT_BYTES
                    h.write(blob)
                h.flush()
                os.fsync(h.fileno())
            os.replace(tmp, out)
            st["experts"] += N_EXPERTS
            # Layout check: with "v + 8" storage the zero value is nibble 8 and nibble 0 (-8) is unused
            # by symmetric min-max; two's complement would put the peak at 0 and 15 instead.
            st[f"L{layer}_peak_nibble"] = int(hist.argmax())
            st[f"L{layer}_nibble0_share"] = round(float(hist[0] / hist.sum()), 5)
        if dense:
            save_file({k[len("language_model."):]: f.get_tensor(k).contiguous() for k in dense},
                      f"{DENSE}/{fn}.tmp", metadata={"source": f"{REPO}@{REV}/{fn}"})
            os.replace(f"{DENSE}/{fn}.tmp", f"{DENSE}/{fn}")
            st["dense"] = len(dense)
    return st


def manifest():
    m = {"repo": REPO, "revision": REV, "format": "q4g32s", "group": GROUP, "levels": "w = scale * (nibble - 8)",
         "packing": "int32 words, 8 nibbles each, value i in bits 4i..4i+3 (compressed-tensors pack-quantized)",
         "record": [f"{p} codes int32 [{o}x{i // 8}], {p} scales bf16 [{o}x{i // GROUP}]" for p, o, i in SHAPES],
         "expert_bytes": EXPERT_BYTES, "experts_per_layer": N_EXPERTS, "moe_layers": "1-60 (layer 0 is dense)",
         "files": {f"L{i:02d}": layer_file(i) for i in range(1, 61)}, "dense": DENSE,
         "license": HOME + "/LICENSE", "notices": HOME + "/THIRD_PARTY_NOTICES.md"}
    json.dump(m, open(HOME + "/manifest.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=0, help="stop after this many new shards (0 = all)")
    args = ap.parse_args()
    low_priority()
    for d in (STAGE, DENSE, EXPERTS):
        os.makedirs(d, exist_ok=True)
    open(HOME + "/REVISION", "w").write(REV)
    idx = json.load(open(HOME + "/model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(idx.values()))
    prog_path = HOME + "/progress.json"
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"done": [], "log": []}
    todo = [s for s in shards if s not in prog["done"]]
    if args.shards:
        todo = todo[: args.shards]
    manifest()
    print(f"{len(prog['done'])} of {len(shards)} shards done; doing {len(todo)}", flush=True)

    ahead: dict[str, threading.Thread] = {}
    got: dict[str, str] = {}

    def start(i):
        if i < len(todo) and todo[i] not in ahead:
            t = threading.Thread(target=lambda fn=todo[i]: got.__setitem__(fn, download(fn)), daemon=True)
            t.start()
            ahead[todo[i]] = t

    start(0)
    for i, fn in enumerate(todo):
        t0 = time.time()
        ahead[fn].join()
        start(i + 1)  # next shard downloads while this one is copied out
        t1 = time.time()
        for attempt in range(6):  # D: is a USB enclosure that sometimes resets: wait it out and redo the shard
            try:
                st = convert(got[fn], fn)
                break
            except Exception as e:  # OSError, and safetensors' own SafetensorError for the same drive reset
                print(f"  copying out {fn} failed ({e}); retry {attempt + 1} in {60 * (attempt + 1)}s", flush=True)
                time.sleep(60 * (attempt + 1))
        else:
            raise RuntimeError(f"could not copy out {fn}: drive keeps failing")
        size = os.path.getsize(got[fn])
        os.remove(got[fn])
        entry = {"shard": fn, "gb": round(size / 1e9, 2), "wait_s": round(t1 - t0), "copy_s": round(time.time() - t1),
                 **st, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        prog["done"].append(fn)
        prog["log"].append(entry)
        json.dump(prog, open(prog_path + ".tmp", "w"), indent=1)
        os.replace(prog_path + ".tmp", prog_path)
        print(json.dumps(entry), flush=True)
    print("finished" if len(prog["done"]) == len(shards) else "stopped", flush=True)


if __name__ == "__main__":
    main()
