"""Download Qwen/Qwen3.8-2.4T-A95B-FP8 one shard at a time and keep it as a 3-bit brain.

The release is 2.5 TB (213 shards). There is room for about 1 TB across D: and E:, so each shard is
downloaded to D:, its experts are quantised to 3 bits and written to per-layer expert files, its
other tensors (attention, DeltaNet, norms, router, shared experts, embeddings, head: all bf16) are
kept byte-exact, and the downloaded shard is removed before the next but one is fetched.

Expert format "q3g64" (one record per expert, gate then up then down, one read per expert):
    for each matrix [out, in]: 3-bit codes, then f16 scales
    codes   groups of 64 along `in`; w ~= scale * (code - 4), code in 0..7; 8 codes per 3 bytes,
            little-endian (code i at bit 3*i of a 24-bit word)
    scales  one f16 per group, [out, in/64]; per group the scale with the lowest squared error
            among absmax/{4, 3.5, 3.2, 2.9, 2.6} is kept
Layers 0..D_LAYERS-1 go to D:, the rest and the MTP layer to E: (see manifest.json).

    python -m aegist.fetch_max [--shards N] [--check]
Resumable: progress.json lists finished shards; a shard that was cut off is simply redone.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import re
import threading
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = "Qwen/Qwen3.8-2.4T-A95B-FP8"
HOME = "D:/aegist/models/qwen3.8-max"
STAGE = HOME + "/stage"
DENSE = "E:/aegist/models/qwen3.8-max/dense"  # exact bf16, read once to build the int8 brain on D:
D_EXPERTS = "D:/aegist/brains/max/experts"
E_EXPERTS = "E:/aegist/brains/max/experts"
D_LAYERS = 4  # D: also holds the int8 dense brain (~50 GB) and two staged shards
D_TAIL = 86   # layers 86-91 also go to D: (2026-11-04): E: (1000 GB) can't hold 89 expert layers (932 GB) + exact dense (~108 GB)
GROUP = 64
HIDDEN, INTER, N_EXPERTS = 8192, 2048, 512
EXPERT = re.compile(r"^(model|mtp)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight(_scale_inv)?$")
CANDIDATES = (4.0, 3.5, 3.2, 2.9, 2.6)


def mat_bytes(out: int, inn: int) -> tuple[int, int]:
    groups = out * inn // GROUP
    return groups * 24, groups * 2


SHAPES = (("gate", INTER, HIDDEN), ("up", INTER, HIDDEN), ("down", HIDDEN, INTER))
EXPERT_BYTES = sum(sum(mat_bytes(o, i)) for _, o, i in SHAPES)


def layer_file(kind: str, layer: int) -> str:
    if kind == "mtp":
        return f"{E_EXPERTS}/mtp{layer}.q3"
    return f"{D_EXPERTS if layer < D_LAYERS or layer >= D_TAIL else E_EXPERTS}/L{layer:02d}.q3"


def quantise(w: torch.Tensor) -> tuple[bytes, bytes, float]:
    """[out, in] float32 -> (codes, f16 scales, relative squared error)."""
    g = w.reshape(-1, GROUP)
    amax = g.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best_err = best_q = best_s = None
    for c in CANDIDATES:
        s = (amax / c).half().float()
        s = torch.where(s > 0, s, torch.ones_like(s))  # f16 underflow: an all-but-zero group decodes to 0
        q = torch.clamp(torch.round(g / s), -4, 3)
        err = ((q * s - g) ** 2).sum(1, keepdim=True)
        if best_err is None:
            best_err, best_q, best_s = err, q, s
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_q = torch.where(take, q, best_q)
            best_s = torch.where(take, s, best_s)
    codes = (best_q + 4).to(torch.int32).reshape(-1, 8)
    word = codes[:, 0]
    for i in range(1, 8):
        word = word | (codes[:, i] << (3 * i))
    packed = word.view(torch.uint8).reshape(-1, 4)[:, :3].contiguous()
    rel = float(best_err.sum() / (g * g).sum().clamp_min(1e-30))
    return packed.numpy().tobytes(), best_s.half().numpy().tobytes(), rel


def dequant_fp8(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    s = scale_inv.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return w.float() * s[: w.shape[0], : w.shape[1]]


def low_priority():
    try:
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        pass


def download(fn: str, rev: str) -> str:
    from huggingface_hub import hf_hub_download
    for attempt in range(20):
        try:
            return hf_hub_download(REPO, fn, revision=rev, local_dir=STAGE)
        except Exception as e:  # network hiccups: wait and retry
            print(f"  download {fn} failed ({e}); retry {attempt + 1}", flush=True)
            time.sleep(min(600, 30 * (attempt + 1)))
    raise RuntimeError(f"could not download {fn}")


CARRY = STAGE + "/carry.safetensors"
# byte offset of each matrix (codes then scales) inside an expert record
MATRIX_AT = {}
_at = 0
for _p, _o, _i in SHAPES:
    MATRIX_AT[_p] = _at
    _at += sum(mat_bytes(_o, _i))


def convert(path: str, fn: str, writes: queue.Queue, stats: dict):
    """Quantise the expert matrices of one shard, each straight to its place in its expert's record
    (a shard can hold the gate and up matrices of a layer and the next one its down matrices). A
    weight whose scale is in the next shard, or the other way round, waits in CARRY."""
    dense_names = []
    carry = {}
    if os.path.exists(CARRY):
        with safe_open(CARRY, framework="pt") as c:
            carry = {k: c.get_tensor(k) for k in c.keys()}
    mats: dict[tuple[str, int, int, str], dict[str, torch.Tensor | str]] = {}
    for k, t in carry.items():
        m = EXPERT.match(k)
        mats.setdefault((m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)), {})["s" if m.group(5) else "w"] = t
    left = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            m = EXPERT.match(k)
            if m:
                mats.setdefault((m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)), {})["s" if m.group(5) else "w"] = k
            else:
                dense_names.append(k)

        def get(v):
            return f.get_tensor(v) if isinstance(v, str) else v

        for key in sorted(mats, key=lambda k: (k[0], k[1], k[2], MATRIX_AT[k[3]])):
            kind, layer, e, p = key
            pair = mats[key]
            if "w" not in pair or "s" not in pair:
                for which, v in pair.items():
                    left[f"{kind}.layers.{layer}.mlp.experts.{e}.{p}_proj.weight" + ("_scale_inv" if which == "s" else "")] = get(v).contiguous()
                continue
            c, s, rel = quantise(dequant_fp8(get(pair["w"]), get(pair["s"])))
            stats["err"] += rel
            stats["n"] += 1
            writes.put((layer_file(kind, layer), e * EXPERT_BYTES + MATRIX_AT[p], c + s))
        dense = {k: f.get_tensor(k) for k in dense_names}  # loaded after the experts, to keep RAM free
    if dense:
        save_file(dense, f"{DENSE}/{fn}", metadata={"source": f"{REPO}/{fn}"})
    if left:
        save_file(left, CARRY + ".tmp")
        os.replace(CARRY + ".tmp", CARRY)
    elif os.path.exists(CARRY):
        os.remove(CARRY)
    stats["experts"] += stats["n"] / 3
    stats["carried"] = len(left)


def sparse(fh):
    """Mark a new file sparse, so writing a record past the end does not zero-fill the gap first."""
    try:
        import msvcrt
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        ret = wintypes.DWORD()
        k.DeviceIoControl(wintypes.HANDLE(msvcrt.get_osfhandle(fh.fileno())), 0x000900C4, None, 0, None, 0,
                          ctypes.byref(ret), None)  # FSCTL_SET_SPARSE
    except Exception as e:
        print(f"  could not mark {fh.name} sparse: {e}", flush=True)


def writer(writes: queue.Queue, stats: dict):
    handles = {}
    while True:
        item = writes.get()
        if item is None:
            break
        path, off, rec = item
        if path not in handles:
            new = not os.path.exists(path)
            handles[path] = open(path, "w+b" if new else "r+b")
            if new:
                sparse(handles[path])
        h = handles[path]
        h.seek(off)
        h.write(rec)
        stats["written"] += len(rec)
    for h in handles.values():
        h.flush()
        os.fsync(h.fileno())
        h.close()


def manifest(rev: str):
    files = {f"L{i:02d}": layer_file("model", i) for i in range(92)}
    files["mtp0"] = layer_file("mtp", 0)
    m = {"repo": REPO, "revision": rev, "format": "q3g64", "group": GROUP, "levels": "w = scale * (code - 4)",
         "packing": "8 codes per 3 bytes, little-endian 24-bit word, code i at bit 3*i",
         "scale_choice": "per group, lowest squared error among absmax/" + ",".join(map(str, CANDIDATES)),
         "record": [f"{p} codes [{o}x{i // GROUP}x24 B], {p} scales f16 [{o}x{i // GROUP}]" for p, o, i in SHAPES],
         "expert_bytes": EXPERT_BYTES, "experts_per_layer": N_EXPERTS, "files": files,
         "dense": DENSE, "license": HOME + "/LICENSE"}
    json.dump(m, open(HOME + "/manifest.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=0, help="stop after this many new shards (0 = all)")
    args = ap.parse_args()
    low_priority()
    for d in (STAGE, DENSE, D_EXPERTS, E_EXPERTS):
        os.makedirs(d, exist_ok=True)
    rev = open(HOME + "/REVISION").read().strip()
    idx = json.load(open(HOME + "/model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(idx.values()))
    prog_path = HOME + "/progress.json"
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"done": [], "log": []}
    todo = [s for s in shards if s not in prog["done"]]
    if args.shards:
        todo = todo[: args.shards]
    manifest(rev)
    print(f"{len(prog['done'])} of {len(shards)} shards done; doing {len(todo)}", flush=True)

    ahead: dict[str, threading.Thread] = {}
    got: dict[str, str] = {}

    def fetch(fn):
        got[fn] = download(fn, rev)

    def start(i):
        if i < len(todo) and todo[i] not in ahead:
            t = threading.Thread(target=fetch, args=(todo[i],), daemon=True)
            t.start()
            ahead[todo[i]] = t

    start(0)
    for i, fn in enumerate(todo):
        t0 = time.time()
        ahead[fn].join()
        start(i + 1)  # next shard downloads while this one converts
        t1 = time.time()
        stats = {"err": 0.0, "n": 0, "experts": 0, "written": 0}
        writes: queue.Queue = queue.Queue(maxsize=24)
        w = threading.Thread(target=writer, args=(writes, stats))
        w.start()
        convert(got[fn], fn, writes, stats)
        writes.put(None)
        w.join()
        size = os.path.getsize(got[fn])
        if not os.environ.get("FETCH_KEEP"):
            os.remove(got[fn])  # the downloaded shard; its content now lives in the expert files and dense/
        t2 = time.time()
        entry = {"shard": fn, "gb": round(size / 1e9, 2), "wait_s": round(t1 - t0), "convert_s": round(t2 - t1),
                 "matrices": stats["n"], "carried": stats.get("carried", 0), "rel_err": round(stats["err"] / max(1, stats["n"]), 5),
                 "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        prog["done"].append(fn)
        prog["log"].append(entry)
        json.dump(prog, open(prog_path + ".tmp", "w"), indent=1)
        os.replace(prog_path + ".tmp", prog_path)
        print(json.dumps(entry), flush=True)
    print("finished" if len(prog["done"]) == len(shards) else "stopped", flush=True)


if __name__ == "__main__":
    main()
