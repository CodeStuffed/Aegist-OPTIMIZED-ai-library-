"""The .nx file format: one file holds an entire Aegist.

    bytes 0..8    magic  b"AEGISTNX"
    bytes 8..12   u32 LE format version (1)
    bytes 12..20  u64 LE header length H
    bytes 20..20+H  UTF-8 JSON header
    padding to a 64-byte boundary, then the data section

The header lists every tensor (dtype, shape, byte offset into the data
section, byte length, group) and every blob (tokenizer, license notice).
Every tensor starts on a 64-byte boundary, so a reader can memory-map the
file and use tensors in place; a core is one contiguous slice of its layer's
core tensor, so cores can be paged in individually without reading the rest.

Groups: "base" (always-on reading/thinking/speaking), "engine" (core routers),
"cores" (switchable cores). The header also carries the config, persona
defaults, generation defaults, lineage (what each learning session changed),
and a SHA-256 per group so tampering or corruption is detectable.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import time

import torch

MAGIC = b"AEGISTNX"
LEGACY_MAGIC = b"AUGISTNX"  # files written before the rename to Aegist
VERSION = 1
ALIGN = 64
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32, "i8": torch.int8, "i64": torch.int64,
          "u8": torch.uint8, "f8e4m3": torch.float8_e4m3fn}
DTYPE_NAMES = {v: k for k, v in DTYPES.items()}


class Lazy:
    """A tensor produced only when the writer reaches it, so a brain larger than RAM can be
    written in one pass. `make()` must return a tensor of exactly `dtype` and `shape`, or (for a
    tensor larger than RAM) an iterator of tensors of that dtype whose rows, in order, make it up."""

    def __init__(self, dtype: torch.dtype, shape, make):
        self.dtype, self.shape, self.make = dtype, tuple(shape), make

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def element_size(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()


def _group(name: str) -> str:
    if ".cores." in name:
        return "cores"
    if ".engine." in name:
        return "engine"
    return "base"


def _pad(n: int) -> int:
    return (-n) % ALIGN


def save(path: str, state: dict[str, torch.Tensor], config: dict, tokenizer_json: str, *, notice: str = "",
         license_text: str = "", persona: dict | None = None, generation: dict | None = None,
         lineage: list | None = None, extra: dict | None = None) -> dict:
    """Write atomically (temp file + rename) so a crash never leaves a half-written brain."""
    tensors, blobs, offset = {}, {}, 0
    order = sorted(state, key=lambda n: (_group(n), n))
    for n in order:
        t = state[n]
        if t.dtype not in DTYPE_NAMES:
            raise ValueError(f"{n}: unsupported dtype {t.dtype}")
        nbytes = t.numel() * t.element_size()
        tensors[n] = {"dtype": DTYPE_NAMES[t.dtype], "shape": list(t.shape), "offset": offset, "nbytes": nbytes,
                      "group": _group(n)}
        offset += nbytes + _pad(nbytes)
    for name, text in (("tokenizer.json", tokenizer_json), ("LICENSE", license_text)):
        data = text.encode("utf-8")
        blobs[name] = {"offset": offset, "nbytes": len(data)}
        offset += len(data) + _pad(len(data))

    hashes = {g: hashlib.sha256() for g in ("base", "engine", "cores")}
    header = {"name": "Aegist", "format": "nx", "format_version": VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "config": config, "notice": notice, "persona": persona or {}, "generation": generation or {},
              "lineage": lineage or [], "tensors": tensors, "blobs": blobs, **(extra or {})}

    def header_bytes(h):
        return json.dumps(h, separators=(",", ":")).encode("utf-8")

    # hashes go into the header, but the header size must be known before writing data:
    # reserve space for them with placeholders of the final length.
    header["sha256"] = {g: "0" * 64 for g in hashes}
    hb = header_bytes(header)
    data_start = 20 + len(hb) + _pad(20 + len(hb))
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(b"\0" * data_start)
        for n in order:
            t = state[n]
            if isinstance(t, Lazy):
                t = t.make()
                if not isinstance(t, torch.Tensor):  # written piece by piece
                    done = 0
                    for piece in t:
                        if DTYPE_NAMES[piece.dtype] != tensors[n]["dtype"]:
                            raise ValueError(f"{n}: piece is {piece.dtype}, header says {tensors[n]['dtype']}")
                        raw = piece.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()
                        hashes[tensors[n]["group"]].update(raw)
                        f.write(raw)
                        done += len(raw)
                        del piece, raw
                    if done != tensors[n]["nbytes"]:
                        raise ValueError(f"{n}: pieces made {done} bytes, header says {tensors[n]['nbytes']}")
                    f.write(b"\0" * _pad(done))
                    continue
                want = (DTYPE_NAMES[t.dtype], list(t.shape))
                if want != (tensors[n]["dtype"], tensors[n]["shape"]):
                    raise ValueError(f"{n}: produced {want}, header says {tensors[n]['dtype']} {tensors[n]['shape']}")
            t = t.detach().contiguous().cpu()
            raw = t.reshape(-1).view(torch.uint8).numpy() if t.numel() else b""
            hashes[tensors[n]["group"]].update(raw)
            f.write(raw)
            f.write(b"\0" * _pad(len(raw)))
            del t, raw
        for name, text in (("tokenizer.json", tokenizer_json), ("LICENSE", license_text)):
            data = text.encode("utf-8")
            f.write(data)
            f.write(b"\0" * _pad(len(data)))
        header["sha256"] = {g: h.hexdigest() for g, h in hashes.items()}
        hb2 = header_bytes(header)
        assert len(hb2) == len(hb)
        f.seek(0)
        f.write(MAGIC + struct.pack("<IQ", VERSION, len(hb2)) + hb2)
        f.write(b"\0" * (data_start - 20 - len(hb2)))
    os.replace(tmp, path)
    return header


class NxFile:
    """Memory-mapped reader. Tensors are views into the file until moved."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        head = self._f.read(20)
        if head[:8] not in (MAGIC, LEGACY_MAGIC):
            raise ValueError(f"{path} is not an .nx file")
        version, hlen = struct.unpack("<IQ", head[8:20])
        if version > VERSION:
            raise ValueError(f"{path} uses .nx format v{version}; this Aegist reads up to v{VERSION}")
        self.header = json.loads(self._f.read(hlen).decode("utf-8"))
        self.data_start = 20 + hlen + _pad(20 + hlen)
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self):
        try:
            self._mm.close()
        except BufferError:
            pass  # tensors still view the map; it is released when they are freed
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    @property
    def config(self) -> dict:
        return self.header["config"]

    def names(self, group: str | None = None) -> list[str]:
        return [n for n, m in self.header["tensors"].items() if group is None or m["group"] == group]

    def tensor(self, name: str) -> torch.Tensor:
        m = self.header["tensors"][name]
        dt = DTYPES[m["dtype"]]
        n = m["nbytes"] // torch.tensor([], dtype=dt).element_size()
        if n == 0:
            return torch.empty(m["shape"], dtype=dt)
        import warnings
        fp8 = dt == torch.float8_e4m3fn  # read as bytes, then reinterpret
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # read-only mmap; we copy before writing
            t = torch.frombuffer(self._mm, dtype=torch.uint8 if fp8 else dt, count=n, offset=self.data_start + m["offset"])
        return (t.view(dt) if fp8 else t).view(m["shape"])

    def core(self, layer: int, core: int) -> dict[str, torch.Tensor]:
        """One core's weights, read without touching any other core."""
        return {k: self.tensor(f"blocks.{layer}.cores.{k}")[core] for k in ("gate", "up", "down")}

    def blob(self, name: str) -> bytes:
        m = self.header["blobs"][name]
        s = self.data_start + m["offset"]
        return bytes(self._mm[s : s + m["nbytes"]])

    def verify(self) -> dict[str, bool]:
        hashes = {g: hashlib.sha256() for g in ("base", "engine", "cores")}
        order = sorted(self.header["tensors"], key=lambda n: (self.header["tensors"][n]["group"], n))
        for n in order:
            m = self.header["tensors"][n]
            s = self.data_start + m["offset"]
            hashes[m["group"]].update(self._mm[s : s + m["nbytes"]])
        return {g: hashes[g].hexdigest() == self.header["sha256"][g] for g in hashes}


# --- whole models -----------------------------------------------------------

def save_model(path: str, model, tokenizer_json: str, *, notice: str = "", license_text: str = "",
               persona: dict | None = None, generation: dict | None = None, lineage: list | None = None,
               extra: dict | None = None) -> dict:
    """Write an AegistLM (resident or paged cores, int8 or full-precision base) to .nx."""
    from .model import QLinear

    if getattr(model.cfg, "arch", None) == "hybrid-moe":
        from .hybrid import hybrid_state
        info = {"quant": "int8" if any(isinstance(m, QLinear) for m in model.modules()) else "none",
                "dtype": DTYPE_NAMES[model.embed.weight.dtype],
                "experts": f"q4a-g{model.cfg.expert_group}", "params": model.param_report(), **(extra or {})}
        return save(path, hybrid_state(model), model.cfg.to_dict(), tokenizer_json, notice=notice,
                    license_text=license_text, persona=persona, generation=generation, lineage=lineage, extra=info)
    state = dict(model.state_dict())
    for i, b in enumerate(model.blocks):
        if b.cores.paged:  # paged cores are not in state_dict; their masters are the real weights
            for k, t in zip(("gate", "up", "down"), b.cores.masters()):
                state[f"blocks.{i}.cores.{k}"] = t
    silent = [not bool(b.cores.masters()[2].detach().any()) for b in model.blocks]
    info = {"quant": "int8" if any(isinstance(m, QLinear) for m in model.modules()) else "none",
            "dtype": DTYPE_NAMES[model.embed.weight.dtype], "cores_silent": silent,
            "params": model.param_report(), **(extra or {})}
    return save(path, state, model.cfg.to_dict(), tokenizer_json, notice=notice, license_text=license_text,
                persona=persona, generation=generation, lineage=lineage, extra=info)


def load_model(path: str, device="cuda", slots: int = 16, cores_in_ram: bool = False, train: bool = False):
    """Load an .nx brain. Returns (model, tokenizer, NxFile).

    Inference (train=False): the always-on base and the core engine go to `device`;
    the cores stay in the file (memory-mapped, read from disk on first use) or, with
    `cores_in_ram`, in pinned RAM; `slots` cores per layer are cached on `device`.
    The NxFile must stay open while the model is in use.
    Learning time (train=True): everything is loaded onto `device` as Parameters.
    """
    from .model import AegistConfig, AegistLM
    from .tokenizer import AegistTokenizer

    nx = NxFile(path)
    h = nx.header
    if h["config"].get("arch") == "hybrid-moe":
        from .hybrid import load_hybrid
        return load_hybrid(nx, device=device, slots=slots, train=train)
    cfg = AegistConfig.from_dict(h["config"])
    dtype = DTYPES[h.get("dtype", "bf16")]
    with torch.device("meta"):
        model = AegistLM(cfg).to(dtype)
        if h.get("quant") == "int8":
            model.quantize_base_int8()
    silent = h.get("cores_silent") or [None] * cfg.n_layers
    if not train:
        for i, b in enumerate(model.blocks):
            masters = tuple(nx.tensor(f"blocks.{i}.cores.{k}") for k in ("gate", "up", "down"))
            if cores_in_ram:
                masters = tuple(m.clone() for m in masters)
            b.cores.page(slots, "meta", masters=masters, pin=False, silent=silent[i])
    model = model.to_empty(device=device)
    sd = model.state_dict()
    missing = [n for n in sd if n not in h["tensors"]]
    if missing:
        nx.close()
        raise ValueError(f"{path} is missing tensors: {missing[:5]}")
    with torch.no_grad():
        for n, t in sd.items():
            t.copy_(nx.tensor(n))
    for b in model.blocks:
        if b.cores.paged:
            b.cores.reset_slots()
            if cores_in_ram and torch.device(device).type == "cuda":
                b.cores.m_gate, b.cores.m_up, b.cores.m_down = (m.pin_memory() for m in b.cores.masters())
    model._nx = nx  # keeps the memory map alive for paged cores
    tok = AegistTokenizer.from_json_str(nx.blob("tokenizer.json").decode("utf-8"))
    model.train(train)
    if train:
        for p in model.base_parameters():
            p.requires_grad_(False)
    return model, tok, nx
