"""Build Aegist's second language core from Qwen3.8-Flash-Next-FP8 (Qwen Community License 1.0).

Nothing is re-quantised: every weight goes into the .nx exactly as released. The experts stay FP8
(with their bf16 128x128 block scales), everything else stays bf16, and the 51 GB hashed n-gram
table stays FP8 with its one scale. What changes is the packing:

    experts            one row per expert (gate, up, down and their scales together), so the engine
                       pages an expert with a single read
    n-gram table       its 128 released pieces joined into one file-backed table (rows are read on demand)
    MTP layer, vision  kept byte-exact under `extra.` for later (multi-token prediction, image input)

`verify_against_source` then compares every byte of the brain with the released files.
"""
from __future__ import annotations

import json
import os
import re
import time

import torch
from safetensors import safe_open

from . import hc as HC
from .convert_moe import Shards, adapt, config_from_hf, rename
from .hybrid import ARCH, HybridLM, expert_bytes, pack_experts_fp8
from .model import CoreEngine

BASE_NOTICE = ("Aegist's language core is derived from Qwen3.8-Flash-Next by Qwen (Alibaba Cloud), "
               "Copyright (c) 2026 Qwen, licensed under the Qwen Community License 1.0 (full text in this brain's "
               "LICENSE). Changes: re-implemented architecture and runtime, weights re-packed for paging "
               "(released FP8/BF16 values unchanged), added core engine and switchable cores, Aegist persona, "
               "confidence and learning.")

TABLE_PIECE = re.compile(r"^(.*)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
EXPERT = re.compile(r"^(.*)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight(_scale_inv)?$")
ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64,
             "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "I8": torch.int8}


class Source(Shards):
    """The released checkpoint: names, shapes and dtypes from the headers, tensors on demand."""

    def __init__(self, path: str):
        super().__init__(path)
        self.meta: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
        for fn in sorted(set(self.where.values())):
            with safe_open(os.path.join(path, fn), framework="pt") as f:
                for k in f.keys():
                    if self.where.get(k) == fn:
                        s = f.get_slice(k)
                        self.meta[k] = (ST_DTYPES[s.get_dtype()], tuple(s.get_shape()))

    def close(self):
        self._open.clear()


def _layer_prefix(src: Source) -> str:
    for n in src.where:
        m = EXPERT.match(n)
        if m:
            return m.group(1)
    raise ValueError("no experts in the checkpoint")


def table_pieces(src: Source, layer: int) -> list[str]:
    """The released pieces of block `layer`'s n-gram table, in row order."""
    found = {}
    for n in src.where:
        m = TABLE_PIECE.match(n)
        if m and m.group(1).endswith(f"layers.{layer}"):
            found[int(m.group(2))] = n
    return [found[i] for i in range(len(found))]


def expert_names(prefix: str, layer: int, e: int) -> dict[str, str]:
    b = f"{prefix}.layers.{layer}.mlp.experts.{e}"
    return {k: f"{b}.{p}_proj.weight{s}" for k, p, s in
            (("g", "gate", ""), ("u", "up", ""), ("d", "down", ""),
             ("gs", "gate", "_scale_inv"), ("us", "up", "_scale_inv"), ("ds", "down", "_scale_inv"))}


def expert_parts(src: Source, prefix: str, layer: int, e: int):
    t = {k: src.get(n) for k, n in expert_names(prefix, layer, e).items()}
    for k in ("g", "u", "d"):
        assert t[k].dtype == HC.FP8, (layer, e, k, t[k].dtype)
    for k in ("gs", "us", "ds"):
        assert t[k].dtype == torch.bfloat16, (layer, e, k, t[k].dtype)
    return (torch.cat([t["g"], t["u"]]), t["d"], torch.cat([t["gs"], t["us"]]), t["ds"])


def brain_spec(src: Source, cfg, seed: int, log=print):
    from .nx import Lazy

    have = src.aegist_names()
    prefix = _layer_prefix(src)
    with torch.device("meta"):
        model = HybridLM(cfg).to(torch.bfloat16)
    torch.manual_seed(seed)
    engines = [{k: v.to(torch.bfloat16) for k, v in CoreEngine(cfg).state_dict().items()} for _ in range(cfg.n_layers)]
    used: set[str] = set()
    spec = {}

    def plain(name, orig):
        def make():
            t = adapt(name, src.get(orig))
            if t.dtype != torch.bfloat16:
                raise ValueError(f"{orig}: expected bf16, found {t.dtype}")
            return t.contiguous()
        return make

    def experts(i):
        def make():
            t = time.time()
            rows = pack_experts_fp8([expert_parts(src, prefix, i, e) for e in range(cfg.n_experts)], cfg)
            log(f"  block {i}: {cfg.n_experts} FP8 experts packed ({time.time() - t:.1f}s)")
            return rows
        return make

    def table(names):
        def make():
            for n in names:
                piece = src.get(n)
                assert piece.dtype == HC.FP8, (n, piece.dtype)
                yield piece
        return make

    for name, t in model.state_dict().items():
        if ".moe.experts." in name:
            continue
        if ".engine." in name:
            i, k = int(name.split(".")[1]), name.split(".engine.")[1]
            spec[name] = Lazy(torch.bfloat16, t.shape, lambda i=i, k=k: engines[i][k])
        elif ".cores." in name:
            spec[name] = Lazy(t.dtype, t.shape, (lambda shape=tuple(t.shape), dt=t.dtype: torch.zeros(shape, dtype=dt)))
        elif name.endswith(".ple.table.weight"):
            i = int(name.split(".")[1])
            names = table_pieces(src, i)
            rows = sum(src.meta[n][1][0] for n in names)
            if not names or rows != t.shape[0] or any(src.meta[n][1][1] != t.shape[1] for n in names):
                raise ValueError(f"block {i}: n-gram table pieces make {rows} rows, expected {tuple(t.shape)}")
            spec[name] = Lazy(HC.FP8, t.shape, table(names))
            scale = names[0].rsplit(".shard_", 1)[0] + ".weight_scale"
            spec[f"blocks.{i}.ple.table_scale"] = Lazy(torch.bfloat16, (1,), lambda s=scale: src.get(s).reshape(1))
            base = names[0].split(".ngram_embedding.")[0]
            ple = model.blocks[i].ple
            for suffix, ours in ((".layer_multipliers", ple.mult), (".ngram_heads_vocab_sizes", ple.sizes),
                                 (".ngram_heads_offsets", ple.offsets)):
                if base + suffix in src.where:  # hash constants are recomputed at load; they must agree
                    if src.get(base + suffix).tolist() != ours.tolist():
                        raise ValueError(f"{base + suffix}: the released n-gram hash constants differ from Aegist's")
                    used.add(base + suffix)
            used.update(names)
            used.add(scale)
        else:
            if name not in have:
                raise ValueError(f"checkpoint has no tensor for {name}")
            spec[name] = Lazy(torch.bfloat16, t.shape, plain(name, have[name]))
            used.add(have[name])
    for i in range(cfg.n_layers):
        spec[f"blocks.{i}.moe.experts.fp8"] = Lazy(torch.uint8, (cfg.n_experts, expert_bytes(cfg)), experts(i))
        for e in range(cfg.n_experts):
            used.update(expert_names(prefix, i, e).values())
    for n, (dt, shape) in src.meta.items():  # MTP, vision and anything else: kept as released
        if n not in used:
            spec["extra." + n] = Lazy(dt, shape, lambda n=n: src.get(n))
    return spec


def build_next(src_dir: str, out: str, *, core_grid=(8, 64), core_rank: int = 64, cores_active: int = 2,
               core_rows_max: int = 85, seed: int = 0, log=print) -> dict:
    from . import persona
    from .nx import save

    raw = json.load(open(os.path.join(src_dir, "config.json"), encoding="utf-8"))
    cfg = config_from_hf(raw, core_grid=tuple(core_grid), core_rank=core_rank, cores_active=cores_active,
                         core_rows_max=core_rows_max, core_seed=seed, expert_format="fp8")
    q = raw.get("quantization_config") or {}
    if q and (q.get("quant_method") != "fp8" or list(q.get("weight_block_size", [])) != [HC.BLOCK, HC.BLOCK]):
        raise ValueError(f"unexpected quantization_config: {q}")
    gen = json.load(open(os.path.join(src_dir, "generation_config.json"), encoding="utf-8"))
    src = Source(src_dir)
    spec = brain_spec(src, cfg, seed, log)
    with torch.device("meta"):
        report = HybridLM(cfg).to(torch.bfloat16).param_report()
    tok_json = open(os.path.join(src_dir, "tokenizer.json"), encoding="utf-8").read()
    lic = open(os.path.join(src_dir, "LICENSE"), encoding="utf-8").read()
    lineage = [{"event": "born", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "base": f"{os.path.basename(os.path.normpath(src_dir))} (Qwen Community License 1.0)", "arch": ARCH,
                "quant": {"experts": "fp8 e4m3, 128x128 block scales (as released)", "linears": "bf16 (as released)",
                          "ngram_table": "fp8 e4m3, one scale (as released)"},
                "core_grid": list(cfg.core_grid), "core_rank": core_rank, "cores_active": cores_active,
                "core_rows_max": core_rows_max, "seed": seed}]
    generation = {"temperature": 0.7, "top_p": 0.8, "top_k": int(gen.get("top_k", 20)),
                  "think_temperature": float(gen.get("temperature", 1.0)), "think_top_p": float(gen.get("top_p", 0.95)),
                  "presence_penalty": 1.5, "open_think": True, "eos": gen.get("eos_token_id")}
    t = time.time()
    header = save(out, spec, cfg.to_dict(), tok_json, notice=BASE_NOTICE, license_text=BASE_NOTICE + "\n\n" + lic,
                  persona={"name": "Aegist", "default_mode": "normal", "modes": list(persona.MODES)},
                  generation=generation, lineage=lineage,
                  extra={"name": "reid1", "version": 1, "quant": "none", "experts": "fp8-b128", "dtype": "bf16", "params": report})
    src.close()
    log(f"wrote {out}: {os.path.getsize(out) / 1e9:.2f} GB in {time.time() - t:.0f}s")
    return header


def verify_against_source(out: str, src_dir: str, log=print) -> dict:
    """Every tensor of the brain against the released bytes. Returns counts; raises on any difference."""
    from .nx import NxFile

    src = Source(src_dir)
    prefix = _layer_prefix(src)
    have = src.aegist_names()
    same = lambda a, b: a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    n_plain = n_extra = n_exp = 0
    with NxFile(out) as nx:
        cfg_d = nx.header["config"]
        from .hybrid import HybridConfig
        cfg = HybridConfig.from_dict(cfg_d)
        for name, m in nx.header["tensors"].items():
            if ".engine." in name or ".cores." in name:
                continue
            got = nx.tensor(name)
            if name.startswith("extra."):
                if not same(got, src.get(name[len("extra."):])):
                    raise ValueError(f"{name} differs from the release")
                n_extra += 1
            elif name.endswith(".moe.experts.fp8"):
                i = int(name.split(".")[1])
                for e in range(cfg.n_experts):
                    want = pack_experts_fp8([expert_parts(src, prefix, i, e)], cfg)[0]
                    if not torch.equal(got[e], want):
                        raise ValueError(f"{name} expert {e} differs from the release")
                n_exp += cfg.n_experts
                log(f"  block {i}: {cfg.n_experts} experts identical")
            elif name.endswith(".ple.table.weight"):
                i, row = int(name.split(".")[1]), 0
                for pn in table_pieces(src, i):
                    piece = src.get(pn)
                    if not same(got[row: row + piece.shape[0]], piece):
                        raise ValueError(f"{name} rows from {pn} differ from the release")
                    row += piece.shape[0]
                assert row == got.shape[0]
                log(f"  block {i}: n-gram table identical ({got.numel() / 1e9:.1f} GB)")
            elif name.endswith(".ple.table_scale"):
                pn = table_pieces(src, int(name.split(".")[1]))[0].rsplit(".shard_", 1)[0] + ".weight_scale"
                if not same(got, src.get(pn).reshape(1)):
                    raise ValueError(f"{name} differs")
            else:
                if not same(got, adapt(name, src.get(have[name]))):
                    raise ValueError(f"{name} differs from {have[name]}")
                n_plain += 1
        hashes = nx.verify()
        if not all(hashes.values()):
            raise ValueError(f"checksums: {hashes}")
    src.close()
    return {"plain": n_plain, "experts": n_exp, "extra": n_extra, "checksums": "ok"}
