"""Build the dense half of the 2.4T deep brain: aegist-max.nx.

fetch_max.py leaves the checkpoint's non-expert tensors byte-exact in DENSE (one safetensors file
per source shard) and the experts as 3-bit per-layer files. This writes the dense tensors to one
.nx file in a single streaming pass, in the names the 35B hybrid brain uses:

    big linears        int8 per output channel (attention, DeltaNet projections, shared expert, output head, mtp.fc)
    everything else    bf16 (embedding, norms, router, DeltaNet gates and convolution)

The multi-token-prediction layer is kept (as mtp.*) for speculative decoding. The header's config
points at the expert files listed in manifest.json.

    python -m aegist.convert_max [--out D:/aegist/brains/max/aegist-max.nx]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import OrderedDict

import torch
from safetensors import safe_open

from .convert_moe import MIXER, adapt, config_from_hf, rename
from .fetch_max import DENSE, EXPERT_BYTES, GROUP, HOME, REPO

OUT = "D:/aegist/brains/max/aegist-max.nx"
NOTICE = ("Aegist's deep-think core is derived from Qwen3.8-2.4T-A95B by the Qwen team (Alibaba Cloud), "
          "licensed under the Qwen3.8-Max License. Changes: re-implemented runtime, 3-bit experts, "
          "8-bit linears, text only, Aegist persona.")
INT8 = ("mixer.q.weight", "mixer.k.weight", "mixer.v.weight", "mixer.o.weight", "mixer.in_qkv.weight",
        "mixer.in_z.weight", "mixer.out.weight", "moe.shared.gate.weight", "moe.shared.up.weight",
        "moe.shared.down.weight")


def rename_max(name: str) -> str | None:
    """rename() plus the MTP layer: mtp.layers.0.X -> mtp.block.<as for a block>."""
    if name.startswith("mtp."):
        rest = name[4:]
        if rest.startswith("layers."):
            _, i, sub = rest.split(".", 2)
            return f"mtp.block{i}.{MIXER[sub]}" if sub in MIXER else None
        return {"fc.weight": "mtp.fc.weight", "norm.weight": "mtp.norm.weight",
                "pre_fc_norm_embedding.weight": "mtp.norm_embed.weight",
                "pre_fc_norm_hidden.weight": "mtp.norm_hidden.weight"}.get(rest)
    return rename(name)


def is_int8(new: str) -> bool:
    return new in ("lm_head.weight", "mtp.fc.weight") or new.endswith(INT8)


def q8(w: torch.Tensor, rows: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row absmax int8, as model.QLinear, in row chunks so a 248k-row head stays small in RAM."""
    q = torch.empty(w.shape, dtype=torch.int8)
    s = torch.empty((w.shape[0], 1), dtype=torch.bfloat16)
    for r in range(0, w.shape[0], rows):
        c = w[r : r + rows].float()
        sc = c.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127.0
        q[r : r + rows] = torch.round(c / sc).to(torch.int8)
        s[r : r + rows] = sc.to(torch.bfloat16)
    return q, s


class Dense:
    """Tensor name -> tensor over the DENSE files, with few files open at once."""

    def __init__(self, path: str):
        self.where: dict[str, str] = {}
        for fn in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(fn, framework="pt") as f:
                for k in f.keys():
                    self.where[k] = fn
        self._open: OrderedDict[str, object] = OrderedDict()

    def get(self, name: str) -> torch.Tensor:
        fn = self.where[name]
        if fn in self._open:
            self._open.move_to_end(fn)
        else:
            while len(self._open) >= 4:
                self._open.popitem(last=False)
            self._open[fn] = safe_open(fn, framework="pt")
        t = self._open[fn].get_tensor(name)
        si = name + "_scale_inv"
        if t.dtype == torch.float8_e4m3fn and si in self.where:  # in case a dense matrix is FP8 too
            s = self.get(si).float()
            t = (t.float() * s.repeat_interleave(128, 0)[: t.shape[0]].repeat_interleave(128, 1)[:, : t.shape[1]])
        return t


def spec(dense: Dense):
    from .nx import Lazy

    out, memo = {}, {}
    for src in sorted(dense.where):
        if src.endswith("_scale_inv"):
            continue
        new = rename_max(src)
        if new is None:
            continue
        with safe_open(dense.where[src], framework="pt") as f:
            shape = list(f.get_slice(src).get_shape())
        if new.endswith(("mixer.conv",)):
            shape = [shape[0], shape[-1]]
        if is_int8(new):
            base = new[: -len(".weight")]

            def part(i, src=src, base=base):
                def make():
                    if base not in memo:
                        memo.clear()
                        memo[base] = q8(dense.get(src))
                    return memo[base][i]
                return make
            out[base + ".qweight"] = Lazy(torch.int8, shape, part(0))
            out[base + ".scale"] = Lazy(torch.bfloat16, [shape[0], 1], part(1))
        else:
            out[new] = Lazy(torch.bfloat16, shape,
                            (lambda src=src, new=new: adapt(new, dense.get(src)).to(torch.bfloat16).contiguous()))
    return out


def main():
    from . import persona
    from .nx import save

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    raw = json.load(open(f"{HOME}/config.json", encoding="utf-8"))
    man = json.load(open(f"{HOME}/manifest.json", encoding="utf-8"))
    gen = json.load(open(f"{HOME}/generation_config.json", encoding="utf-8"))
    c = raw.get("text_config", raw)
    cfg = config_from_hf(raw).to_dict()
    cfg.update(arch="max-moe", expert_format="q3g64", expert_group=GROUP, expert_bytes=EXPERT_BYTES,
               expert_files=man["files"], mtp_layers=c.get("mtp_num_hidden_layers", 1),
               out_gate=c.get("output_gate_type", "swish"), eos_id=248044)
    dense = Dense(DENSE)
    s = spec(dense)
    missing = [n for n in ("embed.weight", "norm.weight", "lm_head.qweight") if n not in s]
    if missing:
        raise SystemExit(f"dense tensors missing (download not finished?): {missing}")
    lic = open(f"{HOME}/LICENSE", encoding="utf-8").read()
    generation = {"temperature": float(gen.get("temperature", 0.6)), "top_p": float(gen.get("top_p", 0.95)),
                  "top_k": int(gen.get("top_k", 20)), "think_temperature": float(gen.get("temperature", 0.6)),
                  "think_top_p": float(gen.get("top_p", 0.95)), "open_think": True, "eos": gen.get("eos_token_id")}
    rev = open(f"{HOME}/REVISION", encoding="utf-8").read().strip()
    lineage = [{"event": "born", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "base": f"{REPO}@{rev}",
                "arch": "max-moe", "quant": {"experts": "q3 group 64, f16 scales", "linears": "int8"}}]
    t = time.time()
    save(a.out, s, cfg, open(f"{HOME}/tokenizer.json", encoding="utf-8").read(), notice=NOTICE,
         license_text=NOTICE + "\n\n" + lic, persona={"name": "Aegist", "default_mode": "deep",
                                                     "modes": list(persona.MODES)},
         generation=generation, lineage=lineage,
         extra={"name": "Aegist-max-2.4T", "version": 1, "quant": "int8", "experts": "q3g64", "dtype": "bf16"})
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e9:.2f} GB in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
