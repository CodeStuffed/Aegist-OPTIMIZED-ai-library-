"""Build the dense half of the Kimi-K2.6 deep brain: aegist-kimi.nx.

fetch_k2.py leaves the non-expert tensors byte-exact in dense/ (one safetensors file per source shard)
and the experts as per-layer int4 files. This writes the dense tensors to one .nx file:

    int8 per output channel   q_a_proj, q_b_proj, kv_a_proj_with_mqa, o_proj, shared expert, layer 0's MLP, head
    kept as published         embedding, norms, kv_b_proj (bf16: folded into attention at run time),
                              router weight and its correction bias

Names: blocks.<i>.<rest> for "model.layers.<i>.<rest>", plus embed.weight, norm.weight, lm_head.*.
The header's config carries the MLA / MoE shape and points at the expert files of manifest.json.

    python -m aegist.convert_k2 [--out D:/aegist/brains/kimi/aegist-kimi.nx]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from safetensors import safe_open

from .convert_max import Dense, q8
from .fetch_k2 import DENSE, EXPERT_BYTES, GROUP, HOME, REPO, REV

OUT = "D:/aegist/brains/kimi/aegist-kimi.nx"
NOTICE = ("Aegist's deep-think core is derived from Kimi-K2.6 by Moonshot AI, licensed under a Modified MIT "
          "License (architecture DeepSeek-V3-like, MIT, DeepSeek). Changes: re-implemented runtime, experts "
          "kept in their published int4 form, 8-bit dense linears, text only, Aegist persona.")
INT8 = ("self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.o_proj.weight", "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
        "mlp.shared_experts.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def rename(src: str) -> str | None:
    if src.startswith("lm_head."):
        return src
    if not src.startswith("model."):
        return None
    rest = src[len("model."):]
    if rest.startswith("layers."):
        _, i, sub = rest.split(".", 2)
        return f"blocks.{i}.{sub}"
    return {"embed_tokens.weight": "embed.weight", "norm.weight": "norm.weight"}.get(rest)


def is_int8(new: str) -> bool:
    return new == "lm_head.weight" or new.endswith(INT8)


def spec(dense: Dense):
    from .nx import Lazy

    out, memo = {}, {}
    for src in sorted(dense.where):
        new = rename(src)
        if new is None:
            continue
        with safe_open(dense.where[src], framework="pt") as f:
            sl = f.get_slice(src)
            shape, dt = list(sl.get_shape()), sl.get_dtype()
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
            # the router's correction bias is read as fp32 by the kernel, whatever it was published in
            f32 = dt in ("F32", "float32") or new.endswith("e_score_correction_bias")
            dtype = torch.float32 if f32 else torch.bfloat16
            out[new] = Lazy(dtype, shape, (lambda src=src, dtype=dtype: dense.get(src).to(dtype).contiguous()))
    return out


def config() -> dict:
    raw = json.load(open(f"{HOME}/config.json", encoding="utf-8"))
    c = raw["text_config"]
    man = json.load(open(f"{HOME}/manifest.json", encoding="utf-8"))
    return {"arch": "kimi-mla", "vocab_size": c["vocab_size"], "d_model": c["hidden_size"],
            "n_layers": c["num_hidden_layers"], "first_dense": c["first_k_dense_replace"],
            "dense_hidden": c["intermediate_size"], "n_heads": c["num_attention_heads"],
            "q_lora": c["q_lora_rank"], "kv_lora": c["kv_lora_rank"], "qk_nope": c["qk_nope_head_dim"],
            "qk_rope": c["qk_rope_head_dim"], "v_head": c["v_head_dim"], "rope_theta": c["rope_theta"],
            "rope_scaling": c["rope_scaling"], "max_positions": c["max_position_embeddings"],
            "n_experts": c["n_routed_experts"], "experts_active": c["num_experts_per_tok"],
            "n_shared": c["n_shared_experts"], "expert_hidden": c["moe_intermediate_size"],
            "routed_scale": c["routed_scaling_factor"], "scoring": c["scoring_func"],
            "norm_topk": c["norm_topk_prob"], "norm_eps": c["rms_norm_eps"],
            "expert_format": "q4g32s", "expert_group": GROUP, "expert_bytes": EXPERT_BYTES,
            "expert_files": man["files"], "bos_id": raw["bos_token_id"], "eos_id": raw["eos_token_id"],
            "chat": "kimi"}


def main():
    from . import persona
    from .nx import save

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dense = Dense(DENSE)
    s = spec(dense)
    cfg = config()
    need = ["embed.weight", "norm.weight", "lm_head.qweight"] + [f"blocks.{i}.self_attn.o_proj.qweight" for i in range(cfg["n_layers"])]
    missing = [n for n in need if n not in s]
    if missing:
        raise SystemExit(f"dense tensors missing (download not finished?): {missing[:5]} ...")
    if not os.path.exists(f"{HOME}/tokenizer.json"):
        raise SystemExit("run python -m aegist.kimi_tokenizer first")
    lic = open(f"{HOME}/LICENSE", encoding="utf-8").read() + "\n\n" + open(f"{HOME}/THIRD_PARTY_NOTICES.md", encoding="utf-8").read()
    generation = {"temperature": 0.6, "top_p": 0.95, "top_k": 0, "think_temperature": 1.0, "think_top_p": 0.95,
                  "open_think": True, "eos": cfg["eos_id"]}  # Moonshot's advice: 1.0 thinking, 0.6 instant, top-p 0.95
    lineage = [{"event": "born", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "base": f"{REPO}@{REV}", "arch": "kimi-mla",
                "quant": {"experts": "int4 group 32 as published (QAT)", "linears": "int8"}}]
    t = time.time()
    save(a.out, s, cfg, open(f"{HOME}/tokenizer.json", encoding="utf-8").read(), notice=NOTICE,
         license_text=NOTICE + "\n\n" + lic,
         persona={"name": "Aegist", "default_mode": "deep", "modes": list(persona.MODES)},
         generation=generation, lineage=lineage,
         extra={"name": "Aegist-kimi-1T", "version": 1, "quant": "int8", "experts": "q4g32s", "dtype": "bf16"})
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e9:.2f} GB in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
