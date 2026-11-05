"""Build an Aegist model from Qwen3 base weights (Apache-2.0).

The base weights become Aegist's always-on reading/thinking/speaking path; the
core engine and cores are new, Aegist-only modules (output starts at zero).
"""
from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open

from .model import AegistConfig, AegistLM

BASE_NOTICE = ("Aegist's base weights are derived from Qwen3-4B by the Qwen team (Alibaba Cloud), "
               "licensed under the Apache License 2.0. Changes: re-implemented architecture, added "
               "core engine and switchable cores, Aegist persona and learning.")


def config_from_qwen(path: str, **overrides) -> AegistConfig:
    c = json.load(open(os.path.join(path, "config.json")))
    cfg = AegistConfig(vocab_size=c["vocab_size"], d_model=c["hidden_size"], n_layers=c["num_hidden_layers"],
                       n_heads=c["num_attention_heads"], n_kv_heads=c["num_key_value_heads"],
                       head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
                       ffn_hidden=c["intermediate_size"], rope_theta=float(c["rope_theta"]),
                       norm_eps=c["rms_norm_eps"], max_seq_len=c["max_position_embeddings"],
                       tie_embeddings=c.get("tie_word_embeddings", False))
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _rename(name: str) -> str | None:
    if name == "model.embed_tokens.weight":
        return "embed.weight"
    if name == "model.norm.weight":
        return "norm.weight"
    if name == "lm_head.weight":
        return "lm_head.weight"
    if not name.startswith("model.layers."):
        return None
    _, _, i, rest = name.split(".", 3)
    table = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn.q.weight", "self_attn.k_proj.weight": "attn.k.weight",
        "self_attn.v_proj.weight": "attn.v.weight", "self_attn.o_proj.weight": "attn.o.weight",
        "self_attn.q_norm.weight": "attn.q_norm.weight", "self_attn.k_norm.weight": "attn.k_norm.weight",
        "mlp.gate_proj.weight": "ffn.gate.weight", "mlp.up_proj.weight": "ffn.up.weight",
        "mlp.down_proj.weight": "ffn.down.weight",
    }
    return f"blocks.{i}.{table[rest]}" if rest in table else None


def iter_qwen_tensors(path: str):
    idx = os.path.join(path, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    for f in files:
        with safe_open(os.path.join(path, f), framework="pt") as sf:
            for name in sf.keys():
                yield name, sf.get_tensor(name)


def load_qwen3(path: str, device="cpu", dtype=torch.bfloat16, n_layers: int | None = None, seed: int = 0,
               **overrides) -> AegistLM:
    cfg = config_from_qwen(path, **overrides)
    if n_layers is not None:
        cfg.n_layers = n_layers
    torch.manual_seed(seed)  # core engine / core init is reproducible
    with torch.device("meta"):
        model = AegistLM(cfg).to(dtype)
    model = model.to_empty(device=device)  # allocate directly in the working dtype
    for b in model.blocks:  # fresh Aegist-only modules
        b.engine.reset_parameters()
        b.cores.reset_parameters()
    sd = model.state_dict()
    loaded = set()
    for name, t in iter_qwen_tensors(path):
        new = _rename(name)
        if new is None or new not in sd:
            continue  # layers beyond n_layers, or an untied head when tied
        sd[new].copy_(t.to(sd[new].dtype))
        loaded.add(new)
    base = {k for k in sd if not (".engine." in k or ".cores." in k)}
    missing = base - loaded
    if missing:
        raise ValueError(f"base weights missing from checkpoint: {sorted(missing)[:5]} ...")
    return model.to(dtype)
