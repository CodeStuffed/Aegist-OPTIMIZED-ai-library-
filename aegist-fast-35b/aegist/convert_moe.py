"""Build a hybrid-MoE Aegist brain from Qwen3.6-35B-A3B-style base weights (Apache-2.0).

The checkpoint is read straight from its safetensors shards and written to one .nx file in a
single streaming pass: each tensor is converted only when the writer reaches it, so the 67 GB
bf16 checkpoint never has to fit in RAM. Text weights only; the vision tower and the
multi-token-prediction layer are left out.

    experts            4-bit asymmetric, 64-value groups, least-squares refit (about 4.5 bits/weight)
    big linears        int8 per output channel (attention, DeltaNet projections, shared expert, output head)
    everything else    bf16 (embedding, norms, router, DeltaNet gates and convolution)
"""
from __future__ import annotations

import json
from collections import OrderedDict
import os
import time

import torch
from safetensors import safe_open

from .hybrid import ARCH, HybridConfig, HybridLM, expert_bytes, pack_experts, unpack_expert
from .model import CoreEngine, QLinear

BASE_NOTICE = ("Aegist's language core is derived from Qwen3.6-35B-A3B by the Qwen team (Alibaba Cloud), "
               "licensed under the Apache License 2.0. Changes: re-implemented architecture and runtime, "
               "4-bit/8-bit quantisation, text only, added core engine and switchable cores, Aegist persona, "
               "confidence and learning.")

MIXER = {
    "self_attn.q_proj.weight": "mixer.q.weight", "self_attn.k_proj.weight": "mixer.k.weight",
    "self_attn.v_proj.weight": "mixer.v.weight", "self_attn.o_proj.weight": "mixer.o.weight",
    "self_attn.q_norm.weight": "mixer.q_norm.weight", "self_attn.k_norm.weight": "mixer.k_norm.weight",
    "linear_attn.in_proj_qkv.weight": "mixer.in_qkv.weight", "linear_attn.in_proj_z.weight": "mixer.in_z.weight",
    "linear_attn.in_proj_b.weight": "mixer.in_b.weight", "linear_attn.in_proj_a.weight": "mixer.in_a.weight",
    "linear_attn.conv1d.weight": "mixer.conv", "linear_attn.dt_bias": "mixer.dt_bias",
    "linear_attn.A_log": "mixer.A_log", "linear_attn.norm.weight": "mixer.norm.weight",
    "linear_attn.out_proj.weight": "mixer.out.weight",
    "input_layernorm.weight": "norm1.weight", "post_attention_layernorm.weight": "norm2.weight",
    "mlp.gate.weight": "moe.router.weight",
    "mlp.experts.gate_up_proj": "moe.experts.gate_up", "mlp.experts.down_proj": "moe.experts.down",
    "mlp.shared_expert.gate_proj.weight": "moe.shared.gate.weight",
    "mlp.shared_expert.up_proj.weight": "moe.shared.up.weight",
    "mlp.shared_expert.down_proj.weight": "moe.shared.down.weight",
    "mlp.shared_expert_gate.weight": "moe.shared_gate.weight",
    # second language core (Qwen3.8-Flash-Next): indexer, hyper-connections, PLE
    "self_attn.indexer.index_qk_proj.weight": "mixer.indexer.qk.weight",
    "self_attn.indexer.q_layernorm.weight": "mixer.indexer.q_norm.weight",
    "self_attn.indexer.k_layernorm.weight": "mixer.indexer.k_norm.weight",
    "ple.ple_embedding.ngram_embedding.weight": "ple.table.weight",
    "ple.key_proj.weight": "ple.key.weight", "ple.value_proj.weight": "ple.value.weight",
    "ple.norm_key.weight": "ple.norm_key.weight", "ple.norm_query.weight": "ple.norm_query.weight",
    "ple.norm_conv.weight": "ple.norm_conv.weight", "ple.conv1d.weight": "ple.conv",
}
for _hc, _ours in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_mlp")):
    MIXER.update({f"{_hc}.hc_norm.weight": f"{_ours}.norm.weight",
                  f"{_hc}.input_mix_weight_down.weight": f"{_ours}.down.weight",
                  f"{_hc}.input_mix_weight_up.weight": f"{_ours}.up.weight",
                  f"{_hc}.block_inject_weight.weight": f"{_ours}.inject.weight"})
OUT_MIXER = {"hyper_connection_mixer.hc_norm.weight": "hc_out.norm.weight",
             "hyper_connection_mixer.input_mix_weight_down.weight": "hc_out.down.weight",
             "hyper_connection_mixer.input_mix_weight_up.weight": "hc_out.up.weight"}
PREFIXES = ("model.language_model.", "model.")


def rename(name: str) -> str | None:
    """Checkpoint tensor name -> Aegist name (None for tensors Aegist does not use)."""
    if name == "lm_head.weight":
        return "lm_head.weight"
    for p in PREFIXES:
        if name.startswith(p):
            rest = name[len(p):]
            break
    else:
        return None
    if rest.startswith(("visual.", "mtp.")):
        return None
    if rest == "embed_tokens.weight":
        return "embed.weight"
    if rest == "norm.weight":
        return "norm.weight"
    if rest in OUT_MIXER:
        return OUT_MIXER[rest]
    if not rest.startswith("layers."):
        return None
    _, i, sub = rest.split(".", 2)
    return f"blocks.{i}.{MIXER[sub]}" if sub in MIXER else None


def adapt(new: str, t: torch.Tensor) -> torch.Tensor:
    return t.squeeze(1) if new.endswith(("mixer.conv", "ple.conv")) else t


def config_from_hf(c: dict, **overrides) -> HybridConfig:
    """`c` is the text config (config.json's "text_config", or the whole file for a text-only model)."""
    c = c.get("text_config", c)
    rope = c.get("rope_parameters") or {}
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    partial = rope.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0))
    kinds = {"linear_attention": "delta", "full_attention": "attn", "indexed_attention": "attn"}
    cfg = HybridConfig(
        arch=ARCH, vocab_size=c["vocab_size"], d_model=c["hidden_size"], n_layers=c["num_hidden_layers"],
        layer_types=[kinds[t] for t in c["layer_types"]], n_heads=c["num_attention_heads"],
        n_kv_heads=c["num_key_value_heads"], head_dim=head_dim, rotary_dim=int(head_dim * partial),
        rope_theta=float(rope.get("rope_theta", c.get("rope_theta", 10000.0))),
        lin_k_heads=c["linear_num_key_heads"], lin_v_heads=c["linear_num_value_heads"],
        lin_k_dim=c["linear_key_head_dim"], lin_v_dim=c["linear_value_head_dim"],
        conv_kernel=c["linear_conv_kernel_dim"], n_experts=c["num_experts"], experts_active=c["num_experts_per_tok"],
        expert_hidden=c["moe_intermediate_size"], shared_hidden=c["shared_expert_intermediate_size"],
        norm_eps=c["rms_norm_eps"], max_seq_len=c["max_position_embeddings"],
        tie_embeddings=c.get("tie_word_embeddings", False))
    if c.get("hc_count", 1) > 1:  # second language core
        eos = c.get("eos_token_id")
        cfg.hc_count, cfg.hc_lowrank = c["hc_count"], c["hc_lowrank"]
        cfg.ple_layers = [i - 1 for i in sorted(set(c.get("ple_layer_ids") or []))]  # config counts from 1
        cfg.ple_dim = c.get("ple_embed_dim") or c["hidden_size"]
        cfg.ple_conv_kernel = c.get("ple_conv_kernel_size", 4)
        cfg.ngram_size, cfg.heads_per_ngram = c.get("ngram_size", 3), c.get("heads_per_ngram", 8)
        cfg.ngram_vocab_base = c.get("ngram_vocab_size_base", 20_000_000)
        cfg.ngram_divisor = c.get("make_ngram_vocab_size_divisible_by", 128)
        cfg.ngram_seed = c.get("seed", 1234)
        cfg.eos_id = eos[0] if isinstance(eos, list) else (eos if eos is not None else cfg.eos_id)
        if c.get("indexer_n_heads"):
            assert c.get("indexer_kv_heads", 1) == 1, "the indexer is written for one key head"
            cfg.idx_heads, cfg.idx_head_dim = c["indexer_n_heads"], c["indexer_head_dim"]
            cfg.idx_budget, cfg.idx_ratio = c["indexer_budget"], c["indexer_compress_ratio"]
        cfg.out_gate = c.get("output_gate_type") or c.get("hidden_act", "silu")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def load_dense(state: dict[str, torch.Tensor], cfg: HybridConfig, dtype=torch.float32, seed: int = 0) -> HybridLM:
    """A dense (unquantised) HybridLM from a checkpoint state dict; used by tests and small models."""
    torch.manual_seed(seed)
    model = HybridLM(cfg).to(dtype)
    sd = model.state_dict()
    loaded = set()
    for name, t in state.items():
        new = rename(name)
        if new is None or (new == "lm_head.weight" and cfg.tie_embeddings):
            continue
        if new not in sd:
            raise KeyError(f"{name} -> {new}: not an Aegist tensor")
        sd[new].copy_(adapt(new, t).to(dtype))
        loaded.add(new)
    base = {k for k in sd if ".engine." not in k and ".cores." not in k}
    if cfg.tie_embeddings:
        base.discard("lm_head.weight")
    missing = base - loaded
    if missing:
        raise ValueError(f"missing from checkpoint: {sorted(missing)[:5]}")
    return model


class Shards:
    """Lazy access to a sharded safetensors checkpoint by tensor name."""

    def __init__(self, path: str):
        self.path = path
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.where = json.load(open(idx, encoding="utf-8"))["weight_map"]
        else:
            with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
                self.where = {k: "model.safetensors" for k in f.keys()}
        self._open: OrderedDict[str, object] = OrderedDict()

    # Open shards are memory maps, and on Windows each one is charged against the commit limit at
    # its full size: keep only the few in use (a 131-shard checkpoint otherwise runs out of memory).
    MAX_OPEN = 6

    def get(self, name: str) -> torch.Tensor:
        fn = self.where[name]
        if fn in self._open:
            self._open.move_to_end(fn)
        else:
            while len(self._open) >= self.MAX_OPEN:
                self._open.popitem(last=False)
            self._open[fn] = safe_open(os.path.join(self.path, fn), framework="pt")
        return self._open[fn].get_tensor(name)

    def aegist_names(self) -> dict[str, str]:
        out = {}
        for n in self.where:
            new = rename(n)
            if new is not None:
                out[new] = n
        return out


def quantize_layer_experts(gate_up: torch.Tensor, down: torch.Tensor, cfg: HybridConfig, device,
                           chunk: int = 32) -> tuple[torch.Tensor, float]:
    """bf16 experts -> uint8 [E, expert_bytes], and the relative RMS weight error."""
    rows, err, ref = [], 0.0, 0.0
    for s in range(0, gate_up.shape[0], chunk):
        gu = gate_up[s : s + chunk].to(device)
        dn = down[s : s + chunk].to(device)
        packed = pack_experts(gu, dn, cfg)
        for j in range(packed.shape[0]):
            g2, d2 = unpack_expert(packed[j], cfg, torch.float32)
            err += float((g2 - gu[j].float()).pow(2).sum() + (d2 - dn[j].float()).pow(2).sum())
            ref += float(gu[j].float().pow(2).sum() + dn[j].float().pow(2).sum())
        rows.append(packed.cpu())
        del gu, dn, packed
    return torch.cat(rows), (err / max(ref, 1e-30)) ** 0.5


def brain_spec(src: str, cfg: HybridConfig, seed: int, device, log=print):
    """Name -> nx.Lazy for every tensor of a fresh brain, in the layout `load_hybrid` expects."""
    from .nx import Lazy

    shards = Shards(src)
    have = shards.aegist_names()
    with torch.device("meta"):
        model = HybridLM(cfg).to(torch.bfloat16)
        model.quantize_base_int8()
    torch.manual_seed(seed)
    engines = [{k: v.to(torch.bfloat16) for k, v in CoreEngine(cfg).state_dict().items()} for _ in range(cfg.n_layers)]
    memo: dict[str, QLinear] = {}
    errors: dict[int, float] = {}

    def plain(name):
        def make():
            return adapt(name, shards.get(have[name])).to(torch.bfloat16).contiguous()
        return make

    def int8(base, part):
        def make():
            if base not in memo:
                memo.clear()
                memo[base] = QLinear(shards.get(have[base + ".weight"]).to(device))
            return getattr(memo[base], part).cpu()
        return make

    def engine(i, k):
        return lambda: engines[i][k]

    def experts(i):
        def make():
            t = time.time()
            gu = shards.get(have[f"blocks.{i}.moe.experts.gate_up"])
            dn = shards.get(have[f"blocks.{i}.moe.experts.down"])
            q, rel = quantize_layer_experts(gu, dn, cfg, device)
            errors[i] = rel
            log(f"  block {i}: experts 4-bit, relative weight error {rel:.4f} ({time.time() - t:.1f}s)")
            return q
        return make

    spec = {}
    for name, t in model.state_dict().items():
        if ".moe.experts." in name:
            continue
        if ".engine." in name:
            i, k = int(name.split(".")[1]), name.split(".engine.")[1]
            spec[name] = Lazy(torch.bfloat16, t.shape, engine(i, k))
        elif ".cores." in name:
            spec[name] = Lazy(t.dtype, t.shape, (lambda shape=tuple(t.shape), dt=t.dtype: torch.zeros(shape, dtype=dt)))
        elif name.endswith((".qweight", ".scale")):
            base, part = name.rsplit(".", 1)
            if base + ".weight" not in have:
                raise ValueError(f"checkpoint has no {base}.weight")
            spec[name] = Lazy(t.dtype, t.shape, int8(base, part))
        else:
            if name not in have:
                raise ValueError(f"checkpoint has no tensor for {name}")
            spec[name] = Lazy(torch.bfloat16, t.shape, plain(name))
    for i in range(cfg.n_layers):
        spec[f"blocks.{i}.moe.experts.q4"] = Lazy(torch.uint8, (cfg.n_experts, expert_bytes(cfg)), experts(i))
    return spec, errors


def build_brain(src: str, out: str, *, core_grid=(8, 64), core_rank: int = 64, cores_active: int = 2,
                core_rows_max: int = 85, expert_group: int = 64, seed: int = 0, device="cuda", log=print) -> dict:
    from . import persona
    from .nx import save

    raw = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    cfg = config_from_hf(raw, core_grid=tuple(core_grid), core_rank=core_rank, cores_active=cores_active,
                         core_rows_max=core_rows_max, core_seed=seed, expert_group=expert_group)
    gen = json.load(open(os.path.join(src, "generation_config.json"), encoding="utf-8"))
    spec, errors = brain_spec(src, cfg, seed, device, log)
    with torch.device("meta"):
        probe = HybridLM(cfg).to(torch.bfloat16)
        probe.quantize_base_int8()
    report = probe.param_report()
    tok_json = open(os.path.join(src, "tokenizer.json"), encoding="utf-8").read()
    lic = os.path.join(src, "LICENSE")
    license_text = BASE_NOTICE + "\n\n" + (open(lic, encoding="utf-8").read() if os.path.exists(lic) else "")
    lineage = [{"event": "born", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "base": f"{os.path.basename(os.path.normpath(src))} (Apache-2.0)", "arch": ARCH,
                "quant": {"experts": f"q4 asymmetric, group {cfg.expert_group}", "linears": "int8"},
                "core_grid": list(cfg.core_grid), "core_rank": core_rank, "cores_active": cores_active,
                "core_rows_max": core_rows_max, "seed": seed}]
    # the base's generation_config holds its thinking settings; the plain-answer ones and the
    # presence penalty are the base model card's recommendations
    generation = {"temperature": 0.7, "top_p": 0.8, "top_k": int(gen.get("top_k", 20)),
                  "think_temperature": float(gen.get("temperature", 1.0)), "think_top_p": float(gen.get("top_p", 0.95)),
                  "presence_penalty": 1.5, "open_think": True, "eos": gen.get("eos_token_id")}
    t = time.time()
    header = save(out, spec, cfg.to_dict(), tok_json, notice=BASE_NOTICE, license_text=license_text,
                  persona={"name": "Aegist", "default_mode": "normal", "modes": list(persona.MODES)},
                  generation=generation, lineage=lineage,
                  extra={"name": "Aegist-nx-35B", "version": 1, "quant": "int8", "experts": f"q4a-g{cfg.expert_group}", "dtype": "bf16",
                         "params": report})
    header["expert_error"] = errors
    log(f"wrote {out}: {os.path.getsize(out) / 1e9:.2f} GB in {time.time() - t:.0f}s")
    return header
