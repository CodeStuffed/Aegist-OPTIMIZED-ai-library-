"""Aegist hybrid language model (arch "hybrid-moe").

Every block has a token mixer, a mixture of experts and Aegist's switchable cores:

    x = x + mixer(norm1(x))                    mixer: gated DeltaNet (3 of every 4 blocks) or gated attention
    h = norm2(x)
    x = x + moe(h) + cores(h, engine(h))       moe: router picks experts_active of n_experts, plus a shared expert

Gated DeltaNet (Yang et al. 2024, "Gated Delta Networks: Improving Mamba2 with Delta Rule") keeps a
fixed-size state per head instead of a growing key/value cache, so 3 of every 4 blocks cost the same
at token 100,000 as at token 1. The gated-attention blocks keep an ordinary KV cache.

The language core starts from Qwen3.6-35B-A3B's open weights (Apache-2.0; notice and license travel in
the .nx file). Experts are stored 4-bit (asymmetric, 64-value groups) and paged: all of them live in
host memory or the memory-mapped brain, and `slots` per block sit on the GPU, least recently used out.

The cores are the same product-key engine as aegist.model, but stored sparsely: a core takes no space
until learning time first trains it, so the core grid (the capacity) can be far larger than what is
stored. Untrained cores add exactly zero, so a fresh brain computes what its base weights compute.

The second language core (from Qwen3.8-Flash-Next) adds hyper-connections, per-layer n-gram
embeddings, a sparse-attention indexer and FP8 weights (aegist/hc.py); with hc_count > 1 a block is

    h, x0, w = hc_attn(x);  x = x0 + w * mixer(h)       (x is hc_count lanes of d_model)
    h, x0, w = hc_mlp(x);   x = x0 + w * (moe(h) + cores(h, engine(h)))

with a PLE added to the lanes first in the blocks that have one.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from . import hc as HC
from .model import CoreEngine, QLinear, SwiGLU

ARCH = "hybrid-moe"


@dataclass
class HybridConfig:
    arch: str = ARCH
    vocab_size: int = 248320
    d_model: int = 2048
    n_layers: int = 40
    layer_types: list[str] = field(default_factory=lambda: (["delta"] * 3 + ["attn"]) * 10)
    # gated attention
    n_heads: int = 16
    n_kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 10_000_000.0
    # gated DeltaNet
    lin_k_heads: int = 16
    lin_v_heads: int = 32
    lin_k_dim: int = 128
    lin_v_dim: int = 128
    conv_kernel: int = 4
    # mixture of experts
    n_experts: int = 256
    experts_active: int = 8
    expert_hidden: int = 512
    shared_hidden: int = 512
    expert_group: int = 64  # 4-bit quantisation group (input values per scale/min pair)
    norm_eps: float = 1e-6
    max_seq_len: int = 262144
    tie_embeddings: bool = False
    # switchable cores (sparse: only trained cores are stored)
    core_grid: tuple[int, int] = (8, 64)  # rows x columns; learning time grows rows, never columns
    core_rank: int = 64
    cores_active: int = 2
    core_aux_coef: float = 0.01
    core_rows_max: int = 85  # capacity limit for growth
    core_seed: int = 0
    # second language core: hyper-connections (hc_count 1 = off), PLE, QSA indexer, FP8
    hc_count: int = 1
    hc_lowrank: int = 0
    ple_layers: list[int] = field(default_factory=list)  # 0-indexed blocks with a per-layer embedding
    ple_dim: int = 0
    ple_conv_kernel: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_base: int = 20_000_000
    ngram_divisor: int = 128
    ngram_seed: int = 1234
    eos_id: int = 248044
    idx_heads: int = 0  # 0 = no sparse-attention indexer
    idx_head_dim: int = 128
    idx_budget: int = 2048
    idx_ratio: int = 4
    out_gate: str = "silu"  # DeltaNet output gate: silu or sigmoid
    expert_format: str = "q4"  # q4 (4-bit groups) or fp8 (float8 with 128x128 block scales)

    @property
    def n_cores(self) -> int:
        return self.core_grid[0] * self.core_grid[1]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["core_grid"] = list(self.core_grid)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HybridConfig":
        d = dict(d)
        if "core_grid" in d:
            d["core_grid"] = tuple(d["core_grid"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --- norms -------------------------------------------------------------------------

class ZNorm(nn.Module):
    """Zero-centred RMSNorm: the stored weight w scales by (1 + w), computed in fp32."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (1.0 + self.weight.float())).type_as(x)


class GatedNorm(nn.Module):
    """RMSNorm (plain weight) of x, then times SiLU(z) (or sigmoid(z))."""

    def __init__(self, dim: int, eps: float, act: str = "silu"):
        super().__init__()
        self.eps = eps
        self.act = act
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, z):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        y = self.weight * xf.to(x.dtype)
        g = torch.sigmoid(z.float()) if self.act == "sigmoid" else F.silu(z.float())
        return (y * g).to(x.dtype)


# --- position ----------------------------------------------------------------------

def rope_tables(cfg: HybridConfig, length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    rd = cfg.rotary_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, rd, 2, device=device, dtype=torch.float32) / rd))
    f = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv)
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def apply_partial_rope(x, cos, sin):
    """Rotate the first cos.shape[-1] dims of each head (rotate-half form); pass the rest through."""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    h = rd // 2
    rot = torch.cat([-xr[..., h:], xr[..., :h]], -1)
    return torch.cat([xr * cos.to(x.dtype) + rot * sin.to(x.dtype), xp], -1)


# --- cache -------------------------------------------------------------------------

class HybridCache:
    """Everything a conversation carries between tokens.

    Attention blocks: a key/value cache that grows in 1024-token blocks.
    DeltaNet blocks: the last conv_kernel-1 inputs of the short convolution and the fp32 state
    (lin_v_heads x lin_k_dim x lin_v_dim per block); both are fixed size.

    A recurrent state cannot be cut back to an earlier position, so `snapshot()` records the
    DeltaNet states at the current length and `crop(n)` returns to the latest snapshot at or
    before n. The caller re-feeds the tokens after the returned length.
    """

    BLOCK = 1024

    def __init__(self, cfg: HybridConfig, batch: int, max_len: int, device, dtype):
        self.cfg, self.batch, self.device, self.dtype = cfg, batch, device, dtype
        self.max_len = max_len
        self.attn_layers = [i for i, t in enumerate(cfg.layer_types) if t == "attn"]
        self.delta_layers = [i for i, t in enumerate(cfg.layer_types) if t == "delta"]
        self.slot = {i: j for j, i in enumerate(self.attn_layers)}
        self.slot.update({i: j for j, i in enumerate(self.delta_layers)})
        conv_dim = 2 * cfg.lin_k_heads * cfg.lin_k_dim + cfg.lin_v_heads * cfg.lin_v_dim
        self.conv = torch.zeros(len(self.delta_layers), batch, conv_dim, cfg.conv_kernel - 1, device=device, dtype=dtype)
        self.state = torch.zeros(len(self.delta_layers), batch, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim,
                                 device=device, dtype=torch.float32)
        self.length = 0
        self.capacity = 0
        self.k = self.v = self.idx_k = None
        self.ple_hist = [None] * len(cfg.ple_layers)  # each block's last ngram_size-1 tokens
        self.ple_conv = [None] * len(cfg.ple_layers)  # each block's PLE convolution inputs
        self.snapshots: dict[int, tuple] = {}
        self.reserve(min(max_len, self.BLOCK))

    def reserve(self, n: int):
        if n <= self.capacity:
            return
        if n > self.max_len:
            raise ValueError(f"context of {n} tokens exceeds this cache's limit of {self.max_len}")
        cap = max(n, 2 * self.capacity)
        cap = min(self.max_len, -(-cap // self.BLOCK) * self.BLOCK)
        c = self.cfg
        shape = (len(self.attn_layers), self.batch, c.n_kv_heads, cap, c.head_dim)
        k = torch.empty(shape, device=self.device, dtype=self.dtype)
        v = torch.empty(shape, device=self.device, dtype=self.dtype)
        if self.length:
            k[:, :, :, : self.length] = self.k[:, :, :, : self.length]
            v[:, :, :, : self.length] = self.v[:, :, :, : self.length]
        if c.idx_heads:
            ik = torch.empty(len(self.attn_layers), self.batch, cap, c.idx_head_dim, device=self.device, dtype=self.dtype)
            if self.length:
                ik[:, :, : self.length] = self.idx_k[:, :, : self.length]
            self.idx_k = ik
        self.k, self.v, self.capacity = k, v, cap

    def snapshot(self, keep: int = 4):
        clone = lambda xs: [None if t is None else t.clone() for t in xs]
        self.snapshots[self.length] = (self.conv.clone(), self.state.clone(), clone(self.ple_hist), clone(self.ple_conv))
        for old in sorted(self.snapshots)[:-keep]:
            del self.snapshots[old]

    def crop(self, n: int) -> int:
        """Go back to at most n tokens; returns the length actually kept (0 if nothing usable)."""
        if n >= self.length:
            return self.length
        ok = [p for p in self.snapshots if p <= n]
        if not ok:
            self.reset()
            return 0
        p = max(ok)
        conv, state, hist, pconv = self.snapshots[p]
        self.conv.copy_(conv)
        self.state.copy_(state)
        self.ple_hist = [None if t is None else t.clone() for t in hist]
        self.ple_conv = [None if t is None else t.clone() for t in pconv]
        self.length = p
        for later in [q for q in self.snapshots if q > p]:
            del self.snapshots[later]
        return p

    def reset(self):
        self.conv.zero_()
        self.state.zero_()
        self.ple_hist = [None] * len(self.ple_hist)
        self.ple_conv = [None] * len(self.ple_conv)
        self.length = 0
        self.snapshots.clear()

    def nbytes(self) -> int:
        kv = 0 if self.k is None else 2 * self.k.numel() * self.k.element_size()
        return kv + self.conv.numel() * self.conv.element_size() + self.state.numel() * 4


# --- token mixers ------------------------------------------------------------------

class GatedAttention(nn.Module):
    """GQA attention with per-head q/k zero-centred RMSNorm, partial RoPE and a sigmoid output gate.
    q projects to heads x (query, gate)."""

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.h, self.kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q = nn.Linear(cfg.d_model, self.h * self.hd * 2, bias=False)
        self.k = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, cfg.d_model, bias=False)
        self.q_norm = ZNorm(self.hd, cfg.norm_eps)
        self.k_norm = ZNorm(self.hd, cfg.norm_eps)
        self.indexer = HC.QSAIndexer(cfg) if cfg.idx_heads else None

    def forward(self, x, cos, sin, cache: HybridCache | None, slot: int):
        B, T, _ = x.shape
        sparse = None
        if self.indexer is not None:
            sparse = self.indexer(x, cos, sin, cache, slot, cache.length if cache is not None else 0)
        q, gate = self.q(x).view(B, T, self.h, 2 * self.hd).chunk(2, -1)
        gate = gate.reshape(B, T, self.h * self.hd)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k(x).view(B, T, self.kv, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        pos = cache.length if cache is not None else 0
        c, s = cos[pos : pos + T], sin[pos : pos + T]
        q, k = apply_partial_rope(q, c, s), apply_partial_rope(k, c, s)
        if cache is not None:
            cache.k[slot, :, :, pos : pos + T] = k
            cache.v[slot, :, :, pos : pos + T] = v
            k, v = cache.k[slot, :, :, : pos + T], cache.v[slot, :, :, : pos + T]
        g = self.h // self.kv
        if sparse is not None:
            k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=sparse)
        elif T == 1:
            y = F.scaled_dot_product_attention(q.reshape(B, self.kv, g, self.hd), k, v).reshape(B, self.h, 1, self.hd)
        else:
            k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
            if pos == 0:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                mask = torch.ones(T, pos + T, dtype=torch.bool, device=x.device).tril(pos)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, self.h * self.hd) * torch.sigmoid(gate)
        return self.o(y)


def l2norm(x, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def delta_rule_step(q, k, v, g, beta, state):
    """One token. q, k [B,H,dk]; v [B,H,dv]; g, beta [B,H]; state [B,H,dk,dv] (fp32, updated in place)."""
    state.mul_(g.exp()[..., None, None])
    kv_mem = (state * k.unsqueeze(-1)).sum(-2)
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    state.add_(k.unsqueeze(-1) * delta.unsqueeze(-2))
    return (state * q.unsqueeze(-1)).sum(-2)


def delta_rule_chunked(q, k, v, g, beta, state, chunk: int = 64):
    """The gated delta rule over a whole sequence, `chunk` tokens at a time (WY/UT form).

    q, k [B,H,T,dk]; v [B,H,T,dv]; g, beta [B,H,T]; all fp32. state [B,H,dk,dv] or None.
    Returns (out [B,H,T,dv], final state). Within a chunk the delta-rule updates are solved
    together as a unit-lower-triangular system; across chunks the state is carried forward.
    """
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    pad = (chunk - T % chunk) % chunk
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        g, beta = (F.pad(t, (0, pad)) for t in (g, beta))
    n = (T + pad) // chunk
    kb = (k * beta.unsqueeze(-1)).view(B, H, n, chunk, dk)
    vb = (v * beta.unsqueeze(-1)).view(B, H, n, chunk, dv)
    q, k = q.view(B, H, n, chunk, dk), k.view(B, H, n, chunk, dk)
    cum = g.view(B, H, n, chunk).cumsum(-1)
    upper = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    decay = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(upper, float("-inf")).exp()  # [i, j] = exp(cum_i - cum_j), j <= i
    system = (kb @ k.transpose(-1, -2)) * decay  # solved as unit lower-triangular: diagonal ignored
    u = torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(system, kb * cum.exp().unsqueeze(-1), upper=False, unitriangular=True)
    intra = (q @ k.transpose(-1, -2)) * decay
    q_dec = q * cum.exp().unsqueeze(-1)
    k_dec = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    S = torch.zeros(B, H, dk, dv, device=q.device, dtype=q.dtype) if state is None else state.to(q.dtype)
    outs = []
    for i in range(n):
        v_new = u[:, :, i] - w[:, :, i] @ S
        outs.append(q_dec[:, :, i] @ S + intra[:, :, i] @ v_new)
        S = S * chunk_decay[:, :, i] + k_dec[:, :, i].transpose(-1, -2) @ v_new
    out = torch.stack(outs, 2).reshape(B, H, n * chunk, dv)[:, :, :T]
    return out, S


class GatedDeltaNet(nn.Module):
    """Linear-attention token mixer with a fixed-size state per value head.

    in_qkv -> depthwise causal conv (kernel conv_kernel) -> SiLU -> q, k (L2-normalised, q scaled
    by 1/sqrt(dk)), v; beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias) (log decay);
    gated delta rule; per-head RMSNorm gated by SiLU(z); out projection."""

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.hk, self.hv, self.dk, self.dv = cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim
        self.key_dim, self.value_dim = self.hk * self.dk, self.hv * self.dv
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.K = cfg.conv_kernel
        self.in_qkv = nn.Linear(cfg.d_model, self.conv_dim, bias=False)
        self.in_z = nn.Linear(cfg.d_model, self.value_dim, bias=False)
        self.in_b = nn.Linear(cfg.d_model, self.hv, bias=False)
        self.in_a = nn.Linear(cfg.d_model, self.hv, bias=False)
        self.conv = nn.Parameter(torch.zeros(self.conv_dim, self.K))
        self.dt_bias = nn.Parameter(torch.ones(self.hv))
        self.A_log = nn.Parameter(torch.zeros(self.hv))
        self.norm = GatedNorm(self.dv, cfg.norm_eps, cfg.out_gate)
        self.out = nn.Linear(self.value_dim, cfg.d_model, bias=False)
        self.chunk = 64

    def forward(self, x, cache: HybridCache | None, slot: int):
        B, T, _ = x.shape
        qkv = self.in_qkv(x).transpose(1, 2)  # [B, C, T]
        z = self.in_z(x).view(B, T, self.hv, self.dv)
        b, a = self.in_b(x), self.in_a(x)
        w = self.conv.to(qkv.dtype)
        prev = cache.conv[slot].to(qkv.dtype) if cache is not None else qkv.new_zeros(B, self.conv_dim, self.K - 1)
        xp = torch.cat([prev, qkv], -1)
        if cache is not None:
            cache.conv[slot].copy_(xp[..., -(self.K - 1):])
        qkv = F.silu(F.conv1d(xp, w.unsqueeze(1), groups=self.conv_dim)).transpose(1, 2)  # [B, T, C]
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], -1)
        q = q.reshape(B, T, self.hk, self.dk)
        k = k.reshape(B, T, self.hk, self.dk)
        v = v.reshape(B, T, self.hv, self.dv)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        r = self.hv // self.hk
        if r > 1:
            q, k = q.repeat_interleave(r, 2), k.repeat_interleave(r, 2)
        q, k, v, beta, g = (t.transpose(1, 2).float() for t in (q, k, v, beta, g))  # heads first
        q, k = l2norm(q), l2norm(k)
        q = q * self.dk ** -0.5
        if cache is not None and T == 1 and cache.length > 0:
            out = delta_rule_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], g[:, :, 0], beta[:, :, 0], cache.state[slot])
            out = out.unsqueeze(2)
        else:
            init = cache.state[slot] if cache is not None and cache.length > 0 else None
            out, S = delta_rule_chunked(q, k, v, g, beta, init, self.chunk)
            if cache is not None:
                cache.state[slot].copy_(S)
        out = out.transpose(1, 2).to(x.dtype)  # [B, T, hv, dv]
        out = self.norm(out, z).reshape(B, T, self.value_dim)
        return self.out(out)


# --- experts -----------------------------------------------------------------------

def q4_quantize(w: torch.Tensor, group: int, iters: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric 4-bit quantisation of w [..., in] in groups of `group` input values.

    Starts from each group's min/max, then refits scale and offset by least squares for the
    current codes a few times, keeping whichever fit has the lowest error per group.
    Returns (packed uint8 [..., in/2] (even index in the low nibble), scale f16 [..., in/group],
    min f16 [..., in/group]); w ~= code * scale + min."""
    *lead, n = w.shape
    x = w.float().reshape(*lead, n // group, group)
    lo, hi = x.amin(-1), x.amax(-1)

    def fit(scale, mn):
        scale, mn = scale.half().float(), mn.half().float()  # quantise against what is stored
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        code = ((x - mn.unsqueeze(-1)) / safe.unsqueeze(-1)).round().clamp(0, 15)
        err = (code * scale.unsqueeze(-1) + mn.unsqueeze(-1) - x).pow(2).sum(-1)
        return scale, mn, code, err

    best = fit((hi - lo) / 15, lo)
    scale, mn, code, err = best
    for _ in range(iters):
        # least squares for x ~= code * s + m within each group
        cm, xm = code.mean(-1), x.mean(-1)
        var = (code - cm.unsqueeze(-1)).pow(2).mean(-1)
        cov = ((code - cm.unsqueeze(-1)) * (x - xm.unsqueeze(-1))).mean(-1)
        s_new = torch.where(var > 0, cov / var.clamp(min=1e-12), scale).clamp(min=0)
        m_new = torch.where(var > 0, xm - s_new * cm, mn)
        cand = fit(s_new, m_new)
        better = cand[3] < best[3]
        best = tuple(torch.where(better.unsqueeze(-1) if t.dim() > better.dim() else better, c, b)
                     for c, b, t in zip(cand, best, cand))
        scale, mn, code, err = best
    code = code.to(torch.uint8).reshape(*lead, n)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous(), best[0].half(), best[1].half()


def q4_dequantize(packed, scale, mn, group: int, dtype=torch.float32):
    lo = (packed & 0xF).to(dtype)
    hi = (packed >> 4).to(dtype)
    code = torch.stack([lo, hi], -1).flatten(-2)
    *lead, n = code.shape
    code = code.view(*lead, n // group, group)
    return (code * scale.to(dtype).unsqueeze(-1) + mn.to(dtype).unsqueeze(-1)).reshape(*lead, n)


def expert_layout(cfg: HybridConfig) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
    """The byte layout of one expert inside its layer's packed row (`experts.q4` or `experts.fp8`).
    4-bit: every section is a multiple of 64 bytes, so each starts aligned."""
    if cfg.expert_format == "fp8":
        return HC.expert_layout_fp8(cfg)
    I, D, G = cfg.expert_hidden, cfg.d_model, cfg.expert_group
    return [("gu_q", (2 * I, D // 2), torch.uint8), ("gu_s", (2 * I, D // G), torch.float16),
            ("gu_m", (2 * I, D // G), torch.float16), ("dn_q", (D, I // 2), torch.uint8),
            ("dn_s", (D, I // G), torch.float16), ("dn_m", (D, I // G), torch.float16)]


def expert_bytes(cfg: HybridConfig) -> int:
    n = sum(math.prod(s) * torch.tensor([], dtype=dt).element_size() for _, s, dt in expert_layout(cfg))
    return -(-n // 64) * 64  # rows start 64-byte aligned


def pack_experts_fp8(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                     cfg: HybridConfig) -> torch.Tensor:
    """[(gate_up f8 [2I, D], down f8 [D, I], gate_up scale, down scale)] per expert -> uint8 [E, expert_bytes].
    The bytes are kept exactly as given (scales must already be bf16)."""
    rows = []
    for p in parts:
        r = torch.cat([t.contiguous().view(torch.uint8).reshape(-1) for t in p])
        rows.append(F.pad(r, (0, expert_bytes(cfg) - r.numel())))
    return torch.stack(rows)


def pack_experts(gate_up: torch.Tensor, down: torch.Tensor, cfg: HybridConfig) -> torch.Tensor:
    """gate_up [E, 2I, D], down [E, D, I] (any float dtype/device) -> uint8 [E, expert_bytes]."""
    if cfg.expert_format == "fp8":
        parts = []
        for gu, dn in zip(gate_up, down):
            gq, gs = HC.fp8_quant(gu)
            dq, ds = HC.fp8_quant(dn)
            parts.append((gq, dq, gs.to(torch.bfloat16), ds.to(torch.bfloat16)))
        return pack_experts_fp8(parts, cfg)
    G = cfg.expert_group
    parts = [*q4_quantize(gate_up, G), *q4_quantize(down, G)]
    E = gate_up.shape[0]
    return torch.cat([p.contiguous().view(torch.uint8).reshape(E, -1) for p in parts], 1)


def unpack_expert(row: torch.Tensor, cfg: HybridConfig, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """One expert's uint8 row -> (gate_up [2I, D], down [D, I]) in `dtype`."""
    out, off = {}, 0
    for name, shape, dt in expert_layout(cfg):
        nb = math.prod(shape) * torch.tensor([], dtype=dt).element_size()
        out[name] = row[off : off + nb].view(dt).view(shape)
        off += nb
    if cfg.expert_format == "fp8":
        return HC.fp8_dequant(out["gu"], out["gu_s"], dtype), HC.fp8_dequant(out["dn"], out["dn_s"], dtype)
    G = cfg.expert_group
    gu = q4_dequantize(out["gu_q"], out["gu_s"], out["gu_m"], G, dtype)
    dn = q4_dequantize(out["dn_q"], out["dn_s"], out["dn_m"], G, dtype)
    return gu, dn


class _ExpertFF(torch.autograd.Function):
    """One expert on its tokens, times the router weights. Backward fetches the expert's weights
    again instead of keeping them: a layer's 512 dequantized experts would not fit in memory until
    backward. The experts are frozen, so only the tokens and router weights get gradients."""

    @staticmethod
    def forward(ctx, x, wt, bank, e):
        gu, dn = bank.weights(e, x.dtype)
        gate, up = F.linear(x, gu).chunk(2, -1)
        y = F.linear(F.silu(gate) * up, dn)
        ctx.save_for_backward(x, wt)
        ctx.bank, ctx.e = bank, e
        return y * wt.unsqueeze(-1).to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        x, wt = ctx.saved_tensors
        gu, dn = ctx.bank.weights(ctx.e, x.dtype)
        f = torch.promote_types(x.dtype, torch.float32)
        gate, up = F.linear(x, gu).to(f).chunk(2, -1)
        sg = torch.sigmoid(gate)
        silu = gate * sg
        z = silu * up
        y = F.linear(z.to(x.dtype), dn).to(f)
        dyf = dy.to(f)
        dwt = (dyf * y).sum(-1).to(wt.dtype) if ctx.needs_input_grad[1] else None
        dz = F.linear((dyf * wt.to(f).unsqueeze(-1)).to(x.dtype), dn.t()).to(f)
        da = torch.cat([dz * up * sg * (1 + gate * (1 - sg)), dz * silu], -1)
        dx = F.linear(da.to(x.dtype), gu.t()) if ctx.needs_input_grad[0] else None
        return dx, dwt, None, None


class ExpertBank(nn.Module):
    """n_experts SwiGLU experts. Dense (gate_up/down tensors, for tests and the bf16 reference) or
    4-bit and paged: `masters` [E, expert_bytes] uint8 in host memory, `slots` rows on the device."""

    def __init__(self, cfg: HybridConfig, dense: bool = True):
        super().__init__()
        self.cfg = cfg
        self.E, self.I = cfg.n_experts, cfg.expert_hidden
        self.dense = dense
        if dense:
            self.gate_up = nn.Parameter(torch.zeros(self.E, 2 * self.I, cfg.d_model))
            self.down = nn.Parameter(torch.zeros(self.E, cfg.d_model, self.I))
        self.page_ins = 0
        # learning time on the 177B: refetch expert weights in backward (see _ExpertFF)
        self.refetch = False

    def page(self, masters: torch.Tensor, slots: int, device):
        """Switch to 4-bit paged storage. `masters` may be a view into a memory-mapped .nx."""
        for n in ("gate_up", "down"):
            self._parameters.pop(n, None)
        self.dense = False
        self.masters = masters
        self.slots = max(self.cfg.experts_active, min(slots, self.E))
        self.register_buffer("s_q4", torch.zeros(self.slots, masters.shape[1], dtype=torch.uint8, device=device),
                             persistent=False)
        self.reset_slots()
        return self

    def reset_slots(self):
        self._lru: OrderedDict[int, int] = OrderedDict()
        self._free = list(range(self.slots))

    def _row(self, e: int) -> torch.Tensor:
        if e in self._lru:
            self._lru.move_to_end(e)
            return self.s_q4[self._lru[e]]
        s = self._free.pop() if self._free else self._lru.pop(next(iter(self._lru)))
        self.s_q4[s].copy_(self.masters[e], non_blocking=True)
        self._lru[e] = s
        self.page_ins += 1
        return self.s_q4[s]

    def weights(self, e: int, dtype):
        if self.dense:
            return self.gate_up[e].to(dtype), self.down[e].to(dtype)
        return unpack_expert(self._row(e), self.cfg, dtype)

    def forward(self, h, idx, w):
        """h [N, D]; idx, w [N, k] -> [N, D]."""
        out = torch.zeros_like(h)
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = torch.arange(h.shape[0], device=h.device).repeat_interleave(idx.shape[1])[order]
        ww = w.reshape(-1)[order]
        uniq, counts = torch.unique_consecutive(flat[order], return_counts=True)
        refetch = self.refetch and torch.is_grad_enabled()
        for e, t, wt in zip(uniq.tolist(), tok.split(counts.tolist()), ww.split(counts.tolist())):
            if refetch:
                out = out.index_add(0, t, _ExpertFF.apply(h[t], wt, self, e))
                continue
            gu, dn = self.weights(e, h.dtype)
            gate, up = F.linear(h[t], gu).chunk(2, -1)
            y = F.linear(F.silu(gate) * up, dn) * wt.unsqueeze(-1).to(h.dtype)
            out.index_add_(0, t, y)
        return out

    def numel(self) -> int:
        return self.E * 3 * self.I * self.cfg.d_model


class MoE(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.k = cfg.experts_active
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = ExpertBank(cfg)
        self.shared = SwiGLU(cfg.d_model, cfg.shared_hidden)
        self.shared_gate = nn.Linear(cfg.d_model, 1, bias=False)
        self.last_experts: torch.Tensor | None = None

    def route(self, h):
        logits = self.router(h)
        p = logits.float().softmax(-1)
        top, idx = p.topk(self.k, -1)
        return idx, (top / top.sum(-1, keepdim=True)).to(logits.dtype)

    def forward(self, h):
        """h [N, D] -> [N, D]."""
        idx, w = self.route(h)
        self.last_experts = idx.detach()
        y = self.experts(h, idx, w)
        return y + torch.sigmoid(self.shared_gate(h)) * self.shared(h)


# --- sparse cores ------------------------------------------------------------------

class SparseCoreBank(nn.Module):
    """Aegist's switchable cores, stored sparsely.

    A core id is row * grid_cols + col of the engine's grid. Only "live" cores (ever trained) are
    stored: `ids` [n] with gate/up [n, d, r] and down [n, r, d]. A core that is not live adds
    exactly zero (its output projection would start at zero), so it costs nothing. When learning
    time first trains a core it is made live with a deterministic start (seeded by layer and id),
    so the same core always starts from the same weights.
    """

    def __init__(self, cfg: HybridConfig, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        self.d, self.r = cfg.d_model, cfg.core_rank
        self.register_buffer("ids", torch.zeros(0, dtype=torch.long))
        self.gate = nn.Parameter(torch.zeros(0, self.d, self.r))
        self.up = nn.Parameter(torch.zeros(0, self.d, self.r))
        self.down = nn.Parameter(torch.zeros(0, self.r, self.d))
        self._index: dict[int, int] | None = None
        self.work_ids = None

    @property
    def n_cores(self) -> int:
        return self.cfg.n_cores

    @property
    def silent(self) -> bool:
        return self.ids.numel() == 0 or not bool(self.down.detach().any())

    def refresh_silent(self):
        pass  # computed on demand

    def index(self) -> dict[int, int]:
        if self._index is None:
            self._index = {int(c): i for i, c in enumerate(self.ids.tolist())}
        return self._index

    def live_ids(self) -> list[int]:
        return self.ids.tolist()

    def numel(self) -> int:
        return self.gate.numel() + self.up.numel() + self.down.numel()

    def start_weights(self, core: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.cfg.core_seed * 1_000_003 + self.layer * 65_537 + core)
        std = self.d ** -0.5
        return (torch.randn(self.d, self.r, generator=g) * std, torch.randn(self.d, self.r, generator=g) * std)

    def make_live(self, cores: list[int]):
        new = [c for c in dict.fromkeys(cores) if c not in self.index()]
        if not new:
            return
        dev, dt = self.gate.device, self.gate.dtype
        starts = [self.start_weights(c) for c in new]
        with torch.no_grad():
            g = torch.stack([s[0] for s in starts]).to(dev, dt)
            u = torch.stack([s[1] for s in starts]).to(dev, dt)
            dn = torch.zeros(len(new), self.r, self.d, device=dev, dtype=dt)
            self.gate = nn.Parameter(torch.cat([self.gate.detach(), g]), requires_grad=False)
            self.up = nn.Parameter(torch.cat([self.up.detach(), u]), requires_grad=False)
            self.down = nn.Parameter(torch.cat([self.down.detach(), dn]), requires_grad=False)
            self.ids = torch.cat([self.ids, torch.tensor(new, dtype=torch.long, device=self.ids.device)])
        self._index = None

    # --- learning time (same contract as aegist.model.CoreBank) ---------------------
    def begin_learning(self, cores: list[int], device):
        self.make_live(cores)
        ix = self.index()
        rows = torch.tensor([ix[c] for c in cores], dtype=torch.long, device=self.gate.device)
        self.work_ids = list(cores)
        self.work_rows = rows
        self.w_gate = nn.Parameter(self.gate.detach()[rows].to(device, torch.float32))
        self.w_up = nn.Parameter(self.up.detach()[rows].to(device, torch.float32))
        self.w_down = nn.Parameter(self.down.detach()[rows].to(device, torch.float32))
        return [self.w_gate, self.w_up, self.w_down]

    def end_learning(self, keep: bool = True):
        if self.work_ids is None:
            return
        if keep:
            with torch.no_grad():
                for w, m in ((self.w_gate, self.gate), (self.w_up, self.up), (self.w_down, self.down)):
                    m.index_copy_(0, self.work_rows.to(m.device), w.detach().to(m.device, m.dtype))
        for n in ("w_gate", "w_up", "w_down"):
            delattr(self, n)
        self.work_ids = None
        self.work_rows = None

    def forward(self, h, core_ids, weights):
        """h [N, D]; core_ids, weights [N, k] -> [N, D]; cores that are not live add zero."""
        out = torch.zeros_like(h)
        if self.ids.numel() == 0:
            return out
        ix = self.index()
        learning = self.work_ids is not None
        work = {c: i for i, c in enumerate(self.work_ids)} if learning else {}
        flat = core_ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = torch.arange(h.shape[0], device=h.device).repeat_interleave(core_ids.shape[1])[order]
        ww = weights.reshape(-1)[order]
        uniq, counts = torch.unique_consecutive(flat[order], return_counts=True)
        dt = h.dtype
        for c, t, wt in zip(uniq.tolist(), tok.split(counts.tolist()), ww.split(counts.tolist())):
            if c in work:
                i = work[c]
                g, u, dn = self.w_gate[i], self.w_up[i], self.w_down[i]
            elif c in ix:
                i = ix[c]
                g, u, dn = self.gate[i], self.up[i], self.down[i]
            else:
                continue
            x = h[t]
            z = F.silu(x @ g.to(dt)) * (x @ u.to(dt))
            out = out.index_add(0, t, (z @ dn.to(dt)) * wt.unsqueeze(-1).to(dt))
        return out

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # live-core tensors change size as cores are added: take the stored shapes
        for n in ("gate", "up", "down"):
            t = state_dict.get(prefix + n)
            if t is not None and t.shape != getattr(self, n).shape:
                setattr(self, n, nn.Parameter(torch.empty(t.shape, dtype=getattr(self, n).dtype,
                                                          device=getattr(self, n).device), requires_grad=False))
        t = state_dict.get(prefix + "ids")
        if t is not None and t.shape != self.ids.shape:
            self.ids = torch.empty(t.shape, dtype=torch.long, device=self.ids.device)
        self._index = None
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


# --- blocks and model --------------------------------------------------------------

class HybridBlock(nn.Module):
    def __init__(self, cfg: HybridConfig, layer: int):
        super().__init__()
        self.kind = cfg.layer_types[layer]
        self.lanes = cfg.hc_count > 1
        if self.lanes:
            self.hc_attn = HC.HyperConnection(cfg.d_model, cfg.hc_count, cfg.hc_lowrank, cfg.norm_eps)
            self.hc_mlp = HC.HyperConnection(cfg.d_model, cfg.hc_count, cfg.hc_lowrank, cfg.norm_eps)
            self.ple = HC.PLE(cfg, cfg.ple_layers.index(layer)) if layer in cfg.ple_layers else None
            self.ple_slot = cfg.ple_layers.index(layer) if layer in cfg.ple_layers else -1
        else:
            self.norm1 = ZNorm(cfg.d_model, cfg.norm_eps)
            self.norm2 = ZNorm(cfg.d_model, cfg.norm_eps)
            self.ple = None
        self.mixer = GatedAttention(cfg) if self.kind == "attn" else GatedDeltaNet(cfg)
        self.moe = MoE(cfg)
        self.engine = CoreEngine(cfg)
        self.cores = SparseCoreBank(cfg, layer)
        self.cores_enabled = True
        self.last_core_ids: torch.Tensor | None = None
        self.core_hits: torch.Tensor | None = None
        self.slot = 0  # index among blocks of the same kind (set by HybridLM)

    def forward(self, x, cos, sin, cache, ids=None):
        if self.lanes:
            if self.ple is not None:
                x = x + self.ple(x, ids, cache, self.ple_slot)
            h, x0, w = self.hc_attn(x)
        else:
            h = self.norm1(x)
        y = self.mixer(h, cos, sin, cache, self.slot) if self.kind == "attn" else self.mixer(h, cache, self.slot)
        if self.lanes:
            x = HC.inject(x0, y, w)
            h, x0, w_lanes = self.hc_mlp(x)
        else:
            x = x + y
            h = self.norm2(x)
        B, T, D = h.shape
        hf = h.reshape(-1, D)
        out = self.moe(hf)
        aux = x.new_zeros((), dtype=torch.float32)
        if self.cores_enabled:
            ids, w, aux = self.engine(hf)
            self.last_core_ids = ids.detach()
            if not self.training:
                if self.core_hits is None or self.core_hits.shape[0] != self.cores.n_cores or \
                        self.core_hits.device != ids.device:
                    self.core_hits = torch.zeros(self.cores.n_cores, dtype=torch.long, device=ids.device)
                flat = ids.reshape(-1)
                self.core_hits.index_add_(0, flat, torch.ones_like(flat))
            if self.training or not self.cores.silent:
                out = out + self.cores(hf, ids, w)
        if self.lanes:
            return HC.inject(x0, out.view(B, T, D), w_lanes), aux
        return x + out.view(B, T, D), aux


class HybridLM(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(HybridBlock(cfg, i) for i in range(cfg.n_layers))
        counters = {"attn": 0, "delta": 0}
        for b in self.blocks:
            b.slot = counters[b.kind]
            counters[b.kind] += 1
        if cfg.hc_count > 1:
            self.hc_out = HC.HyperConnection(cfg.d_model, cfg.hc_count, cfg.hc_lowrank, cfg.norm_eps, combine=False)
        else:
            self.norm = ZNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._rope: tuple | None = None

    def rope(self, length: int, device):
        if self._rope is None or self._rope[0].shape[0] < length or self._rope[0].device != torch.device(device):
            self._rope = rope_tables(self.cfg, min(max(length, 4096), self.cfg.max_seq_len), device)
        return self._rope

    def new_cache(self, batch: int, max_len: int, device=None, dtype=None) -> HybridCache:
        device = device or self.embed.weight.device
        return HybridCache(self.cfg, batch, max_len, device, dtype or self.embed.weight.dtype)

    def set_cores(self, enabled: bool):
        for b in self.blocks:
            b.cores_enabled = enabled

    def refresh_core_flags(self):
        pass

    def core_usage(self) -> torch.Tensor:
        dev = self.embed.weight.device
        rows = [b.core_hits if b.core_hits is not None and b.core_hits.shape[0] == self.cfg.n_cores
                else torch.zeros(self.cfg.n_cores, dtype=torch.long, device=dev) for b in self.blocks]
        return torch.stack([r.to(rows[0].device) for r in rows])

    def page_experts(self, masters: list[torch.Tensor], slots: int, device=None):
        device = device or self.embed.weight.device
        for b, m in zip(self.blocks, masters):
            b.moe.experts.page(m, slots, device)
        return self

    def logits(self, h):
        if self.lm_head is None:
            return F.linear(h, self.embed.weight.to(h.device, h.dtype))
        return self.lm_head(h)

    def hidden(self, idx, cache: HybridCache | None = None, checkpoint: bool = False):
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        if cache is not None:
            cache.reserve(start + T)
        cos, sin = self.rope(start + T, idx.device)
        x = self.embed(idx.to(self.embed.weight.device)).to(idx.device)
        lanes = self.cfg.hc_count > 1
        if lanes:
            x = x.repeat(1, 1, self.cfg.hc_count)
        aux = x.new_zeros((), dtype=torch.float32)
        for b in self.blocks:
            if checkpoint and torch.is_grad_enabled():
                x, a = torch.utils.checkpoint.checkpoint(b, x, cos, sin, cache, idx, use_reentrant=False)
            else:
                x, a = b(x, cos, sin, cache, idx)
            aux = aux + a
        if cache is not None:
            cache.length += T
        return (self.hc_out(x) if lanes else self.norm(x)), aux / len(self.blocks)

    CE_CHUNK = 256

    def forward(self, idx, targets=None, cache: HybridCache | None = None, last_only=False, checkpoint=False):
        h, aux = self.hidden(idx, cache, checkpoint)
        out = {"aux": aux}
        if targets is None:
            out["logits"] = self.logits(h[:, -1:] if last_only else h)
            return out
        keep = targets != -100
        hs, ts = h[keep], targets[keep]
        if hs.shape[0] == 0:
            ce = h.sum() * 0.0
        elif torch.is_grad_enabled() and hs.shape[0] > self.CE_CHUNK:
            ce = sum(torch.utils.checkpoint.checkpoint(self._ce_sum, a, b, use_reentrant=False)
                     for a, b in zip(hs.split(self.CE_CHUNK), ts.split(self.CE_CHUNK))) / hs.shape[0]
        else:
            ce = self._ce_sum(hs, ts) / hs.shape[0]
        out["ce"] = ce
        out["loss"] = ce + self.cfg.core_aux_coef * aux
        return out

    def _ce_sum(self, h, t):
        return F.cross_entropy(self.logits(h).float(), t, reduction="sum")

    def core_parameters(self):
        for b in self.blocks:
            yield from b.engine.parameters()
            yield from b.cores.parameters()

    def base_parameters(self):
        core = {id(p) for p in self.core_parameters()}
        return (p for p in self.parameters() if id(p) not in core)

    BIG_LINEARS = {"attn": ("q", "k", "v", "o"), "delta": ("in_qkv", "in_z", "out")}

    def quantize_base_int8(self):
        """int8 weight-only for the big always-on linears and the output head. Router, the small
        DeltaNet gates (in_a, in_b), the shared-expert gate and the embedding stay full precision."""
        for b in self.blocks:
            for n in self.BIG_LINEARS[b.kind]:
                lin = getattr(b.mixer, n)
                if isinstance(lin, nn.Linear):
                    setattr(b.mixer, n, QLinear(lin.weight.data))
            for n in ("gate", "up", "down"):
                lin = getattr(b.moe.shared, n)
                if isinstance(lin, nn.Linear):
                    setattr(b.moe.shared, n, QLinear(lin.weight.data))
        if isinstance(self.lm_head, nn.Linear):
            self.lm_head = QLinear(self.lm_head.weight.data)
        return self

    def param_report(self) -> dict:
        cfg = self.cfg
        per_core = 3 * cfg.d_model * cfg.core_rank
        experts = sum(b.moe.experts.numel() for b in self.blocks)
        engine = sum(p.numel() for b in self.blocks for p in b.engine.parameters())
        live = sum(b.cores.ids.numel() for b in self.blocks)
        qlin = sum(m.qweight.numel() for m in self.modules() if isinstance(m, QLinear))
        dense = sum(p.numel() for n, p in self.named_parameters()
                    if ".cores." not in n and ".engine." not in n and ".experts." not in n)
        base = dense + qlin + experts
        capacity = cfg.n_layers * cfg.core_rows_max * cfg.core_grid[1] * per_core
        active_experts = cfg.n_layers * cfg.experts_active * 3 * cfg.d_model * cfg.expert_hidden
        # a token reads one row per n-gram head of each table (ple_dim values), not the whole table
        table = sum(p.numel() for n, p in self.named_parameters() if ".ple.table." in n)
        active = base - experts + active_experts - table + len(cfg.ple_layers or []) * cfg.ple_dim
        return {"base_total": base, "base_active_per_token": active,
                "cores_trained": live * per_core, "cores_trained_count": live,
                "core_grid_now": cfg.n_layers * cfg.n_cores, "engine": engine,
                "total_stored": base + live * per_core + engine,
                "core_capacity": capacity, "total_capacity": base + capacity + engine}


# --- .nx -----------------------------------------------------------------------------

def hybrid_state(model: HybridLM) -> dict[str, torch.Tensor]:
    """Everything an .nx brain stores for a hybrid model (paged 4-bit experts included)."""
    state = dict(model.state_dict())
    key = "experts.fp8" if model.cfg.expert_format == "fp8" else "experts.q4"
    for i, b in enumerate(model.blocks):
        ex = b.moe.experts
        if not ex.dense:
            state[f"blocks.{i}.moe.{key}"] = ex.masters
        if b.ple is not None and b.ple.table is None:  # the file-backed n-gram table
            state[f"blocks.{i}.ple.table.weight"] = b.ple.file_table.view(HC.FP8)
            state[f"blocks.{i}.ple.table_scale"] = b.ple.file_scale
    return state


def load_hybrid(nx, device="cuda", slots: int = 32, train: bool = False, experts_in_ram: bool = False,
                dtype_override=None):
    """Load a hybrid-MoE brain from an open NxFile. Returns (model, tokenizer, nx).

    The always-on base, router and core engine go to `device`; experts stay in the file
    (memory-mapped) or, with `experts_in_ram`, in RAM, with `slots` per block cached on `device`.
    Live cores are loaded where the model is. With train=True the base is frozen and learning
    time trains working copies of cores (SparseCoreBank.begin_learning)."""
    from .nx import DTYPES
    from .tokenizer import AegistTokenizer

    h = nx.header
    cfg = HybridConfig.from_dict(h["config"])
    dtype = DTYPES[h.get("dtype", "bf16")]
    with torch.device("meta"):
        model = HybridLM(cfg).to(dtype)
        if h.get("quant") == "int8":
            model.quantize_base_int8()
        for b in model.blocks:
            for n in ("gate_up", "down"):
                b.moe.experts._parameters.pop(n, None)
            b.moe.experts.dense = False
    model = model.to_empty(device=device)
    expert_key = "experts.fp8" if cfg.expert_format == "fp8" else "experts.q4"
    file_tables = {n for n in h["tensors"] if n.endswith(".ple.table.weight") and h["tensors"][n]["dtype"] == "f8e4m3"}
    for n in file_tables:  # never materialised: rows are read from the file
        model.blocks[int(n.split(".")[1])].ple.use_file_table(nx.tensor(n), nx.tensor(n.replace(".weight", "_scale")))
    skip = lambda n: (n.endswith("." + expert_key) or n in file_tables or n.endswith(".ple.table_scale")
                      or n.startswith("extra."))
    sd = {n: nx.tensor(n) for n in h["tensors"] if not skip(n)}
    if dtype_override is not None:
        model = model.to(dtype_override)
        sd = {n: t.to(dtype_override) if t.is_floating_point() and t.dtype != HC.FP8 else t for n, t in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        nx.close()
        raise ValueError(f"{nx.path}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    for i, b in enumerate(model.blocks):
        m = nx.tensor(f"blocks.{i}.moe.{expert_key}")
        b.moe.experts.page(m.clone() if experts_in_ram else m, slots, device)
    model._nx = nx
    tok = AegistTokenizer.from_json_str(nx.blob("tokenizer.json").decode("utf-8"))
    model.train(train)
    if train:
        for p in model.base_parameters():
            p.requires_grad_(False)
    return model, tok, nx
