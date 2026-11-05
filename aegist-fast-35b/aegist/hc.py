"""The parts of Aegist's second language core (from Qwen3.8-Flash-Next) that the first did not have.

    hyper-connections   the residual stream is hc_count lanes wide. Before each token mixer and each
                        mixture of experts, a learned low-rank mix of the normalised lanes forms the
                        sublayer's input; its output is injected back into every lane with learned weights.
    PLE                 per-layer n-gram embeddings: the last 1-2 tokens are hashed into large embedding
                        tables (8 heads per n-gram order), projected, gated by the lanes, and passed
                        through a short dilated convolution. Lookups are a few rows per token, so the
                        table can stay on disk.
    QSA indexer         sparse attention for long contexts: keys are averaged in blocks of
                        `idx_ratio` tokens, a small multi-head scorer picks the best `idx_budget / idx_ratio`
                        blocks for each query, and the query attends to those plus the incomplete tail.
                        Below idx_budget tokens of context every block is picked (ordinary attention).
    FP8 weights         float8 e4m3 with one float32 scale per 128 x 128 block, kept as released.

Each module mirrors the reference implementation (transformers' qwen4_exp) operation for operation;
tests/test_hc.py checks the whole model against it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8 = torch.float8_e4m3fn
BLOCK = 128


# --- norms -------------------------------------------------------------------------

class GroupZNorm(nn.Module):
    """Zero-centred RMSNorm applied to each `group`-wide slice separately (one weight for all)."""

    def __init__(self, dim: int, group: int, eps: float):
        super().__init__()
        self.group, self.eps = group, eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = x.float()
        g = xf.reshape(*xf.shape[:-1], -1, self.group)
        g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + self.eps)
        return (g.flatten(-2) * (1.0 + self.weight.float())).type_as(x)


# --- hyper-connections -------------------------------------------------------------

class HyperConnection(nn.Module):
    """Gated residual over hc lanes. forward(x [..., hc*D]) -> (input [..., D], x, inject [..., hc]);
    with combine=False (the model's last one) only the mixed input."""

    def __init__(self, d: int, hc: int, rank: int, eps: float, combine: bool = True):
        super().__init__()
        self.d, self.hc = d, hc
        self.norm = GroupZNorm(hc * d, d, eps)
        self.down = nn.Linear(hc * d, rank, bias=False)
        self.up = nn.Linear(rank, hc * d, bias=False)
        self.inject = nn.Linear(hc * d, hc, bias=False) if combine else None

    def forward(self, x):
        xn = self.norm(x)
        w = torch.sigmoid(self.up(F.silu(self.down(xn) / self.hc)))
        mixed = (w.unflatten(-1, (self.hc, self.d)) * xn.unflatten(-1, (self.hc, self.d))).mean(-2)
        if self.inject is None:
            return mixed
        return mixed, x, 2 * torch.sigmoid(self.inject(xn) / self.hc)


def inject(x, y, weights):
    """x [..., hc*D] + y [..., D] times each lane's weight [..., hc]."""
    return x + (y.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)


# --- PLE: hashed n-gram embeddings --------------------------------------------------

_MASK64 = (1 << 64) - 1
_GAMMA, _M1, _M2, _PRIME_1 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 10007


def _splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def ngram_multipliers(vocab: int, n: int, ple_index: int, seed: int) -> list[int]:
    half = max(1, (((1 << 63) - 1) // max(vocab, 1)) // 2)
    base = seed + _PRIME_1 * ple_index
    return [2 * (_splitmix64((base + _GAMMA * (i + 1)) & _MASK64) % half) + 1 for i in range(n)]


def _is_prime(v: int) -> bool:
    if v < 2 or v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def ngram_head_sizes(base: int, heads: int, ple_index: int) -> list[int]:
    """Each head's table size: the (global head index + 1)-th prime after base - 1."""
    out, p, want = [], base - 1, ple_index * heads
    k = 0
    while len(out) < heads:
        p += 1
        while not _is_prime(p):
            p += 1
        if k >= want:
            out.append(p)
        k += 1
    return out


class PLE(nn.Module):
    """Per-layer embedding for one block; adds [..., hc*D] to the lanes."""

    def __init__(self, cfg, ple_index: int):
        super().__init__()
        self.d, self.hc = cfg.d_model, cfg.hc_count
        self.n = cfg.ngram_size
        self.ctx = self.n - 1
        self.heads = (self.n - 1) * cfg.heads_per_ngram
        self.per_ngram = cfg.heads_per_ngram
        self.eos = cfg.eos_id
        sizes = ngram_head_sizes(cfg.ngram_vocab_base, self.heads, ple_index)
        offsets = [sum(sizes[:i]) for i in range(len(sizes))]
        total = -(-sum(sizes) // cfg.ngram_divisor) * cfg.ngram_divisor
        # hash constants: recomputed, never stored (and so never left uninitialised by a meta-device load)
        self._hash = (ngram_multipliers(cfg.vocab_size, self.n, ple_index, cfg.ngram_seed), sizes, offsets)
        self._hash_t: tuple | None = None
        self.table = nn.Embedding(total, cfg.ple_dim // self.heads)
        hcd = self.hc * self.d
        self.key = nn.Linear(cfg.ple_dim, hcd, bias=False)
        self.value = nn.Linear(cfg.ple_dim, self.d, bias=False)
        self.norm_key = GroupZNorm(hcd, self.d, cfg.norm_eps)
        self.norm_query = GroupZNorm(hcd, self.d, cfg.norm_eps)
        self.norm_conv = GroupZNorm(hcd, self.d, cfg.norm_eps)
        self.K, self.dil = cfg.ple_conv_kernel, self.n
        self.state_len = (self.K - 1) * self.dil
        self.conv = nn.Parameter(torch.zeros(hcd, self.K))

    def hash_tables(self, device="cpu"):
        device = torch.device(device)
        if self._hash_t is None or self._hash_t[0].device != device:
            self._hash_t = tuple(torch.tensor(x, dtype=torch.long, device=device) for x in self._hash)
        return self._hash_t

    @property
    def mult(self):
        return self.hash_tables()[0]

    @property
    def sizes(self):
        return self.hash_tables()[1]

    @property
    def offsets(self):
        return self.hash_tables()[2]

    def _shift(self, ids, s):
        """ids shifted right by s within each eos-delimited segment; eos where there is no token."""
        if s == 0:
            return ids
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        eos_pos = torch.where(ids == self.eos, pos, -1)
        prev_incl = torch.cummax(eos_pos, 1).values
        prev = torch.cat([eos_pos.new_full((B, 1), -1), prev_incl[:, :-1]], 1)
        in_seg = pos.unsqueeze(0) - (prev + 1)
        src = pos - s
        shifted = ids.gather(1, src.clamp_min(0).unsqueeze(0).expand(B, -1))
        ok = (in_seg >= s) & (src.unsqueeze(0) >= 0)
        return torch.where(ok, shifted, ids.new_full((), self.eos))

    def rows(self, ids, history):
        """Table rows [B, T, heads] for ids [B, T] after `history` [B, ctx] earlier tokens."""
        full = torch.cat([history, ids], -1).long()
        mult, sizes, offsets = self.hash_tables(full.device)
        sh = [self._shift(full, s) for s in range(self.n)]
        blocks = []
        for n in range(2, self.n + 1):
            a = (n - 2) * self.per_ngram
            mixed = sh[0] * mult[0]
            for p in range(1, n):
                mixed = torch.bitwise_xor(mixed, sh[p] * mult[p])
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes[a:a + self.per_ngram].view(1, 1, -1))
                          + offsets[a:a + self.per_ngram].view(1, 1, -1))
        return torch.cat(blocks, -1)[:, -ids.shape[1]:]

    def use_file_table(self, table: torch.Tensor, scale: torch.Tensor):
        """Read rows from `table` (FP8 [rows, dim], e.g. memory-mapped from the brain) instead of
        holding it: each token needs only (n-1) * heads_per_ngram rows."""
        self.table = None  # drops the in-memory table
        self.file_table, self.file_scale = table.view(torch.uint8), scale

    def embed(self, rows):
        if self.table is None:
            got = self.file_table[rows.reshape(-1).cpu()].to(rows.device).view(FP8)
            dt = self.key.weight.dtype
            e = got.to(dt) * self.file_scale.to(rows.device, dt)  # the reference rescales gathered rows
            return e.view(*rows.shape, -1).flatten(-2)
        w = self.table.weight
        return self.table(rows.to(w.device)).to(rows.device).flatten(-2)

    def forward(self, x, ids, cache, slot: int):
        B, T = ids.shape
        if cache is not None and cache.length > 0:
            history = cache.ple_hist[slot]
        else:
            history = ids.new_full((B, self.ctx), self.eos)
        if cache is not None:
            cache.ple_hist[slot] = torch.cat([history, ids.long()], -1)[:, -self.ctx:]
        e = self.embed(self.rows(ids, history)).to(x.dtype)
        k = self.norm_key(self.key(e)).unflatten(-1, (self.hc, self.d))
        v = self.value(e)
        q = self.norm_query(x).unflatten(-1, (self.hc, self.d))
        gate = (k * q).sum(-1, keepdim=True) / math.sqrt(self.d)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = torch.sigmoid(gate) * v.unsqueeze(-2)
        gvn = self.norm_conv(gv.flatten(-2))
        gv = gv.flatten(-2)
        h = gvn.transpose(1, 2)  # [B, C, T]
        if cache is not None and cache.length > 0:
            prev = cache.ple_conv[slot].to(h.dtype)
        else:
            prev = h.new_zeros(B, h.shape[1], self.state_len)
        hp = torch.cat([prev, h], -1)
        if cache is not None:
            cache.ple_conv[slot] = hp[..., -self.state_len:].clone()
        y = F.silu(F.conv1d(hp, self.conv.to(h.dtype).unsqueeze(1), groups=h.shape[1], dilation=self.dil))
        return gv + y.transpose(1, 2)


# --- QSA indexer -------------------------------------------------------------------

class QSAIndexer(nn.Module):
    """Which keys each query may attend to (a boolean mask), for one attention block."""

    def __init__(self, cfg):
        super().__init__()
        self.h, self.d = cfg.idx_heads, cfg.idx_head_dim
        self.budget, self.r = cfg.idx_budget, cfg.idx_ratio
        self.top = self.budget // self.r
        self.qk = nn.Linear(cfg.d_model, (self.h + 1) * self.d, bias=False)
        from .hybrid import ZNorm
        self.q_norm = ZNorm(self.d, cfg.norm_eps)
        self.k_norm = ZNorm(self.d, cfg.norm_eps)

    def forward(self, x, cos, sin, cache, slot: int, pos: int):
        """x [B, T, D] at positions pos..pos+T-1. Returns None (every earlier key is visible) or a
        bool mask [B, 1, T, pos+T]."""
        from .hybrid import apply_partial_rope
        B, T, _ = x.shape
        q, k = self.qk(x).split([self.h * self.d, self.d], -1)
        if cache is not None:
            cache.idx_k[slot, :, pos:pos + T] = k
            keys = cache.idx_k[slot, :, :pos + T]
        else:
            keys = k
        L = pos + T
        if L // self.r <= self.top:  # every complete block is picked for every query
            return None
        q = apply_partial_rope(self.q_norm(q.view(B, T, self.h, self.d)), cos[pos:L].unsqueeze(1), sin[pos:L].unsqueeze(1))
        nb = L // self.r
        pooled = keys[:, :nb * self.r].float().view(B, nb, self.r, self.d).mean(2).to(keys.dtype)
        starts = torch.arange(nb, device=x.device) * self.r
        bk = apply_partial_rope(self.k_norm(pooled), cos[starts], sin[starts])
        scores = torch.relu(torch.einsum("bthd,bnd->btnh", q.float(), bk.float())).sum(-1) / math.sqrt(self.d)
        p = torch.arange(pos, L, device=x.device)
        complete = (p + 1) // self.r  # complete blocks visible to each query
        scores = scores.masked_fill(torch.arange(nb, device=x.device).view(1, 1, -1) >= complete.view(1, -1, 1),
                                    float("-inf"))
        k_top = min(self.top, nb)
        top = scores.topk(k_top, -1).indices  # [B, T, k]
        chosen = torch.zeros(B, T, nb, dtype=torch.bool, device=x.device)
        chosen.scatter_(-1, top, True)
        chosen &= torch.isfinite(scores)
        tok = chosen.repeat_interleave(self.r, -1)  # [B, T, nb*r]
        tail = torch.arange(L, device=x.device).view(1, 1, -1) >= (complete * self.r).view(1, -1, 1)
        causal = torch.arange(L, device=x.device).view(1, -1) <= p.view(-1, 1)
        mask = F.pad(tok, (0, L - nb * self.r)) | tail
        return (mask & causal).unsqueeze(1)


# --- FP8 ---------------------------------------------------------------------------

def fp8_dequant(w: torch.Tensor, scale: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """w float8 [out, in] with scale [ceil(out/128), ceil(in/128)] -> dtype."""
    o, i = w.shape
    s = scale.float().repeat_interleave(BLOCK, 0)[:o].repeat_interleave(BLOCK, 1)[:, :i]
    return (w.float() * s).to(dtype)


def fp8_quant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For tests: w [out, in] -> (float8, scale) with the released checkpoints' block layout."""
    o, i = w.shape
    bo, bi = -(-o // BLOCK), -(-i // BLOCK)
    p = F.pad(w.float(), (0, bi * BLOCK - i, 0, bo * BLOCK - o)).view(bo, BLOCK, bi, BLOCK)
    s = p.abs().amax((1, 3)).clamp(min=1e-12) / torch.finfo(FP8).max
    q = (p / s[:, None, :, None]).view(bo * BLOCK, bi * BLOCK)[:o, :i].to(FP8)
    return q, s


class F8Linear(nn.Module):
    """A linear layer kept in FP8 with block scales; dequantised on the fly."""

    def __init__(self, out_f: int, in_f: int, device=None):
        super().__init__()
        self.register_buffer("weight", torch.empty(out_f, in_f, dtype=FP8, device=device))
        self.register_buffer("scale", torch.ones(-(-out_f // BLOCK), -(-in_f // BLOCK), device=device))

    def forward(self, x):
        return F.linear(x, fp8_dequant(self.weight, self.scale, x.dtype))


def expert_layout_fp8(cfg) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
    """One FP8 expert's bytes: gate_up [2I, D] (gate rows, then up rows) and down [D, I] in float8,
    then their bf16 block scales: the released bytes, unchanged."""
    I, D = cfg.expert_hidden, cfg.d_model
    assert I % BLOCK == 0, "gate and up must each fill whole 128-row blocks to share one scale grid"
    b = lambda n: -(-n // BLOCK)
    return [("gu", (2 * I, D), FP8), ("dn", (D, I), FP8),
            ("gu_s", (b(2 * I), b(D)), torch.bfloat16), ("dn_s", (b(D), b(I)), torch.bfloat16)]
