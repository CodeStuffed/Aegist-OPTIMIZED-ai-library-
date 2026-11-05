"""Aegist language model.

Every block has an always-on path and a switchable path:

    x = x + attn(norm(x))                  always on  ("reading": attention over the context)
    h = norm(x)
    x = x + ffn(h) + cores(h, engine(h))   ffn always on ("thinking"); cores switched per token

The embedding table reads tokens in and, tied, speaks them out. The core engine
is a product-key router (Lample et al. 2019, "Large Memory Layers with Product
Keys"): two small score tables of size n1 and n2 address n1*n2 cores, so picking
cores costs O(sqrt(N)) and N can grow to whatever the cores' storage allows.

Cores are low-rank SwiGLU experts whose output projection starts at zero, so a
freshly upcycled Aegist computes exactly what its base weights compute; the
cores only change behaviour once learning time trains them.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class AegistConfig:
    vocab_size: int = 151936
    d_model: int = 2560
    n_layers: int = 36
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    ffn_hidden: int = 9728
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 40960
    tie_embeddings: bool = True
    # switchable cores
    core_grid: tuple[int, int] = (8, 8)  # n1 x n2 cores per layer
    core_rank: int = 32
    cores_active: int = 2  # cores switched on per token per layer
    core_aux_coef: float = 0.01

    @property
    def n_cores(self) -> int:
        return self.core_grid[0] * self.core_grid[1]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["core_grid"] = list(self.core_grid)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AegistConfig":
        d = dict(d)
        if "core_grid" in d:
            d["core_grid"] = tuple(d["core_grid"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(x.dtype)


class QLinear(nn.Module):
    """Frozen int8 weight-only linear (per-output-channel absmax scale).

    Works on the meta device too, so a quantized model can be laid out without
    allocating and then filled straight from an .nx file."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        w = weight.float()
        scale = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer("qweight", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale.to(torch.bfloat16))
        self.out_features, self.in_features = weight.shape

    def forward(self, x):
        return F.linear(x, self.qweight.to(x.dtype) * self.scale.to(x.dtype))


def rope_tables(cfg: AegistConfig, length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device, dtype=torch.float32) / cfg.head_dim))
    f = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv)
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos.to(x.dtype) + rot * sin.to(x.dtype)


class KVCache:
    """Per-layer key/value cache that grows in blocks as the conversation does.

    Appending writes in place. When capacity runs out the buffers are reallocated
    at double the size (whole blocks, capped at `max_len`), so memory follows the
    context actually in use instead of the longest context allowed.
    """

    BLOCK = 1024

    def __init__(self, cfg: AegistConfig, batch: int, max_len: int, device, dtype):
        self.cfg, self.batch, self.device, self.dtype = cfg, batch, device, dtype
        self.max_len = max_len
        self.length = 0
        self.capacity = 0
        self.k = self.v = None
        self.reserve(min(max_len, self.BLOCK))

    def reserve(self, n: int):
        if n <= self.capacity:
            return
        if n > self.max_len:
            raise ValueError(f"context of {n} tokens exceeds this cache's limit of {self.max_len}")
        cap = max(n, 2 * self.capacity)
        cap = min(self.max_len, -(-cap // self.BLOCK) * self.BLOCK)
        shape = (self.cfg.n_layers, self.batch, self.cfg.n_kv_heads, cap, self.cfg.head_dim)
        k = torch.empty(shape, device=self.device, dtype=self.dtype)
        v = torch.empty(shape, device=self.device, dtype=self.dtype)
        if self.length:
            k[:, :, :, : self.length] = self.k[:, :, :, : self.length]
            v[:, :, :, : self.length] = self.v[:, :, :, : self.length]
        self.k, self.v, self.capacity = k, v, cap

    def crop(self, n: int):
        self.length = min(self.length, n)

    def nbytes(self) -> int:
        return 0 if self.k is None else 2 * self.k.numel() * self.k.element_size()


class Attention(nn.Module):
    def __init__(self, cfg: AegistConfig):
        super().__init__()
        self.cfg = cfg
        self.h, self.kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q = nn.Linear(cfg.d_model, self.h * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.norm_eps)
        self.k_norm = RMSNorm(self.hd, cfg.norm_eps)

    def forward(self, x, cos, sin, layer: int, cache: KVCache | None):
        B, T, _ = x.shape
        q = self.q_norm(self.q(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(B, T, self.kv, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        pos = cache.length if cache is not None else 0
        q, k = apply_rope(q, cos[pos : pos + T], sin[pos : pos + T]), apply_rope(k, cos[pos : pos + T], sin[pos : pos + T])
        if cache is not None:
            cache.k[layer, :, :, pos : pos + T] = k
            cache.v[layer, :, :, pos : pos + T] = v
            k, v = cache.k[layer, :, :, : pos + T], cache.v[layer, :, :, : pos + T]
        g = self.h // self.kv
        if T == 1:
            # Decode: fold the query heads of each KV group into the query axis, so the
            # cache is never expanded (no copy of the whole context per step).
            y = F.scaled_dot_product_attention(q.reshape(B, self.kv, g, self.hd), k, v)
            y = y.reshape(B, self.h, 1, self.hd)
        else:
            k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
            if pos == 0:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:  # chunk appended to an existing cache: query i sees cache + chunk[:i+1]
                mask = torch.ones(T, pos + T, dtype=torch.bool, device=x.device).tril(pos)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CoreEngine(nn.Module):
    """Product-key router: picks `cores_active` of n1*n2 cores per token."""

    def __init__(self, cfg: AegistConfig):
        super().__init__()
        self.n1, self.n2 = cfg.core_grid
        self.k = cfg.cores_active
        self.key1 = nn.Linear(cfg.d_model, self.n1, bias=False)
        self.key2 = nn.Linear(cfg.d_model, self.n2, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in (self.key1, self.key2):
            nn.init.normal_(lin.weight, std=lin.in_features ** -0.5)

    def forward(self, h):
        h = h.to(self.key1.weight.dtype)
        s1, s2 = self.key1(h).float(), self.key2(h).float()  # [N, n1], [N, n2]
        k1, k2 = min(self.k, self.n1), min(self.k, self.n2)
        v1, i1 = s1.topk(k1, -1)
        v2, i2 = s2.topk(k2, -1)
        cand = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(1)  # [N, k1*k2]
        ids = (i1.unsqueeze(-1) * self.n2 + i2.unsqueeze(-2)).flatten(1)
        top, pos = cand.topk(self.k, -1)
        core_ids = ids.gather(1, pos)
        weights = top.softmax(-1)
        if not self.training:
            return core_ids, weights, s1.new_zeros(())
        # Switch-style balance loss on each half of the key, so no row/column of the grid starves.
        p1, p2 = s1.softmax(-1), s2.softmax(-1)
        f1 = F.one_hot(core_ids // self.n2, self.n1).float().sum(1).mean(0) / self.k
        f2 = F.one_hot(core_ids % self.n2, self.n2).float().sum(1).mean(0) / self.k
        aux = self.n1 * (f1 * p1.mean(0)).sum() + self.n2 * (f2 * p2.mean(0)).sum()
        return core_ids, weights, aux


class CoreBank(nn.Module):
    """N low-rank SwiGLU cores stored as stacked tensors (one slice per core).

    Two storage modes:
    - resident (training): gate/up/down are Parameters on the compute device.
    - paged (inference): every core lives in host memory (pinned RAM, or a
      memory-mapped .nx file, so the bank can be far larger than the GPU), and
      only `slots` cores per layer sit on the GPU. A core is copied in when the
      engine first switches it on and evicted least-recently-used.
    """

    def __init__(self, cfg: AegistConfig):
        super().__init__()
        N, d, r = cfg.n_cores, cfg.d_model, cfg.core_rank
        self.n_cores, self.d, self.r = N, d, r
        self.gate = nn.Parameter(torch.empty(N, d, r))
        self.up = nn.Parameter(torch.empty(N, d, r))
        self.down = nn.Parameter(torch.empty(N, r, d))
        self.small_batch = 64  # at or below this many tokens, gather weights instead of dispatching
        self.paged = False
        self.silent = False  # True when every core's output projection is zero (set by refresh_silent)
        self.page_ins = 0
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.gate.normal_(std=self.d ** -0.5)
            self.up.normal_(std=self.d ** -0.5)
            self.down.zero_()  # silent until trained

    def masters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.m_gate, self.m_up, self.m_down) if self.paged else (self.gate, self.up, self.down)

    def numel(self) -> int:
        return sum(t.numel() for t in self.masters())

    def refresh_silent(self):
        if self.paged and getattr(self, "_silent_checked", False):
            return
        down = self.masters()[2]
        if down.device.type != "meta":
            self.silent = not bool(down.detach().any())

    # --- paging -----------------------------------------------------------
    def page(self, slots: int, device, masters=None, pin: bool = True, silent: bool | None = None):
        """Switch to paged storage. `masters` (gate, up, down) may be any host tensors,
        e.g. views into a memory-mapped .nx; by default the current weights are used.
        `silent`, when known (an .nx header records it), avoids scanning the masters."""
        if masters is None:
            masters = tuple(getattr(self, n).detach().to("cpu") for n in ("gate", "up", "down"))
        if pin and torch.cuda.is_available() and torch.device(device).type == "cuda":
            masters = tuple(m if m.is_pinned() else m.pin_memory() for m in masters)
        for n in ("gate", "up", "down"):
            if n in self._parameters:
                del self._parameters[n]
        self.m_gate, self.m_up, self.m_down = masters
        self.paged = True
        self.slots = max(1, min(slots, self.n_cores))
        dt = masters[0].dtype
        self.register_buffer("s_gate", torch.zeros(self.slots, self.d, self.r, device=device, dtype=dt), persistent=False)
        self.register_buffer("s_up", torch.zeros(self.slots, self.d, self.r, device=device, dtype=dt), persistent=False)
        self.register_buffer("s_down", torch.zeros(self.slots, self.r, self.d, device=device, dtype=dt), persistent=False)
        self.register_buffer("slot_of", torch.full((self.n_cores,), -1, device=device, dtype=torch.long), persistent=False)
        self.reset_slots()
        if silent is None:
            self._silent_checked = False
            self.refresh_silent()
        else:
            self.silent = silent
        self._silent_checked = True  # paged cores do not change, so this never needs rescanning
        return self

    def reset_slots(self):
        """Forget what is resident (after the slot buffers were (re)allocated)."""
        self.slot_of.fill_(-1)
        self._lru: OrderedDict[int, int] = OrderedDict()  # core -> slot, least recently used first
        self._free = list(range(self.slots))

    def unpage(self, device):
        """Back to resident Parameters (for learning time)."""
        if not self.paged:
            return self
        for n in ("gate", "up", "down"):
            self._parameters[n] = nn.Parameter(getattr(self, "m_" + n).to(device).clone())
            delattr(self, "m_" + n)
        for n in ("s_gate", "s_up", "s_down", "slot_of"):
            del self._buffers[n]
        self.paged = False
        self._silent_checked = False
        return self

    def _ensure(self, cores: list[int]):
        """Make `cores` resident (len(cores) <= slots). Copies are queued on the current
        stream, so a slot being overwritten is never read by a kernel queued after it."""
        need = set(cores)
        changed: list[tuple[int, int]] = []
        for c in cores:
            if c in self._lru:
                self._lru.move_to_end(c)
                continue
            if self._free:
                s = self._free.pop()
            else:
                old = next(o for o in self._lru if o not in need)
                s = self._lru.pop(old)
                changed.append((old, -1))
            for src, dst in ((self.m_gate, self.s_gate), (self.m_up, self.s_up), (self.m_down, self.s_down)):
                dst[s].copy_(src[c], non_blocking=True)
            self._lru[c] = s
            changed.append((c, s))
            self.page_ins += 1
        if changed:
            idx, val = zip(*changed)
            dev = self.slot_of.device
            self.slot_of[torch.tensor(idx, device=dev)] = torch.tensor(val, device=dev)

    def resident(self) -> list[int]:
        return list(self._lru) if self.paged else list(range(self.n_cores))

    # --- learning time ----------------------------------------------------
    def begin_learning(self, cores: list[int], device):
        """Make `cores` trainable as fp32 working copies on `device`; every other core
        stays frozen where it lives. Optimizer memory therefore scales with the working
        set, not with the size of the bank. Masters must be writable (not a file map)."""
        g, u, d = self.masters()
        idx = torch.tensor(cores, dtype=torch.long, device=g.device)
        self.work_ids = list(cores)
        self.work_index = {c: i for i, c in enumerate(cores)}
        # frozen cores that are still silent add exactly zero: skip them while learning
        self.live = set(d.detach().reshape(d.shape[0], -1).any(1).nonzero().flatten().tolist())
        self.w_gate = nn.Parameter(g.detach()[idx].to(device, torch.float32))
        self.w_up = nn.Parameter(u.detach()[idx].to(device, torch.float32))
        self.w_down = nn.Parameter(d.detach()[idx].to(device, torch.float32))
        return [self.w_gate, self.w_up, self.w_down]

    def end_learning(self, keep: bool = True):
        """Write the working copies back into the bank (or drop them with keep=False)."""
        if getattr(self, "work_ids", None) is None:
            return
        if keep:
            with torch.no_grad():
                for w, m in ((self.w_gate, self.masters()[0]), (self.w_up, self.masters()[1]),
                             (self.w_down, self.masters()[2])):
                    idx = torch.tensor(self.work_ids, dtype=torch.long, device=m.device)
                    m.index_copy_(0, idx, w.detach().to(m.device, m.dtype))
            if self.paged:
                self.reset_slots()  # resident copies of changed cores are stale
        for n in ("w_gate", "w_up", "w_down"):
            delattr(self, n)
        self.work_ids = None
        self._silent_checked = False
        self.refresh_silent()

    def _core_weights(self, e: int, device):
        if getattr(self, "work_ids", None) is not None:
            if e in self.work_index:
                i = self.work_index[e]
                return self.w_gate[i], self.w_up[i], self.w_down[i]
            # frozen core during learning: a fresh copy, so no later page-in can overwrite
            # a tensor autograd saved for the backward pass
            return tuple(m[e].detach().to(device, non_blocking=True) for m in self.masters())
        if self.paged:
            self._ensure([e])
            s = self._lru[e]
            return self.s_gate[s], self.s_up[s], self.s_down[s]
        return self.gate[e], self.up[e], self.down[e]

    # --- compute ----------------------------------------------------------
    @staticmethod
    def _gather(h, ids, weights, gate, up, down):
        dt = h.dtype
        g, u, dn = gate[ids].to(dt), up[ids].to(dt), down[ids].to(dt)
        z = F.silu(torch.einsum("nd,nkdr->nkr", h, g)) * torch.einsum("nd,nkdr->nkr", h, u)
        y = torch.einsum("nkr,nkrd->nkd", z, dn)
        return (y * weights.unsqueeze(-1).to(dt)).sum(1)

    def _dispatch(self, h, core_ids, weights):
        N, k = core_ids.shape
        dt = h.dtype
        flat = core_ids.reshape(-1)
        tok = torch.arange(N, device=h.device).repeat_interleave(k)
        w = weights.reshape(-1)
        order = torch.argsort(flat, stable=True)
        flat, tok, w = flat[order], tok[order], w[order]
        uniq, counts = torch.unique_consecutive(flat, return_counts=True)
        out = torch.zeros_like(h)
        counts = counts.tolist()
        learning = getattr(self, "work_ids", None) is not None
        for e, x, t, ww in zip(uniq.tolist(), h[tok].split(counts), tok.split(counts), w.split(counts)):
            if learning and e not in self.work_index and e not in self.live:
                continue
            g, u, dn = self._core_weights(e, h.device)
            z = F.silu(x @ g.to(dt)) * (x @ u.to(dt))
            out = out.index_add(0, t, (z @ dn.to(dt)) * ww.unsqueeze(-1).to(dt))
        return out

    def forward(self, h, core_ids, weights):
        N = core_ids.shape[0]
        if getattr(self, "work_ids", None) is not None:
            return self._dispatch(h, core_ids, weights)
        if not self.paged:
            if N <= self.small_batch:
                return self._gather(h, core_ids, weights, self.gate, self.up, self.down)
            return self._dispatch(h, core_ids, weights)
        if N <= self.small_batch:
            need = core_ids.unique().tolist()
            if len(need) <= self.slots:
                self._ensure(need)
                return self._gather(h, self.slot_of[core_ids], weights, self.s_gate, self.s_up, self.s_down)
        return self._dispatch(h, core_ids, weights)


class Block(nn.Module):
    def __init__(self, cfg: AegistConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.engine = CoreEngine(cfg)
        self.cores = CoreBank(cfg)
        self.cores_enabled = True
        self.last_core_ids: torch.Tensor | None = None
        self.core_hits: torch.Tensor | None = None  # inference-time switch-on counts per core (on device, no sync)

    def forward(self, x, cos, sin, layer, cache):
        x = x + self.attn(self.attn_norm(x), cos, sin, layer, cache)
        h = self.ffn_norm(x)
        out = self.ffn(h)
        aux = x.new_zeros((), dtype=torch.float32)
        if self.cores_enabled:
            B, T, D = h.shape
            hf = h.reshape(-1, D)
            ids, w, aux = self.engine(hf)
            self.last_core_ids = ids.detach()
            if not self.training:
                if self.core_hits is None or self.core_hits.device != ids.device:
                    self.core_hits = torch.zeros(self.cores.n_cores, dtype=torch.long, device=ids.device)
                flat = ids.reshape(-1)
                self.core_hits.index_add_(0, flat, torch.ones_like(flat))
            # A silent bank (all output projections zero) adds exactly zero; skip its compute
            # at inference. Training always runs it so the output projections get gradients.
            if self.training or not self.cores.silent:
                out = out + self.cores(hf, ids, w).view(B, T, D)
        return x + out, aux


class AegistLM(nn.Module):
    def __init__(self, cfg: AegistConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._rope: tuple | None = None

    def rope(self, length: int, device):
        if self._rope is None or self._rope[0].shape[0] < length or self._rope[0].device != device:
            n = max(length, 4096)
            self._rope = rope_tables(self.cfg, min(max(n, length), self.cfg.max_seq_len), device)
        return self._rope

    def set_cores(self, enabled: bool):
        for b in self.blocks:
            b.cores_enabled = enabled

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.refresh_core_flags()
        return self

    def refresh_core_flags(self):
        for b in self.blocks:
            b.cores.refresh_silent()

    def page_cores(self, slots: int = 16, device=None, pin: bool = True):
        """Inference: keep all cores in host memory and `slots` per layer on `device`."""
        device = device or self.embed.weight.device
        for b in self.blocks:
            b.cores.page(slots, device, pin=pin)
        return self

    def unpage_cores(self, device=None):
        device = device or self.embed.weight.device
        for b in self.blocks:
            b.cores.unpage(device)
        return self

    def core_usage(self) -> torch.Tensor:
        """[n_layers, n_cores] counts of how often the engine switched each core on."""
        rows = [b.core_hits if b.core_hits is not None else torch.zeros(self.cfg.n_cores, dtype=torch.long,
                device=self.embed.weight.device) for b in self.blocks]
        return torch.stack([r.to(rows[0].device) for r in rows])

    def logits(self, h):
        w = self.embed.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(h, w.to(h.dtype))

    def hidden(self, idx, cache: KVCache | None = None, checkpoint: bool = False):
        """Final normed hidden states [B, T, D] and the mean core-balance loss.
        `checkpoint` recomputes each block during backward (learning time on a small GPU)."""
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        if cache is not None:
            cache.reserve(start + T)
        cos, sin = self.rope(start + T, idx.device)
        x = self.embed(idx)
        aux = x.new_zeros((), dtype=torch.float32)
        for i, b in enumerate(self.blocks):
            if checkpoint and torch.is_grad_enabled():
                x, a = torch.utils.checkpoint.checkpoint(b, x, cos, sin, i, cache, use_reentrant=False)
            else:
                x, a = b(x, cos, sin, i, cache)
            aux = aux + a
        if cache is not None:
            cache.length += T
        return self.norm(x), aux / len(self.blocks)

    def forward(self, idx, targets=None, cache: KVCache | None = None, last_only=False, checkpoint=False):
        h, aux = self.hidden(idx, cache, checkpoint)
        out = {"aux": aux}
        if targets is None:
            out["logits"] = self.logits(h[:, -1:] if last_only else h)
            return out
        # Only scored positions get logits: a 151k-entry row per token is the biggest tensor in training.
        keep = targets != -100
        hs, ts = h[keep], targets[keep]
        if hs.shape[0] == 0:
            ce = h.sum() * 0.0
        elif torch.is_grad_enabled() and hs.shape[0] > self.CE_CHUNK:
            # chunked + recomputed in backward, so the full logits never exist at once
            ce = sum(torch.utils.checkpoint.checkpoint(self._ce_sum, a, b, use_reentrant=False)
                     for a, b in zip(hs.split(self.CE_CHUNK), ts.split(self.CE_CHUNK))) / hs.shape[0]
        else:
            ce = self._ce_sum(hs, ts) / hs.shape[0]
        out["ce"] = ce
        out["loss"] = ce + self.cfg.core_aux_coef * aux
        return out

    CE_CHUNK = 256

    def _ce_sum(self, h, t):
        return F.cross_entropy(self.logits(h).float(), t, reduction="sum")

    def core_parameters(self):
        for b in self.blocks:
            yield from b.engine.parameters()
            yield from b.cores.parameters()

    def base_parameters(self):
        core = {id(p) for p in self.core_parameters()}
        return (p for p in self.parameters() if id(p) not in core)

    def quantize_base_int8(self):
        """Swap the frozen base's big linears for int8 weight-only versions (halves their memory)."""
        for b in self.blocks:
            for parent, names in ((b.attn, ("q", "k", "v", "o")), (b.ffn, ("gate", "up", "down"))):
                for n in names:
                    lin = getattr(parent, n)
                    if isinstance(lin, nn.Linear):
                        setattr(parent, n, QLinear(lin.weight.data))
        return self

    def param_report(self) -> dict:
        cores = sum(b.cores.numel() + sum(p.numel() for p in b.engine.parameters()) for b in self.blocks)
        paged = sum(b.cores.numel() for b in self.blocks if b.cores.paged)
        total = sum(p.numel() for p in self.parameters()) + paged + sum(
            m.qweight.numel() for m in self.modules() if isinstance(m, QLinear))
        per_core = 3 * self.cfg.d_model * self.cfg.core_rank
        return {"total": total, "base_always_on": total - cores, "cores_total": cores,
                "cores_active_per_token": self.cfg.n_layers * self.cfg.cores_active * per_core,
                "n_cores": self.cfg.n_layers * self.cfg.n_cores}
