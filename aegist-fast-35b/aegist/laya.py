"""Laya as an optional judge, loaded with our own code.

Laya (convaiinnovations/laya on Hugging Face, Apache-2.0, by Convai Innovations) is a 421M-parameter
decision model. ModernBERT-large reads one sequence holding a question, its options and a context
("state"), and a small head scores each option. It answers three kinds of question:
  choice  pick one of several options        -> probabilities over the options
  score   pick a level on a scale             -> probabilities over the levels
  noul    is a statement true?                -> P(true)
It does not write text. This module rebuilds its head here and loads the published weights into
it (the encoder is transformers' own ModernBERT); none of Laya's Python files are used. The
sequence layout follows Laya's published format: [CLS] "<type> question: <instructions>" [SEP],
then [MASK] + text for each option, [SEP], then the state and [SEP]. Each option's [MASK]
position is scored, and the scores are divided by Laya's fitted temperature before the softmax.

    from aegist.laya import Laya
    j = Laya("D:/aegist/models/laya")
    j.choice("Which city is the capital of Peru?", ["Lima", "Quito", "Bogota"])     # {option: p}
    j.noul("Is this statement true?", state="water expands when it freezes")       # P(true)
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
MASK = "[MASK]"


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{QTYPE_NAMES[qtype]}:{size}"


def render_options(t: str, crit) -> list[str]:
    if t == "choice":
        return [k if not v else f"{k}: {v}" for k, v in crit.items()]
    if t == "score":
        return [f"level {i}: {c}" for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


class LayaNet(nn.Module):
    """ModernBERT encoder, a type embedding added at every position, two transformer layers, an
    option scorer at each [MASK] and an act head on [CLS]. Parameter names match the release."""

    def __init__(self, enc_cfg, head_layers: int, n_act: int):
        super().__init__()
        from transformers import ModernBertModel

        d = enc_cfg.hidden_size
        self.encoder = ModernBertModel(enc_cfg)
        self.type_emb = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, 0.0, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, ids, attn, markers, marker_mask, qtype):
        h = self.encoder(input_ids=ids, attention_mask=attn).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        pad = ~attn.bool()
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        m = torch.gather(h, 1, markers.clamp(min=0)[..., None].expand(-1, -1, h.shape[-1]))
        logits = self.scorer(m).squeeze(-1).float().masked_fill(~marker_mask, -1e4)
        p = logits.detach().softmax(-1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        top2 = p.topk(2, -1).values
        ent = -(p * p.clamp(min=1e-12).log()).sum(-1) / k.log()
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act = self.act_head(torch.cat([h[:, 0].float(), feats], -1).to(h.dtype))
        return logits, act.float()


class Laya:
    def __init__(self, model_dir: str, device: str | None = None):
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        from transformers import ModernBertConfig

        with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        enc_cfg = ModernBertConfig.from_json_file(os.path.join(model_dir, "encoder", "config.json"))
        enc_cfg._attn_implementation = "sdpa"
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer", "tokenizer.json"))
        self.cls, self.sep, self.mask = (self.tok.token_to_id(t) for t in ("[CLS]", "[SEP]", MASK))
        n_act = len(self.cfg.get("act_costs", {})) + 1
        self.net = LayaNet(enc_cfg, self.cfg["head_layers"], n_act)
        self.net.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.net.to(self.device, self.dtype).eval()
        self.temperature = [float(t) for t in self.cfg.get("temperature", [1.0, 1.0, 1.0])]
        self.temp_by_options = self.cfg.get("temperature_by_options", {})

    def _enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    @staticmethod
    def _plain(text: str) -> str:
        """Text from the user cannot add Laya's own markers (the Rust reader does the same)."""
        for m in ("[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"):
            text = text.replace(m, " ")
        return text

    def build(self, t: str, instructions: str, crit, state, truncate_left: bool = False):
        max_len, head_max = self.cfg["max_len"], self.cfg["head_max_len"]
        opts = render_options(t, crit)
        head = self._enc(f"{t} question: {instructions}")
        opt_ids = [[self.mask] + self._enc(" " + self._plain(o))[:48] for o in opts]
        budget = head_max - sum(map(len, opt_ids))
        if budget < 16:  # long options: each gets an equal share, at least 4 tokens
            share = max(4, (head_max - 16) // len(opt_ids))
            opt_ids = [o[:share] for o in opt_ids]
            budget = head_max - sum(map(len, opt_ids))
        ids = [self.cls] + head[:max(8, budget)] + [self.sep]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids += o
        ids.append(self.sep)
        st = self._enc(self._plain(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)))
        room = max(0, max_len - len(ids) - 1)
        st = (st[-room:] if truncate_left else st[:room]) if room else []
        ids += st + [self.sep]
        markers = [m for m in markers if m < max_len]
        if len(markers) != len(opts):
            raise ValueError("the options do not fit in Laya's head")
        return ids[:max_len], markers

    @torch.no_grad()
    def run(self, items: list[tuple[str, str, object, object]], batch: int = 16) -> list[tuple[list[float], float]]:
        """items: (type, instructions, criteria, state). Returns (option probabilities, act p[0]) each."""
        out = []
        for a in range(0, len(items), batch):
            part = items[a:a + batch]
            seqs = [self.build(*it) for it in part]
            n, k = max(len(s) for s, _ in seqs), max(len(m) for _, m in seqs)
            ids = torch.zeros(len(seqs), n, dtype=torch.long)
            attn = torch.zeros(len(seqs), n, dtype=torch.long)
            mk = torch.full((len(seqs), k), -1, dtype=torch.long)
            for i, (s, m) in enumerate(seqs):
                ids[i, :len(s)], attn[i, :len(s)], mk[i, :len(m)] = torch.tensor(s), 1, torch.tensor(m)
            qt = torch.tensor([QTYPES[t] for t, *_ in part])
            dev = lambda x: x.to(self.device)  # noqa: E731
            logits, act = self.net(dev(ids), dev(attn), dev(mk), dev(mk >= 0), dev(qt))
            for i, (_, m) in enumerate(seqs):
                q, kk = int(qt[i]), len(m)
                z = logits[i, :kk] / self.temp_by_options.get(temp_bucket(q, kk), self.temperature[q])
                out.append((z.softmax(-1).tolist(), float(act[i].softmax(-1)[0])))
        return out

    def choice(self, question: str, options: list[str], state="") -> dict[str, float]:
        p, _ = self.run([("choice", question, {o: None for o in options}, state)])[0]
        return dict(zip(options, p))

    def noul(self, question: str, state="", crit=None) -> float:
        p, _ = self.run([("noul", question, crit, state)])[0]
        return p[1]
