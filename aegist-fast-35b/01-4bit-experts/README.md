# Step 01: 4-bit experts

Gain: 3.3× smaller: about 69 GB at bf16 (computed from 34.66B parameters) down to a 21.1 GB file, for a mean KL of 0.0079 per token.

Builds on the Qwen3.6-35B-A3B checkpoint. Everything after this runs on the 4-bit file.

Previous: none, this is the first step. Next: [step 02, Rust and CUDA engine](../02-rust-cuda-engine/). Main README: [../README.md](../README.md).

The converter reads the Qwen checkpoint once and writes one `.nx` file:

- routed experts in 4 bits: asymmetric, groups of 64 with an f16 scale and minimum, least-squares refit.
  That is about 4.5 bits per weight, with a mean relative weight error of 0.086.
- big linears (attention, DeltaNet projections, shared expert, head) in int8, one scale per output channel.
- the embedding and small tensors in bf16.
- the tokenizer, licence text, persona, generation defaults and lineage, inside the same file.

At this size most experts fit on a 16 GB card, which step 03 depends on.

Measured against the original bf16 checkpoint on 4 texts of 384 tokens:

| what | result |
|---|---|
| mean KL per token | 0.0079 |
| top-1 agreement | 87.8-100% |
| file size | 21.1 GB |

```powershell
python -I scripts/make_aegist_moe.py --base D:/aegist/models/qwen3.6-35b-a3b --out D:/aegist/brains/aegist-nx-35b.nx
python -I scripts/check_moe_quality.py
```

The second command repeats the KL check.

Code: [`scripts/make_aegist_moe.py`](../scripts/make_aegist_moe.py), [`aegist/convert_moe.py`](../aegist/convert_moe.py)
(`build_brain`), [`aegist/nx.py`](../aegist/nx.py) (the file format).
