# Step 04: Reading core

Gain: 80% of fresh prompts read right, against 36% for a keyword rule (2.2×), for 180 ms of CPU time and no GPU.

Builds on step 02: its answer sets the engine's sampling for each reply.

Previous: [step 03, VRAM expert cache](../03-vram-expert-cache/). Next: [step 05, Confidence probe](../05-confidence-probe/). Main README: [../README.md](../README.md).

Laya (convaiinnovations/laya, Apache-2.0, 421M parameters) scores options; it doesn't write text.
`laya.rs` rebuilds its ModernBERT-large encoder and head in Rust with AVX2, so it runs on the CPU and
takes nothing from the GPU. Before every reply it answers "Is this request asking for a fact, something
creative, or other help?", and the answer sets the reply's settings:

| read as | sampling | unsure marks |
|---|---|---|
| fact | temperature 0.3 at most, greedy after the first unsure sentence | yes |
| creative | temperature 1.0, top-p 0.95 | none |
| other | the usual settings | yes |

It reads a prompt in about 180 ms and matches the Python reference's token ids and pick, with a largest
probability gap of 1.1e-5. On 45 prompts written after the reader was fixed, it read 80% right (fact
80%, creative 93%, other 67%) and never mixed up fact and creative. A keyword rule got 36% of the same
prompts right.

Laya can't check facts: on 80 true/false claims it scored an AUROC of 0.55-0.62. It decides what a
prompt asks for and never judges whether an answer is true.

Code: [`aegist_rs/src/laya.rs`](../aegist_rs/src/laya.rs), [`aegist/laya.py`](../aegist/laya.py),
[`scripts/eval_prompt_reader.py`](../scripts/eval_prompt_reader.py), [`scripts/eval_laya.py`](../scripts/eval_laya.py).
