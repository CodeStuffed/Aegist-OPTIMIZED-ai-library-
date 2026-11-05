# Aegist fast brain: Qwen3.6-35B-A3B on one 16 GB GPU

Aegist is a local chat AI. This repo holds its fast brain: a 35B mixture-of-experts model with 3.45B
parameters active per token, running on a single RTX 5080.

Measured on RTX 5080 (16 GB) and RTX 5060 Ti (16 GB, PCIe x4), 40 GB RAM, Windows 11. Estimates are labelled as estimates. The brain knows what its Qwen base knows: these
steps make it smaller, faster and clearer about doubt, and they add no knowledge.

## At a glance

| | |
|---|---|
| Base model | Qwen3.6-35B-A3B (Apache-2.0): 40 layers, attention every 4th and gated DeltaNet between, 256 experts per layer with 8 active |
| Parameters | 34.66B, 3.45B active per token |
| Size on disk | 21.1 GB, one `.nx` file |
| Quality vs the bf16 original | mean KL 0.0079 per token |
| Decode speed | 38-53 tokens/s (Python reference: 1.79) |
| VRAM | 13.56 GB, holding 6,544 of 10,240 experts |
| Confidence probe | AUROC 0.988 on held-out answers |
| Everyday prompts | 16 prompts in 141 s with both brains (1,738 s with the 177B alone) |
| HumanEval | 150 of 164 (91.5%), thinking off |
| CQ | 89.6 (36 of 40) |

## Optimisations

Each change has its own folder and builds on the ones before it.

| # | folder | what it adds | measured | gain |
|---|---|---|---|---|
| 01 | [4-bit experts](01-4bit-experts/) | Experts in 4 bits, big linears in int8 | 21.1 GB file; mean KL 0.0079 vs bf16 | 3.3× smaller |
| 02 | [Rust and CUDA engine](02-rust-cuda-engine/) | The model rewritten in Rust with custom kernels | 38-53 tok/s vs 1.79 in Python | 21-30× faster |
| 03 | [VRAM expert cache](03-vram-expert-cache/) | Experts paged into a pool of GPU slots | 6,544 of 10,240 experts in 13.56 GB | 64% of experts in VRAM |
| 04 | [Reading core](04-reading-core/) | A 421M CPU model reads each prompt as fact, creative or other | 180 ms per prompt; 80% right on fresh prompts | 2.2× more prompts read right |
| 05 | [Confidence probe](05-confidence-probe/) | Marks likely-wrong sentences with an estimated chance | AUROC 0.988 | AUROC 0.778 → 0.988 |
| 06 | [Tools and second look](06-tools-and-second-look/) | Calculator, speller, word parts, web lookup; a fresh answer when unsure | AQ 86 → 90 on the 177B brain | AQ +4, maths +17 |
| 07 | [Two-brain routing](07-two-brain-routing/) | Everyday prompts go to this brain, hard ones to the 177B | 16 prompts: 1,738 s → 141 s | 12.3× faster |
| 08 | [Shorter answers, less thinking](08-shorter-answers/) | Answer-length and thinking budgets | spider legs: 614 thinking tokens → 0 | 1.25×; 53% fewer answer tokens |
| 09 | [Test-and-fix loop](09-test-and-fix-loop/) | A `run_python` tool so the brain can test its code | built and safe; the model hasn't called it yet | none yet |
| 10 | [Shield against hidden text](10-shield/) | Invisible and encoded text removed and reported before the brain reads it | runs on every message | safety, no speed gain |
| 11 | [Modes and idea mode](11-modes-and-idea-mode/) | Normal, blunt and idea modes that mirror the user's tone | idea mode: probe flagged 14 of 15 review sentences | a feature, no speed gain |
| 12 | [Runtime governor](12-runtime-governor/) | Sampling temperature adjusted from the unsure marks as chat goes on | cools by 0.05 when over 30% of sentences are marked | a feature, not measured |
| 13 | [Core engine and learning time](13-core-engine/) | A bank of switchable low-rank cores in every layer, trained only behind gates | 4B brain: held-out loss 3.8718 → 3.8105 | room for 85.56B more parameters |
| 14 | [Benchmarks](14-benchmarks/) | CQ (40 own problems) and HumanEval | CQ 89.6; HumanEval 150 of 164 | results |

Each step folder gives its own measurement. The steps were measured at different times, so their
speed-ups don't multiply into one total.

## Change log

### Step 01: 4-bit experts

Gain: 3.3× smaller: about 69 GB at bf16 (computed from 34.66B parameters) down to a 21.1 GB file, for a mean KL of 0.0079 per token. Builds on the Qwen3.6-35B-A3B checkpoint. Everything after this runs on the 4-bit file. [Details](01-4bit-experts/)

### Step 02: Rust and CUDA engine

Gain: 21-30× faster decoding than the Python reference on the same GPU (1.79 to 38-53 tok/s). Builds on step 01: the engine reads the 4-bit file directly. [Details](02-rust-cuda-engine/)

### Step 03: VRAM expert cache

Gain: 64% of all experts held in VRAM (6,544 of 10,240) on one 16 GB card. Speed without the cache was not measured separately. Builds on step 02: the engine's expert kernels read from the pool's slots. [Details](03-vram-expert-cache/)

### Step 04: Reading core

Gain: 80% of fresh prompts read right, against 36% for a keyword rule (2.2×), for 180 ms of CPU time and no GPU. Builds on step 02: its answer sets the engine's sampling for each reply. [Details](04-reading-core/)

### Step 05: Confidence probe

Gain: AUROC up from 0.778 (logit statistics alone) to 0.988. Sampling 4 extra answers scores 0.894 and costs 4× the work. Builds on step 04: sentences in fact and other replies get marks; creative replies don't. [Details](05-confidence-probe/)

### Step 06: Tools and second look

Gain: AQ up from 86 to 90: maths 83 to 100, letters 61 to 94. Measured on the 177B brain, which runs the same code. Builds on step 05: a mark on a fact answer triggers the second look. [Details](06-tools-and-second-look/)

### Step 07: Two-brain routing

Gain: 12.3× faster on 16 everyday prompts (1,738 s to 141 s), with all 16 answers still right. Builds on steps 03 and 06: an 8 GB pool for this brain, and unsure fact answers go to the big brain. [Details](07-two-brain-routing/)

### Step 08: Shorter answers, less thinking

Gain: 1.25× faster on the 177B alone (1,738 s to 1,388 s), with 53% fewer answer tokens (4,893 to 2,323) and 98% fewer thinking tokens (937 to 23). Builds on step 07: shorter replies make both brains faster. [Details](08-shorter-answers/)

### Step 09: Test-and-fix loop

Gain: none yet. CQ went from 89.6 to 88.3 with the same 36 of 40 right, because the model never called the tool. Builds on step 06: another tool in the same tool loop. [Details](09-test-and-fix-loop/)

### Step 10: Shield against hidden text

Gain: a safety change, not a speed one: hidden or encoded instructions reach the brain only as quoted, flagged text. Builds on step 02: the engine runs the scan on each message before prefill. [Details](10-shield/)

### Step 11: Modes and idea mode

Gain: a feature, not a speed change: idea mode uses made-up ideas on purpose and labels them, instead of hiding them behind unsure marks. Builds on step 05: idea mode switches the probe's marks off, for a measured reason. [Details](11-modes-and-idea-mode/)

### Step 12: Runtime governor

Gain: not measured as a speed or score change: it lowers the temperature when too many sentences get marked, and raises it again when few do. Builds on step 05: it reads the probe's marks. [Details](12-runtime-governor/)

### Step 13: Core engine and learning time

Gain: room to grow from 34.66B to 120.23B parameters. No cores are trained on this brain yet, so it is no smarter than its base today. Builds on step 02: the engine uploads only cores that have been trained. [Details](13-core-engine/)

### Step 14: Benchmarks

Gain: a measurement of everything above, not an optimisation: CQ 89.6, HumanEval 91.5%. Builds on every step before it: these runs use the whole stack. [Details](14-benchmarks/)

## Code

| folder | what it holds |
|---|---|
| `aegist_rs/` | the Rust engine and launcher with its CUDA kernels. Build it with `cargo build --release` (needs the CUDA toolkit and MSVC). |
| `aegist/` | the Python side: converters, the reference model, the confidence probe, tests and benchmarks |
| `scripts/` | build and check scripts |

All three Aegist repos share the engine and the Python package, so each carries a full copy. Every step
folder links to the file it changed. There are no weights here: you build a brain (`.nx` file) yourself
from the base model with the scripts.

## Quick start

```powershell
python -I scripts/make_aegist_moe.py --base D:/aegist/models/qwen3.6-35b-a3b --out D:/aegist/brains/aegist-nx-35b.nx
cd aegist_rs; cargo build --release; cd ..
aegist_rs/target/release/aegist.exe chat --brain D:/aegist/brains/aegist-nx-35b.nx
```

You need the Qwen3.6-35B-A3B download from Hugging Face only to build the brain.

## Sister repos

- aegist-reid1-177b, the medium brain: a 177B mixture-of-experts streamed from an SSD on two consumer GPUs.
- aegist-kimi-1t, the big brain: Kimi-K2.6 (1T), in progress.

## Licence

Code: Apache-2.0 (`LICENSE`). Base model: Apache-2.0, see `NOTICE.md`. This repo has no weights.
