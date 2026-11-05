# Step 07: Two-brain routing

Gain: 12.3× faster on 16 everyday prompts (1,738 s to 141 s), with all 16 answers still right.

Builds on steps 03 and 06: an 8 GB pool for this brain, and unsure fact answers go to the big brain.

Previous: [step 06, Tools and second look](../06-tools-and-second-look/). Next: [step 08, Shorter answers, less thinking](../08-shorter-answers/). Main README: [../README.md](../README.md).

`chat` and `ask` load this brain next to the 177B brain: this one first, on the last GPU, with an 8 GB
expert pool (`AEGIST_FAST_GB`), then the 177B on what is left. Each prompt goes to one of them:

- prompts with no hard cue and under 800 characters go to this brain;
- hard cues (prove, theorem, algorithm, step by step, puzzle, riddle, in depth) and long texts go to the 177B;
- when this brain marks a fact answer unsure, the 177B answers it again.

`/brain auto|fast|big` in chat overrides the choice, and `AEGIST_FAST=off` loads the 177B alone.

Measured on 16 everyday prompts (chat, facts, how-to, a poem, code, SQL, word problems, an explanation, a
comparison and a study plan), thinking on auto:

| setup | total time |
|---|---|
| 177B alone, before | 1,738 s |
| 177B alone, with step 08 | 1,388 s |
| both brains, 4 GB for this brain | 292 s |
| both brains, 8 GB for this brain (default) | 141 s |

Everyday answers now take 1-12 s instead of 17-212 s. The cost: the 177B keeps 2,715 expert slots
instead of 6,563, so the prompts it still gets run slower. This brain also got more probe questions
wrong than the 177B (10% against 6%). Sixteen prompts is a small sample.

Code: [`aegist_rs/src/chat.rs`](../aegist_rs/src/chat.rs) (`pick_brain`).
