# Step 13: Core engine and learning time

Gain: room to grow from 34.66B to 120.23B parameters. No cores are trained on this brain yet, so it is no smarter than its base today.

Builds on step 02: the engine uploads only cores that have been trained.

Previous: [step 12, Runtime governor](../12-runtime-governor/). Next: [step 14, Benchmarks](../14-benchmarks/). Main README: [../README.md](../README.md).

Every layer has a growable bank of switchable low-rank cores. Learning time trains only the cores the core
engine switches on for the new material, which comes from files in `learn/inbox`, rated replies
(`/good`, `/bad`, `/fix`), identity examples and a replay buffer of earlier lessons.

Drift from the frozen base is measured on 17 probe conversations every 2 steps, and training stops before
drift passes 0.9 × the limit. A new version is kept only if all of these hold:

- held-out loss drops at least 1%;
- drift stays at or below 0.1 nats per token;
- it still says it is Aegist.

Versions go to `brains/history/` and `brains/current.txt` points at the live one, so nothing is
overwritten. Each version of this brain is a full 21 GB file, and the last 5 are kept.

The Rust engine supports up to 128 × 64 cores per layer, rank 64 at most and 8 active, and uploads only
live cores. The bank's 85.56B parameters are capacity, filled only as learning time trains them. On the
earlier 4B brain, the first accepted learning time took held-out loss from 3.8718 to 3.8105 with drift 0.074.

Code: [`aegist/learn.py`](../aegist/learn.py), [`aegist/model.py`](../aegist/model.py), [`aegist_rs/src/model.rs`](../aegist_rs/src/model.rs).
