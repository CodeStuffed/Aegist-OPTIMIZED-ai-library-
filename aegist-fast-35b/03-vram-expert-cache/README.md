# Step 03: VRAM expert cache

Gain: 64% of all experts held in VRAM (6,544 of 10,240) on one 16 GB card. Speed without the cache was not measured separately.

Builds on step 02: the engine's expert kernels read from the pool's slots.

Previous: [step 02, Rust and CUDA engine](../02-rust-cuda-engine/). Next: [step 04, Reading core](../04-reading-core/). Main README: [../README.md](../README.md).

The always-on weights live in VRAM. The 10,240 experts (40 layers × 256) stay in the file, and a pool of
expert slots sized to free VRAM holds the recently used ones. The least recently used slot goes first.
Each layer, the router runs on the GPU, its picks come back to the host, and missing experts are copied
in. A long prompt streams its misses through the pool's coldest slots, so it doesn't push out the experts
that decoding keeps using.

DeltaNet blocks keep a fixed-size state instead of a KV cache. A state can't be cut back to an earlier
position, so the engine snapshots it before each reply, and `crop` returns to the latest snapshot.

Measured: 13.56 GB in VRAM holding 6,544 of 10,240 experts on the RTX 5080, decoding at 38-53 tok/s.
When this brain shares the PC with the 177B brain (step 07), it gets an 8 GB pool (`AEGIST_FAST_GB`).

To compare cache policies, record a routing trace with `AEGIST_ROUTE_TRACE=FILE` and replay it with
`python -I scripts/sim_expert_cache.py FILE`. The script tries LRU, wider windows and Belady's optimum.

Code: [`aegist_rs/src/hybrid.rs`](../aegist_rs/src/hybrid.rs) (`Pool`), [`aegist_rs/src/reader.rs`](../aegist_rs/src/reader.rs),
[`scripts/sim_expert_cache.py`](../scripts/sim_expert_cache.py).
