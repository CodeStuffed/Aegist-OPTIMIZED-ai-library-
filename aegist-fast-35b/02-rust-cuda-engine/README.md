# Step 02: Rust and CUDA engine

Gain: 21-30× faster decoding than the Python reference on the same GPU (1.79 to 38-53 tok/s).

Builds on step 01: the engine reads the 4-bit file directly.

Previous: [step 01, 4-bit experts](../01-4bit-experts/). Next: [step 03, VRAM expert cache](../03-vram-expert-cache/). Main README: [../README.md](../README.md).

The model was first written in PyTorch as a reference (`aegist/hybrid.py`). The engine re-implements it in
Rust with hand-written CUDA kernels: gated DeltaNet with its recurrent state, gated attention, int8
gemv and gemm, 4-bit expert kernels that find their experts through a pointer table, and sampling. The
`.nx` file is memory-mapped, and embedding rows are looked up on the host.

Measured on the RTX 5080:

| what | result |
|---|---|
| Rust decode, thinking off | 38-53 tok/s |
| Python reference decode | 1.79 tok/s |
| prefill, 748 tokens, cold expert cache | about 4.45 s |
| Rust vs Python logits, chat (25 positions) | top-1 agreement 100%, mean KL 5.06e-5 |
| Rust vs Python, long context | top-1 agreement 98.82%, median KL 4.4e-7 |

```powershell
$env:AEGIST_GPU_TESTS = 1; pytest tests/test_hybrid_rust.py -s
```

Code: [`aegist_rs/src/hybrid.rs`](../aegist_rs/src/hybrid.rs), [`aegist_rs/kernels/hybrid.cu`](../aegist_rs/kernels/hybrid.cu),
[`aegist_rs/src/model.rs`](../aegist_rs/src/model.rs), [`aegist_rs/src/cuda.rs`](../aegist_rs/src/cuda.rs) (a small CUDA
driver binding, no extra crates), [`aegist/hybrid.py`](../aegist/hybrid.py) (the reference).
