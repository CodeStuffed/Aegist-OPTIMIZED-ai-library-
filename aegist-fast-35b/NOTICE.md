# Notice

Aegist's fast brain is derived from Qwen3.6-35B-A3B by the Qwen team (Alibaba Cloud), licensed under the
Apache License 2.0 (`LICENSES/Qwen3.6-35B-A3B-Apache-2.0.txt`).

Changes: re-implemented architecture and runtime, 4-bit/8-bit quantisation, text only, added core engine
and switchable cores, Aegist persona, confidence and learning.

This repository holds code and documentation only, with no Qwen weights. The reading core rebuilds Laya
(convaiinnovations/laya, Apache-2.0) in Rust; its weights aren't included either.

The code in this repository is under the Apache License 2.0 (`LICENSE`).
