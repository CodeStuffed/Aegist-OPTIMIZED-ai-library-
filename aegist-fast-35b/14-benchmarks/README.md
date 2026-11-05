# Step 14: Benchmarks

Gain: a measurement of everything above, not an optimisation: CQ 89.6, HumanEval 91.5%.

Builds on every step before it: these runs use the whole stack.

Previous: [step 13, Core engine and learning time](../13-core-engine/). Main README: [../README.md](../README.md).

CQ (coding qualification) is 40 programming problems written for Aegist in six sections, asked through
the whole chat pipeline. Code runs against hidden tests in the sandbox, and code-reading answers must
match exactly. CQ is 100 × the mean of the section scores.

| run, thinking off | CQ | basics | algorithms | structures | bug fixing | careful specs | code reading |
|---|---|---|---|---|---|---|---|
| 2026-11-05, before `run_python` | 89.6 (36/40) | 87.5 | 100 | 100 | 100 | 100 | 50 |
| 2026-11-05, with `run_python` | 88.3 (36/40) | 100 | 100 | 100 | 100 | 80 | 50 |

HumanEval (OpenAI, MIT licence; 164 problems, added with `--suite`), thinking off, one answer each:

| what | result |
|---|---|
| solved | 150 of 164 (91.5%) |
| speed | 12.78 words/s, 4.76 code lines/s, 5.4 s per answer (median, including reading the question) |
| wall time | 25 min |
| missed | 13 failed their tests (89, 93, 99, 106, 125, 126, 129, 132, 134, 145, 147, 158, 163); 160 ran past the 20 s limit |

`--selftest` first checks that every reference solution passes its own tests. HumanEval/160's
reference uses `eval`, which the sandbox refuses, so it is reported as gated.

```powershell
python -m aegist.cq --brain D:/aegist/brains/aegist-nx-35b.nx
python -m aegist.cq --suite humaneval.jsonl --selftest
```

Code: [`aegist/cq.py`](../aegist/cq.py), [`aegist/cq_bank.py`](../aegist/cq_bank.py). HumanEval itself isn't
included; get it from openai/human-eval.
