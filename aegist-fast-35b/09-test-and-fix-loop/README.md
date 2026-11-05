# Step 09: Test-and-fix loop

Gain: none yet. CQ went from 89.6 to 88.3 with the same 36 of 40 right, because the model never called the tool.

Builds on step 06: another tool in the same tool loop.

Previous: [step 08, Shorter answers, less thinking](../08-shorter-answers/). Next: [step 10, Shield against hidden text](../10-shield/). Main README: [../README.md](../README.md).

The tool lets the brain run code it wrote before it answers:

1. The Rust tool call sends the code on stdin to `python -m aegist.sandbox`.
2. A safety gate refuses code that imports anything outside a safe list (no `os`, `subprocess` or
   `socket`) or calls `open`, `eval`, `exec`, `getattr` and similar.
3. Allowed code runs with `python -I` in a fresh, empty temporary folder with a minimal environment
   and a 10 s limit.
4. What it printed, or the traceback, goes back to the brain as text.

`AEGIST_RUN=off` removes the tool. `AEGIST_PYTHON` and `AEGIST_PYROOT` pick the Python and the project.

In the CQ run after it was added, the brain called it 0 times in 40 problems. CQ was 89.6 before and
88.3 after, with 36 of 40 right both times (the sections weighed the misses differently). Code requests
need a stronger hint in the system prompt, or an engine-driven loop that always runs the tests.

```powershell
"print(sum(range(10)))" | python -m aegist.sandbox
```

Code: [`aegist/sandbox.py`](../aegist/sandbox.py), [`aegist/cq.py`](../aegist/cq.py) (`unsafe`, `SAFE_IMPORTS`),
[`aegist_rs/src/tools.rs`](../aegist_rs/src/tools.rs) (`run_python`, `runner`).
