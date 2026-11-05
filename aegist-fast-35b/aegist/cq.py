"""Aegist's CQ test (coding qualification): the AQ test's coding sibling.

The live brain gets programming problems through the chat pipeline (`aegist exam`), as a user would ask them.
Its code is run against hidden tests; code-reading answers must match the exact output. CQ is 100 x the mean of
the section scores (each section = share of its problems solved), so sections weigh the same.

  basics       strings, lists, small functions
  algorithms   binary search, intervals, DP, graphs
  structures   LRU cache, trie, union-find, min-stack
  bugfix       a broken function is given: return the fixed one
  careful      specs with edge cases (parsing, rounding money, rectangular matrices)
  reading      "what does this print?" (exact output)

It also times every answer: words per second and code lines per second, measured from the question going in to
the whole answer coming out (so it includes reading the question, not only writing).

A standard AI coding benchmark in HumanEval's format can be added with --suite FILE.jsonl (fields task_id, prompt,
entry_point, test); each such file is scored as its own section.

Safety: the brain's code is checked before it runs. Imports or calls that could touch files, processes or the
network (os, subprocess, socket, open, eval, ...) fail the problem without running. Allowed code runs in a fresh
empty folder, with `python -I` and a time limit.

Run:  aegist cq [--brain FILE] [--pick auto|fast|big] [--think off|on] [--suite FILE.jsonl] [--selftest]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time

from .cq_bank import PROBLEMS

ASK = ("\n\nReply with only the Python code in one ```python block (no explanation, no example usage). "
       "Use only the standard library; no file, network or system access.")
ASK_READ = ""  # reading prompts already say "Answer with only the exact output."
SECTIONS = ("basics", "algorithms", "structures", "bugfix", "careful", "reading")
SAFE_IMPORTS = {"collections", "heapq", "itertools", "functools", "math", "re", "bisect", "string", "typing",
                "dataclasses", "operator", "statistics", "fractions", "decimal", "random", "copy", "enum",
                "abc", "array", "numbers", "__future__", "hashlib"}
BAD_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals", "vars",
             "getattr", "setattr", "delattr"}


def exe_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "aegist_rs", "target", "release", "aegist.exe" if os.name == "nt" else "aegist")


def extract_code(answer: str) -> str:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", answer, flags=re.S)
    if blocks:
        return max(blocks, key=len)
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*)", answer, flags=re.S)  # cut off before the closing fence
    return m.group(1) if m else answer


def unsafe(code: str) -> str | None:
    """Why this code must not run, or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e.msg} (line {e.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in SAFE_IMPORTS:
                    return f"import {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in SAFE_IMPORTS:
                return f"from {node.module} import"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BAD_CALLS:
            return f"calls {node.func.id}()"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ("__init__", "__name__"):
            return f"uses {node.attr}"
    return None


def run_tests(code: str, tests: str, timeout: float = 20.0) -> tuple[bool, str]:
    why = unsafe(code)
    if why:
        return False, why
    with tempfile.TemporaryDirectory(prefix="cq-") as d:
        path = os.path.join(d, "solution.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code + "\n\n# ---- tests\n" + tests + "\nprint('CQ-PASS')\n")
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True, timeout=timeout,
                               env={"SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PYTHONIOENCODING": "utf-8"})
        except subprocess.TimeoutExpired:
            return False, f"took over {timeout:.0f}s"
    if "CQ-PASS" in r.stdout:
        return True, ""
    err = (r.stderr.strip().splitlines() or ["no output"])[-1]
    return False, err[:200]


def norm_out(s: str) -> str:
    s = s.strip()
    m = re.search(r"```(?:\w+)?\s*\n?(.*?)```", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    return " ".join(s.replace("`", "").split())


def grade(p: dict, answer: str) -> tuple[bool, str, str]:
    """-> (right, why wrong, code that was run)"""
    if "gold" in p:
        got = norm_out(answer)
        ok = got == p["gold"] or (got.splitlines()[-1:] == [p["gold"]])
        return ok, "" if ok else f"expected {p['gold']!r}", ""
    code = complete(p, extract_code(answer))
    ok, why = run_tests(code, p["tests"])
    return ok, why, code


def load_suite(path: str) -> list[dict]:
    """HumanEval-format problems -> CQ problems (one section named after the file)."""
    name = os.path.splitext(os.path.basename(path))[0]
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        t = json.loads(line)
        out.append({"id": t["task_id"], "section": name,
                    "prompt": "Complete this Python function:\n\n" + t["prompt"].rstrip(),
                    "tests": t["test"] + f"\ncheck({t['entry_point']})",
                    "stub": t["prompt"], "entry": t["entry_point"],
                    "ref": t["prompt"] + t.get("canonical_solution", "")})
    return out


def complete(p: dict, code: str) -> str:
    """HumanEval answers: a bare function body goes under the given signature; a full function keeps the
    stub's imports and helper functions (the stub is put first, the answer's own def overrides it)."""
    if "stub" not in p:
        return code
    if not re.search(rf"^\s*def {re.escape(p['entry'])}\s*\(", code, flags=re.M):
        body = textwrap.indent(textwrap.dedent(code.expandtabs(4)).strip("\n"), "    ")
        return p["stub"] + body + "\n"
    return p["stub"] + "    pass\n\n" + code


def selftest(suites=()) -> int:
    bad = 0
    for f in suites:  # each reference must pass both as a bare body and as a whole function in a code block
        probs, sbad, gated = load_suite(f), 0, set()
        for p in probs:
            body = p["ref"][len(p["stub"]):]
            for ans in (body, "Here it is:\n```python\n" + p["ref"] + "\n```"):
                ok, why, code = grade(p, ans)
                if not ok and unsafe(code):  # the reference itself uses something the safety gate refuses
                    gated.add(p["id"])
                elif not ok:
                    sbad += 1
                    print(f"  REF FAILS {p['id']}: {why}")
        print(f"selftest {os.path.basename(f)}: {len(probs)} problems, {sbad} reference failures"
              + (f"; reference refused by the safety gate (solvable without it): {', '.join(sorted(gated))}" if gated else ""))
        bad += sbad
    own = 0
    for p in PROBLEMS:
        if "gold" in p:
            continue
        ok, why = run_tests(p["ref"], p["tests"])
        if not ok:
            own += 1
            print(f"  REF FAILS {p['id']}: {why}")
    # the gate itself: these must be refused without running
    leaks = [c for c in ("import os\nos.listdir('.')", "open('x', 'w')", "__import__('os')", "import subprocess",
                         "().__class__.__bases__[0].__subclasses__()") if not unsafe(c)]
    for c in leaks:
        print(f"  NOT BLOCKED: {c!r}")
    n = sum("gold" not in p for p in PROBLEMS)
    print(f"selftest: {n - own}/{n} CQ reference solutions pass; unsafe-code gate {'FAILED' if leaks else 'ok'}")
    return 1 if bad + own + len(leaks) else 0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aegist cq", description="Aegist's CQ test (coding qualification).")
    ap.add_argument("--brain", help="the brain to test (default: the live one)")
    ap.add_argument("--pick", choices=["auto", "fast", "big"], default=None,
                    help="with two brains: let chat choose (auto) or force one; default: the brain given / live one")
    ap.add_argument("--think", choices=["off", "on", "auto"], default="off")
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--suite", action="append", default=[], help="extra HumanEval-format .jsonl file(s)")
    ap.add_argument("--only-suite", action="store_true", help="skip CQ's own problems")
    ap.add_argument("--limit", type=int, default=None, help="first N problems only (quick check)")
    ap.add_argument("--out", default="cq")
    ap.add_argument("--selftest", action="store_true", help="check the tests against the reference solutions")
    ap.add_argument("--grade", help="re-grade an existing answers.json in this run folder (no brain)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.suite)

    probs = ([] if a.only_suite else list(PROBLEMS)) + [p for f in a.suite for p in load_suite(f)]
    probs = probs[: a.limit] if a.limit else probs
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = a.grade or os.path.join(a.out, f"cq-{stamp}")
    os.makedirs(work, exist_ok=True)
    inp, outp = os.path.join(work, "exam.json"), os.path.join(work, "answers.json")
    brain_label = a.brain or "live brain"
    t0 = time.time()
    if not a.grade:
        spec = {"prompts": [p["prompt"] + (ASK_READ if "gold" in p else ASK) for p in probs], "think": a.think,
                "seed": 1, "max_new": a.max_new}
        if a.pick:
            spec["pick"] = a.pick
        json.dump(spec, open(inp, "w", encoding="utf-8"))
        print(f"CQ test: {len(probs)} problems for {brain_label} (think {a.think}"
              + (f", pick {a.pick}" if a.pick else "") + ")", flush=True)
        cmd = [exe_path(), "exam", inp, outp, "--resume"] + (["--brain", a.brain] if a.brain else [])
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"aegist exam exited with {r.returncode}")
    got = json.load(open(outp, encoding="utf-8"))

    rows = []
    for p, g in zip(probs, got):
        ok, why, code = grade(p, g["answer"])
        secs = g.get("seconds") or float("nan")
        words = len(g["answer"].split())
        lines = len([l for l in (code or "").splitlines() if l.strip()])
        rows.append({"id": p["id"], "section": p["section"], "right": ok, "why": why, "seconds": round(secs, 2),
                     "words": words, "code_lines": lines, "brain": g.get("brain"), "answer": g["answer"]})
    secs_all = [r["seconds"] for r in rows if r["seconds"] == r["seconds"] and r["seconds"] > 0]
    sections = {}
    for s in dict.fromkeys(r["section"] for r in rows):
        rs = [r for r in rows if r["section"] == s]
        sections[s] = round(100 * mean(r["right"] for r in rs), 1)
    wps = [r["words"] / r["seconds"] for r in rows if r["seconds"] and r["seconds"] > 0]
    lps = [r["code_lines"] / r["seconds"] for r in rows if r["code_lines"] and r["seconds"] > 0]
    res = {"cq": round(mean(sections.values()), 1), "brain": a.brain or "live", "pick": a.pick, "think": a.think,
           "problems": len(rows), "right": sum(r["right"] for r in rows), "sections": sections,
           "speed": {"median_words_per_s": round(statistics.median(wps), 2) if wps else None,
                     "median_code_lines_per_s": round(statistics.median(lps), 2) if lps else None,
                     "median_seconds_per_answer": round(statistics.median(secs_all), 1) if secs_all else None,
                     "brains_used": {b: sum(r["brain"] == b for r in rows) for b in {r["brain"] for r in rows}}},
           "when": stamp, "wall_seconds": round(time.time() - t0), "rows": rows}
    json.dump(res, open(os.path.join(work, "result.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(a.out, "history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in res.items() if k != "rows"}) + "\n")

    print(f"\nCQ {res['cq']:.0f} / 100   ({res['right']} of {res['problems']} solved, {brain_label})\n")
    for s, v in sections.items():
        print(f"  {s:12} {v:5.1f}  {'#' * round(v / 5)}")
    sp = res["speed"]
    print(f"\n  speed: {sp['median_words_per_s']} words/s, {sp['median_code_lines_per_s']} code lines/s, "
          f"{sp['median_seconds_per_answer']} s per answer (median; includes reading the question)")
    print(f"  brains used: {sp['brains_used']}")
    wrong = [r for r in rows if not r["right"]]
    if wrong:
        print("\n  missed:")
        for r in wrong[:12]:
            print(f"    {r['id']:12} {r['why'][:100]}")
    print(f"\nFull answers: {os.path.abspath(os.path.join(work, 'result.json'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
