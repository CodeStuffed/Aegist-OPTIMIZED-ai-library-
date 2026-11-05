"""The run_python tool: the brain tests code it wrote before answering (the test-and-fix loop).

Code arrives on stdin. It is refused without running if the safety gate in cq.py finds an import outside the
safe list (no os, subprocess, socket, ...) or a call such as open/eval/exec/getattr. Allowed code runs with
`python -I` in an empty temporary folder, with a minimal environment and a time limit. What it printed and any
error come back as text for the brain.

    echo print(1+1) | python -m aegist.sandbox
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

from .cq import SAFE_IMPORTS, unsafe

TIMEOUT = 10.0
MAX_CODE = 20_000
MAX_OUT = 1500


def clip(s: str, n: int = MAX_OUT) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n // 3] + "\n...\n" + s[-(2 * n) // 3:]


def run(code: str) -> str:
    if len(code) > MAX_CODE:
        return f"refused: the code is longer than {MAX_CODE} characters"
    why = unsafe(code)
    if why:
        if why.startswith("syntax error"):
            return why
        return (f"refused without running: {why} is not allowed in the sandbox. It has no files, network or OS access; "
                f"imports allowed: {', '.join(sorted(SAFE_IMPORTS - {'__future__'}))}. Test the logic without it.")
    with tempfile.TemporaryDirectory(prefix="aegist-run-") as d:
        path = os.path.join(d, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        t = time.time()
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True, timeout=TIMEOUT,
                               encoding="utf-8", errors="replace",
                               env={"SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PYTHONIOENCODING": "utf-8"})
        except subprocess.TimeoutExpired:
            return f"error: still running after {TIMEOUT:.0f} s (an endless loop, or far too slow); nothing finished"
        secs = time.time() - t
        out, err = r.stdout.replace(path, "main.py"), r.stderr.replace(path, "main.py").replace(d, ".")
    if r.returncode == 0:
        return f"ran OK in {secs:.2f} s" + (f"; printed:\n{clip(out)}" if out.strip() else "; printed nothing") + \
            (f"\nwarnings:\n{clip(err, 400)}" if err.strip() else "")
    return f"FAILED (exit {r.returncode}) in {secs:.2f} s" + (f"; printed:\n{clip(out, 500)}" if out.strip() else "") + \
        f"\nerror:\n{clip(err)}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print(run(sys.stdin.buffer.read().decode("utf-8", errors="replace")))


if __name__ == "__main__":
    main()
