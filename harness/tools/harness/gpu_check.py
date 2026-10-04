#!/usr/bin/env python3
"""GPU (CUDA) probe + outcome classifier for Julia targets.

Every GPU check has exactly one outcome: PASS, FAIL or UNAVAILABLE.
**The exit code of the test command alone never yields PASS** -- a command that
exits 0 without running any GPU testset (e.g. the GPU tests were skipped, the
flag was ignored, CUDA was not loaded) is FAIL, because "skipped" must never be
counted as "passed".

Classification rules (`classify`):
  1. probe NOT_FUNCTIONAL                                   -> UNAVAILABLE
     probe ERROR (CUDA.jl could not even be loaded/resolved) -> FAIL
       (a probe *error* is NOT evidence that there is no GPU: it is a setup
        problem to investigate / a blocker, never an UNAVAILABLE pass-through)
  2. output contains the UNAVAILABLE marker
     (default `HARNESS_GPU_UNAVAILABLE`)                    -> UNAVAILABLE
  3. any GPU testset row (name matches --gpu-testset, default word /cuda|gpu/i)
     in a Julia `Test Summary:` table has Fail/Error > 0    -> FAIL
  4. exit code != 0                                          -> FAIL
  5. GPU tests demonstrably ran: the RAN marker (default
     `HARNESS_GPU_TESTS_RAN`) is present, or a GPU testset row
     reports Pass > 0                                        -> PASS
  6. otherwise (exit 0 but zero GPU tests executed)         -> FAIL

Usage:
  gpu_check.py probe <target-root> [--no-develop] [--online]
      Probe CUDA in a *temporary* Julia environment (develops the target and
      adds CUDA from the local depot), so it also works when CUDA is only a
      weak dependency / test extra of the target (IntervalMDP.jl layout).
      Prints GPU_PROBE=FUNCTIONAL|NOT_FUNCTIONAL|ERROR; exit 0 / 3 / 4.
  gpu_check.py classify --output FILE --exit-code N [--probe STATE] [...]
  gpu_check.py run <target-root> --cmd "<configured gpu.test>" [--no-probe] [...]
      Probe, run the configured GPU test command in the target root and
      classify. Prints GPU_OUTCOME=PASS|FAIL|UNAVAILABLE and REASON=...;
      exit 0 = PASS, 1 = FAIL, 3 = UNAVAILABLE.

Packages are resolved offline (JULIA_PKG_OFFLINE=true) unless --online is
given; nothing is installed into the target project.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

RAN_MARKER = "HARNESS_GPU_TESTS_RAN"
UNAVAILABLE_MARKER = "HARNESS_GPU_UNAVAILABLE"
GPU_TESTSET = r"(?i)\b(?:cuda|gpu)\b"
OUTCOMES = ("PASS", "FAIL", "UNAVAILABLE")
EXIT = {"PASS": 0, "FAIL": 1, "UNAVAILABLE": 3}

PROBE_JL = r"""
using Pkg
Pkg.activate(; temp=true)
try
    root = ARGS[1]
    if ARGS[2] == "develop"
        Pkg.develop(path=root; io=devnull)
    end
    Pkg.add("CUDA"; io=devnull)
    @eval using CUDA
catch e
    msg = replace(sprint(showerror, e), '\n' => ' ')
    println("GPU_PROBE=ERROR ", first(msg, 400))
    exit(4)
end
ok = try
    CUDA.functional() && length(CUDA.devices()) > 0
catch e
    println("GPU_PROBE=ERROR CUDA.functional() threw: ", first(replace(sprint(showerror, e), '\n' => ' '), 300))
    exit(4)
end
if ok
    println("GPU_PROBE=FUNCTIONAL device=", CUDA.name(CUDA.device()))
    exit(0)
else
    println("GPU_PROBE=NOT_FUNCTIONAL CUDA.functional()=", CUDA.functional())
    exit(3)
end
"""


def _env(online):
    env = dict(os.environ)
    if not online:
        env["JULIA_PKG_OFFLINE"] = "true"
    return env


def probe(root, develop=True, online=False, julia="julia", timeout=1200):
    """Return (state, detail) with state in FUNCTIONAL / NOT_FUNCTIONAL / ERROR."""
    if not shutil.which(julia):
        return "ERROR", "julia not found on PATH"
    with tempfile.TemporaryDirectory(prefix="gpu-probe-") as tmp:
        script = os.path.join(tmp, "probe.jl")
        with open(script, "w") as fh:
            fh.write(PROBE_JL)
        env = _env(online)
        env.pop("JULIA_PROJECT", None)
        try:
            p = subprocess.run([julia, "--startup-file=no", script, os.path.abspath(root),
                                "develop" if develop else "nodevelop"],
                               capture_output=True, text=True, env=env, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return "ERROR", "probe timed out after %ss" % timeout
    out = p.stdout + p.stderr
    m = re.search(r"GPU_PROBE=(FUNCTIONAL|NOT_FUNCTIONAL|ERROR)(.*)", out)
    if not m:
        return "ERROR", "probe produced no GPU_PROBE line (exit %d): %s" % (p.returncode, out.strip()[-400:])
    return m.group(1), m.group(2).strip()


def parse_test_summaries(text):
    """Parse Julia `Test Summary:` tables -> list of (name, {col: int})."""
    rows = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(Test Summary:\s*)\|(.*)$", line)
        if not m:
            i += 1
            continue
        bar = len(m.group(1))
        cols = []
        for cm in re.finditer(r"\S+", m.group(2)):
            cols.append((cm.group(0), bar + 1 + cm.end()))  # (name, end column)
        i += 1
        while i < len(lines) and len(lines[i]) > bar and lines[i][bar:bar + 1] == "|":
            row = lines[i]
            name = row[:bar].strip()
            counts = {}
            for nm in re.finditer(r"\d+", row[bar + 1:]):
                end = bar + 1 + nm.end()
                for cname, cend in cols:
                    if cname != "Time" and abs(cend - end) <= 1:
                        counts[cname] = int(nm.group(0))
            rows.append((name, counts))
            i += 1
    return rows


def classify(output, exit_code, probe_state=None, probe_detail="", ran_marker=RAN_MARKER,
             unavailable_marker=UNAVAILABLE_MARKER, gpu_testset=GPU_TESTSET):
    """Return (outcome, reason). Exit code alone never yields PASS."""
    if probe_state == "NOT_FUNCTIONAL":
        return "UNAVAILABLE", "probe: no functional CUDA device (%s)" % probe_detail
    if probe_state == "ERROR":
        return "FAIL", ("probe ERROR (not evidence of missing GPU -- setup problem to investigate / blocker): %s"
                        % probe_detail)
    output = output or ""
    m = re.search(re.escape(unavailable_marker) + r"[^\n]*", output)
    if m:
        return "UNAVAILABLE", "test reported %s" % m.group(0).strip()
    gpu_rx = re.compile(gpu_testset)
    gpu_rows = [(n, c) for n, c in parse_test_summaries(output) if gpu_rx.search(n)]
    bad = [(n, c) for n, c in gpu_rows if c.get("Fail", 0) or c.get("Error", 0)]
    if bad:
        return "FAIL", "GPU testset failures: %s" % "; ".join("%s %s" % (n, c) for n, c in bad)
    if exit_code != 0:
        return "FAIL", "GPU test command exited %d" % exit_code
    ran_rows = [(n, c) for n, c in gpu_rows if c.get("Pass", 0) > 0]
    if ran_marker and ran_marker in output:
        return "PASS", "GPU tests ran and passed (%s marker; rows: %s)" % (ran_marker, ran_rows or "n/a")
    if ran_rows:
        return "PASS", "GPU testset(s) ran and passed: %s" % ran_rows
    return "FAIL", ("exit 0 but no GPU testset ran (no %s marker, no GPU Test Summary row): "
                    "skipped GPU tests are never a pass" % ran_marker)


def run(root, cmd, do_probe=True, develop=True, online=False, timeout=3600, env_extra=None, **kw):
    """Probe, run the configured GPU test command in `root`, classify.

    Returns dict(outcome, reason, probe, exit_code, output)."""
    pstate, pdetail = (None, "")
    if do_probe:
        pstate, pdetail = probe(root, develop=develop, online=online)
        if pstate != "FUNCTIONAL":
            o, r = classify("", 0, pstate, pdetail, **kw)
            return {"outcome": o, "reason": r, "probe": pstate, "exit_code": None, "output": ""}
    env = _env(online)
    env.update(env_extra or {})
    try:
        p = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True, env=env, timeout=timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        return {"outcome": "FAIL", "reason": "GPU test command timed out", "probe": pstate, "exit_code": None,
                "output": ""}
    o, r = classify(out, rc, None, "", **kw)
    return {"outcome": o, "reason": r, "probe": pstate, "exit_code": rc, "output": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    pp = sub.add_parser("probe")
    pp.add_argument("root")
    pc = sub.add_parser("classify")
    pc.add_argument("--output", required=True)
    pc.add_argument("--exit-code", type=int, required=True)
    pc.add_argument("--probe", choices=["FUNCTIONAL", "NOT_FUNCTIONAL", "ERROR"])
    pr = sub.add_parser("run")
    pr.add_argument("root")
    pr.add_argument("--cmd", required=True)
    pr.add_argument("--no-probe", action="store_true")
    for p in (pp, pr):
        p.add_argument("--no-develop", action="store_true")
        p.add_argument("--online", action="store_true")
    for p in (pc, pr):
        p.add_argument("--ran-marker", default=RAN_MARKER)
        p.add_argument("--unavailable-marker", default=UNAVAILABLE_MARKER)
        p.add_argument("--gpu-testset", default=GPU_TESTSET)
    a = ap.parse_args(argv)
    if a.mode == "probe":
        state, detail = probe(a.root, develop=not a.no_develop, online=a.online)
        print("GPU_PROBE=%s %s" % (state, detail))
        return {"FUNCTIONAL": 0, "NOT_FUNCTIONAL": 3}.get(state, 4)
    kw = dict(ran_marker=a.ran_marker, unavailable_marker=a.unavailable_marker, gpu_testset=a.gpu_testset)
    if a.mode == "classify":
        with open(a.output, encoding="utf-8", errors="replace") as fh:
            o, r = classify(fh.read(), a.exit_code, a.probe, "", **kw)
    else:
        res = run(a.root, a.cmd, do_probe=not a.no_probe, develop=not a.no_develop, online=a.online, **kw)
        sys.stdout.write(res["output"][-4000:])
        o, r = res["outcome"], res["reason"]
    print("\nGPU_OUTCOME=%s" % o)
    print("REASON=%s" % r)
    return EXIT[o]


if __name__ == "__main__":
    sys.exit(main())
