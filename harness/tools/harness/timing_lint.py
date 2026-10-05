#!/usr/bin/env python3
"""Flag brittle wall-clock thresholds in Julia unit tests.

Performance claims belong in reproducible benchmark evidence (baseline vs
change), not in ordinary unit tests. This lint scans `<root>/test/**/*.jl`
for `@test` / `@assert` lines that compare a timing measurement against a
threshold, e.g.

    @test @elapsed(solve(prob)) < 0.5
    t = @belapsed f(x); @test t < 1e-3
    @test (time_ns() - t0) / 1e9 < 2
    b = @benchmark f(x); @test median(b).time < 1e6

Allocation checks (`@allocated`, `@allocations`, `@ballocated`) are NOT flagged: they are
deterministic and acceptable correctness/regression guards.

Usage: timing_lint.py <julia-project-root> [--tests-dir test]
Exit 0 = clean, 1 = brittle timing assertions found.
"""
import os
import re
import sys

TIMING = re.compile(r"@(?:elapsed|belapsed|btime|benchmark|benchmarkable|time|timed)\b"
                    r"|\btime_ns\s*\(|\btime\s*\(\s*\)|\bnow\s*\(\s*\)")
ASSERT = re.compile(r"@test\b|@assert\b|@test_broken\b")
CMP = re.compile(r"<=?|>=?")


def lint(root, tests_dir="test"):
    findings = []
    base = os.path.join(root, tests_dir)
    timing_vars = set()
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames.sort()
        for fn in sorted(filenames):
            if not fn.endswith(".jl"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            timing_vars.clear()
            for i, line in enumerate(lines, 1):
                code = line.split("#", 1)[0]
                m = re.match(r"\s*(?:local\s+)?([A-Za-z_]\w*)\s*=\s*(.*)$", code)
                if m and (TIMING.search(m.group(2)) or
                          any(re.search(r"\b%s\b" % re.escape(v), m.group(2)) for v in timing_vars)):
                    # timing measurement, BenchmarkTools trial (`@benchmark`/`@benchmarkable`)
                    # or a value derived from one (`m = median(b)`)
                    timing_vars.add(m.group(1))
                if not ASSERT.search(code) or not CMP.search(code):
                    continue
                uses_var = any(re.search(r"\b%s\b" % re.escape(v), code) for v in timing_vars)
                if TIMING.search(code) or uses_var:
                    findings.append("%s:%d: wall-clock threshold in unit test: %s" % (rel, i, line.strip()))
    return findings


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    td = argv[argv.index("--tests-dir") + 1] if "--tests-dir" in argv else "test"
    f = lint(argv[1], td)
    for x in f:
        print("FAIL " + x)
    print("RESULT: %s (%d brittle timing assertion(s))" % ("FAIL" if f else "PASS", len(f)))
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
