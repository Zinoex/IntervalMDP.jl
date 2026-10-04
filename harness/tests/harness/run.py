#!/usr/bin/env python3
"""Run the harness test suite and report PASS / FAIL / UNAVAILABLE separately.

A test skipped with a reason beginning "UNAVAILABLE:" is a check that could not
run because a tool/hardware is missing. It is reported distinctly and is NEVER
counted as passed.

Exit codes / RESULT line:
  0  RESULT: PASS                        every check ran and passed
  1  RESULT: FAIL                        at least one failure/error
  2  RESULT: PASS-INCOMPLETE (n UNAVAILABLE)  no failures, but n checks could not
                                         run (missing tool/hardware) -- NOT a full pass
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class Result(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.passed = []
        self.partially_skipped = set()

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        # a skipped subTest makes its parent test incomplete: never count the parent as passed
        self.partially_skipped.add(getattr(test, "test_case", test).id())

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None and issubclass(err[0], unittest.SkipTest):
            self.partially_skipped.add(test.id())

    def addSuccess(self, test):
        super().addSuccess(test)
        if test.id() not in self.partially_skipped:
            self.passed.append(test)


def main(start_dir=HERE):
    suite = unittest.defaultTestLoader.discover(start_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2, resultclass=Result, stream=sys.stdout)
    res = runner.run(suite)
    unavailable = [(t, r) for t, r in res.skipped if r.startswith("UNAVAILABLE")]
    other_skips = [(t, r) for t, r in res.skipped if not r.startswith("UNAVAILABLE")]
    print("\n==== Harness test summary ====")
    for t, r in unavailable:
        print("UNAVAILABLE %s -- %s" % (t.id().split(".", 1)[-1], r))
    for t, r in other_skips:
        print("SKIPPED     %s -- %s" % (t.id().split(".", 1)[-1], r))
    for t, _ in res.failures + res.errors:
        print("FAIL        %s" % t.id().split(".", 1)[-1])
    failed = len(res.failures) + len(res.errors)
    print("SUMMARY: total=%d pass=%d fail=%d unavailable=%d skipped=%d"
          % (res.testsRun, len(res.passed), failed, len(unavailable), len(other_skips)))
    override = os.environ.get("HARNESS_LEAN_TOOLCHAIN_OVERRIDE", "").strip()
    if override:
        print("NOTE: Lean checks ran with HARNESS_LEAN_TOOLCHAIN_OVERRIDE=%s -- NON-PINNED evidence: "
              "acceptance under each fixture's pinned lean-toolchain is NOT shown" % override)
    if failed:
        print("RESULT: FAIL (%d failure(s)%s)" % (failed, ", %d UNAVAILABLE" % len(unavailable) if unavailable else ""))
        return 1
    if unavailable:
        print("RESULT: PASS-INCOMPLETE (%d UNAVAILABLE check(s) not executed -- not counted as pass)" % len(unavailable))
        return 2
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
