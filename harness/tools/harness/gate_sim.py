#!/usr/bin/env python3
"""Executable model of the /harness orchestrator gate logic.

`.claude/commands/harness.md` is the normative definition; this module encodes
the same rules so they can be regression-tested without running Claude:

  * Stage order: Dev -> Formal Verification -> QE -> Ops   ([n/4] status lines)
  * Verify may be "not required" only for non-algorithm changes that touch no
    Lean/proof dependency; that is recorded as gate result "skip" (never "pass").
  * A code failure in Dev, Verify or QE loops back to Dev with failure evidence.
    At most MAX_CYCLES (2) Dev->gate cycles in total; then harness_failed.
  * An environment BLOCKER (required toolchain / GPU UNAVAILABLE, command not
    inferable) is terminal immediately: harness_failed, no further cycles.
  * Ops runs only if Dev passed, Verify passed (or was legitimately skipped),
    and QE passed in the SAME cycle. Ops never runs after any failed gate.

Stage outcomes are strings: "pass" | "fail" | "blocked".

CLI: gate_sim.py scenario.json [--db PATH]   (prints events; optional telemetry write)
scenario.json: {"verify_required": true,
                "cycles": [{"dev": "pass", "verify": "fail", "qe": "pass"}, ...],
                "ops": "pass"}
"""
import json
import sys

MAX_CYCLES = 2
STAGES = ["dev", "verify", "qe", "ops"]
LABEL = {"dev": "Dev", "verify": "Formal Verification", "qe": "QE", "ops": "Ops"}
INDEX = {"dev": 1, "verify": 2, "qe": 3, "ops": 4}

# GPU / criterion outcomes reported by QE
GPU_OUTCOMES = ("PASS", "FAIL", "UNAVAILABLE")


def qe_gate(criteria):
    """Aggregate QE acceptance criteria into a gate outcome.

    criteria: list of {"id": str, "result": "pass"|"fail"|"unavailable", "required": bool}
    - any "fail"                    -> "fail"
    - any required "unavailable"    -> "blocked" (unmet criterion; NEVER pass)
    - non-required "unavailable"    -> recorded, does not count as pass
    """
    out = "pass"
    for c in criteria:
        r = c["result"].lower()
        if r not in ("pass", "fail", "unavailable"):
            raise ValueError("unknown criterion result %r" % r)
        if r == "fail":
            return "fail"
        if r == "unavailable" and c.get("required", True):
            out = "blocked"
    return out


def gpu_criterion(outcome, required):
    """Map a GPU test outcome to an acceptance_criterion result."""
    if outcome not in GPU_OUTCOMES:
        raise ValueError("GPU outcome must be one of %s" % (GPU_OUTCOMES,))
    return {"id": "gpu", "result": outcome.lower(), "required": required}


def simulate(verify_required, cycles, ops="pass"):
    events, status = [], []

    def ev(name, **d):
        events.append((name, d))

    ev("harness_started")
    cycle = 0
    passed = False
    for spec in cycles[:MAX_CYCLES]:
        cycle += 1
        failure = None
        for stage in ("dev", "verify", "qe"):
            if stage == "verify" and not verify_required:
                reason = "no VI/Bellman semantic change and no Lean/proof dependency touched"
                ev("gate_decision", gate="verify", result="skip", cycle=cycle, reason="not required: " + reason)
                status.append("– [2/4] Formal Verification — not required (%s)" % reason)
                continue
            prefix = "↻" if (stage == "dev" and cycle > 1) else "▶"
            status.append("%s [%d/4] %s" % (prefix, INDEX[stage], LABEL[stage]))
            ev("delegation_start", stage=stage, cycle=cycle)
            ev(stage + "_started", cycle=cycle)
            outcome = spec.get(stage, "pass")
            ev(stage + "_finished", status=outcome, cycle=cycle)
            ev("delegation_end", stage=stage, cycle=cycle, status=outcome)
            if outcome == "pass":
                ev("gate_decision", gate=stage, result="pass", cycle=cycle)
                continue
            failure = (stage, outcome)
            break
        if failure is None:
            passed = True
            break
        stage, outcome = failure
        if outcome == "blocked":
            ev("gate_decision", gate=stage, result="fail", cycle=cycle, blocker=True)
            ev("harness_failed", stage=stage, cycle=cycle, reason="blocker")
            return {"events": events, "status": status, "terminal": "harness_failed", "ops_ran": False, "cycles": cycle}
        if cycle < MAX_CYCLES and cycle < len(cycles):
            ev("gate_decision", gate=stage, result="retry", cycle=cycle)
            ev("loopback", cycle=cycle + 1, from_gate=stage)
            continue
        ev("gate_decision", gate=stage, result="fail", cycle=cycle)
        ev("harness_failed", stage=stage, cycle=cycle, reason="%s failed; retry policy exhausted" % stage)
        return {"events": events, "status": status, "terminal": "harness_failed", "ops_ran": False, "cycles": cycle}
    if not passed:
        ev("harness_failed", stage="dev", cycle=cycle, reason="no passing cycle")
        return {"events": events, "status": status, "terminal": "harness_failed", "ops_ran": False, "cycles": cycle}
    # All required gates passed in this cycle -> Ops.
    assert ops_allowed(events, cycle), "invariant violated: Ops would run after failed gate"
    status.append("▶ [4/4] Ops")
    ev("delegation_start", stage="ops", cycle=cycle)
    ev("ops_started", cycle=cycle)
    ev("ops_finished", status=ops, cycle=cycle)
    ev("delegation_end", stage="ops", cycle=cycle, status=ops)
    if ops == "pass":
        ev("harness_completed", cycles=cycle)
        return {"events": events, "status": status, "terminal": "harness_completed", "ops_ran": True, "cycles": cycle}
    status.append("✗ Ops failed — %s" % ops)
    ev("harness_failed", stage="ops", cycle=cycle, reason="ops " + ops)
    return {"events": events, "status": status, "terminal": "harness_failed", "ops_ran": True, "cycles": cycle}


def ops_allowed(events, cycle):
    """Ops allowed iff dev, verify and qe gates in `cycle` are pass (verify may be skip)."""
    got = {e[1]["gate"]: e[1]["result"] for e in events
           if e[0] == "gate_decision" and e[1].get("cycle") == cycle}
    return got.get("dev") == "pass" and got.get("verify") in ("pass", "skip") and got.get("qe") == "pass"


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    with open(argv[1]) as fh:
        sc = json.load(fh)
    res = simulate(sc.get("verify_required", True), sc["cycles"], sc.get("ops", "pass"))
    db = argv[argv.index("--db") + 1] if "--db" in argv else None
    if db:
        sys.path.insert(0, __import__("os").path.dirname(__file__))
        from record_event import record
        for name, d in res["events"]:
            record(name, dict(d, simulated=True), db)
    for line in res["status"]:
        print(line)
    for name, d in res["events"]:
        print(name, json.dumps(d, sort_keys=True))
    print("TERMINAL:", res["terminal"])
    return 0 if res["terminal"] == "harness_completed" else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
