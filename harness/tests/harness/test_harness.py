"""Harness-level tests for the Julia/Lean adaptation.

Test names are prefixed with the Test Specification case they cover
(case1 .. case8, see specs/intervalmdp-harness-update.md) or `static_` /
`tool_` for cross-cutting checks. Tests that need a tool that is not installed
(pinned Lean toolchain, a functional CUDA GPU) are skipped with a
reason starting with "UNAVAILABLE:" -- the runner reports those separately,
never as PASS.

Real runs always operate on temporary copies of the fixtures (never in the
repo fixture dirs, so no Manifest.toml / .lake / lake-manifest.json is left
behind):
  * Lean: only with an ALREADY INSTALLED toolchain, invoked via its own
    `bin/lake` (never the elan proxy -> no download). By default the pinned
    toolchain of each fixture is required. The opt-in
    HARNESS_LEAN_TOOLCHAIN_OVERRIDE=<toolchain> rewrites the pin in the temp
    copy to another installed toolchain: that is NON-PINNED evidence and the
    runner labels it as such.
  * GPU: probe in a temp Julia env (gpu_check.py), then the configured
    gpu.test on temp copies of the GPU PASS and GPU FAIL fixtures, classified
    by gpu_check.classify (exit code alone never yields PASS).
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))  # the harness/ directory
# The harness is vendored into a host repo: .claude/ lives one level above harness/.
PROJECT = os.path.abspath(os.path.join(REPO, ".."))
FIX = os.path.join(HERE, "fixtures")
TOOLS = os.path.join(REPO, "tools", "harness")
sys.path.insert(0, TOOLS)

import discover  # noqa: E402
import gate_sim  # noqa: E402
import gpu_check  # noqa: E402
import proof_hygiene  # noqa: E402
import record_event  # noqa: E402
import timing_lint  # noqa: E402

THM = "Fixture.bellman_monotone"


def read(rel):
    base = PROJECT if rel.startswith(".claude/") else REPO
    with open(os.path.join(base, rel), encoding="utf-8") as fh:
        return fh.read()


def disc(root, **kw):
    return discover.Discovery(root, **kw).run()


def unavailable(tool):
    return "UNAVAILABLE: `%s` not installed in this environment" % tool


def copy_fixture(name):
    tmp = tempfile.mkdtemp(prefix="harness-fixture-")
    dst = os.path.join(tmp, name)
    shutil.copytree(os.path.join(FIX, name), dst)
    return dst


def run_julia_tests(fixture):
    """Run the *discovered* Julia test command on a temp copy of a fixture."""
    root = copy_fixture(fixture)
    d = disc(root)
    assert "julia" in d.values["kinds"].split(","), d.values
    cmd = d.values["julia.test"]
    env = dict(os.environ, JULIA_PKG_OFFLINE="true")
    p = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True, env=env, timeout=900)
    shutil.rmtree(os.path.dirname(root), ignore_errors=True)
    return cmd, p.returncode, p.stdout + p.stderr


LEAN_OVERRIDE = os.environ.get("HARNESS_LEAN_TOOLCHAIN_OVERRIDE", "").strip()


def lean_prepare(testcase, fixture):
    """Temp copy of a Lean fixture + bin dir of an installed toolchain, or skip UNAVAILABLE.

    Never downloads: the toolchain must already exist under $ELAN_HOME/toolchains and
    its own bin/lake is used (not the elan proxy)."""
    src = os.path.join(FIX, fixture)
    with open(os.path.join(src, "lean-toolchain")) as fh:
        pinned = fh.read().strip()
    if LEAN_OVERRIDE:
        tc = discover.toolchain_dir(LEAN_OVERRIDE)
        if not tc:
            testcase.skipTest("UNAVAILABLE: HARNESS_LEAN_TOOLCHAIN_OVERRIDE=%s is not installed under %s "
                              "(never downloaded)" % (LEAN_OVERRIDE, discover.elan_home()))
        root = copy_fixture(fixture)
        with open(os.path.join(root, "lean-toolchain"), "w") as fh:
            fh.write(LEAN_OVERRIDE + "\n")
        testcase.addCleanup(shutil.rmtree, os.path.dirname(root), True)
        sys.stderr.write("[NON-PINNED evidence: %s built with %s instead of pinned %s] " % (fixture, LEAN_OVERRIDE, pinned))
        return root, os.path.join(tc, "bin")
    d = disc(src)
    tb = d.values.get("lean.toolchain_bin")
    if d.values.get("lean.toolchain_installed") != "yes" or not tb:
        testcase.skipTest("UNAVAILABLE: pinned Lean toolchain %s not installed (lean.toolchain_installed=%s; "
                          "never downloaded) -- real lake build/#print axioms NOT verified; opt-in "
                          "HARNESS_LEAN_TOOLCHAIN_OVERRIDE=<installed toolchain> gives non-pinned evidence"
                          % (pinned, d.values.get("lean.toolchain_installed")))
    root = copy_fixture(fixture)
    testcase.addCleanup(shutil.rmtree, os.path.dirname(root), True)
    return root, tb


def lean_verdict(results):
    return "FAIL" if any(r[0] == "FAIL" for r in results) else "PASS"


def names(events):
    return [e[0] for e in events]


# ---------------------------------------------------------------------------
# Static: instruction files encode the required rules
# ---------------------------------------------------------------------------
class StaticInstructionTests(unittest.TestCase):
    def setUp(self):
        self.h = read(".claude/commands/harness.md")

    def test_static_stage_order_and_status_lines(self):
        h = self.h
        self.assertIn("Dev → Formal Verification → QE → Ops", h)
        for line in ("[1/4] Dev", "[2/4] Formal Verification", "[3/4] QE", "[4/4] Ops"):
            self.assertIn(line, h)
        self.assertNotIn("[1/3]", h)
        idx = [h.index(s) for s in ("1. **Dev**", "2. **Formal Verification**", "3. **QE**", "4. **Ops**")]
        self.assertEqual(idx, sorted(idx), "gated workflow steps out of order")
        self.assertIn("subagent_type: verifier", h)

    def test_static_verify_telemetry_events_and_gate(self):
        h = self.h
        for ev in ("verify_started", "verify_finished", '"gate": "dev|verify|qe"',
                   '"stage": "dev|verify|qe|ops"', '"result": "pass|fail|retry|skip"'):
            self.assertIn(ev, h)
        self.assertIn("never** recorded as `pass`", h)

    def test_static_target_root_and_discovery(self):
        h = self.h
        self.assertIn("Identify the target root", h)
        self.assertIn("tools/harness/discover.py", h)
        self.assertIn("Pkg.test()", h)
        self.assertIn("lake build", h)
        self.assertNotIn("npm", h)
        self.assertIn("Precedence", h)

    def test_static_verifier_least_privilege(self):
        v = read(".claude/agents/verifier.md")
        tools = re.search(r"^tools:\s*(.+)$", v, re.M).group(1)
        toolset = {t.strip() for t in tools.split(",")}
        self.assertEqual(toolset, {"Read", "Glob", "Grep", "Bash"})
        for phrase in ("lean-toolchain", "lake build", "#print axioms", "sorry", "admit", "axiom",
                       "Never download, install, update, or switch toolchains",
                       "blocker / fail — never a pass", "never edit", "verify_started", "verify_finished",
                       "proof_obligation", "abstract/mathematical", "verified floating-point implementation",
                       "proof_hygiene.py", "not** trust Dev"):
            self.assertIn(phrase.lower(), v.lower(), phrase)

    def test_static_dev_rules(self):
        d = read(".claude/agents/dev.md")
        for phrase in ("Pkg.test()",
                       "Lean theorem statement and a complete proof", "Julia↔Lean traceability",
                       "No wall-clock thresholds in unit tests", "verified floating-point implementation",
                       "legacy algorithm without a proof", "MDP, interval MDP, L1-MDP, mixtures, factored",
                       "dev_started", "dev_finished", "test_run", "lake build", "record_event.py"):
            self.assertIn(phrase, d, phrase)
        self.assertNotIn("npm", d)

    def test_static_planner_onboarding_and_extensibility(self):
        p = read(".claude/agents/planner.md")
        for phrase in ("onboarding inventory", "legacy verification gaps", "Model-family extensibility",
                       "TEMPLATE-julia-lean.md", "TEMPLATE-onboarding-inventory.md", "discover.py"):
            self.assertIn(phrase, p, phrase)
        self.assertNotIn("/Users/andrewevans", p)

    def test_static_readme_and_templates(self):
        r = read("README.md")
        for phrase in ("Dev → Formal Verification → QE → Ops", "verifier.md", "harness.config.toml",
                       "CPU / GPU matrix", "TEMPLATE-julia-lean.md", "Precedence".lower()):
            self.assertIn(phrase.lower(), r.lower(), phrase)
        for gone in ("Node.js", "npm", "telemetry-mcp", "mcp__telemetry", ".mcp.json"):
            self.assertNotIn(gone, r, gone)
        t = read("specs/TEMPLATE-julia-lean.md")
        for sec in ("## Objective", "## Commands / Toolchain", "## Julia Behavior & Tests",
                    "## Algorithm ↔ Theorem Mapping", "## Proof Obligations & Limitations", "## Proof Policy",
                    "## CPU / GPU Matrix", "## Performance Evidence", "## Acceptance Criteria", "## Out of Scope"):
            self.assertIn(sec, t, sec)
        inv = read("specs/TEMPLATE-onboarding-inventory.md")
        self.assertIn("Legacy verification gaps", inv)
        self.assertIn("not** fully verified", inv)
        for f in ("harness.config.example.toml",):
            discover.load_toml(os.path.join(REPO, f))  # parses


    def test_static_no_node_tooling(self):
        # The harness is Julia/Lean only: no Node.js MCP server, no Node fixtures, no npm in the agents.
        self.assertFalse(os.path.exists(os.path.join(PROJECT, ".mcp.json")))
        self.assertFalse(os.path.exists(os.path.join(REPO, "tools", "telemetry-mcp")))
        for dirpath, dirnames, filenames in os.walk(REPO):
            dirnames[:] = [d for d in dirnames if d not in (".lake", "__pycache__")]
            self.assertNotIn("package.json", filenames, dirpath)
        for agent in ("dev", "qe", "ops", "verifier", "planner"):
            a = read(".claude/agents/%s.md" % agent)
            self.assertNotIn("mcp__telemetry", a, agent)
            self.assertNotIn("npm", a, agent)
        self.assertTrue(record_event.DEFAULT_DB.endswith(os.path.join("telemetry", "telemetry.db")))

# ---------------------------------------------------------------------------
# Case 1: Julia-only non-algorithm change -> Julia command, no npm, no Lean proof
# ---------------------------------------------------------------------------
class Case1JuliaOnly(unittest.TestCase):
    def test_case1_discovery_julia_only(self):
        d = disc(os.path.join(FIX, "julia-only"))
        v = d.values
        self.assertEqual(v["kinds"], "julia")
        self.assertIn("Pkg.test()", v["julia.test"])
        self.assertIn("--project=", v["julia.test"])
        self.assertEqual(v["julia.compat"], "1.9")
        self.assertFalse([k for k in v if k.startswith("node.")], "no npm/node command for Julia target")
        self.assertFalse([k for k in v if k.startswith("lean.")], "no Lean requirement for Julia-only target")
        self.assertNotIn("npm", " ".join(str(x) for x in v.values() if not str(x).startswith(("available", "UNAVAILABLE"))))
        self.assertEqual(d.blockers, [])

    def test_case1_verify_gate_skipped_not_passed(self):
        r = gate_sim.simulate(verify_required=False, cycles=[{"dev": "pass", "qe": "pass"}])
        gd = [e[1] for e in r["events"] if e[0] == "gate_decision" and e[1]["gate"] == "verify"]
        self.assertEqual(len(gd), 1)
        self.assertEqual(gd[0]["result"], "skip")
        self.assertTrue(gd[0]["reason"])
        self.assertNotIn("verify_started", names(r["events"]))
        self.assertEqual(r["terminal"], "harness_completed")
        line = [x for x in r["status"] if "[2/4]" in x][0]
        self.assertTrue(line.startswith("– [2/4] Formal Verification — not required ("), line)
        self.assertIn("`– [2/4] Formal Verification — not required (<reason>)`", read(".claude/commands/harness.md"))

    def test_case1_julia_tests_run(self):
        if not shutil.which("julia"):
            self.skipTest(unavailable("julia"))
        cmd, rc, out = run_julia_tests("julia-only")
        self.assertEqual(rc, 0, out[-2000:])
        self.assertRegex(out, r"JuliaOnly fixture\s*\|\s*3\s+3")
        self.assertNotIn("npm", cmd)


# ---------------------------------------------------------------------------
# Case 2: algorithm change with valid theorem: Julia tests + Lean build; QE only after Dev+Verify
# ---------------------------------------------------------------------------
class Case2ValidTheorem(unittest.TestCase):
    def test_case2_discovery_julia_and_lean(self):
        d = disc(os.path.join(FIX, "julia-lean"))
        v = d.values
        self.assertEqual(v["kinds"], "julia,lean")
        self.assertEqual(v["lean.build"], "lake build")
        self.assertEqual(v["lean.toolchain"], "leanprover/lean4:v4.15.0")
        self.assertTrue(v["lean.root"].endswith(os.path.join("julia-lean", "lean")))
        self.assertEqual(d.source["lean.root"], "config")

    def test_case2_static_precheck_passes(self):
        res = proof_hygiene.scan(os.path.join(FIX, "julia-lean", "lean"), [THM], proof_hygiene.DEFAULT_APPROVED)
        self.assertTrue(all(r[0] == "PASS" for r in res), res)

    def test_case2_axioms_output_clean(self):
        for f in ("clean.txt", "classical.txt"):
            with open(os.path.join(FIX, "axioms-output", f)) as fh:
                res = proof_hygiene.check_axioms_output(fh.read(), [THM], proof_hygiene.DEFAULT_APPROVED)
            self.assertEqual([r[0] for r in res], ["PASS"], (f, res))

    def test_case2_order_qe_after_dev_and_verify(self):
        r = gate_sim.simulate(True, [{"dev": "pass", "verify": "pass", "qe": "pass"}])
        n = names(r["events"])
        self.assertLess(n.index("dev_finished"), n.index("verify_started"))
        self.assertLess(n.index("verify_finished"), n.index("qe_started"))
        self.assertLess(n.index("qe_finished"), n.index("ops_started"))
        self.assertEqual(r["status"][:3], ["▶ [1/4] Dev", "▶ [2/4] Formal Verification", "▶ [3/4] QE"])
        self.assertEqual(r["terminal"], "harness_completed")

    def test_case2_julia_tests_run(self):
        if not shutil.which("julia"):
            self.skipTest(unavailable("julia"))
        _, rc, out = run_julia_tests("julia-lean")
        self.assertEqual(rc, 0, out[-2000:])

    def test_case2_lake_build_valid(self):
        prepared = [(fx,) + lean_prepare(self, fx) for fx in ("lean-valid", "lean-valid-nextline")]
        for fx, root, tb in prepared:
            with self.subTest(fixture=fx):
                res = proof_hygiene.lean_pipeline(root, ["Fixture"], [THM], proof_hygiene.DEFAULT_APPROVED, tb)
                by = {r[1]: r for r in res}
                self.assertEqual(by["lake-build"][0], "PASS", res)
                self.assertEqual(by["print-axioms:" + THM][0], "PASS", res)
                self.assertEqual(lean_verdict(res), "PASS", res)
                self.assertFalse(os.path.exists(os.path.join(FIX, fx, ".lake")), "repo fixture polluted")


# ---------------------------------------------------------------------------
# Case 3: missing theorem / failed build / sorry / admit / unapproved axiom fail Verify
# ---------------------------------------------------------------------------
class Case3VerifyFailures(unittest.TestCase):
    def scan(self, fixture):
        return {r[1]: r for r in proof_hygiene.scan(os.path.join(FIX, fixture), [THM], proof_hygiene.DEFAULT_APPROVED)}

    def test_case3_valid_fixture_passes(self):
        res = self.scan("lean-valid")
        self.assertTrue(all(r[0] == "PASS" for r in res.values()), res)

    def test_case3_sorry_fails(self):
        res = self.scan("lean-sorry")
        self.assertEqual(res["no-sorry-admit-placeholder"][0], "FAIL")
        self.assertIn("sorry", res["no-sorry-admit-placeholder"][2])

    def test_case3_admit_fails(self):
        res = self.scan("lean-admit")
        self.assertEqual(res["no-sorry-admit-placeholder"][0], "FAIL")
        self.assertIn("admit", res["no-sorry-admit-placeholder"][2])

    def test_case3_unapproved_axiom_fails(self):
        res = self.scan("lean-axiom")
        self.assertEqual(res["no-unapproved-axioms"][0], "FAIL")
        self.assertIn("Fixture.bellman_mono_ax", res["no-unapproved-axioms"][2])
        # ... unless the proof policy explicitly approves it
        ok = proof_hygiene.scan(os.path.join(FIX, "lean-axiom"), [THM],
                                proof_hygiene.DEFAULT_APPROVED + ["Fixture.bellman_mono_ax"])
        self.assertTrue(all(r[0] == "PASS" for r in ok), ok)

    def test_case3_missing_theorem_fails(self):
        res = self.scan("lean-missing-theorem")
        self.assertEqual(res["theorem-present:" + THM][0], "FAIL")

    def test_case3_placeholder_true_fails(self):
        for fx in ("lean-placeholder", "lean-placeholder-multiline"):
            res = self.scan(fx)
            self.assertEqual(res["theorem-present:" + THM][0], "FAIL", fx)
            self.assertIn("True", res["theorem-present:" + THM][2])

    def test_case3_char_literal_does_not_hide_sorry(self):
        res = self.scan("lean-charlit-sorry")
        self.assertEqual(res["no-sorry-admit-placeholder"][0], "FAIL", res)
        self.assertIn("Bellman.lean:9 uses `sorry`", res["no-sorry-admit-placeholder"][2])
        strip = proof_hygiene.strip_comments_and_strings
        for src in ("def q : Char := '\"'\nsorry\ndef r : Char := '\"'",
                    "def q : Char := '\\''\nsorry\ndef r : String := \"'\"",
                    "#eval s!\"{sorry}\"",
                    "def q := r\"\\\" sorry \"x\"",
                    "def q := ⟨'\"', sorry, '\"'⟩"):
            self.assertIn("sorry", strip(src), src)
        for src in ("def q := \"sorry\"", "def q := r#\"a \" sorry\"#", "-- sorry", "/- sorry -/"):
            self.assertNotIn("sorry", strip(src), src)
        self.assertIn("h'", strip("theorem t (h' : p) : p := h'"))

    def test_case3_namespaced_core_axiom_name_fails(self):
        res = self.scan("lean-ns-propext")
        self.assertEqual(res["no-unapproved-axioms"][0], "FAIL", res)
        self.assertIn("Fixture.propext", res["no-unapproved-axioms"][2])

    def test_case3_kernel_bypass_needs_approval(self):
        res = self.scan("lean-kernel-bypass")
        self.assertEqual(res["no-kernel-bypass"][0], "FAIL", res)
        for tag in ("debug.skipKernelTC", "implemented_by", "extern", "unsafe"):
            self.assertIn("`%s`" % tag, res["no-kernel-bypass"][2], tag)
        ok = proof_hygiene.scan(os.path.join(FIX, "lean-kernel-bypass"), [THM], proof_hygiene.DEFAULT_APPROVED,
                                ["debug.skipKernelTC", "implemented_by", "extern", "unsafe"])
        self.assertTrue(all(r[0] == "PASS" for r in ok), ok)

    def test_case3_name_on_next_line_and_multiline_statement(self):
        res = self.scan("lean-valid-nextline")
        self.assertTrue(all(r[0] == "PASS" for r in res.values()), res)

    def test_case3_axioms_output_sorryax_and_custom_fail(self):
        for f in ("sorryax.txt", "custom.txt"):
            with open(os.path.join(FIX, "axioms-output", f)) as fh:
                res = proof_hygiene.check_axioms_output(fh.read(), [THM], proof_hygiene.DEFAULT_APPROVED)
            self.assertEqual(res[0][0], "FAIL", f)
        res = proof_hygiene.check_axioms_output("", [THM], proof_hygiene.DEFAULT_APPROVED)
        self.assertEqual(res[0][0], "FAIL", "missing #print axioms output must fail")

    def test_case3_cli_exit_codes(self):
        py = sys.executable
        script = os.path.join(TOOLS, "proof_hygiene.py")
        ok = subprocess.run([py, script, os.path.join(FIX, "lean-valid"), "--theorem", THM], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stdout)
        self.assertIn("static pre-check only", ok.stdout)
        for bad in ("lean-sorry", "lean-admit", "lean-axiom", "lean-missing-theorem", "lean-placeholder",
                    "lean-charlit-sorry", "lean-ns-propext", "lean-placeholder-multiline", "lean-kernel-bypass"):
            p = subprocess.run([py, script, os.path.join(FIX, bad), "--theorem", THM], capture_output=True, text=True)
            self.assertEqual(p.returncode, 1, (bad, p.stdout))
        emit = subprocess.run([py, script, ".", "--emit-axiom-check", "Fixture", "--theorem", THM], capture_output=True, text=True)
        self.assertEqual(emit.stdout.split("\n")[:2], ["import Fixture", "#print axioms " + THM])

    def test_case3_missing_toolchain_is_blocker(self):
        d = disc(os.path.join(FIX, "lean-no-toolchain"))
        self.assertTrue(any("no pinned lean-toolchain" in b for b in d.blockers), d.blockers)
        d = disc(os.path.join(FIX, "lean-ambiguous"))
        self.assertTrue(any("ambiguous" in b for b in d.blockers), d.blockers)
        d = disc(os.path.join(FIX, "lean-valid"), sets={"lean.toolchain": "leanprover/lean4:v4.0.0"})
        self.assertTrue(any("never switch toolchains" in b for b in d.blockers), d.blockers)
        if not shutil.which("lake"):
            d = disc(os.path.join(FIX, "julia-lean"), require=["lean"])
            self.assertTrue(any("UNAVAILABLE" in b for b in d.blockers), d.blockers)

    def test_case3_verify_failure_blocks_qe_and_ops(self):
        # fails twice -> retry policy exhausted, QE never started, Ops never ran
        r = gate_sim.simulate(True, [{"dev": "pass", "verify": "fail"}, {"dev": "pass", "verify": "fail"}])
        n = names(r["events"])
        self.assertNotIn("qe_started", n)
        self.assertNotIn("ops_started", n)
        self.assertEqual(n[-1], "harness_failed")
        self.assertEqual(r["events"][-1][1]["stage"], "verify")
        # fails once then passes -> QE only in cycle 2, then Ops
        r = gate_sim.simulate(True, [{"dev": "pass", "verify": "fail"}, {"dev": "pass", "verify": "pass", "qe": "pass"}])
        qe = [e[1]["cycle"] for e in r["events"] if e[0] == "qe_started"]
        self.assertEqual(qe, [2])
        self.assertEqual(r["terminal"], "harness_completed")
        # blocker (toolchain unavailable) -> immediate terminal failure
        r = gate_sim.simulate(True, [{"dev": "pass", "verify": "blocked"}, {"dev": "pass", "verify": "pass", "qe": "pass"}])
        self.assertEqual(r["cycles"], 1)
        self.assertFalse(r["ops_ran"])

    def test_case3_lake_pipeline_rejects(self):
        """Real `lake build` + `#print axioms` verdicts. Note `lake build` itself SUCCEEDS
        for sorry (only a warning) -- the axiom check is what rejects it."""
        # fixture -> (obligation that must FAIL, substring of its detail)
        cases = {
            "lean-sorry": ("print-axioms:" + THM, "sorryAx"),
            "lean-charlit-sorry": ("print-axioms:" + THM, "sorryAx"),
            "lean-axiom": ("print-axioms:" + THM, "Fixture.bellman_mono_ax"),
            "lean-ns-propext": ("print-axioms:" + THM, "Fixture.propext"),
            "lean-placeholder-multiline": ("theorem-present:" + THM, "True"),
            "lean-kernel-bypass": ("no-kernel-bypass", "debug.skipKernelTC"),
        }
        prepared = {fx: lean_prepare(self, fx) for fx in cases}  # skips the whole test if UNAVAILABLE
        for fx, (ob, needle) in cases.items():
            with self.subTest(fixture=fx):
                root, tb = prepared[fx]
                res = proof_hygiene.lean_pipeline(root, ["Fixture"], [THM], proof_hygiene.DEFAULT_APPROVED, tb)
                by = {r[1]: r for r in res}
                self.assertEqual(lean_verdict(res), "FAIL", res)
                self.assertEqual(by[ob][0], "FAIL", res)
                self.assertIn(needle, by[ob][2])
                if fx in ("lean-sorry", "lean-axiom"):
                    self.assertEqual(by["lake-build"][0], "PASS", "build alone accepts it: %s" % res)


# ---------------------------------------------------------------------------
# Case 4: Julia test failure fails QE and prevents Ops
# ---------------------------------------------------------------------------
class Case4JuliaFailure(unittest.TestCase):
    def test_case4_qe_failure_prevents_ops(self):
        r = gate_sim.simulate(True, [{"dev": "pass", "verify": "pass", "qe": "fail"},
                                     {"dev": "pass", "verify": "pass", "qe": "fail"}])
        self.assertFalse(r["ops_ran"])
        self.assertNotIn("ops_started", names(r["events"]))
        self.assertEqual(r["events"][-1][1]["stage"], "qe")

    def test_case4_real_julia_failure(self):
        if not shutil.which("julia"):
            self.skipTest(unavailable("julia"))
        cmd, rc, out = run_julia_tests("julia-failing")
        self.assertNotEqual(rc, 0, "failing fixture must fail Pkg.test()")
        qe = gate_sim.qe_gate([{"id": "Pkg.test() on CPU", "result": "pass" if rc == 0 else "fail", "required": True}])
        self.assertEqual(qe, "fail")
        r = gate_sim.simulate(False, [{"dev": "pass", "qe": qe}])
        self.assertFalse(r["ops_ran"])


# ---------------------------------------------------------------------------
# Case 5: GPU PASS / FAIL / UNAVAILABLE distinguished; UNAVAILABLE never pass
# ---------------------------------------------------------------------------
class Case5Gpu(unittest.TestCase):
    def test_case5_gpu_outcomes(self):
        cpu = {"id": "cpu", "result": "pass", "required": True}
        self.assertEqual(gate_sim.qe_gate([cpu, gate_sim.gpu_criterion("PASS", True)]), "pass")
        self.assertEqual(gate_sim.qe_gate([cpu, gate_sim.gpu_criterion("FAIL", True)]), "fail")
        self.assertEqual(gate_sim.qe_gate([cpu, gate_sim.gpu_criterion("FAIL", False)]), "fail")
        self.assertEqual(gate_sim.qe_gate([cpu, gate_sim.gpu_criterion("UNAVAILABLE", True)]), "blocked")
        self.assertNotEqual(gate_sim.qe_gate([gate_sim.gpu_criterion("UNAVAILABLE", True)]), "pass")
        self.assertEqual(gate_sim.qe_gate([cpu, gate_sim.gpu_criterion("UNAVAILABLE", False)]), "pass")
        with self.assertRaises(ValueError):
            gate_sim.gpu_criterion("SKIPPED", True)

    def test_case5_unavailable_required_gpu_blocks_ops(self):
        qe = gate_sim.qe_gate([gate_sim.gpu_criterion("UNAVAILABLE", True)])
        r = gate_sim.simulate(False, [{"dev": "pass", "qe": qe}, {"dev": "pass", "qe": "pass"}])
        self.assertFalse(r["ops_ran"])
        gd = [e[1] for e in r["events"] if e[0] == "gate_decision" and e[1]["gate"] == "qe"]
        self.assertEqual(gd[-1]["result"], "fail")
        self.assertTrue(gd[-1]["blocker"])

    def test_case5_gpu_discovery(self):
        d = disc(os.path.join(FIX, "julia-gpu"), require=["gpu"])
        self.assertEqual(d.values["gpu.code_present"], "yes")
        self.assertEqual(d.values["gpu.backends"], "CUDA")
        self.assertEqual(d.source["gpu.test"], "config")
        self.assertEqual(d.blockers, [])
        # CUDA is only a weakdep/test extra: the probe must not `using CUDA` in the target project
        self.assertIn("gpu_check.py probe", d.values["gpu.probe"])
        self.assertNotIn("using CUDA", d.values["gpu.probe"])
        self.assertIn("gpu_check.py run", d.values["gpu.classifier"])
        root = copy_fixture("julia-gpu")
        os.remove(os.path.join(root, "harness.config.toml"))
        d = disc(root, require=["gpu"])
        self.assertTrue(any("GPU UNAVAILABLE" in b for b in d.blockers), d.blockers)
        shutil.rmtree(os.path.dirname(root), ignore_errors=True)
        self.assertEqual(disc(os.path.join(FIX, "julia-only")).values["gpu.code_present"], "no")

    def test_case5_qe_instructions(self):
        q = read(".claude/agents/qe.md")
        self.assertIn("**PASS**", q)
        self.assertIn("**FAIL**", q)
        self.assertIn("**UNAVAILABLE**", q)
        self.assertIn("UNAVAILABLE is an unmet criterion / blocker — never a pass", q)
        self.assertIn("A skipped GPU test set must never be reported as passed", q)
        self.assertIn("gpu_check.py", q)
        self.assertIn("The exit code alone never yields PASS", q)
        self.assertIn("a probe error is not evidence of \"no GPU\"", q)

    def test_case5_gpu_classifier(self):
        c = gpu_check.classify
        summary_ok = ("Test Summary:      | Pass  Total  Time\n"
                      "JuliaGpu CUDA path |    3      3  7.3s\n")
        summary_fail = ("Test Summary:             | Pass  Fail  Total  Time\n"
                        "JuliaGpuFailing CUDA path |    2     1      3  9.3s\n")
        cpu_only = ("Test Summary:     | Pass  Total  Time\n"
                    "JuliaGpu CPU path |    1      1  0.2s\n     Testing JuliaGpu tests passed\n")
        self.assertEqual(c(cpu_only, 0)[0], "FAIL", "exit 0 with zero GPU tests must never be PASS")
        self.assertEqual(c("", 0)[0], "FAIL")
        self.assertEqual(c(summary_ok, 0)[0], "PASS")
        self.assertEqual(c("HARNESS_GPU_TESTS_RAN: device=x", 0)[0], "PASS")
        self.assertEqual(c(summary_ok, 1)[0], "FAIL")
        self.assertEqual(c(summary_fail, 1)[0], "FAIL")
        self.assertEqual(c(summary_fail + "HARNESS_GPU_TESTS_RAN", 0)[0], "FAIL")
        self.assertEqual(c("HARNESS_GPU_UNAVAILABLE: CUDA.functional()=false", 1)[0], "UNAVAILABLE")
        self.assertEqual(c(summary_ok, 0, "NOT_FUNCTIONAL", "x")[0], "UNAVAILABLE")
        o, r = c(summary_ok, 0, "ERROR", "Package CUDA not found")
        self.assertEqual(o, "FAIL")
        self.assertIn("not evidence of missing GPU", r)
        rows = dict(gpu_check.parse_test_summaries(summary_fail + cpu_only))
        self.assertEqual(rows["JuliaGpuFailing CUDA path"], {"Pass": 2, "Fail": 1, "Total": 3})
        self.assertEqual(rows["JuliaGpu CPU path"], {"Pass": 1, "Total": 1})
        for o in ("PASS", "FAIL", "UNAVAILABLE"):
            gate_sim.gpu_criterion(o, True)  # classifier outcomes map onto the QE gate

    def test_case5_real_cpu_only_command_is_not_gpu_pass(self):
        """The skipped-counted-as-pass hazard, for real: plain Pkg.test() on the GPU fixture
        exits 0 but runs no GPU testset -> classifier says FAIL, never PASS."""
        if not shutil.which("julia"):
            self.skipTest(unavailable("julia"))
        root = copy_fixture("julia-gpu")
        self.addCleanup(shutil.rmtree, os.path.dirname(root), True)
        res = gpu_check.run(root, disc(root).values["julia.test"], do_probe=False)
        self.assertEqual(res["exit_code"], 0, res["output"][-2000:])
        self.assertEqual(res["outcome"], "FAIL", res["reason"])
        self.assertIn("no GPU testset ran", res["reason"])


class Case5RealGpu(unittest.TestCase):
    """Real GPU PASS / FAIL / UNAVAILABLE classification on temp fixture copies."""
    probe_state = None

    @classmethod
    def setUpClass(cls):
        if shutil.which("julia"):
            tmp = copy_fixture("julia-gpu")
            try:
                cls.probe_state = gpu_check.probe(tmp)  # temp env: develop copy + add CUDA (offline)
            finally:
                shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)

    def gpu_run(self, fixture, env_extra=None):
        if not shutil.which("julia"):
            self.skipTest(unavailable("julia"))
        state, detail = self.probe_state
        if state == "NOT_FUNCTIONAL":
            self.skipTest("UNAVAILABLE: GPU probe NOT_FUNCTIONAL (%s) -- real GPU PASS/FAIL not exercised" % detail)
        if state != "FUNCTIONAL":
            self.skipTest("UNAVAILABLE: GPU probe ERROR -- CUDA.jl could not be resolved/loaded offline from the "
                          "local depot (%s); real GPU PASS/FAIL not exercised" % detail)
        root = copy_fixture(fixture)
        self.addCleanup(shutil.rmtree, os.path.dirname(root), True)
        d = disc(root, require=["gpu"])
        self.assertEqual(d.source["gpu.test"], "config")
        res = gpu_check.run(root, d.values["gpu.test"], do_probe=False, env_extra=env_extra)
        sys.stderr.write("[GPU %s: %s -- %s] " % (fixture, res["outcome"], res["reason"][:160]))
        return res

    def test_case5_real_gpu_pass(self):
        res = self.gpu_run("julia-gpu")
        self.assertEqual(res["outcome"], "PASS", res["reason"] + res["output"][-3000:])
        self.assertRegex(res["output"], r"JuliaGpu CUDA path\s*\|\s*3\s+3")
        self.assertIn("HARNESS_GPU_TESTS_RAN", res["output"])

    def test_case5_real_gpu_fail(self):
        res = self.gpu_run("julia-gpu-failing")
        self.assertEqual(res["outcome"], "FAIL", res["reason"] + res["output"][-3000:])
        self.assertIn("GPU testset failures", res["reason"])
        self.assertNotIn("HARNESS_GPU_TESTS_RAN", res["output"])

    def test_case5_real_gpu_masked_device_unavailable(self):
        res = self.gpu_run("julia-gpu", env_extra={"CUDA_VISIBLE_DEVICES": "-1"})
        self.assertEqual(res["outcome"], "UNAVAILABLE", res["reason"] + res["output"][-3000:])
        self.assertIn("HARNESS_GPU_UNAVAILABLE", res["reason"])
        self.assertEqual(gate_sim.qe_gate([gate_sim.gpu_criterion(res["outcome"], True)]), "blocked")


# ---------------------------------------------------------------------------
# Case 6: performance-scoped change: benchmark evidence, no brittle timing thresholds
# ---------------------------------------------------------------------------
class Case6Performance(unittest.TestCase):
    def test_case6_timing_lint(self):
        bad = timing_lint.lint(os.path.join(FIX, "julia-perf-bad"))
        self.assertEqual(len(bad), 4, bad)
        self.assertTrue(any("median(b).time" in f for f in bad), bad)
        self.assertTrue(any("m.time" in f for f in bad), bad)
        self.assertFalse(any("@ballocated" in f for f in bad), "allocation checks must not be flagged")
        self.assertEqual(timing_lint.lint(os.path.join(FIX, "julia-perf-good")), [])
        self.assertEqual(timing_lint.lint(os.path.join(FIX, "julia-only")), [])

    def test_case6_instructions_require_benchmark_evidence(self):
        q = read(".claude/agents/qe.md")
        self.assertIn("Performance-scoped specs only", q)
        self.assertIn("baseline ref vs changed ref", q)
        self.assertIn("Do not turn benchmarks into timing assertions", q)
        self.assertIn("timing_lint.py", q)
        t = read("specs/TEMPLATE-julia-lean.md")
        self.assertIn("Baseline ref vs changed ref", t)
        self.assertIn("No timing thresholds in unit tests", t)
        self.assertTrue(os.path.isfile(os.path.join(FIX, "julia-perf-good", "benchmark", "benchmarks.jl")))

    def test_case6_benchmark_is_configurable(self):
        d = disc(os.path.join(FIX, "julia-perf-good"), sets={"benchmark.cmd": "julia --project=. benchmark/benchmarks.jl"})
        self.assertEqual(d.source["benchmark.cmd"], "spec")

    def test_case6_correct_fixture_tests_pass(self):
        if not shutil.which("julia"):
            self.skipTest(unavailable("julia"))
        _, rc, out = run_julia_tests("julia-perf-good")
        self.assertEqual(rc, 0, out[-2000:])


# ---------------------------------------------------------------------------
# Case 7: loop-back, two-cycle maximum, terminal telemetry, Ops-only-after-all-gates
# ---------------------------------------------------------------------------
class Case7GateStateMachine(unittest.TestCase):
    def test_case7_max_cycles_matches_docs(self):
        self.assertEqual(gate_sim.MAX_CYCLES, 2)
        self.assertIn("At most **2** Dev→gate cycles in total", read(".claude/commands/harness.md"))
        self.assertIn("Maximum **2** Dev → gate cycles", read("README.md"))

    def test_case7_loopback_then_success(self):
        r = gate_sim.simulate(True, [{"dev": "pass", "verify": "pass", "qe": "fail"},
                                     {"dev": "pass", "verify": "pass", "qe": "pass"}])
        lb = [e for e in r["events"] if e[0] == "loopback"]
        self.assertEqual(len(lb), 1)
        self.assertEqual(lb[0][1], {"cycle": 2, "from_gate": "qe"})
        self.assertIn("↻ [1/4] Dev", r["status"])
        self.assertEqual(r["terminal"], "harness_completed")

    def test_case7_two_cycle_maximum(self):
        always_fail = [{"dev": "pass", "verify": "pass", "qe": "fail"}] * 5
        r = gate_sim.simulate(True, always_fail)
        self.assertEqual(names(r["events"]).count("dev_started"), 2)
        self.assertEqual(r["cycles"], 2)
        self.assertEqual(r["terminal"], "harness_failed")
        self.assertIn("exhausted", r["events"][-1][1]["reason"])

    def test_case7_exhaustive_invariants(self):
        outcomes = ["pass", "fail", "blocked"]
        combos = [{"dev": d, "verify": v, "qe": q} for d in outcomes for v in outcomes for q in outcomes]
        scenarios = [[c] for c in combos] + [[a, b] for a in combos for b in combos]
        for req in (True, False):
            for sc in scenarios:
                r = gate_sim.simulate(req, sc)
                n = names(r["events"])
                terminals = [x for x in n if x in ("harness_completed", "harness_failed")]
                self.assertEqual(len(terminals), 1, sc)
                self.assertEqual(n[-1], terminals[0], sc)
                self.assertLessEqual(n.count("dev_started"), 2)
                if r["ops_ran"]:
                    last = r["cycles"]
                    self.assertTrue(gate_sim.ops_allowed(r["events"], last), sc)
                    spec = sc[last - 1]
                    self.assertEqual(spec["dev"], "pass")
                    self.assertEqual(spec["qe"], "pass")
                    if req:
                        self.assertEqual(spec["verify"], "pass")
                else:
                    self.assertNotIn("ops_started", n)
                # QE never starts in a cycle whose verify gate did not pass/skip
                for cyc in (1, 2):
                    v = [e[1]["result"] for e in r["events"] if e[0] == "gate_decision"
                         and e[1]["gate"] == "verify" and e[1]["cycle"] == cyc]
                    qe = [e for e in r["events"] if e[0] == "qe_started" and e[1]["cycle"] == cyc]
                    if qe:
                        self.assertIn(v[0], ("pass", "skip"), sc)

    def test_case7_events_written_to_telemetry_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            sc = os.path.join(tmp, "s.json")
            db = os.path.join(tmp, "t.db")
            with open(sc, "w") as fh:
                json.dump({"verify_required": True, "cycles": [{"dev": "pass", "verify": "fail"},
                                                               {"dev": "pass", "verify": "fail"}]}, fh)
            p = subprocess.run([sys.executable, os.path.join(TOOLS, "gate_sim.py"), sc, "--db", db],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
            rows = [r[0] for r in sqlite3.connect(db).execute("select event_name from events order by id")]
            self.assertEqual(rows[-1], "harness_failed")
            self.assertIn("verify_finished", rows)
            self.assertIn("loopback", rows)
            self.assertNotIn("ops_started", rows)

    def test_case7_ops_failure_is_harness_failed(self):
        for ops in ("fail", "blocked"):
            r = gate_sim.simulate(True, [{"dev": "pass", "verify": "pass", "qe": "pass"}], ops=ops)
            n = names(r["events"])
            self.assertEqual(n[-1], "harness_failed")
            self.assertNotIn("harness_completed", n)
            self.assertEqual([e[1]["status"] for e in r["events"] if e[0] == "ops_finished"], [ops])
        h = read(".claude/commands/harness.md")
        self.assertIn("record `ops_finished` with `status: \"fail\"` or `status: \"blocked\"`", h)
        self.assertIn("**never** `harness_completed`", h)

    def test_case7_ops_refuses_and_detects_env(self):
        o = read(".claude/agents/ops.md")
        self.assertIn("refuse unless every required gate passed", o)
        self.assertIn("not a git repository", o)
        self.assertIn("`gh` is missing", o)
        self.assertIn("blocker", o)


# ---------------------------------------------------------------------------
# Tool-level checks
# ---------------------------------------------------------------------------
class ToolTests(unittest.TestCase):
    def test_tool_config_precedence(self):
        root = os.path.join(FIX, "julia-config")
        d = disc(root)
        self.assertEqual(d.values["julia.test"], "julia --project=. test/runtests.jl")
        self.assertEqual(d.source["julia.test"], "config")
        d = disc(root, sets={"julia.test": "julia --project=. -e 'using Pkg; Pkg.test(; coverage=false)'"})
        self.assertEqual(d.source["julia.test"], "spec")
        d = disc(os.path.join(FIX, "julia-only"))
        self.assertEqual(d.source["julia.test"], "discovered")

    def test_tool_empty_target_is_blocker(self):
        p = subprocess.run([sys.executable, os.path.join(TOOLS, "discover.py"), os.path.join(FIX, "empty")],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("blocker=no Julia/Lean project detected", p.stdout)
        p = subprocess.run([sys.executable, os.path.join(TOOLS, "discover.py"), "/nonexistent/target"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)

    def test_tool_julia_compat(self):
        vs = discover.version_satisfies
        self.assertTrue(vs("1.10.4", "1.9"))
        self.assertTrue(vs("1.9.0", "1.9"))
        self.assertFalse(vs("1.8.5", "1.9"))
        self.assertFalse(vs("2.0.0", "1.9"))
        self.assertTrue(vs("1.6.7", "1.6, 1.9"))
        self.assertFalse(vs("1.10.0", "~1.9"))
        self.assertTrue(vs("1.11.0", ">= 1.10"))
        self.assertTrue(vs("1.11.2", "1.9 - 1.11"))
        self.assertFalse(vs("1.12.0", "1.9 - 1.11"))

    def test_tool_julia_version_blocker(self):
        if not shutil.which("julia"):
            self.skipTest(unavailable("julia"))
        d = disc(os.path.join(FIX, "julia-only"), sets={"julia.compat": "0.7"})
        self.assertTrue(any("does not satisfy compat" in b for b in d.blockers), d.blockers)

    def test_tool_runner_exit_codes(self):
        import run as runner  # tests/harness/run.py
        src = ("import unittest\n"
               "class T(unittest.TestCase):\n"
               "    def test_a(self): pass\n"
               "    def test_b(self): self.skipTest('UNAVAILABLE: x')\n")
        for body, want, label in ((src, 2, "PASS-INCOMPLETE"), (src.replace("self.skipTest('UNAVAILABLE: x')", "pass"), 0, "RESULT: PASS"),
                                  (src + "    def test_c(self): self.fail('x')\n", 1, "RESULT: FAIL"),
                                  # a test whose subTests were all skipped must not count as passed
                                  ("import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n"
                                   "        with self.subTest(i=0):\n            self.skipTest('UNAVAILABLE: y')\n",
                                   2, "pass=0 fail=0 unavailable=1")):
            with tempfile.TemporaryDirectory() as tmp:
                with open(os.path.join(tmp, "test_fake.py"), "w") as fh:
                    fh.write(body)
                p = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); import run; "
                                    "sys.exit(run.main(%r))" % (HERE, tmp)], capture_output=True, text=True)
                self.assertEqual(p.returncode, want, p.stdout + p.stderr)
                self.assertIn(label, p.stdout)
        self.assertTrue(callable(runner.main))

    def test_tool_comments_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "A.lean"), "w") as fh:
                fh.write('/- sorry /- nested admit -/ axiom foo : False -/\n-- sorry\n'
                         'def s : String := "sorry admit"\ntheorem t : 1 = 1 := rfl\n'
                         'theorem sorry_free : 2 = 2 := rfl\n')
            res = proof_hygiene.scan(tmp, ["t", "sorry_free"], proof_hygiene.DEFAULT_APPROVED)
            self.assertTrue(all(r[0] == "PASS" for r in res), res)
            with open(os.path.join(tmp, "B.lean"), "w") as fh:
                fh.write("theorem u : 1 = 1 := by native_decide\n")
            res = {r[1]: r[0] for r in proof_hygiene.scan(tmp, [], proof_hygiene.DEFAULT_APPROVED)}
            self.assertEqual(res["no-sorry-admit-placeholder"], "FAIL")


if __name__ == "__main__":
    unittest.main()
