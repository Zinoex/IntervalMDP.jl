# Harness Learnings

This file is the harness's accumulated memory. It is the counterpart to the
SQLite telemetry database: telemetry records *what happened* (which agent ran,
when, and with what result), while this file records *what we learned from it*.
Together they are how the harness improves itself over time.

## How agents use this file

- **On start / when an issue comes up.** Every agent (Planner, Dev, QE, Ops)
  reads this file before it begins, and consults it again whenever it hits a
  problem — a failing test, an ambiguous spec, a broken command, a flaky
  environment. If a relevant lesson is recorded here, apply it instead of
  rediscovering the fix.
- **At the end of each harness run.** The workflow appends any new lessons from
  the run to the log below — what went wrong, the root cause, and the fix or
  guardrail that resolved it. Prefer durable, reusable lessons over one-off
  notes.

## Format

Each entry should be dated and attributed to the stage that learned it:

```
### YYYY-MM-DD — <stage> — <short title>
- **Problem:** what went wrong.
- **Root cause:** why it happened.
- **Fix / guardrail:** what to do next time (and where it was encoded — prompt,
  spec template, command, etc.).
```

## Lessons learned


### 2026-10-05 — Orchestrator — Telemetry MCP needs node; use a fallback recorder
- **Problem:** The telemetry MCP server did not start, so `mcp__telemetry__recordTelemetry` was not available.
- **Root cause:** `node` is not installed on the host, and `.mcp.json` launches the server with `node`.
- **Fix / guardrail:** Use `python3 harness/tools/harness/record_event.py <eventName> '<json>'`, which writes the same SQLite schema to `harness/tools/telemetry-mcp/telemetry.db`. Check `command -v node` at run start and switch to the fallback right away.

### 2026-10-05 — QE — UNAVAILABLE GPU verdict given with a false reason
- **Problem:** In QE cycle 1, a GPU check was marked UNAVAILABLE and the reason given was wrong: CUDA actually worked on the host.
- **Root cause:** QE assumed the GPU was unavailable without probing it. A CPU-only test run that exits 0 was then treated as acceptable.
- **Fix / guardrail:** Probe the GPU for real (`harness/tools/harness/gpu_check.py`). Report PASS/FAIL/UNAVAILABLE with the probe output as evidence. A CPU-only exit 0 is never a GPU pass.

### 2026-10-05 — QE — `using CUDA` fails when CUDA is a weakdep
- **Problem:** `using CUDA` failed in the target project environment.
- **Root cause:** CUDA is declared as a weak dependency (package extension), not a direct dependency, so it cannot be loaded in the project env directly.
- **Fix / guardrail:** Probe CUDA in a temporary environment. In projects, list CUDA under `[extras]` and in the `test` target so test runs load the extension.

### 2026-10-05 — Formal Verification — `lake build` passes with `sorry`; `#print axioms` is mandatory
- **Problem:** Proofs that contained `sorry` (or custom axioms) still built successfully.
- **Root cause:** Lean reports `sorry` only as a warning. Static text scanners can be bypassed: char literals, namespaced axioms (e.g. `Foo.propext`), multi-line placeholders, and custom `elab` tactics calling `admitGoal`.
- **Fix / guardrail:** Run `#print axioms` on every required theorem and allow only the standard axioms (`proof_hygiene.py`). Treat the static scan as a pre-check only.

### 2026-10-05 — Formal Verification — The elan proxy silently downloads pinned toolchains
- **Problem:** Running `lake`/`lean` in a directory pinned to a toolchain that is not installed starts a silent toolchain download.
- **Root cause:** The elan proxy binaries auto-install whatever `lean-toolchain` names. Also, `~/.elan/bin` is not on the default PATH on this host.
- **Fix / guardrail:** Never run the elan proxy in such directories. Use the absolute `~/.elan/toolchains/<tc>/bin/lake` on temporary copies. If the pinned toolchain is missing, report UNAVAILABLE.

### 2026-10-05 — QE — UNAVAILABLE or skipped tests must not count as passes
- **Problem:** Tests that were skipped or UNAVAILABLE could make the run look fully green.
- **Root cause:** The runner reported success whenever no test failed.
- **Fix / guardrail:** `harness/tests/harness/run.py` exits with 0 = PASS, 1 = FAIL and 2 = PASS-INCOMPLETE (some tests UNAVAILABLE). Report the incomplete count and the reason for each. (Follow-up N5: skips not marked "UNAVAILABLE:" still do not affect RESULT.)

### 2026-10-05 — Orchestrator — A spec file vanished mid-run
- **Problem:** `harness/specs/hello-world-api.md` disappeared during the run. No agent removed it.
- **Root cause:** Something outside the agents changed the working tree.
- **Fix / guardrail:** At run start, snapshot (or hash) the spec and fixture files, and re-check them before each gate. If inputs change unexpectedly, report it instead of carrying on silently.

### 2026-10-05 — Ops — Check that the git repo and remote really exist in the target dir
- **Problem:** The operator said a repo had been initialised, but `/home/fresen/Documents/rmdp` was not a git repository and had no remote.
- **Root cause:** The operator's "repo initialised" referred to a different directory (or to a step not yet done).
- **Fix / guardrail:** Ops runs `git rev-parse --is-inside-work-tree` and `git remote -v` in the target root before doing anything. With orchestrator authorisation, a local `git init` is allowed. Never create a remote or GitHub repo, or fork upstream, without the operator's explicit choice. Commit locally and report the push/PR commands as a blocker.

### 2026-10-05 — Ops — Harness moved into IntervalMDP.jl under `harness/`
- **Problem:** The harness began as a standalone repo (`rmdp`) with `tools/`, `tests/`, `specs/` and `README.md` at its root. Those collide with the package's own `test/`, `README.md` and `.gitignore`, and JuliaFormatter's `format(".")` CI step would reformat the deliberately unformatted Julia fixtures.
- **Root cause:** Every harness path was written relative to a standalone repo root.
- **Fix / guardrail:** `.claude/` and `.mcp.json` stay at the repo root, because Claude Code only finds them there; everything else lives under `harness/`. Docs and agents use `harness/...` paths, and the tests resolve `.claude/` one level above `harness/`. `.JuliaFormatter.toml` has `ignore = ["harness"]`. The default target root is now the repository root, never `harness/` or its fixtures. Run `sh harness/tests/harness/run.sh` from the repo root after moving files.
