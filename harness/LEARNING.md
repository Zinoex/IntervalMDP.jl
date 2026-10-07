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

### 2026-10-05 — Ops — Check that the git repo and remote really exist in the target dir
- **Problem:** The operator said a repo had been initialised, but `/home/fresen/Documents/rmdp` was not a git repository and had no remote.
- **Root cause:** The operator's "repo initialised" referred to a different directory (or to a step not yet done).
- **Fix / guardrail:** Ops runs `git rev-parse --is-inside-work-tree` and `git remote -v` in the target root before doing anything. With orchestrator authorisation, a local `git init` is allowed. Never create a remote or GitHub repo, or fork upstream, without the operator's explicit choice. Commit locally and report the push/PR commands as a blocker.

### 2026-10-05 — Ops — Harness moved into IntervalMDP.jl under `harness/`
- **Problem:** The harness began as a standalone repo (`rmdp`) with `tools/`, `tests/`, `specs/` and `README.md` at its root. Those collide with the package's own `test/`, `README.md` and `.gitignore`, and JuliaFormatter's `format(".")` CI step would reformat the deliberately unformatted Julia fixtures.
- **Root cause:** Every harness path was written relative to a standalone repo root.
- **Fix / guardrail:** `.claude/` and `.mcp.json` stay at the repo root, because Claude Code only finds them there; everything else lives under `harness/`. Docs and agents use `harness/...` paths, and the tests resolve `.claude/` one level above `harness/`. `.JuliaFormatter.toml` has `ignore = ["harness"]`. The default target root is now the repository root, never `harness/` or its fixtures. Run `sh harness/tests/harness/run.sh` from the repo root after moving files.

### 2026-10-05 — QE — Julia 1.13 changed `CartesianIndex` printing
- **Problem:** `test/base/specification.jl` failed 11 tests on Julia 1.13.1.
- **Root cause:** Julia 1.13 prints `CartesianIndex` without the trailing comma, and the expected `show` strings were hard-coded for the old format.
- **Fix / guardrail:** Build expected `show` strings by interpolating the same value (`"... $(CartesianIndex(...)) ..."`) instead of hard-coding the printed form. Keep such fixes in their own commit, separate from feature work.

### 2026-10-05 — QE/Ops — `Pkg.test()` rewrites a tracked data file
- **Problem:** Running the test suite modifies `test/data/multiObj_robotIMDP.nc`, which leaves a dirty working tree.
- **Root cause:** A test writes the NetCDF file back to the tracked fixture path.
- **Fix / guardrail:** After test runs, `git checkout -- test/data/multiObj_robotIMDP.nc`. Ops must never commit it as part of an unrelated change.

### 2026-10-05 — Dev/Formal Verification — Choose a Mathlib tag that matches an installed toolchain
- **Problem:** A Mathlib tag whose `lean-toolchain` is not installed triggers a silent toolchain download through the elan proxy.
- **Root cause:** Each Mathlib tag pins its own Lean toolchain.
- **Fix / guardrail:** Pick a Mathlib tag whose `lean-toolchain` equals an installed toolchain (Phase 0 used Mathlib v4.33.0-rc2). Run `lake` from `~/.elan/toolchains/<tc>/bin`, never through the proxy. Commit `lake-manifest.json`, and add `lean/.lake/` to `.gitignore`.

### 2026-10-05 — QE — Spec docstring rules need a machine check
- **Problem:** QE failed cycle 1 because 83 Lean docstrings had no Julia counterpart reference, even though the spec required one.
- **Root cause:** The rule was prose in the spec, so neither Dev nor Verify checked it mechanically.
- **Fix / guardrail:** `lean/scripts/check_julia_refs.py` enforces it now. Any spec rule about documentation or traceability should come with a script that Dev runs before handing off.

### 2026-10-05 — Dev/Planner — Raise modeling conflicts with the literature as Findings
- **Problem:** Dev initially modeled the factored IMDP ambiguity set as a convex hull, but the literature (arXiv:2411.11803, arXiv:2508.00707) uses the literal, non-convex product set.
- **Root cause:** A modeling choice that made proofs easier quietly diverged from the reference semantics.
- **Fix / guardrail:** Surface such conflicts as Findings (F-numbered) for the operator, not silent resolutions. Phase 0 resolved F1 by modeling the literal `productSet` and proving `productSet_not_convex`.

### 2026-10-05 — Ops — Check that local main is not ahead of origin before opening a PR
- **Problem:** If local `main` has unpushed commits (e.g. local merges), a feature branch from HEAD would carry them into the PR.
- **Root cause:** Merges made locally on `main` are not on `origin/main`.
- **Fix / guardrail:** Run `git fetch origin` and `git rev-list --count origin/main..main` before branching. If the count is non-zero, do not push or open a PR. Report the commits and the operator commands (`git push origin main`, or rebase the branch onto `origin/main`). In this run the count was 0, so the PR went ahead. Also, the shell here is zsh: `R="python3 x.py"; $R ...` does not word-split, so define a function `R(){ python3 x.py "$@"; }` for the telemetry wrapper.

### 2026-10-06 — Orchestrator — Split long phases into sub-phase spec files
- **Problem:** A single Phase 0 run outgrew the Dev context window.
- **Root cause:** One spec covered the whole phase (suite, baseline, profiles, report), so Dev had to load and act on all of it in one session.
- **Fix / guardrail:** Specs are now one file per sub-phase (`harness/specs/perf-vi-bellman/`, `harness/specs/lean-proofs/`) with a shared `common.md`, a "read only these sections" line and a context-budget section. Sub-phase 0a ran with Dev at about 65k tokens.

### 2026-10-06 — Ops — Committing a sub-phase out of a pre-existing working tree
- **Problem:** The working tree held changes for several sub-phases and for the harness itself, so a broad `git add` would mix unrelated work into the 0a commit.
- **Root cause:** Sub-phases share one branch (`perf/phase0`) and one working tree. Some files are also ignored: `benchmark/Manifest.toml` is in `.gitignore`.
- **Fix / guardrail:** Stage explicit paths only and check `git diff --cached --name-status` against the expected list. Include every file the entry point `include`s, even if a later sub-phase owns it, so the commit runs on its own. Run `git check-ignore -v` on files the spec says to commit. `benchmark/Manifest.toml` needs `git add -f`.

### 2026-10-06 — Ops — Feature branch built on unpushed local commits blocks the PR
- **Problem:** `perf/phase0` was branched from a local HEAD that carried 2 commits not on `origin/main` (`91bc0c8` "Remove leftover", which deletes `harness/specs/hello-world-api.md`, and merge `f775c17`). `origin/main..main` was 0, so the existing check passed while the branch was still not clean.
- **Root cause:** The check looked at local `main`, but the branch had been created from another local branch (`lean/phase0-models`) after its PR merged.
- **Fix / guardrail:** Always check `git rev-list --count origin/main..HEAD` as well as `origin/main..main`. If it is non-zero, commit locally only and hand the operator the commands. Either push the leftover commit to main first, or rebase the sub-phase commits onto `origin/main`. Planners should create phase branches from a freshly fetched `origin/main`.

### 2026-10-07 — Ops — Rebase a sub-phase commit off unpushed local commits with `--autostash`
- **Problem:** The 0a commit on `perf/phase0` sat on top of 2 local-only commits (`91bc0c8`, merge `f775c17`). The working tree also held many unrelated uncommitted changes that had to survive.
- **Root cause:** The branch had been created from a local branch after its PR merged, not from a freshly fetched `origin/main`.
- **Fix / guardrail:** Run `git fetch origin`, record `git rev-parse HEAD` as a backup ref, then run `git rebase --autostash --onto origin/main <old-base> <branch>`. Autostash stashes the uncommitted work and puts it back afterwards. If there is a conflict, run `git rebase --abort` and check `git stash list`. Before pushing, verify that `git log --oneline origin/main..HEAD` is exactly the sub-phase commit. Also check `git diff --name-status origin/main HEAD` against the expected paths and compare `git status --short | wc -l` with its earlier count. Files deleted by the dropped commits come back into the tree; this is expected. In this run it was clean: `08958a0` became `333f674`, the 21 status lines were unchanged, and draft PR #112 was opened.

### 2026-10-07 — Operator — Node.js support removed from the harness
- **Problem:** The Node.js parts of the harness (the hello-world sample spec and fixture, Node discovery, and the telemetry MCP server launched with `node` via `.mcp.json`) were unused on this host (no `node`) and kept Node assumptions alive in agents, tests and docs.
- **Root cause:** The harness started as a Node.js sample project and kept that workflow after moving to Julia/Lean.
- **Fix / guardrail:** The operator asked to remove all Node.js support. Telemetry is now written only by `python3 harness/tools/harness/record_event.py <eventName> '<json>'` to `harness/tools/telemetry/telemetry.db` (gitignored; the existing DB was moved there). This supersedes the 2026-10-05 lessons that mention the telemetry MCP server and `.mcp.json`. `test_static_no_node_tooling` checks that no `package.json`, `.mcp.json`, `telemetry-mcp/` or `mcp__telemetry` reference comes back.
