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

### 2026-10-07 — QE — `compare.jl` gives verdicts from a single A/B round
- **Problem:** `benchmark/compare.jl` prints speedup/regression verdicts (5% / 3% thresholds) from one A/B run pair and gives no warning, while the spec § Evidence Protocol requires at least 3 interleaved rounds before a performance claim.
- **Root cause:** Sub-phase 0b only scoped the threshold and CLOCK logic; the round-count rule lives in a different spec section.
- **Fix / guardrail:** Noted for 0h and listed as a known limitation in PR #112. Until it is fixed, QE must check the round count by hand before accepting any speedup claim, and `compare.jl` should warn (or refuse a verdict) when fewer than 3 interleaved rounds are given.

### 2026-10-07 — Dev/Planner — Specs must state complete commands
- **Problem:** The 0b spec's reference-write command (`run.jl --write-reference --reference-only`) left out `--out`, which `run.jl` required, so the command as written failed.
- **Root cause:** The spec was written against the intended interface, not the one 0a shipped.
- **Fix / guardrail:** Dev made `--out` optional with `--reference-only` and defaulted it into the gitignored `benchmark/results/logs/`. Planners should copy commands from a dry run (or `--help`) of the current tool, and Dev should raise a mismatch as a Finding rather than guess.

### 2026-10-07 — QE — Check for competing processes before benchmarking
- **Problem:** The quick suite took 43 minutes in 0b. The clock-stability probe ratio was about 2.2 because other `julia -t12` sessions and VS Code were running on the host.
- **Root cause:** No check for load from other processes before a timed run.
- **Fix / guardrail:** Before any timed run, check `uptime` and `ps -eo pid,pcpu,comm --sort=-pcpu | head` and stop or wait for other heavy processes (especially other Julia sessions). Treat a probe ratio well above 1 as a reason to rerun, not as data.

### 2026-10-07 — Ops — Sub-phase on an already-pushed branch: push and update the existing PR
- **Problem:** Later sub-phases share `perf/phase0` and draft PR #112, so a new PR per sub-phase would split the review.
- **Root cause:** One branch and one PR per phase, many sub-phase commits.
- **Fix / guardrail:** Check `gh pr list --head <branch> --state all` and `git log origin/<branch>..HEAD` / `HEAD..origin/<branch>`. If both are empty and a PR exists, commit with explicit-path staging, push normally, then fetch the body with `gh pr view N --json body -q .body`, insert a `### 0x` section before `### Checklist`, tick the checklist item, and apply with `gh pr edit N --body-file`. In 0b this went cleanly (`a9dc02e`). LEARNING.md edits are left uncommitted for the operator, who commits them with the harness changes.

### 2026-10-07 — QE — Check AC power, EPP and governor before timed spot-checks
- **Problem:** The 0c QE spot-check (t=1, 14 entries) came out 7–18% slower than the stored baseline, and 10 of the 14 entries were flagged CLOCK.
- **Root cause:** First attributed to the power state: the host was on battery with EPP `balance_power` during QE, while the baseline had run on AC with EPP `balance_performance`. **Corrected after 0d:** an AC re-run with the baseline's environment (EPP `balance_performance`, governor `powersave`, platform_profile `balanced`, load ~1) reproduced the same offsets (e.g. robot bellman ×1.186 vs ×1.166 on battery) with the clock probe at ~435.8 µs, the same 2.3 GHz-cap state as the baseline. Battery/EPP was **not** the cause; see "Stored baselines recorded with an uncommitted harness may not be reproducible" below.
- **Fix / guardrail:** Before any timed run, check and record the power state: `cat /sys/class/power_supply/*/online`, `cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference` and `scaling_governor`. Compare them with the environment block of the file you compare against, and run only on AC with the same EPP. This is still good practice because it rules out one confounder, but a matching power state does not by itself make stored numbers reproducible. Absolute numbers stored from an earlier session are not comparable to new ones. Later phases must make claims from at least 3 interleaved A/B rounds run in one session, never against stored baseline medians.

### 2026-10-07 — Dev/QE — The clock guard flags nearly every t=8/t=16 entry
- **Problem:** In the CPU baseline, 54–59 of 131 entries at t=8 and 128–129 at t=16 are clock-deviating (marked with a dagger), so the flag means almost nothing at high thread counts.
- **Root cause:** The run-start probe reference is taken unloaded, while the per-trial probe runs under all-core load. That load lowers the clock, so the probe is about 1.28x slower. The deviation is systematic, not noise.
- **Fix / guardrail:** Noted for 0h: take the reference probe under the same thread load, or use a per-thread-count tolerance. Until then, read t=16 CLOCK as "consistent within run" and don't treat it as a reason to rerun. Still treat CLOCK at t=1/t=4 as real.

### 2026-10-07 — Orchestrator — Resume, don't redo: validate existing measurement files
- **Problem:** Sub-phase 0c (CPU baseline) called for about 10 h of full-suite runs at t=1/4/8/16 with re-runs.
- **Root cause:** The eight baseline files already existed from an earlier session, but the spec was written as if starting from scratch.
- **Fix / guardrail:** Before re-running long measurements, check the existing result files with a script: suite, entry count, invalid count, environment block, recorded sha and src/ext tree equality with the base ref. Reuse them if they pass. 0c finished in minutes this way. Specs for measurement sub-phases should say "validate existing files first; regenerate only the ones that fail".

### 2026-10-07 — Orchestrator/QE — A spot-check that cannot run under baseline power conditions is UNAVAILABLE, not a pass
- **Problem:** The 0d spec has a Performance Evidence spot-check, but it is not one of the acceptance criteria. QE ran it on battery with EPP `balance_power` (t=1, 12 entries, 0 invalid: 4 speedup, 3 regression, 5 no change, 0 CLOCK against the AC baseline). That left it unclear whether the gate depended on it. A later re-run on AC in the baseline's environment gave the same offsets and a FAIL against the stored baseline, which the operator declared non-gating for 0d and which was disclosed in PR #112.
- **Root cause:** The spec did not say whether the spot-check gates the stage. The power mismatch made the battery run formally non-comparable, but it did not explain the offsets (see the corrected AC/EPP lesson and the uncommitted-harness lesson).
- **Fix / guardrail:** Report a spot-check run in a different environment as UNAVAILABLE, with the numbers and the reason, in the QE verdict, the commit body and the PR. Never count it as passed. Once it has been re-run in the matching environment, report the real verdict (including FAIL) and, if the operator rules it non-gating, record that decision in the PR instead of leaving UNAVAILABLE in place. Planners must mark a spot-check either as an explicit acceptance criterion or as explicitly non-gating, so the gate is unambiguous. If it gates the stage, QE must block until AC with the same EPP is available rather than run it anyway.

### 2026-10-07 — Planner — Run `git ls-files` on every path in a spec's File List
- **Problem:** The 0d spec's File List called `benchmark/results/noise-before-fix.md` "local only", but the file has been tracked since `8722f05`. Ops needed an orchestrator decision before committing it.
- **Root cause:** The planner assumed everything under `benchmark/results/` is gitignored, but only the JSON and `logs/` are.
- **Fix / guardrail:** When writing a File List, run `git ls-files -- <path>` and `git check-ignore -v <path>` on each entry, and label it as tracked/commit, new/commit, or ignored/local. Tracked Markdown summaries are committed.

### 2026-10-07 — QE — One a/b pair spread can understate process-to-process noise
- **Problem:** In 0d the same vertex `bellman` entry had a 2.5% spread in one a/b pair and 11.8% in another. A single pair under 5% is not proof that an entry is quiet.
- **Root cause:** Two separate processes give only one sample of the between-process variation. Clock state, page placement and thread-to-core assignment differ from run to run.
- **Fix / guardrail:** Treat a single-pair spread as a lower bound. For evidence, use at least 3 interleaved A/B rounds in one clock state, and report the maximum spread across rounds, not one pair's spread. Entries known to be noisy (the 72 listed in REPORT.md § 4) are not usable as evidence from a single pair.

### 2026-10-07 — QE — Stored baselines recorded with an uncommitted harness may not be reproducible
- **Problem:** The 0d spot-check on AC, in the same clock state as the baseline (probe ~435.8 µs), still failed against the stored 0c baseline: `real-multiObj_robotIMDP` bellman ×1.186, `product-imdp-sparse-n1000-nnz10-a1-dfa4` bellman ×1.152, and `workspace` entries (robotIMDP ×0.944 / ×0.930, `fimdp-dense` omax ×0.871). Minimum times shifted too, so the offset is systematic, not tail noise. A battery run gave the same offsets, so power state was not the cause.
- **Root cause (likely, unverified):** The stored baselines (REPORT.md § 3) were recorded at `91bc0c8` on `lean/phase0-models` while the benchmark harness was still uncommitted. Sample counts differ strongly from the committed harness (robot workspace 57 vs ~190, product bellman 618 vs 880–1134), so the stored medians were measured with a different harness than the one that now reproduces them.
- **Fix / guardrail:** Record baselines only from a committed harness, and store the harness commit (`git rev-parse HEAD` plus a clean `git status -- benchmark/`) and per-entry sample counts in the result metadata. Before trusting stored medians, reproduce them with at least 3 interleaved rounds using the current harness; if sample counts or medians differ systematically, re-baseline instead of comparing. Re-baselining 0c with the committed harness is noted for 0h.

### 2026-10-07 — QE — Re-run measurement tools into the scratchpad, never `--append` into committed summaries
- **Problem:** In 0e, QE re-ran the `real-multiObj_robotIMDP` t1 profile to check reproducibility. `benchmark/profile.jl` can `--append` sections to the committed summaries under `benchmark/profiles/`, so a check run can silently change files that Ops is about to commit.
- **Root cause:** The same tool both produces the committed artefacts and serves as the verification re-run, and its append mode writes to the tracked file by default.
- **Fix / guardrail:** For comparison runs, always pass `--out <scratchpad path>` and never `--append`. Before handing off, QE checks `git status --short benchmark/` against the Dev hand-off list; an unexpected diff in a committed summary blocks the gate until it is reverted.

### 2026-10-07 — Dev/Planner — A tool-side pseudo-entry is a weak substitute for a missing registry entry
- **Problem:** The 0e spec required a `bellman` profile for control-synthesis cases, but the case registry has no `bellman` entry for `cs-*` cases. Dev added a `bellman_cs` pseudo-entry in `profile.jl`. It works, but it has no stored reference, so its correctness check only compares the strategy-cache kernel against the same kernel with the default cache (max |ΔV| = 0), which cannot catch a shared bug.
- **Root cause:** The registry (owned by 0a/0b) did not cover everything later sub-phases needed, and the gap was found only in 0e.
- **Fix / guardrail:** When a spec needs an entry the registry lacks, prefer adding the registry entry (with a stored reference) in the sub-phase that owns the registry; if a pseudo-entry is used, disclose the weak check in the PR and schedule the registry entry (here: 0h). Planners should cross-check each sub-phase's required entries against `benchmark/cases/registry.jl` when writing the spec.

### 2026-10-07 — Dev/QE — 16-thread sampling profiles are dominated by idle `wait()`
- **Problem:** In 0e, the t16 CPU profiles showed `wait()` at 68–80% of samples, so the top-frames table said little about where busy threads spend time.
- **Root cause:** `Profile` samples all threads, and most threads are idle (waiting at the task barrier) for the small per-call work in these cases.
- **Fix / guardrail:** For threaded profiles, filter out idle/`wait` frames (busy-only view) or report per-thread top frames, and state the idle share separately. Do not compare raw t16 self-time shares with t1 shares. Noted for 0f/0h.

### 2026-10-08 — Dev/QE — Machine roofs on a clock-switching host are an upper bound over clock states
- **Problem:** In 0f, `benchmark/stream.jl` roofs (L2/L3/DRAM read, FMA) differed by up to ~×2.2 between single rounds for the in-cache roofs, because the host switches clock states between rounds. A single-round roof could make a kernel look memory-bound or compute-bound depending on when the roof was measured.
- **Root cause:** In-cache bandwidth and FMA throughput scale with core clock. The host moves between boost and capped states, and one round samples only one of them.
- **Fix / guardrail:** Measure roofs in at least 3 rounds, store a clock probe per round, and state in the analysis that roof = max over rounds, which is an upper bound over clock states (done in 0f). Better (0h): keep per-state roofs and compare each benchmark entry with the roof for the clock state its own probe shows.

### 2026-10-08 — Dev — A tool run with no arguments must not silently print empty tables
- **Problem:** The 0f spec gave `julia benchmark/analyze.jl` with no arguments. The old tool then matched no result files and printed empty tables without any error, which looked like a valid (empty) analysis.
- **Root cause:** The tool had no default input set and no check that it had found any data.
- **Fix / guardrail:** Analysis tools default to the spec's base sha and files (0f: `analyze.jl` now reads the `20fc03b` files by default, with `--sha` to override) and fail loudly when no input matches. Planners should give full commands (see "Specs must state complete commands") and Dev should check that a command's output is non-empty before trusting it.

### 2026-10-08 — Dev — Always pass `--entries`/`--filter` on check and spot runs
- **Problem:** In 0f, a check run left out `--entries bellman` and so ran the whole suite for the selected cases, which may cost a lot of wall time for data that is not needed.
- **Root cause:** `run.jl` defaults to all entries; the check was meant to cover only `bellman`.
- **Fix / guardrail:** Spot and check runs always pass `--entries` and/or `--filter` explicitly, with `--out` into the scratchpad. Specs should write the spot-check command with these flags.

### 2026-10-08 — QE — A run-start clock reference taken while boosting gives false CLOCK flags at t=1
- **Problem:** In the 0f QE spot-check, t=1 entries were flagged CLOCK with probe ratios around 2.2, even though the t1 scaling and sizes timings matched the stored files within noise (ratios 0.991–1.000).
- **Root cause:** `benchmark/lib/clock.jl` takes the run-start reference probe while the core is still boosting. Later per-trial probes run at the sustained clock, so the ratio is large without any change in kernel speed. This is the t=1 counterpart of "The clock guard flags nearly every t=8/t=16 entry" and limits its advice to "still treat CLOCK at t=1/t=4 as real".
- **Fix / guardrail:** Before treating a t=1 CLOCK flag as real, compare the per-trial probes with each other and with the stored file's probes, not only with the run-start reference. Noted for 0h: take the reference after a warm-up at sustained clock (or as the median of several probes during the run). Separately, a CLOCK-flagged spot-check result (0f: t8 sparse k10 ratio 0.611) is not evidence either way.
