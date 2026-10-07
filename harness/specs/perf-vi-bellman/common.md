# Performance of the VI/Bellman Algorithms — Shared Reference

> **Not a runnable spec.** This file holds the rules shared by every sub-phase of the performance work. Run the harness
> on one sub-phase spec in this directory at a time, e.g. `/harness harness/specs/perf-vi-bellman/0a-suite-skeleton.md`.
> Each sub-phase spec lists the sections of this file its agents must read; they read only those.

Run order:

| Spec | Sub-phase |
|---|---|
| `0a-suite-skeleton.md` | Benchmark environment, case generators, `run.jl`, README |
| `0b-reference-compare.md` | Correctness reference values, `compare.jl` |
| `0c-cpu-baseline.md` | CPU baseline at 1/4/8/16 threads, with re-runs |
| `0d-noise.md` | Noise analysis and fixes |
| `0e-cpu-profiles.md` | CPU profiles |
| `0f-scaling-roofline.md` | Strong/size scaling and roofline |
| `0g-cuda.md` | CUDA baseline and profiles (or UNAVAILABLE) |
| `0h-hypotheses-close.md` | Ranked hypotheses, Findings, Phase 0 close |
| `1a-prepare.md` → `1b.1-h1-sort-ordering.md` → `1b.2-h7-sparse-gather.md` → `1b.3-h6-dense-gap-walk.md` → `1c-close.md` | Phase 1 — single-threaded CPU |
| `2a-prepare.md` → `2b.1-h2-dynamic-scheduling.md` → `2b.2-h8-threaded-threshold.md` → `2c-close.md` | Phase 2 — multi-threaded CPU |
| `3a-prepare.md` → `3f.1-b2-ivi-cuda.md` → `3f.2-b3-cuda-fimdp-sparse.md` → `3b.1-h5-host-sync.md` → `3c-close.md` | Phase 3 — CUDA (needs a passing GPU probe) |
| `4a-prepare.md` → `4f.1-b1-strategy-cache.md` → `4b.1-h4-vertex-support.md` → `4b.2-h3-mccormick-lp-reuse.md` → `4b.3-h1-factored-ordering.md` → `4b.4-h9-fimdp-omax-temporaries.md` → `4c-close.md` | Phase 4 — factored algorithms and VI loop |

The experiments `Nb.k` implement the ranked hypotheses H1–H9 of `benchmark/REPORT.md` § 7 (H10 is not scheduled); the
fixes `Nf.k` resolve the Findings B-1–B-3 of § 8. To add an experiment (e.g. a second variant of H6), copy the closest
`Nb.k` file to the next free number, change its hypothesis, measurement, change, prediction and target cases, and add
the row to the phase plan in `REPORT.md`.

A sub-phase may not start until the previous one has been committed and pushed; a phase may not start until the
previous phase has merged.

## Objective

Make the Bellman update and the value-iteration loops in IntervalMDP.jl faster on single-core CPU, multi-core CPU and CUDA. Every change that lands must be **justified by measurements** taken with the benchmark suite and protocol in this spec. Rules of thumb (cache locality, branch divergence, load balance, allocation pressure, and so on) may be used to decide **what to try**. They are never enough to decide **what to keep**. A change is kept only if the evidence protocol (§ Evidence Protocol) shows a real speedup and no unacceptable regression.

The work has three parts:

1. **Measure first** (Phase 0). Build a reproducible benchmark suite that covers every algorithm, storage type and backend. Record a baseline and profile it. Write down where the time actually goes, before changing any code in `src/` or `ext/`.
2. **Optimize by experiment** (Phases 1–4). Each candidate optimization is a hypothesis with a predicted effect. It is implemented, measured against the baseline with the protocol, and then either accepted or rejected. Rejected experiments are recorded too.
3. **Keep the code modular and the interface stable** (§ Interface Stability, § Modularity).

Adds or semantically changes a VI/Bellman algorithm: **no**. Every optimization must compute the same mathematical result as the baseline. The only allowed differences are floating-point rounding within the tolerances in § Julia Behavior & Tests. If an optimization changes **what** is computed, it is out of scope for this spec and must be raised as a Finding. Examples: a different reduction order or tree shape in the factored recursive O-max (this changes the approximation), a different McCormick relaxation, a different stopping rule, or a different default algorithm.

Lean dependencies touched: none semantically. The Lean docstrings cite Julia functions and files (`lean/scripts/check_julia_refs.py`). If a cited function is renamed, moved or split, its docstrings must be updated in the same phase, and `lake build` plus the reference check must stay green.

Model families: interval MDP (dense and sparse), factored interval MDP (O-max, LP McCormick, vertex enumeration), product process (IMDP × DFA). Algorithms: `RobustValueIteration`, `IntervalValueIteration`, and the standalone `IntervalMDP.bellman` / `IntervalMDP.bellman!`.

## Commands / Toolchain

- Target root: `/home/fresen/.julia/dev/IntervalMDP`
- Julia version / compat: `[compat] julia = "1.11"`; local `julia` is 1.13.1. The benchmark environment records the exact version.
- Julia instantiate: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- Julia test: `julia --project=. -e 'using Pkg; Pkg.test()'`
- Julia test, multi-threaded: `julia --project=. --threads=auto -e 'using Pkg; Pkg.test()'` (the threaded workspaces are only used when `Threads.nthreads() > 1`, so the single-threaded run alone does not exercise them).
- GPU test: run the `:cuda`-tagged test items, i.e. `test/runtests.jl` with `:cuda` removed from `EXCLUDED_TAGS` (do not commit that change), classified through `python3 harness/tools/harness/gpu_check.py run . --cmd "<cmd>"`.
- Lean root: `<root>/lean`; build with `~/.elan/toolchains/<pinned>/bin/lake build` (never through the elan proxy, see `harness/LEARNING.md`); reference check `python3 lean/scripts/check_julia_refs.py`.
- Benchmark: `julia --project=benchmark --threads=<T> benchmark/run.jl --suite <name> --out benchmark/results/<file>.json` (created in Phase 0; § Benchmark Suite).
- Benchmark comparison: `julia --project=benchmark benchmark/compare.jl <baseline.json> <candidate.json>` (created in Phase 0).
- After every test run: `git checkout -- test/data/multiObj_robotIMDP.nc` (see `harness/LEARNING.md`).

## Phases

| Phase | Scope | `src/`/`ext/` changes allowed? |
|---|---|---|
| 0 | Benchmark suite, comparison tool, environment capture, **baseline** results, **profiles**, and a ranked hypothesis list (§ Phase 0 Deliverables), run as sub-phases `0a`–`0h` (one spec file each) | **No.** `src/` and `ext/` must be byte-identical to the base ref |
| 1 | Single-threaded CPU: dense and sparse O-max, IMDP and product | yes |
| 2 | Multi-threaded CPU: work partitioning, scheduling, workspace thresholds, false sharing | yes |
| 3 | CUDA: dense, sparse, factored kernels; host↔device traffic in the VI loop | yes |
| 4 | Factored algorithms (O-max, McCormick, vertex enumeration) and VI/IVI loop overhead (residual, copies, strategy cache, termination check) | yes |

The order of Phases 1–4 may be changed by the operator once the Phase 0 profiles are in, since the profiles may show that the time is spent somewhere other than expected. The operator sets the order by choosing which phase's spec files to run next; the `Na` sub-phase of whichever phase runs first creates `test/base/perf_equivalence.jl` and `benchmark/ab.jl`.

### Sub-phases (Phase 0)

Phase 0 is split into eight sub-phases, each a separate spec file and `/harness` run with its own Dev → QE → Ops gate. A single
Phase 0 run outgrew the Dev agent's context window, so each sub-phase owns a small, self-contained part of the
deliverables and a fixed section of `benchmark/REPORT.md`. `src/` and `ext/` stay byte-identical to the base ref
(`20fc03b`) in every one of them. Phase 0 uses the existing branch `perf/phase0`.

| Sub-phase | Scope (files it owns) | `REPORT.md` section | Ops outcome |
|---|---|---|---|
| `0a` — suite skeleton | `benchmark/Project.toml` + `Manifest.toml` (no Revise/Cthulhu/ProfileView), removal of the stale `benchmark/bellman/`, `benchmark/imdp/`; `cases/*.jl` (generators, registry), `lib/measure.jl`, `lib/environment.jl`, `lib/clock.jl`, `run.jl` (all options, `--list`), `README.md` | § 1 Environment, § 2 Suite | Commit, push `perf/phase0`, open a **draft** PR |
| `0b` — reference + compare | `lib/reference.jl`, `--write-reference [--reference-only]`, `benchmark/reference/**` (CPU Float64), `compare.jl` (A/B verdicts, `--spread`, clock-aware `CLOCK (not evidence)`) | § 2 (reference paragraph) | Commit, push (same draft PR) |
| `0c` — CPU baseline | `benchmark/baseline.sh`, `results/baseline-<sha>-cpu-t{1,4,8,16}.json` and one separate-process re-run of each, `results/reference-run-*-cpu-t1.json` | § 3 Baseline files (CPU rows, selected medians, VI-loop overhead) | Commit, push |
| `0d` — noise | spread analysis with `compare.jl --spread`, `results/noise-before-fix.md`, budget changes in `cases/registry.jl`, `results/noisefix/*.json` | § 4 Noise (incl. the result table at `<!--NOISEFIX-->`) | Commit, push |
| `0e` — CPU profiles | `profile.jl`, `profiles/<case>.md` for every case-matrix row (CPU) | § 5 Profiles | Commit, push |
| `0f` — scaling + roofline | `stream.jl`, `analyze.jl`, `results/scaling-*.json`, `results/sizes-*.json`, `results/stream-*.json`, `results/scaling-roofline-<sha>.md` | § 6.1, § 6.2 | Commit, push |
| `0g` — CUDA | GPU probe; `reference/cuda-*`, `results/baseline-<sha>-cuda*.json` (+ re-runs), `profiles/cuda-*.md`; or UNAVAILABLE with the probe output | § 6.3 (incl. `<!--CUDAPROF-->`), CUDA rows of § 3/§ 4 | Commit, push |
| `0h` — hypotheses + close | Ranked hypothesis list, Findings, Limitations; consistency pass over `REPORT.md` (no placeholders left, every number traceable to a file); `Pkg.test()` 1 thread and `--threads=auto`; `src/`/`ext/` check | § 7, § 8, § 9 | Push, mark the PR ready (`gh pr ready`) |

Rules for every Phase 0 sub-phase:

- **Resume, don't redo.** Much of Phase 0 exists in the working tree (uncommitted, from the stopped single run). Dev starts
  by checking which of the sub-phase's deliverables already exist and are valid (they run against the current API, the
  correctness check passes, the environment block is complete), and only fills the gaps. Valid result files are **not**
  regenerated. Dev lists in its hand-off which files it reused, which it changed and which it created.
- **Stay in scope.** Dev edits only the files and the `REPORT.md` section of the active sub-phase. A defect found in an
  earlier sub-phase's file is fixed only if the active sub-phase cannot be done without it, in a minimal edit noted in the
  hand-off; otherwise it is written down for `0h`.
- **Commit only your own files.** Ops stages the paths of the active sub-phase explicitly (`git add <paths>`), never
  `git add -A`, so that later sub-phases' untracked files and unrelated changes (`.claude/`, `harness/`) stay out of the
  commit.
- **QE checks the sub-phase, not the phase.** QE verifies the acceptance criteria of the sub-phase's own spec file (its § Acceptance
  Criteria). For long measurements (`0c`, `0f`, `0g`) QE re-runs a **spot-check** subset (`--filter`/`--entries`, a few
  cases per family, minutes not hours) instead of the full suite.

Status after the stopped single Phase 0 run (2026-10-06, inventory of the working tree; to be confirmed by each sub-phase's
Dev/QE): `0a`, `0b`, `0c`, `0e`, `0f` deliverables present; `0d` measured (`results/noisefix/`) but the result table at
`<!--NOISEFIX-->` is missing; `0g` measured but `<!--CUDAPROF-->` is missing; `0h` drafted (§ 7–9). Nothing is committed yet.

### Sub-phases (Phases 1–4)

Each of Phases 1–4 is split into sub-phases, each one a separate spec file (`Na-prepare.md`, `Nf.k-*.md`, `Nb.k-*.md`, `Nc-close.md`) and `/harness` run with its own Dev → QE gate. This keeps each agent's context small, so a stopped run (rate limit, server error) loses little work, and QE catches mistakes before later sub-phases build on them. `N` is the phase number.

| Sub-phase | Scope | `src/`/`ext/` changes? | Ops outcome |
|---|---|---|---|
| `Na` — prepare | Pick this phase's hypotheses from the ranked list in `REPORT.md` and record the chosen order as the phase plan in `REPORT.md`. Re-run the noise check on the phase's merge base for the target cases; fix any case with spread > 5%. Add or extend `test/base/perf_equivalence.jl` (and its `:cuda` counterpart) so it covers this phase's code paths. **`1a` only** (or whichever phase runs first): create `test/base/perf_equivalence.jl` with reference values from the base ref, and the interleaved A/B runner `benchmark/ab.jl` if Phase 0 did not provide one. | **No** | Branch `perf/phase-N` from `origin/main`, commit, open a **draft** PR |
| `Nf.k` — Finding fix `k` | Fix one `B-<n>` Finding from Phase 0 in its own commit with a regression test (§ Findings policy), before the phase's experiments, so they are measured on fixed code. Regenerates the affected reference values from the fixed commit. | yes (only that fix) | Push the commit to `perf/phase-N` (same draft PR) |
| `Nb.k` — experiment `k` | **Exactly one** hypothesis from the phase plan: implement it, measure it alone against the commit before it (interleaved A/B, § Evidence Protocol), and decide accept/reject. Accepted: kept as its own commit. Rejected: reverted, with the code diff summarised in the entry. Either way, add the `REPORT.md` entry. Interacting changes get their own `Nb.k` that measures the combination. | yes (only that one change) | Push the commit(s) to `perf/phase-N` (same draft PR) |
| `Nc` — close | No new optimizations. Run the full matrix on every backend and thread count the phase touched, phase head vs merge base. Check allocations, JET, formatter, Lean build and reference check, `docs/src/developer.md`, and resolve or list open Findings. | Only fixes for regressions found here, each with A/B evidence | Push, mark the PR ready for review (`gh pr ready`) |

The `Nf.k` and `Nb.k` runs of each phase are fixed by its spec files (see the run-order table) and recorded as the phase plan in `REPORT.md` by `Na`. The operator may add, drop or reorder experiments between runs by adding, deleting or renumbering spec files; the plan follows. Sub-phase runs never open a second PR for the same phase.

### Context budget (every agent, every phase)

The agents must keep their context small. In particular:

- Never print a whole results JSON, `Manifest.toml`, profile dump or long log. Read JSON through a short `julia`/`python3`
  one-liner that prints only the fields needed (or `compare.jl`/`analyze.jl` output), and large Markdown files by section
  (`grep -n '^#'`, then `sed -n` a range).
- Run benchmark and profile commands with output redirected to a log file in the scratchpad (or `benchmark/results/logs/`,
  gitignored) and read back only the tail or a `grep` of it. Runs longer than ~10 minutes go to the background and are
  waited for with a monitor, not polled in a loop.
- Do not re-read a file after editing it to check the edit. Do not read files of other sub-phases unless needed.
- Keep the hand-off short: per acceptance criterion one line with the evidence (file path, command, number).

## Committed vs local benchmark outputs

Raw benchmark output is large (≈ 90 MB after Phase 0) and stays **local**: `benchmark/results/**/*.json`,
`benchmark/results/logs/` and `benchmark/reference/` are in `.gitignore`. What is committed is the code and the
human-readable evidence: `benchmark/REPORT.md`, `benchmark/README.md`, `benchmark/profiles/*.md` and Markdown summaries
under `benchmark/results/*.md` (e.g. `scaling-roofline-<sha>.md`, `noise-before-fix.md`, `<tag>-sweep.md`).

- Every number in `REPORT.md` still names the local JSON file it comes from, so QE on this host can trace it.
- Results tables in `REPORT.md` entries carry the numbers themselves (medians, ratios, verdicts), so the committed
  report stands on its own without the JSON.
- A heuristic constant links to a committed sweep summary `benchmark/results/<tag>-sweep.md` (the table of the sweep,
  generated from the local JSON), not to the JSON itself.
- Reference values are regenerated from the base ref with `run.jl --suite full --write-reference --reference-only`
  (CPU) and `--backend cuda --eltype all` (CUDA); `README.md` says so. The package's own equivalence test
  (`test/base/perf_equivalence.jl`) stores its small reference values in the test and is committed.

## Phase 0 Deliverables

1. **Benchmark suite** under `benchmark/` that replaces the stale scripts (`benchmark/bellman/` and `benchmark/imdp/` use the removed `IntervalProbabilities`/`construct_ordering` API). § Benchmark Suite gives the requirements.
2. **Baseline results** for the base ref: `benchmark/results/baseline-<short-sha>-<backend>-t<threads>.json`, for `threads ∈ {1, 4, 8, 16}` on CPU and for CUDA when it is available.
3. **Profiles** of the main cases (at least one per row of the case matrix): CPU sampling profiles (`Profile` + a flame graph or a text tree), allocation profiles (`Profile.Allocs`), type-stability checks (`JET.@report_opt` on the hot functions), and for CUDA an Nsight Systems/Compute or `CUDA.@profile` trace. Save the summaries (not the raw dumps) in `benchmark/profiles/<case>.md`.
4. **Scaling data**: strong scaling (fixed size, threads 1→16) and size scaling (fixed threads, increasing `n`) for dense and sparse O-max, plus achieved memory bandwidth or FLOP rate compared with the machine's peak (a roofline estimate). This tells later phases whether a case is memory-bound, compute-bound or overhead-bound.
5. **Hypothesis list** in `benchmark/REPORT.md`, ranked by expected gain. Each entry gives: the case it targets, the evidence from the profile that points to it (for example "62% of samples in `sort!` in `bellman_precomputation!`"), the change proposed, the predicted effect, and the phase it belongs to. Rules of thumb are allowed here, but each entry must cite at least one measurement.

Phase 0 passes when all of the above exist, `benchmark/run.jl` reproduces the baseline numbers within the noise bound of § Evidence Protocol on a re-run, and `src/` and `ext/` are unchanged.

## Benchmark Suite

- **Location and form.** `benchmark/run.jl` (entry point with command-line options for suite, backend, element type and output path), `benchmark/cases/*.jl` (seeded problem generators), `benchmark/compare.jl`, and `benchmark/README.md` explaining how to run and read it. It uses `BenchmarkTools`, and its environment is `benchmark/Project.toml` with a committed `Manifest.toml`. Remove `Revise`, `Cthulhu` and `ProfileView` from the benchmark environment, or move them into a separate profiling environment, so that they do not affect timings.
- **Case matrix.** Every case is generated from a fixed RNG seed and identified by a stable name.

  | Family | Storage | Sizes | Actions | Entry points timed |
  |---|---|---|---|---|
  | IMDP | dense | `n ∈ {100, 1 000, 4 000}` | `{1, 4}` | `bellman!`, `solve` (RVI, infinite reachability), `solve` (IVI) |
  | IMDP | sparse | `n ∈ {10⁴, 10⁵}`, nnz/column `∈ {10, 100}` | `{1, 4}` | same |
  | fIMDP | dense and sparse marginals | 2 and 3 state variables, 10–50 values each | `{1, 4}` | `bellman!` and `solve` for each of `OMaximization`, `LPMcCormickRelaxation`, `VertexEnumeration` (vertex enumeration only on the small sizes) |
  | Product IMDP × DFA | dense and sparse | `n ∈ {1 000, 10⁴}`, DFA with 3–5 states | `{1, 4}` | `solve` (DFA reachability) |
  | Real model | sparse | `test/data/multiObj_robotIMDP.*` | as in file | `solve` |
  | Control synthesis | dense and sparse | middle sizes above | 4 | `solve(ControlSynthesisProblem)`, stationary and time-varying |

  Keep each case within memory on this host (15 GB RAM). Float64 is the default element type, and Float32 is run for the dense and sparse IMDP cases on CUDA.
- **Separate what is timed.** Report the per-iteration Bellman cost (`bellman!` with a pre-built workspace and strategy cache) separately from whole-solve cost, so loop overhead can be told apart from kernel cost. Workspace construction is timed on its own.
- **Output.** One JSON file per run. For each case it holds the median, minimum, mean, standard deviation, number of samples, allocations (count and bytes) and GC time. The file also holds the environment block: git SHA and dirty flag, Julia version, `Threads.nthreads()`, threadpool sizes, BLAS threads, CPU model and governor, GPU model, driver and CUDA runtime version, `CUDA.versioninfo()` digest, and the date.
- **Correctness check inside the suite.** Each run also checks the result of every case against a stored reference value from the baseline, with the tolerances in § Julia Behavior & Tests. A benchmark result whose correctness check fails is invalid and may not be used as evidence.

## Evidence Protocol *(required for every accepted change)*

The goal is to tell a real effect from noise on a laptop CPU (Intel Core Ultra 7 255H: 16 cores of mixed P-, E- and LP-E type, one NUMA node, 24 MiB L3), which throttles under heat and varies in clock speed.

- **Environment.** On AC power, with the performance governor (or the closest available setting), no other heavy processes, and the same Julia flags for baseline and candidate. Record whether threads are pinned and how (e.g. `ThreadPinning.jl`, or `JULIA_EXCLUSIVE=1`). The thread-to-core-type mapping matters on this hybrid CPU, so it must be recorded, not left implicit.
- **A/B interleaving.** Measure baseline and candidate in alternating runs (A B A B …, at least 3 rounds each, in separate Julia processes), not as one block of A followed by one block of B. This cancels out thermal and clock drift.
- **Statistics.** Compare medians. A change is a **speedup** on a case only if the candidate median is at least **5% faster**, and the difference holds in every interleaved round (or a Mann–Whitney U test gives p < 0.01 over the pooled samples). It is a **regression** if it is more than **3% slower** under the same test. Everything in between counts as "no change". `BenchmarkTools.judge` with `time_tolerance = 0.05` may be used as an equivalent check.
- **Noise bound.** Phase 0 measures the run-to-run spread of the unchanged baseline for each case. A case whose spread is larger than the 5% threshold must be fixed (more samples, longer runs, a larger size) before it can serve as evidence.
- **Scope of acceptance.** A change is **accepted** if it is a speedup on the cases it targets and causes no regression on any other case in the matrix, on any backend or thread count it touches. A change that trades a regression in one case for a speedup in another needs the operator's explicit approval (§ Findings). The `REPORT.md` entry must give the trade-off in numbers.
- **Attribution.** Each accepted change is measured on its own against the commit before it. Changes are not bundled. If two changes interact, measure each alone and then the combination.
- **Allocations.** No change may increase allocations per `bellman!` call in the steady state. The steady state of `bellman!` with a pre-built workspace should allocate zero bytes on CPU. Phase 0 records where it does not.
- **Heuristic thresholds** (for example the `threshold = 10` in `construct_workspace` that decides between threaded and single-threaded workspaces, or CUDA launch parameters) may only be set from a measured sweep. Keep the sweep data in `benchmark/results/` (raw JSON local) with a committed summary `benchmark/results/<tag>-sweep.md`, and put a comment next to the constant pointing to that summary.
- **Record everything.** `benchmark/REPORT.md` has one entry per experiment, **accepted or rejected**: hypothesis, rationale, change (commit or diff summary), cases, results table (baseline median, candidate median, ratio, verdict), environment, and the name of the raw JSON file (local only, see § Committed vs local benchmark outputs). Rejected experiments stay in the report so they are not retried blindly.

## Julia Behavior & Tests

- **Same results.** For every case in the matrix and every existing test, the optimized code must return the same values as the base ref, within these limits:
  - O-max (dense and sparse), product, RVI and IVI value functions: `‖V_new − V_base‖∞ ≤ 1e-12` in Float64 for a single `bellman!` call, and `≤ 10 · ε_conv` for converged infinite-horizon solves (where `ε_conv` is the convergence threshold of the spec); Float32: `≤ 1e-5`.
  - Factored O-max, McCormick and vertex enumeration: the same tolerance as above. Larger deviations mean the approximation itself has changed, which is a Finding (§ Objective).
  - Iteration counts for infinite-horizon solves may differ by at most 1 (because of rounding at the stopping check).
  - Strategies: identical, except where actions tie in value within the tolerance. Where they differ, the test checks that the value of the new strategy (by policy evaluation) equals the value of the baseline strategy within tolerance.
- **New tests.** Add `test/base/perf_equivalence.jl` (and a `:cuda`-tagged counterpart) that runs a small version of each case-matrix row and compares it against stored reference values produced by the base ref. It covers dense and sparse, single- and multi-threaded workspaces (by calling the threaded workspace constructors directly, so the test does not depend on `--threads`), and every factored algorithm.
- Any new code path (for example a new workspace type, a new kernel, a new scheduling strategy) gets its own test item that would fail if the path were not selected or gave wrong results.
- No wall-clock thresholds in unit tests. Performance is checked only by the benchmark suite.

## Interface Stability

The interface must not change unless there is a strong, measured performance reason **and** the operator has been warned and has approved it **before** the change is made.

**Frozen (tier 1): may not change without approval.**

- Every exported name and every name declared `public` in `src/IntervalMDP.jl` and `src/Data/Data.jl`: their names, positional and keyword arguments, default values, return types, and documented behavior.
- Everything documented under `docs/src/` (including `docs/src/reference/`), and the docstrings of the above.
- The semantics of `IntervalMDP.cu` / `IntervalMDP.cpu` and what they accept.
- The default algorithm and default Bellman algorithm chosen for each model (`default_algorithm`, `default_bellman_algorithm`).

**Semi-public (tier 2): treated as frozen.** These are not exported, but they have docstrings and are used by tests, benchmarks and (very likely) downstream users: `IntervalMDP.bellman`, `IntervalMDP.bellman!`, `IntervalMDP.construct_workspace` (including its `threshold` and `num_actions` keywords), `IntervalMDP.construct_strategy_cache`. New keyword arguments with defaults that keep the current behavior are allowed here without approval, but must be documented in the docstring and listed in the phase's PR description.

**Internal (tier 3): free to change.** Workspace struct types and fields, `_bellman_helper!`, `state_bellman!`, `state_action_bellman`, `gap_value`, `@threadstid`, CUDA kernels and their launch logic, sorting helpers, and other unexported helpers without docstrings in the docs.

**Procedure when a tier 1 or tier 2 change seems worth it.** Dev stops work on that item and raises a Finding (`I-<n>`) in `benchmark/REPORT.md` and in the hand-off. It gives:

- the measured gain (evidence protocol), achieved with a prototype on a scratch branch that is not merged;
- the cases it helps and what it costs other cases;
- the exact interface change, who it breaks, and a migration path (for example a deprecation);
- the alternatives tried that keep the interface, with their numbers.

The change is not merged in that run. It waits for the operator's decision.

## Modularity

The existing structure dispatches on model type (`IsIMDP`, `IsFIMDP`, product), storage (dense, sparse, CUDA), Bellman algorithm (`OMaximization`, `LPMcCormickRelaxation`, `VertexEnumeration`) and threading (single or threaded workspace). Optimizations must fit this structure:

- **Select through dispatch, not branches.** A new strategy (for example a new scheduling method or a different kernel for some sizes) is a new workspace type or a new method, chosen in `construct_workspace` or in the corresponding `_bellman_helper!` dispatch. Do not add `if` branches on sizes or thread counts inside per-state hot loops.
- **One place per decision.** Heuristic constants (thresholds, block sizes, chunk sizes) are named constants or keyword defaults defined in one place, with the comment that links to the sweep data (§ Evidence Protocol).
- **Backends stay separate.** CUDA code stays in `ext/` (`IntervalMDPCudaExt`). `src/` gains no CUDA-specific code. Shared logic between CPU and GPU (for example the O-max inner step) stays as shared functions, not copies.
- **No copy-paste variants.** A faster variant of a function must replace the old one or share its core. Two near-identical copies that differ only in an optimization detail are not allowed. If both must remain (for example because each is faster on different sizes), the shared part is factored out and the choice is made by dispatch.
- **The O-max step stays one recognizable unit**, since the Lean proofs (Phase 1 of `harness/specs/lean-proofs/`, sub-phases `1c`/`1d`) model it. If its implementation is restructured (for example a full sort replaced by a partial sort or a selection), the docstring must still state which permutation/ordering it computes, and the Lean-side docstrings that cite it must be kept accurate.
- `JET.@report_opt` reports no new runtime dispatch in the hot functions compared with the baseline.
- `JuliaFormatter` (`.JuliaFormatter.toml`) passes on the changed files.

## Candidate Hypotheses (starting points, not conclusions)

These are rules of thumb to guide what to try. **None of them may be implemented and kept without measurements that support it.** Phase 0 profiles may make some of them irrelevant and bring up others.

- *Single-threaded CPU*: sorting cost in O-max (full `sortperm!` each call vs. reusing the previous ordering, insertion sort when values change little between iterations, partial sort up to the budget); memory layout and access order of `lower`/`gap` (column-major traversal); `Int32` vs `Int` indices; SIMD of the budget accumulation; bounds checks; type instabilities; allocations in the steady state.
- *Multi-threaded CPU*: `@threadstid` partitions the range statically into equal chunks, which may leave P-cores idle while E-cores finish on this hybrid CPU (compare with dynamic/chunked scheduling and `:greedy`); the `threshold = 10` cut-off; false sharing in per-thread workspaces and in the output vector; the cost of the shared dense sort done before the parallel loop; scaling past the number of P-cores.
- *CUDA*: occupancy and launch configuration; shared-memory sorting and warp-level reductions; memory coalescing for sparse columns; kernel fusion (Bellman + residual + strategy extraction); host synchronisation and host↔device copies per VI iteration (e.g. the residual check); Float32 vs Float64 throughput on the GPU model in use.
- *Factored*: re-use of the JuMP model in McCormick (rebuilding vs. modifying constraints); expectation cache in factored O-max; enumeration order in vertex enumeration.
- *VI loop*: cost of `lastdiff!`, copying the value function, the termination check and strategy-cache updates relative to `bellman!`.

## CPU / GPU Matrix

| Check | Backend | Required? | Command | Notes |
|---|---|---|---|---|
| Package tests | CPU, 1 thread | yes | Julia test command | every phase |
| Package tests | CPU, `--threads=auto` | yes | multi-threaded Julia test command | every phase |
| GPU tests | CUDA | yes in Phase 3, and in any phase that touches `ext/` or code the CUDA path calls | GPU test command | PASS / FAIL / UNAVAILABLE; UNAVAILABLE on a required row is an unmet criterion |
| Benchmarks | CPU 1/4/8/16 threads | yes | benchmark command | evidence protocol |
| Benchmarks | CUDA | yes in Phases 0 and 3 if the probe passes; Phase 0 records UNAVAILABLE otherwise | benchmark command `--backend cuda` | |

**GPU status at the time of writing (2026-10-05):** `nvidia-smi` fails with "Driver/library version mismatch" (NVML 615.71), so CUDA is currently not functional on this host. This usually means the kernel module and the user-space driver are out of step, and a reboot fixes it. Phase 0 may finish with CUDA rows marked UNAVAILABLE; **Phase 3 cannot start until the GPU probe passes**. The GPU model is recorded in the environment block once it works.

## Performance Evidence

§ Benchmark Suite and § Evidence Protocol apply. For each phase, QE re-runs the comparison independently (baseline = the merge base of the phase, candidate = the head of the phase branch), using the interleaved protocol, and checks that its numbers support every "accepted" verdict in `REPORT.md`. Correctness tests pass first. A speedup claimed in `REPORT.md` that QE cannot reproduce within the noise bound is an unmet criterion.

## Findings policy

Raise a numbered Finding to the operator (in `REPORT.md` and the hand-off), do not resolve it silently, when:

- an optimization would change a tier 1 or tier 2 interface (`I-<n>`, § Interface Stability);
- an optimization would change what is computed (the approximation, the default algorithm, tie-breaking that changes the strategy beyond ties, the stopping rule) (`S-<n>`);
- an optimization causes a regression elsewhere that is worth accepting (`T-<n>`, trade-off);
- profiling exposes a correctness bug (`B-<n>`). Fix it only in its own commit with a regression test, and only if the fix does not change documented behavior; otherwise stop and report.

## File List

- `benchmark/run.jl`, `benchmark/compare.jl`, `benchmark/cases/*.jl`, `benchmark/README.md`, `benchmark/REPORT.md`, `benchmark/Project.toml`, `benchmark/Manifest.toml`, `benchmark/results/*.json`, `benchmark/profiles/*.md`
- Phases 1–4: `src/bellman.jl`, `src/workspace.jl`, `src/threading.jl`, `src/robust_value_iteration.jl`, `src/interval_value_iteration.jl`, `src/strategy_cache.jl`, `ext/cuda/**`, `ext/IntervalMDPCudaExt.jl`
- `test/base/perf_equivalence.jl`, `test/cuda/**` (CUDA counterpart)
- `lean/IntervalMDPProofs/**` docstrings only, where a cited Julia function moved or was renamed
- `docs/src/developer.md` if the description of the algorithms' implementation (threading, CUDA design) becomes inaccurate

## Out of Scope

- New algorithms, new approximations, or changes to the approximation quality of the factored algorithms.
- Changing default algorithm choices or any tier 1 interface without an approved Finding.
- Distributed or multi-GPU execution; non-CUDA GPU backends.
- Optimizing model construction, file I/O (`src/Data/`) or the Lean project.
- Wall-clock assertions in unit tests or CI.
