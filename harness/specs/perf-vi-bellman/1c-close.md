# Performance of the VI/Bellman Algorithms — 1c: Close — Specification (Julia + Lean)

> Sub-phase **1c** (close) of Phase 1 — Single-threaded CPU. One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Evidence Protocol, § Interface Stability, § Modularity, § Performance Evidence, § Findings policy.
> Phase branch: `perf/phase-1` (one draft PR for the whole phase). The previous sub-phase of this phase must be pushed
> first; the first sub-phase of a phase starts only after the previous phase has merged into `main`.

**Active phase: 1c**

## Objective *(required)*

Close Phase 1. No new optimizations. Run the full matrix on every backend and thread count this phase touched,
phase head vs merge base, interleaved; check allocations, JET, formatter, Lean build and reference check, and
`docs/src/developer.md`; resolve or list the open Findings; write the phase summary in `benchmark/REPORT.md`
(accepted/rejected experiments with their ratios, net effect per case family). Only fixes for regressions found here,
each in its own commit with A/B evidence.

Adds or semantically changes a VI/Bellman algorithm: **no**. Every change computes the same result as the merge base
within the tolerances of `common.md` § Julia Behavior & Tests; anything else is raised as a Finding (`S-<n>`) and not
merged. Tier 1/2 interface changes: only additive keywords with defaults, documented; anything wider is an `I-<n>`
Finding (`common.md` § Interface Stability). Lean: if a Julia function cited by a Lean docstring is renamed, moved or
split, update the docstring in this run and keep `lake build` and `check_julia_refs.py` green.
Follow `common.md` § Context budget. Ops: push, then mark the PR ready for review (`gh pr ready`).

## Commands / Toolchain *(required)*

- Target root: `/home/fresen/.julia/dev/IntervalMDP`
- Julia version / compat: `[compat] julia = "1.11"`; local `julia` is 1.13.1.
- Julia instantiate: `julia --project=. -e 'using Pkg; Pkg.instantiate()'` and `julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'`
- Julia test: `julia --project=. -e 'using Pkg; Pkg.test()'`; multi-threaded: `julia --project=. --threads=auto -e 'using Pkg; Pkg.test()'`
- After every test run: `git checkout -- test/data/multiObj_robotIMDP.nc` (see `harness/LEARNING.md`).
- GPU test: the `:cuda`-tagged test items (`test/runtests.jl` with `:cuda` removed from `EXCLUDED_TAGS`, not committed), classified through `python3 harness/tools/harness/gpu_check.py run . --cmd "<cmd>"`; probe: `python3 harness/tools/harness/gpu_check.py probe`.
- Benchmark: `julia --project=benchmark --threads=<T> benchmark/run.jl --suite full --filter '<regex>' [--entries …] [--backend cuda --eltype all] --out benchmark/results/<file>.json`. Redirect output to a log file and read only its tail.
- Interleaved A/B of two refs: `julia --project=benchmark benchmark/ab.jl …` (created in the first prepare sub-phase; see its header). Comparison: `julia --project=benchmark benchmark/compare.jl <a.json> <b.json>`.
- Formatter: JuliaFormatter with `.JuliaFormatter.toml` on changed files. JET: `JET.@report_opt` on the hot functions (as in `benchmark/profile.jl`).
- Lean (only when a cited Julia function moved/was renamed, and in close sub-phases): `~/.elan/toolchains/<pinned>/bin/lake build` in `lean/`, never through the elan proxy; `python3 lean/scripts/check_julia_refs.py`.
- Branch: `perf/phase-1`. Merge base: `git merge-base origin/main HEAD`.

## Julia Behavior & Tests *(required)*

`common.md` § Julia Behavior & Tests applies in full: same results within tolerance, `test/base/perf_equivalence.jl`
and its `:cuda` counterpart, a test item for every new code path, no wall-clock assertions in tests.

## Algorithm ↔ Theorem Mapping *(required)*

none.

## CPU / GPU Matrix *(required)*

| Check | Backend | Required? | Notes |
|---|---|---|---|
| Package tests | CPU, 1 thread | yes | Julia test command |
| Package tests | CPU, `--threads=auto` | yes | multi-threaded Julia test command |
| GPU tests | CUDA | not required unless a change touches code the CUDA path calls (Dev states in the hand-off whether it does; QE verifies with `grep` over `ext/`) | UNAVAILABLE on a required row is unmet |
| Benchmarks | CPU, 1 thread | yes | full matrix, interleaved, merge base vs head |
| Benchmarks | CPU, 16 threads | yes | full matrix, interleaved (regression) |
| Lean | — | yes | `lake build` + `check_julia_refs.py` |

## Performance Evidence *(required)*

`common.md` § Performance Evidence: QE re-runs the comparison independently (merge base vs phase head, interleaved, primary: CPU, 1 thread; regression: CPU, 16 threads (the kernels are shared with the threaded path)) and checks that its numbers support every "accepted" verdict of this phase in `REPORT.md`.

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `Pkg.test()` passes on CPU with 1 thread and with `--threads=auto`, including `test/base/perf_equivalence.jl`.
- [ ] The benchmark suite's correctness checks pass for every case on every backend/thread count measured.
- [ ] Every change kept in the phase has a `REPORT.md` entry with interleaved A/B evidence, and QE reproduces it.
- [ ] No case regresses by more than 3% on CPU, 1 thread or CPU, 16 threads (the kernels are shared with the threaded path), merge base vs phase head, unless an approved `T-<n>` Finding covers it.
- [ ] Steady-state allocations of `bellman!` do not increase for any case.
- [ ] No tier 1 change; tier 2 changes only additive keywords with defaults, documented and listed in the PR description; any wider change is an open `I-<n>` Finding, not merged code.
- [ ] Modularity rules hold over the whole phase diff (dispatch not branches; constants in one place with sweep data; no CUDA code in `src/`; no copy-paste variants; no new runtime dispatch per JET; formatter clean).
- [ ] Every heuristic constant added or changed in the phase links to its committed sweep summary `benchmark/results/<tag>-sweep.md`.
- [ ] Rejected experiments of the phase are recorded in `REPORT.md`.
- [ ] GPU tests: not required unless a change touches code the CUDA path calls (Dev states in the hand-off whether it does; QE verifies with `grep` over `ext/`).
- [ ] Lean: `lake build` passes and `python3 lean/scripts/check_julia_refs.py` passes.
- [ ] `docs/src/developer.md` is accurate for what the phase changed (threading/CUDA/algorithm implementation notes).

## File List

- Regression fixes only (files of this phase's experiments), each with A/B evidence
- `benchmark/REPORT.md` (Phase summary), `benchmark/results/1c-*.json` — local only, not committed
- `docs/src/developer.md` if needed
- `lean/IntervalMDPProofs/**` docstrings only, if needed
- Raw `benchmark/results/**/*.json` and `benchmark/reference/**` are **local only** (gitignored; `common.md` § Committed vs local benchmark outputs): produce and use them, but do not stage them. Commit the Markdown summaries.

## Out of Scope

- New algorithms or approximations; changes to approximation quality, default algorithms, stopping rule or tie-breaking beyond ties.
- Distributed or multi-GPU execution; non-CUDA GPU backends; model construction, file I/O (`src/Data/`), the Lean project.
- Wall-clock assertions in unit tests or CI.
- Work belonging to another sub-phase (see the run-order table in `common.md`).
