# Performance of the VI/Bellman Algorithms — 1a: Prepare — Specification (Julia + Lean)

> Sub-phase **1a** (prepare) of Phase 1 — Single-threaded CPU. One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Sub-phases (Phases 1–4), § Evidence Protocol, § Julia Behavior & Tests, § Candidate Hypotheses.
> Phase branch: `perf/phase-1` (one draft PR for the whole phase). The previous sub-phase of this phase must be pushed
> first; the first sub-phase of a phase starts only after the previous phase has merged into `main`.

**Active phase: 1a**

## Objective *(required)*

Prepare Phase 1 without changing `src/` or `ext/`.

1. Write the phase plan into `benchmark/REPORT.md` (new section "Phase 1 plan"): the table below, each row citing its
   Phase 0 measurement from § 3–6 and its target cases. The plan is fixed by the spec files of this phase; to add, drop
   or reorder experiments the operator edits or adds spec files, and the plan follows.

   | Sub-phase | Spec | Hypothesis |
   |---|---|---|
   | `1b.1` | `1b.1-h1-sort-ordering.md` | H1 — type-stable ordering in the sparse O-max sort (IMDP/product) |
   | `1b.2` | `1b.2-h7-sparse-gather.md` | H7 — sparse gather / tuple construction |
   | `1b.3` | `1b.3-h6-dense-gap-walk.md` | H6 — dense gap walk |

2. Re-run the noise check on the merge base for this phase's target cases (all `imdp-*`, `product-imdp-*`, `cs-imdp-*` and `real-multiObj_robotIMDP` entries `bellman` and `solve_rvi` at 1 thread, and `bellman` at 16 threads): ≥ 3 separate-process runs,
   `compare.jl --spread`; fix any case with spread > 5% (budgets/sizes in `benchmark/cases/registry.jl`).
3. Extend `test/base/perf_equivalence.jl` and its `:cuda` counterpart so they cover this phase's code paths
   (dense and sparse IMDP O-max, product, control synthesis, single- and multi-threaded workspaces constructed directly), with reference values computed from the merge base. **If they do not exist yet** (this is the first
   phase to run), create them, covering a small version of each case-matrix row as described in `common.md`
   § Julia Behavior & Tests.
4. **If `benchmark/ab.jl` does not exist yet**, create the interleaved A/B runner: it takes two git refs, a thread count/
   backend and `run.jl` filter options, checks out each ref into a separate worktree, runs A B A B … (≥ 3 rounds, separate
   processes) and writes per-round JSON plus a summary that `compare.jl` reads.


Adds or semantically changes a VI/Bellman algorithm: **no**. Every change computes the same result as the merge base
within the tolerances of `common.md` § Julia Behavior & Tests; anything else is raised as a Finding (`S-<n>`) and not
merged. Tier 1/2 interface changes: only additive keywords with defaults, documented; anything wider is an `I-<n>`
Finding (`common.md` § Interface Stability). Lean: if a Julia function cited by a Lean docstring is renamed, moved or
split, update the docstring in this run and keep `lake build` and `check_julia_refs.py` green.
Follow `common.md` § Context budget. Ops: branch `perf/phase-1` from `origin/main`, commit, open a **draft** PR.

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
| Benchmarks | CPU, 1 thread | yes | noise check on the merge base, target cases |

## Performance Evidence *(required)*

Noise check only (`common.md` § Evidence Protocol → Noise bound). No speedup is claimed.

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to the merge base.
- [ ] `REPORT.md` has the Phase 1 plan with every row of the table above, each citing its Phase 0 measurement and target cases.
- [ ] Noise check re-run on the merge base for the target cases; every target case is within the 5% spread bound (or is listed as unusable with the reason and removed from the target list).
- [ ] `test/base/perf_equivalence.jl` and its `:cuda` counterpart cover this phase's code paths (dense and sparse IMDP O-max, product, control synthesis, single- and multi-threaded workspaces constructed directly) with reference values from the merge base; `Pkg.test()` passes with 1 thread and with `--threads=auto`.
- [ ] `benchmark/ab.jl` exists and runs an interleaved A/B of two refs (demonstrated on merge base vs merge base for one case: verdict "no change"), writing a summary `compare.jl` reads.

## File List

- `benchmark/REPORT.md` (Phase plan section, noise results)
- `benchmark/cases/registry.jl` (noise fixes only)
- `test/base/perf_equivalence.jl`, `test/cuda/**` counterpart (and their `test/runtests.jl` registration if needed)
- `benchmark/ab.jl`, `benchmark/README.md` (A/B paragraph)
- `benchmark/results/1a-noise-*.json` — local only, not committed
- Raw `benchmark/results/**/*.json` and `benchmark/reference/**` are **local only** (gitignored; `common.md` § Committed vs local benchmark outputs): produce and use them, but do not stage them. Commit the Markdown summaries.

## Out of Scope

- New algorithms or approximations; changes to approximation quality, default algorithms, stopping rule or tie-breaking beyond ties.
- Distributed or multi-GPU execution; non-CUDA GPU backends; model construction, file I/O (`src/Data/`), the Lean project.
- Wall-clock assertions in unit tests or CI.
- Work belonging to another sub-phase (see the run-order table in `common.md`).
