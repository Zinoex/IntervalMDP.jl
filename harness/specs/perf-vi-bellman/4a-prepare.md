# Performance of the VI/Bellman Algorithms — 4a: Prepare — Specification (Julia + Lean)

> Sub-phase **4a** (prepare) of Phase 4 — Factored algorithms and the VI loop. One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Sub-phases (Phases 1–4), § Evidence Protocol, § Julia Behavior & Tests, § Candidate Hypotheses.
> Phase branch: `perf/phase-4` (one draft PR for the whole phase). The previous sub-phase of this phase must be pushed
> first; the first sub-phase of a phase starts only after the previous phase has merged into `main`.

**Active phase: 4a**

## Objective *(required)*

Prepare Phase 4 without changing `src/` or `ext/`.

1. Write the phase plan into `benchmark/REPORT.md` (new section "Phase 4 plan"): the table below, each row citing its
   Phase 0 measurement from § 3–6 and its target cases. The plan is fixed by the spec files of this phase; to add, drop
   or reorder experiments the operator edits or adds spec files, and the plan follows.

   | Sub-phase | Spec | Content |
   |---|---|---|
   | `4f.1` | `4f.1-b1-strategy-cache.md` | Finding B-1 — stationary strategy cache |
   | `4b.1` | `4b.1-h4-vertex-support.md` | H4 — vertex enumeration over the support only |
   | `4b.2` | `4b.2-h3-mccormick-lp-reuse.md` | H3 — rebuild-free McCormick LP |
   | `4b.3` | `4b.3-h1-factored-ordering.md` | H1 (factored part) — type-stable ordering in `orthogonal_inner_bellman!` (skip if done in `1b.1`) |
   | `4b.4` | `4b.4-h9-fimdp-omax-temporaries.md` | H9 — per-state temporaries in factored O-max |
   | — | — | H10 — VI loop: not scheduled (loop overhead ≤ 3%, `REPORT.md` § 3); revisit only if `bellman!` became ≥ 10× faster |

2. Re-run the noise check on the merge base for this phase's target cases (every `fimdp-*` entry `bellman` and `solve_rvi` at 1 thread, and `cs-imdp-*` `solve_cs_*`): ≥ 3 separate-process runs,
   `compare.jl --spread`; fix any case with spread > 5% (budgets/sizes in `benchmark/cases/registry.jl`).
3. Extend `test/base/perf_equivalence.jl` and its `:cuda` counterpart so they cover this phase's code paths
   (factored O-max (dense and sparse marginals), McCormick and vertex enumeration, 2 and 3 state variables, stationary and time-varying strategy caches), with reference values computed from the merge base. **If they do not exist yet** (this is the first
   phase to run), create them, covering a small version of each case-matrix row as described in `common.md`
   § Julia Behavior & Tests.
4. **If `benchmark/ab.jl` does not exist yet**, create the interleaved A/B runner: it takes two git refs, a thread count/
   backend and `run.jl` filter options, checks out each ref into a separate worktree, runs A B A B … (≥ 3 rounds, separate
   processes) and writes per-round JSON plus a summary that `compare.jl` reads.
5. **If `benchmark/lib/clock.jl` has not been fixed yet** (carry-over item 17 of `0h-hypotheses-close.md`; whichever
   prepare sub-phase runs first does it): take the clock-probe reference after warm-up, once the clock has settled, and
   under the same thread load as the trials. Today it is taken unloaded and, at 1 thread, while the core still boosts
   (≈ 197 µs vs ≈ 436 µs), so nearly every 8/16-thread entry and some 1-thread entries are flagged CLOCK for no reason.
   Record the change and a before/after count of CLOCK flags in `benchmark/README.md`.
6. Add a `bellman` entry for the control-synthesis cases (`cs-imdp-*`) to `benchmark/cases/registry.jl`, with stored
   reference values from the merge base (Float64 CPU; CUDA where the case runs on CUDA), so that the strategy-cache
   experiments (`4f.1`) are checked against a stored reference. Today `benchmark/profile.jl` uses a `bellman_cs`
   pseudo-entry whose check compares the strategy-cache kernel with the same kernel using the default cache (carry-over
   item 13 of `0h-hypotheses-close.md`); point `profile.jl` at the new entry and drop the pseudo-entry.


Adds or semantically changes a VI/Bellman algorithm: **no**. Every change computes the same result as the merge base
within the tolerances of `common.md` § Julia Behavior & Tests; anything else is raised as a Finding (`S-<n>`) and not
merged. Tier 1/2 interface changes: only additive keywords with defaults, documented; anything wider is an `I-<n>`
Finding (`common.md` § Interface Stability). Lean: if a Julia function cited by a Lean docstring is renamed, moved or
split, update the docstring in this run and keep `lake build` and `check_julia_refs.py` green.
Follow `common.md` § Context budget. Ops: branch `perf/phase-4` from `origin/main`, commit, open a **draft** PR.

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
- Branch: `perf/phase-4`. Merge base: `git merge-base origin/main HEAD`.

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
| GPU tests | CUDA | required whenever a change touches code the CUDA path calls (the factored O-max inner step is shared); otherwise not required | UNAVAILABLE on a required row is unmet |
| Benchmarks | CPU, 1 thread | yes | noise check on the merge base, target cases |

## Performance Evidence *(required)*

Noise check only (`common.md` § Evidence Protocol → Noise bound). No speedup is claimed.

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to the merge base.
- [ ] `REPORT.md` has the Phase 4 plan with every row of the table above, each citing its Phase 0 measurement and target cases.
- [ ] Noise check re-run on the merge base for the target cases; every target case is within the 5% spread bound (or is listed as unusable with the reason and removed from the target list).
- [ ] `test/base/perf_equivalence.jl` and its `:cuda` counterpart cover this phase's code paths (factored O-max (dense and sparse marginals), McCormick and vertex enumeration, 2 and 3 state variables, stationary and time-varying strategy caches) with reference values from the merge base; `Pkg.test()` passes with 1 thread and with `--threads=auto`.
- [ ] `benchmark/ab.jl` exists and runs an interleaved A/B of two refs (demonstrated on merge base vs merge base for one case: verdict "no change"), writing a summary `compare.jl` reads.
- [ ] `benchmark/lib/clock.jl` takes its reference after warm-up and under the trial thread load (fixed here or by an earlier prepare sub-phase): one case run on the merge base at 1 and at 16 threads, in a steady clock state, has no CLOCK flags caused by the reference.
- [ ] `registry.jl` has a `bellman` entry for every `cs-imdp-*` case with a stored reference from the merge base; `run.jl --filter '^cs-imdp' --entries bellman` passes its correctness check; `profile.jl` uses it and no longer has the `bellman_cs` pseudo-entry.

## File List

- `benchmark/REPORT.md` (Phase plan section, noise results)
- `benchmark/cases/registry.jl` (noise fixes; `bellman` entry for `cs-imdp-*`), `benchmark/profile.jl` (switch from `bellman_cs` to that entry)
- `benchmark/reference/cpu-Float64/cs-imdp-*`, `benchmark/reference/cuda-*/cs-imdp-*` — local only, not committed
- `test/base/perf_equivalence.jl`, `test/cuda/**` counterpart (and their `test/runtests.jl` registration if needed)
- `benchmark/ab.jl`, `benchmark/README.md` (A/B paragraph, clock-reference note)
- `benchmark/lib/clock.jl` (clock reference fix only, if not done yet)
- `benchmark/results/4a-noise-*.json` — local only, not committed
- Raw `benchmark/results/**/*.json` and `benchmark/reference/**` are **local only** (gitignored; `common.md` § Committed vs local benchmark outputs): produce and use them, but do not stage them. Commit the Markdown summaries.

## Out of Scope

- New algorithms or approximations; changes to approximation quality, default algorithms, stopping rule or tie-breaking beyond ties.
- Distributed or multi-GPU execution; non-CUDA GPU backends; model construction, file I/O (`src/Data/`), the Lean project.
- Wall-clock assertions in unit tests or CI.
- Work belonging to another sub-phase (see the run-order table in `common.md`).
