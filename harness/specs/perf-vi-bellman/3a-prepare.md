# Performance of the VI/Bellman Algorithms — 3a: Prepare — Specification (Julia + Lean)

> Sub-phase **3a** (prepare) of Phase 3 — CUDA. One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Sub-phases (Phases 1–4), § Evidence Protocol, § Julia Behavior & Tests, § Candidate Hypotheses.
> Phase branch: `perf/phase-3` (one draft PR for the whole phase). The previous sub-phase of this phase must be pushed
> first; the first sub-phase of a phase starts only after the previous phase has merged into `main`.

**Active phase: 3a**

## Objective *(required)*

Prepare Phase 3 without changing `src/` or `ext/`.

1. Write the phase plan into `benchmark/REPORT.md` (new section "Phase 3 plan"): the table below, each row citing its
   Phase 0 measurement from § 3–6 and its target cases. The plan is fixed by the spec files of this phase; to add, drop
   or reorder experiments the operator edits or adds spec files, and the plan follows.

   | Sub-phase | Spec | Content |
   |---|---|---|
   | `3f.1` | `3f.1-b2-ivi-cuda.md` | Finding B-2 — `IntervalValueIteration` on CUDA |
   | `3f.2` | `3f.2-b3-cuda-fimdp-sparse.md` | Finding B-3 — CUDA factored O-max with sparse marginals |
   | `3f.3` | `3f.3-c1-cuda-coverage.md` | Finding C-1 — CUDA coverage gaps: measure throw vs CPU fallback, make the behaviour explicit |
   | `3b.1` | `3b.1-h5-host-sync.md` | H5 — host synchronisation per VI iteration |

2. Re-run the noise check on the merge base for this phase's target cases (every CUDA `bellman` and `solve_rvi` entry, Float64 and Float32): ≥ 3 separate-process runs,
   `compare.jl --spread`; fix any case with spread > 5% (budgets/sizes in `benchmark/cases/registry.jl`).
3. Extend `test/base/perf_equivalence.jl` and its `:cuda` counterpart so they cover this phase's code paths
   (CUDA dense and sparse IMDP O-max, dense-marginal factored O-max, control synthesis, in Float64 and Float32), with reference values computed from the merge base. **If they do not exist yet** (this is the first
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
6. Make `benchmark/compare.jl` GPU-aware (carry-over item 16 of `0h-hypotheses-close.md`). Its CLOCK verdict uses only
   the host CPU probe, so in 0g a ×30 difference that was really a CUDA entry flipping between kernel speed and the
   ≈ 6 ms host floor (dense n1000 a1 Float64 `bellman`: 0.193 ms in one process, 6.05 ms in the next) was labelled
   CLOCK. For `--backend cuda` entries, flag a bimodal entry (median far from the other process's, or a sample
   distribution with two modes) as `BIMODAL (not evidence)` instead of CLOCK, and say in the summary that CUDA entries
   need ≥ 3 separate processes. Also warn (or refuse a verdict) when fewer than 3 interleaved rounds are given
   (carry-over item 15), unless `ab.jl` already does so. Record the CUDA entries that are bimodal on the merge base.
7. **Validate the CUDA timing method** (carry-over item 19 of `0h-hypotheses-close.md`). Phase 0 timed CUDA entries only
   with the host wall clock around a call that ends in `CUDA.synchronize()` (`benchmark/lib/measure.jl`, backend in
   `run.jl`). The ≈ 6 ms floor that is independent of size, the bimodal entries (0.193 ms vs ≈ 6 ms for the same entry
   in two processes) and minima of 0.29–3.3 ms suggest that the floor may come from the synchronisation method (CUDA.jl's
   default `synchronize()` is non-blocking and yields to the Julia scheduler; confirm this for the installed CUDA.jl
   version and record it), not from the package. For at least one dense, one sparse and one dense-marginal factored
   case (`bellman` and `solve_rvi`, Float64 and Float32), on the merge base, measure:
   - (a) host wall clock with the default `synchronize()` (the Phase 0 method);
   - (b) host wall clock with `synchronize(blocking = true)`;
   - (c) device time with CUDA events (`CUDA.@elapsed`), including all kernels of the call;
   - (d) the same with `julia --threads=1` and `--threads=auto`, since the yielding wait depends on the scheduler.
   Record the table (median, min, samples) in `REPORT.md` (Phase 3 plan section, "CUDA timing method"). If (a) differs from
   (b)/(c) by more than the 5% noise bound, the floor is (at least partly) a measurement artefact: change
   `lib/measure.jl` so CUDA entries report device time from events next to the host time (keep host time: it is what a
   user's synchronised call costs), and make the noise check and `ab.jl` use the event time as the primary CUDA metric.
   Re-run the noise check of step 2 with the new method.
8. **Fine-grained CUDA profiles** (carry-over item 20). Add a mode to `benchmark/profile.jl` that, for CUDA entries:
   - wraps each `bellman!` call and each VI iteration in NVTX ranges and runs the case under Nsight Systems
     (`CUDA.@profile external = true` inside `nsys profile --trace=cuda,nvtx,osrt …`), and summarises the timeline:
     per-iteration kernel time, gaps between kernels, synchronisation and memcpy time, CPU-side time between launches;
   - runs Nsight Compute (`ncu --set full`, limited with `--kernel-name`/`--launch-count` to the Bellman kernels) and
     summarises per kernel: duration, achieved DRAM throughput vs the device peak, compute throughput, occupancy
     (achieved vs theoretical), registers and shared memory per thread/block, and the limiter ncu reports;
   - records registers/shared memory from CUDA.jl as well (`CUDA.registers`, `CUDA.memory` on the compiled kernel, or
     `@device_code_ptx`/`@device_code_sass` excerpts where useful).
   Run it for the CUDA rows of the case matrix (at least `imdp-dense-n4000-a1`, `imdp-sparse-n100000-nnz10-a1`,
   `fimdp-dense-v2-d50-a1-omax`, `real-multiObj_robotIMDP`) and append the summaries to the existing
   `benchmark/profiles/cuda-*.md` (new section "Nsight"), not the raw reports (`.nsys-rep`/`.ncu-rep` stay local under
   `benchmark/results/logs/`). If `ncu` cannot read the performance counters (e.g.
   `ERR_NVGPUCTRPERM`), record the exact error and the fix (driver option `NVreg_RestrictProfilingToAdminUsers=0` or
   running as admin) as a blocker for the operator; never run it with elevated rights on your own.
9. **Re-check H5 against steps 7–8** before `3b.1` runs: write into the Phase 3 plan whether the ≈ 6 ms floor is a
   property of the package's synchronisation (H5 stands, with the prediction restated in event time and host time) or of
   the measuring method (H5's prediction is revised or the hypothesis is dropped, with the operator's decision). Add the
   kernel-level numbers from step 8 (achieved bandwidth vs peak) as the CUDA counterpart of the CPU roofline in § 6.

**The GPU probe must pass** (`gpu_check.py probe` → `GPU_PROBE=FUNCTIONAL`). If it does not, stop: Phase 3 is blocked.

Adds or semantically changes a VI/Bellman algorithm: **no**. Every change computes the same result as the merge base
within the tolerances of `common.md` § Julia Behavior & Tests; anything else is raised as a Finding (`S-<n>`) and not
merged. Tier 1/2 interface changes: only additive keywords with defaults, documented; anything wider is an `I-<n>`
Finding (`common.md` § Interface Stability). Lean: if a Julia function cited by a Lean docstring is renamed, moved or
split, update the docstring in this run and keep `lake build` and `check_julia_refs.py` green.
Follow `common.md` § Context budget. Ops: branch `perf/phase-3` from `origin/main`, commit, open a **draft** PR.

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
- Nsight Systems: `/usr/local/bin/nsys` (2026.1.3). Nsight Compute: `/usr/local/cuda/bin/ncu`. `nvidia-smi` fails on this host (NVML driver/library mismatch, see `harness/LEARNING.md`); `gpu_check.py probe` is the source of truth for GPU availability. Profiling runs write to the scratchpad or `benchmark/results/logs/`.
- Branch: `perf/phase-3`. Merge base: `git merge-base origin/main HEAD`.

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
| GPU tests | CUDA | **required**: the GPU probe must pass; UNAVAILABLE blocks the phase | UNAVAILABLE on a required row is unmet |
| Benchmarks | CUDA, Float64 and Float32 | yes | noise check on the merge base, target cases |

## Performance Evidence *(required)*

Noise check only (`common.md` § Evidence Protocol → Noise bound). No speedup is claimed. The timing-method comparison of step 7 is a measurement of the tool, not a speedup claim; it uses ≥ 3 separate processes per method.

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to the merge base.
- [ ] `REPORT.md` has the Phase 3 plan with every row of the table above, each citing its Phase 0 measurement and target cases.
- [ ] Noise check re-run on the merge base for the target cases; every target case is within the 5% spread bound (or is listed as unusable with the reason and removed from the target list).
- [ ] `test/base/perf_equivalence.jl` and its `:cuda` counterpart cover this phase's code paths (CUDA dense and sparse IMDP O-max, dense-marginal factored O-max, control synthesis, in Float64 and Float32) with reference values from the merge base; `Pkg.test()` passes with 1 thread and with `--threads=auto`.
- [ ] `benchmark/ab.jl` exists and runs an interleaved A/B of two refs (demonstrated on merge base vs merge base for one case: verdict "no change"), writing a summary `compare.jl` reads.
- [ ] `benchmark/lib/clock.jl` takes its reference after warm-up and under the trial thread load (fixed here or by an earlier prepare sub-phase): one case run on the merge base at 1 and at 16 threads, in a steady clock state, has no CLOCK flags caused by the reference.
- [ ] `benchmark/compare.jl` labels a CUDA entry that flips between processes as `BIMODAL (not evidence)`, not CLOCK (demonstrated on two merge-base CUDA runs), and warns when fewer than 3 interleaved rounds are given; the bimodal CUDA entries on the merge base are listed in `REPORT.md`.
- [ ] The GPU probe passes; its output is recorded in `REPORT.md`.
- [ ] The CUDA timing method is validated: the table of step 7 ((a) default sync, (b) blocking sync, (c) CUDA events, (d) 1 thread vs `--threads=auto`) is in `REPORT.md` for the named cases, with the CUDA.jl synchronisation mode recorded; if (a) deviates from (b)/(c) beyond the noise bound, `lib/measure.jl` reports event time for CUDA entries, `ab.jl`/the noise check use it, and the noise check was re-run with it.
- [ ] `profiles/cuda-*.md` for the four named cases have an "Nsight" section with an `nsys` timeline summary (NVTX ranges per `bellman!`/VI iteration) and an `ncu` kernel summary (duration, DRAM throughput vs peak, occupancy, registers/shared memory, limiter) — or, for `ncu`, the exact permission error recorded as an operator blocker.
- [ ] The Phase 3 plan states, from steps 7–8, whether the ≈ 6 ms floor is a package property or a measurement artefact, and H5's prediction is restated (or revised/dropped with the operator's decision) accordingly.

## File List

- `benchmark/REPORT.md` (Phase plan section, noise results)
- `benchmark/cases/registry.jl` (noise fixes only)
- `test/base/perf_equivalence.jl`, `test/cuda/**` counterpart (and their `test/runtests.jl` registration if needed)
- `benchmark/ab.jl`, `benchmark/README.md` (A/B paragraph, clock-reference note)
- `benchmark/lib/clock.jl` (clock reference fix only, if not done yet)
- `benchmark/compare.jl` (GPU-aware verdict, round-count warning), `benchmark/README.md` (compare paragraph)
- `benchmark/lib/measure.jl` (CUDA event timing, only if step 7 shows the host method is distorted), `benchmark/run.jl` (CUDA backend sync / timing option), `benchmark/ab.jl` (CUDA primary metric)
- `benchmark/profile.jl` (Nsight/NVTX mode), `benchmark/Project.toml`/`Manifest.toml` (only if `NVTX.jl` is needed), `benchmark/profiles/cuda-*.md` ("Nsight" section)
- `benchmark/results/logs/*.nsys-rep`, `*.ncu-rep`, `benchmark/results/3a-timing-*.json` — local only, not committed
- `benchmark/results/3a-noise-*.json` — local only, not committed
- Raw `benchmark/results/**/*.json` and `benchmark/reference/**` are **local only** (gitignored; `common.md` § Committed vs local benchmark outputs): produce and use them, but do not stage them. Commit the Markdown summaries.

## Out of Scope

- New algorithms or approximations; changes to approximation quality, default algorithms, stopping rule or tie-breaking beyond ties.
- Distributed or multi-GPU execution; non-CUDA GPU backends; model construction, file I/O (`src/Data/`), the Lean project.
- Wall-clock assertions in unit tests or CI.
- Work belonging to another sub-phase (see the run-order table in `common.md`).
