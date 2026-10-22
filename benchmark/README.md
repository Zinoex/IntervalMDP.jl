# IntervalMDP.jl benchmark suite

Reproducible benchmarks for the Bellman update and the value-iteration loops
(spec: `harness/specs/perf-vi-bellman/`, shared rules in `common.md`, one spec per sub-phase). Results, profiles and the ranked
hypothesis list live in [`REPORT.md`](REPORT.md).

## Layout

| Path | Purpose |
|---|---|
| `run.jl` | entry point: runs a suite, checks correctness, writes one JSON file per run |
| `compare.jl` | A/B and interleaved comparison (evidence protocol) and run-to-run spread (noise bound) |
| `profile.jl` | CPU sampling profile, `Profile.Allocs`, `JET.@report_opt`, `CUDA.@profile` → `profiles/<case>.md` |
| `stream.jl` | measured machine roofs (DRAM / L2 bandwidth, in-L1 FMA rate) for the roofline estimate |
| `analyze.jl` | strong-scaling, size-scaling and roofline tables from result files |
| `cases/*.jl` | seeded problem generators and the case registry (stable names) |
| `lib/*.jl` | environment capture, reference store and correctness check, timing protocol |
| `reference/<backend>-<eltype>/` | reference values written once from the base ref (20fc03b = main; src/ and ext/ tree hashes 290f675 / 5a4cf21) |
| `results/*.json` | result files (`baseline-<sha>-<backend>-t<threads>[-<tag>].json`, scaling, sizes, stream) |
| `profiles/*.md` | profile summaries |

The environment (`Project.toml` + committed `Manifest.toml`) contains only what the
suite needs. `Revise`, `Cthulhu` and `ProfileView` were removed. `JET` is only loaded by
`profile.jl`, and `CUDA`/`cuSPARSE` only with `--backend cuda`, so neither can affect CPU timings.
`IntervalMDP` is taken from the parent directory (`[sources] IntervalMDP = {path = ".."}`).

## Running

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'

# full suite, CPU, 1/4/8/16 threads
julia --project=benchmark --threads=1  benchmark/run.jl --suite full --out benchmark/results/<name>-t1.json
julia --project=benchmark --threads=16 benchmark/run.jl --suite full --out benchmark/results/<name>-t16.json

# CUDA (Float64 and Float32 where the case supports it)
julia --project=benchmark benchmark/run.jl --suite full --backend cuda --out benchmark/results/<name>-cuda.json

# compare a candidate with the baseline (one round) ...
julia --project=benchmark benchmark/compare.jl BASE.json CAND.json
# ... or interleaved rounds A1 B1 A2 B2 A3 B3 (separate processes, run in that order)
julia --project=benchmark benchmark/compare.jl --base A1.json,A2.json,A3.json --cand B1.json,B2.json,B3.json
# run-to-run spread of repeated runs of the same commit (also shows the clock-probe spread)
julia --project=benchmark benchmark/compare.jl --spread R1.json R2.json R3.json

# interleaved A/B of two git refs (worktrees, A1 B1 A2 B2 A3 B3, summary via compare.jl)
julia --project=benchmark benchmark/ab.jl --a <base-ref> --b <cand-ref> --threads 1 \
    --filter '^imdp-sparse-n10000-nnz10-a1$' --entries bellman,solve_rvi

# profile a case
julia --project=benchmark --threads=1 benchmark/profile.jl --case imdp-dense-n1000-a4
```

Options of `run.jl` (see the usage text at the top of the file): `--suite full|quick|scaling|sizes`
(comma-separated), `--backend cpu|cuda`, `--eltype Float64|Float32|all`, `--out`,
`--filter REGEX`, `--entries bellman,solve_rvi,...`, `--pin compact|none`,
`--budget-scale X`, `--max-entry-seconds S`, `--list` (with `--suite`: the cases of that suite; alone: every
registered case, 53), `--write-reference [--reference-only]`.

Suites: `full` = the case matrix of the spec (40 cases, 131 CPU entries); `quick` = a 16-case subset for smoke
tests (1 and 4 threads); `scaling` = the fixed-size dense/sparse O-max cases used for strong scaling;
`sizes` = the size-scaling series (`bellman!` only).

## What is timed

Every case has a stable name and is generated from `StableRNG(crc32c(name))`
(stream stable across Julia versions). Entries per case:

| Entry | Call |
|---|---|
| `workspace` | `IntervalMDP.construct_workspace(model, alg)` |
| `bellman` | `IntervalMDP.bellman!(ws, cache, Vres, V, model; upper_bound = false, maximize = true)` with the workspace and `construct_strategy_cache(model)` built beforehand (steady state; `V` is a seeded uniform vector) |
| `solve_rvi` | `solve(VerificationProblem(model, InfiniteTimeReachability(reach 20%, 1e-6), Pessimistic, Maximize), RobustValueIteration(alg))` |
| `solve_ivi` | `solve(VerificationProblem(model, InfiniteTimeReachAvoid(reach 10%, avoid 10%, 1e-6), …), IntervalValueIteration(alg))` |
| `solve_dfa` | product IMDP × 4-state DFA, `InfiniteTimeDFAReachability([3], 1e-6)`, RVI |
| `solve_cs_stationary` / `solve_cs_timevarying` | `solve(ControlSynthesisProblem(...))`, infinite (stationary strategy) / horizon 10 (time-varying) |

`construct_workspace` picks the threaded workspace when `Threads.nthreads() > 1` and the
model has more than 10 source states, so thread count changes the code path; the
workspace type is recorded per entry (`workspace_type`).

### Timing protocol and parameters

Per entry (`lib/measure.jl`): one warm-up call (compilation; its result is checked for
correctness), one estimate call `t` (the warm-up time is reused when > 5 s), then
BenchmarkTools `run` with

* `samples = clamp(round(budget / t), min_samples, 10 000)`, capped so that
  `samples · t ≤ --max-entry-seconds` (default 90 s) but never below 3;
* budgets: `workspace` 3 s / ≥ 50 samples, `bellman` 4 s / ≥ 40 samples, solves 4 s / ≥ 5 samples (the first two were raised from 1 s / 10 and 2 s / 20 by the Phase 0 noise fix, see `REPORT.md`); the `imdp-*` cases set their `bellman` budget to 2 s in `cases/imdp.jl`, and since 1a `NOISE_BUDGET_SCALE` in `cases/registry.jl` multiplies the `bellman` budget of every `imdp-*` and `product-imdp-*` case by 3, so it is **6 s for `imdp-*` and 12 s for `product-imdp-*`** (`REPORT.md` § 10.2; the `size-*`, `fimdp-*`, `cs-imdp-*` and real-model budgets are unchanged);
* `evals = ceil(50 µs / t)` for calls shorter than 50 µs, else 1;
* `gctrial = true`, `gcsample = false`; `seconds` is set high enough never to be the binding limit;
* **clock guard** (`lib/clock.jl`): the run starts by taking a clock-probe reference (a fixed chain of
  dependent integer multiply–adds, ≈0.5 ms) under the trial thread load after the clock has settled (see "Clock
  reference" below). Every trial is bracketed by two probes (each after 30 ms of
  spinning to leave idle states); if the probe *before* the trial deviates by more than 5% from the reference the
  trial is repeated after a 20 s pause (up to `--clock-retries` times, default 3; only for
  entries with ≤ 30 s of sampling) and the attempt closest to the reference is kept. Each result records
  `clock_probe_before_ns`, `clock_probe_after_ns`, `clock_probe_ratio`, `clock_state` (`nominal`/`deviating`, from the
  before-probe), `clock_after_deviation` (informational: after long memory-heavy trials the core is often ≈28%
  slower, the package power being shared with the memory traffic) and `clock_attempts`. This was added because the reference laptop switches between ≈5 GHz and a firmware cap
  of ≈2.3 GHz for minutes at a time (see `REPORT.md`), which moves compute-bound timings by ≈2.1×.

`compare.jl` reports a difference as `CLOCK (not evidence)` instead of speedup/regression when the clock probes
of the two sides differ by more than 5% or an entry was measured with a deviating clock.

### Clock reference

Since sub-phase 1a (carry-over item 17 of the Phase 0 close) the reference is no longer the median of 20 probes taken
unloaded right at process start. `init_clock_reference!` now (1) busy-spins all `Threads.nthreads(:default)` threads for
1 s and then takes a probe on the main thread, i.e. the same thread load and the same "probe right after threaded work"
position as the pre-trial probe of each trial (the probe that decides the CLOCK flag); (2) repeats this until 5
consecutive probes agree within 2% and at least 10 s have
passed (settled), or 60 s have passed (logged as `NOT settled`); (3) takes the median of 9 further load+probe rounds as
the reference. The settle and reference probes are printed in the run log (`[ Info: clock reference …`). Since 1a
cycle 2 the reference is also logged as `NOT settled`, with a `clock reference not settled` warning, when the range of
those 9 probes, (max − min)/median, exceeds 2% (`CLOCK_SETTLE_TOL`); this is a flag only, the reference value and the
timing procedure are unchanged. A run with a `NOT settled` reference is not evidence.
`measurement_parameters` in `lib/measure.jl` still records only `reference_probe_ns`; the settle series is kept in
`CLOCK_REF_INFO` but not yet written to the result JSON (`REPORT.md` § 10.5, 1a Finding F-3), so `ab.jl` detects a
not-settled reference from the log text; logs written before cycle 2 print `settled` even when only the range check
would fail, and are not flagged. The cost is ≈ 20–70 s per process: the settle phase (10–60 s, plus at most one
≈ 1 s round past 60 s) and the 9 reference rounds of ≈ 1 s each.

Why: the old reference was taken unloaded, so at t = 16 the pre-trial probes (taken after threaded work, ≈ 1.28×
slower) flagged almost every entry, and at t = 1 it could be taken while the core was still boosting (196.5 µs vs
≈ 436 µs later in the run). CLOCK flags with the old reference, from the stored Phase 0 files (`benchmark/results/`,
local): `baseline-20fc03b-cpu-t16.json` 128 / 131 and `-t16-rerun1.json` 129 / 131 entries, `sizes-20fc03b-cpu-t16.json`
12 / 13, `scaling-20fc03b-cpu-t1.json` 4 / 4 (reference 196.5 µs, boosting); `baseline-20fc03b-cpu-t1.json` 2 / 131.
After (1a, 2026-10-20, Performance power profile, `real-multiObj_robotIMDP` with entries `workspace`, `bellman`,
`solve_rvi`, `solve_cs_stationary` on the merge base, `REPORT.md` § 10.3): new reference 0 / 4 CLOCK flags at t = 1 and
0 / 4 at t = 16 (every pre-trial probe ratio 1.000, informational post-trial probe up to 37.8% above the reference at
t = 16; reference 196.5 µs, settled in ≈ 10 s). The old reference, run back-to-back from a `bcbf0df` worktree in the
same session, also gave 0 / 4 and 0 / 4, but with pre-trial probe ratios up to 1.046 at t = 1 and 3 of 8 entries
re-measured after pre-trial deviations of 6.2–22.2%. In the 24 noise-check and `ab.jl` runs of the 1a session (2026-10-20 ≈ 10:57–12:33 UTC) the reference (median) was 196.5 µs in every run and every kept
pre-trial probe (`clock_probe_ratio`) was within 4.1% of it (0.9999–1.041; 0 CLOCK flags in 366 entries), but the
informational post-trial probe (`clock_after_deviation`, which does not affect the flag) was more than 5% slower than
the reference in 67 of 366 entries (64 of them 16-thread `bellman`, a per-case pattern that does not mark the unusable
t = 16 entries) and up to 63.4% slower (`REPORT.md` § 10.3). Transient clock drops also occurred: 7 of the 24
runs had reference-probe ranges of 4.6–24.4%, and 16 trials were re-measured after pre-trial deviations of 5.4–20.6%
(`REPORT.md` § 10.3). There was no systematic offset of the Phase 0 kind. Clock behaviour is session-dependent: a
QE re-run later the same day (≈ 12:45–14:03 UTC) under the same Performance profile showed clock-state switches, with
the reference probes of one run ranging over 196.5–425.6 µs (`REPORT.md` § 10.2). The new protocol removes the cause of
the systematic offsets (an unloaded or boosting reference) in either profile; it does not prevent clock-state switches
during a run, which the `NOT settled` flag and the Phase 1 evidence rule below catch.

With these parameters the full CPU suite takes an estimated ≈ 47–48 min at 1 thread and ≈ 17–28 min at 4–16 threads
on the reference host (dominated by the `imdp-sparse-n100000-nnz100-a4` and `imdp-dense-n4000-a4` solves,
which get 3–5 samples). The full suite was not re-timed in 1a; the estimate is the Phase 0 run time (≈ 45 min and
≈ 15–25 min, with the pre-1a `bellman` budgets and the old clock reference) plus the extra sampling time of the 3×
`bellman` budget, measured as +74–76 s at t = 1 and +76–80 s at t = 16 (the t = 16 value is used for 4–16 threads; sum of samples × evals × median over the 20
`imdp-*`/`product-imdp-*` `bellman` entries, run *i* of `results/1a-noise-fix-t{1,16}-r*.json` against run *i* of
`results/1a-noise-t{1,16}-r*.json`, local; at most 14 × 4 s + 6 × 8 s = 104 s, less where the 10 000-sample cap
binds), plus ≈ 20–70 s for the clock reference ("Clock reference" above; ≈ 20 s when it settles at once, ≈ 70 s when
the settle phase times out), each end rounded up to the next minute: 45 min + 74 s + 20 s ≈ 46.6 min → 47 min to
45 min + 76 s + 70 s ≈ 47.4 min → 48 min at t = 1; 15 min + 76 s + 20 s ≈ 16.6 min → 17 min to 25 min + 80 s + 70 s
≈ 27.5 min → 28 min at 4–16 threads.

### Thread pinning

`run.jl` pins Julia thread *i* to logical CPU *i−1* with ThreadPinning.jl (`--pin compact`,
the default). On the Intel Core Ultra 7 255H CPUs 0–5 are P-cores (5.1 GHz), 6–13 E-cores
(4.4 GHz) and 14–15 low-power E-cores (2.5 GHz), so `t ≤ 6` runs on P-cores only, `t = 8`
adds two E-cores and `t = 16` includes the two LP-E cores. The mapping and the core type of
every thread are recorded in the environment block. `--pin none` leaves placement to the OS.

## Output

One JSON file per run:

* `environment`: git SHA, branch, dirty flag and whether `src/`/`ext/` are dirty; `base_ref`, the git tree hashes of `src/` and `ext/` (`git rev-parse HEAD:src`, `HEAD:ext`) and whether they equal those of the base ref, so the measured source is identified independently of the checked-out commit; Julia version,
  commit, optimisation level, bounds checking; `nthreads`, threadpool sizes, GC threads, BLAS threads
  and config; pinning and thread → CPU → core-type mapping; CPU model, per-CPU max frequency and core type,
  governor, energy-performance preference, platform profile, turbo, AC power, load average and free
  memory at start; GPU block (model, driver, CUDA runtime, `CUDA.versioninfo()` SHA-256 digest and head)
  for CUDA runs; for CPU runs CUDA is deliberately not loaded, so the block holds the GPU model and kernel driver
  version read from `/proc/driver/nvidia` and records CUDA runtime and `versioninfo` digest as `unavailable`; date.
* `parameters`: the timing parameters above, including the clock-guard settings and the reference probe.
* `results[]`: per case/entry/eltype: `median_ns`, `min_ns`, `mean_ns`, `std_ns`, `max_ns`,
  `samples`, `evals`, `allocs`, `memory_bytes`, `gc_median_ns`, `gc_mean_ns`,
  `gc_fraction_of_mean`, raw `times_ns`/`gctimes_ns` (needed for the Mann–Whitney test),
  `iterations` (solves), `workspace_type`, case `meta` (sizes for the roofline model),
  `correctness` and `valid`.
* `summary`: number of entries, invalid entries, wall time.

## Correctness check

`reference/<backend>-<eltype>/` (local only, gitignored) holds the outcome of every entry produced by the
base ref, written with

```sh
julia --project=benchmark --threads=1 benchmark/run.jl --suite full --write-reference --reference-only
```

`--write-reference` refuses to run (exit 3, no override) unless `src/` and `ext/` have no committed difference
to the base ref (`git diff --quiet $BENCH_BASE_REF HEAD -- src ext`, default `20fc03b`) and no uncommitted,
staged or untracked changes (`git status --porcelain -- src ext`). `--reference-only` skips timing; without
`--write-reference` it only checks every outcome against the reference (a fast full correctness pass).
`--out` is optional with `--reference-only` (default `results/logs/reference-{write,check}-<sha>-<backend>.json`).
Every run compares against the reference (`lib/reference.jl`) with the tolerances of the spec:

| Outcome | Bound on ‖V − V_ref‖∞ | Iterations |
|---|---|---|
| single `bellman!` (Float64) | 1e-12 | – |
| converged infinite-horizon solve | 10 · ε_conv (= 1e-5 for ε = 1e-6) | ± 1 |
| finite-horizon solve, horizon H (Float64) | H · 1e-12 | exact |
| any Float32 outcome | max(1e-5, bound above) | as above |

Control-synthesis strategies are compared entry by entry. A differing strategy is accepted
only if its policy-evaluated value (`VerificationProblem(model, spec, strategy)`) is within the
tolerance of the reference value (ties). A failed or missing check sets `valid = false`; such a
result may not be used as evidence and `compare.jl` reports it as `INVALID`.

The reference values are raw little-endian `Float64` (values) / `Int32` (strategies) files
plus an `index.json` with lengths, iteration counts, kinds and tolerances. Vectors longer than 2¹⁷ are stored
(and compared) as the stride subsample `v[1:stride:end]` (only the n = 10⁶ size case). Entries without an
observable outcome (`workspace`) are `not-applicable` and stay valid.

## Comparing results

`compare.jl BASE.json CAND.json` (or interleaved rounds with `--base A1,A2,… --cand B1,B2,…`) prints one row per
case/entry/eltype present in all files: base and candidate median, ratio cand/base, the per-round ratios, the
Mann–Whitney U p-value on the pooled samples, the verdict and the allocations. Verdicts (§ Evidence Protocol):
`speedup` if the ratio is ≤ 0.95 and holds in every round (or p < 0.01); `regression` if the ratio is ≥ 1.03
under the same test; `no change` otherwise; `INVALID` if either side failed its correctness check;
`CLOCK (not evidence)` if the difference would count but the clock probes of the two sides differ by > 5% or an
entry ran with a deviating clock. `ALLOC+` flags more allocations in a steady-state `bellman` entry. The exit code
is 1 if any entry is a regression or invalid. `compare.jl --spread R1.json R2.json …` prints, per entry, the
medians of the repeated runs, their spread (max − min)/min, the clock-probe spread and `NOISY` above 5%
(`--noise`). Both modes accept `--md FILE` and `--json FILE` to save the table.

### Interleaved A/B of two refs (`ab.jl`)

`benchmark/ab.jl` produces the multi-round evidence of § Evidence Protocol in one command. It checks out `--a` and `--b`
into two detached git worktrees in a temporary directory (removed afterwards unless `--keep-worktrees`), overlays the
benchmark harness of the invoking checkout (`run.jl`, `lib/`, `cases/`, `Project.toml`, `Manifest.toml`; local
`reference/` linked) onto both, so the two sides differ only in the package source, precompiles each side once and then
runs `run.jl` in separate processes in the order A1 B1 A2 B2 A3 B3 … (`--rounds`, at least 3) with the given `--threads`,
`--backend`, `--eltype`, `--suite`, `--filter`, `--entries` (both required) and `--run-args`. Output in `--out-dir`
(default `benchmark/results/logs/ab/<label>/`, gitignored; copy a summary you want to commit): the per-round run files `<label>-r<i>-<A|B>.json` with
their logs, `<label>-summary.md`/`.json` = `compare.jl --base A1,A2,… --cand B1,B2,…` (verdicts that require the
difference in every round or MWU p < 0.01), `<label>-spread-A.md`/`-B.md` = `compare.jl --spread` per side (the noise of
each side within the session), and `<label>-manifest.json` with the refs, commits, command lines and the harness
identity (commit, `git status --porcelain -- benchmark`, SHA-256 of the copied harness files). Completed round files are
reused, so an interrupted A/B resumes with the same `--label`/`--out-dir`. A verdict from a single `compare.jl` pair is a
lower bound on the noise (REPORT.md § 4); use the `ab.jl` summary for claims.

**Phase 1 evidence rule** (REPORT.md § 10.2; the 5% noise bound is session-dependent): an entry of a Phase 1 A/B counts
as evidence only if its A-side (merge-base) spread over the A/B's own rounds is ≤ 5% and no round logged a `NOT settled`
clock reference. Otherwise re-run in a steadier session or report the entry as UNAVAILABLE. `ab.jl` appends an
"Evidence check" section to `<label>-summary.md` (and `evidence_check` to the manifest) that lists every entry with
"A-side spread > 5%: not evidence" and every round whose log has a not-settled reference (`<label>-spread-A.json` holds
the per-entry A-side spreads).

## Case sizing and deviations from the case matrix

Sizes were chosen from a sizing probe at 1 thread (numbers in `REPORT.md`):

* **IMDP dense** n ∈ {100, 1000, 4000} × a ∈ {1, 4}; **IMDP sparse** n ∈ {10⁴, 10⁵} ×
  nnz/column ∈ {10, 100} × a ∈ {1, 4}: as specified.
* **fIMDP**: O-max on dense marginals (2 vars × 10 and × 50 values, 3 vars × 10) and sparse marginals
  (2 × 50 with 10 non-zeros, 3 × 20 with 5), a ∈ {1, 4}. 3 vars × 20 dense takes 1.4 s per
  `bellman!` call at 1 thread and is left out. McCormick: 2 × 10 dense (a = 1), 2 × 10 with 4 non-zeros
  (a ∈ {1, 4}), 3 × 10 with 3 non-zeros (`bellman!` only, 1.3 s per call). Vertex enumeration: small
  supports only (2 × 10 with 4 non-zeros, a ∈ {1, 4}; 3 × 10 with 3 non-zeros, `bellman!` only);
  on a dense 10-value marginal one call did not finish in 10 minutes.
* **Product IMDP × DFA**: dense n = 1000, sparse n ∈ {1000, 10⁴} (10 non-zeros), a ∈ {1, 4}, 4-state DFA.
  Dense n = 10⁴ is left out: the dense IMDP alone needs 1.6 GB (a = 1) / 6.4 GB (a = 4), which does
  not fit next to the desktop workload in the 15 GB of this host.
* **Real model**: `test/data/multiObj_robotIMDP.{sta,tra,lab,pctl}` (read with `read_prism_file`;
  the `.nc` copy is rewritten by `Pkg.test()` and is therefore not used).
* **Control synthesis**: dense n = 1000 and sparse n = 10⁴ (100 non-zeros), 4 actions.
* **CUDA**: IMDP dense/sparse in Float64 and Float32, fIMDP O-max, control synthesis and the real model;
  product processes and McCormick/vertex enumeration have no CUDA implementation in the package.
