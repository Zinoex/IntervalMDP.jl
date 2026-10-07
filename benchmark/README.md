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
* budgets: `workspace` 3 s / ≥ 50 samples, `bellman` 4 s / ≥ 40 samples, solves 4 s / ≥ 5 samples (the first two were raised from 1 s / 10 and 2 s / 20 by the Phase 0 noise fix, see `REPORT.md`);
* `evals = ceil(50 µs / t)` for calls shorter than 50 µs, else 1;
* `gctrial = true`, `gcsample = false`; `seconds` is set high enough never to be the binding limit;
* **clock guard** (`lib/clock.jl`): the run starts by taking the median of 20 clock probes (the prevailing clock state) (a fixed chain of
  dependent integer multiply–adds, ≈0.5 ms) as reference. Every trial is bracketed by two probes (each after 30 ms of
  spinning to leave idle states); if the probe *before* the trial deviates by more than 5% from the reference the
  trial is repeated after a 20 s pause (up to `--clock-retries` times, default 3; only for
  entries with ≤ 30 s of sampling) and the attempt closest to the reference is kept. Each result records
  `clock_probe_before_ns`, `clock_probe_after_ns`, `clock_probe_ratio`, `clock_state` (`nominal`/`deviating`, from the
  before-probe), `clock_after_deviation` (informational: after long memory-heavy trials the core is often ≈28%
  slower, the package power being shared with the memory traffic) and `clock_attempts`. This was added because the reference laptop switches between ≈5 GHz and a firmware cap
  of ≈2.3 GHz for minutes at a time (see `REPORT.md`), which moves compute-bound timings by ≈2.1×.

`compare.jl` reports a difference as `CLOCK (not evidence)` instead of speedup/regression when the clock probes
of the two sides differ by more than 5% or an entry was measured with a deviating clock.

With these parameters the full CPU suite takes ≈ 45 min at 1 thread and ≈ 15–25 min at 4–16 threads
on the reference host (dominated by the `imdp-sparse-n100000-nnz100-a4` and `imdp-dense-n4000-a4` solves,
which get 3–5 samples).

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
