# Performance of the VI/Bellman algorithms — Phase 0 report

Spec: `harness/specs/perf-vi-bellman/` (Phase 0, sub-phases `0a`–`0h`). Base ref: **`20fc03b`** (`main` = `origin/main`).
The measurements were taken in a checkout of `91bc0c8`, whose `src/` and `ext/` trees are identical to those of
`20fc03b` (`git rev-parse HEAD:src` = `290f675a…`, `HEAD:ext` = `5a4cf216…`; every result file records both
hashes, the `base_ref` and `src_ext_equal_to_base_ref = true`). `src/` and `ext/` were not modified.

How to run and read the suite: [`README.md`](README.md). Raw data: `results/*.json`. Profiles: `profiles/*.md`.

## 1. Environment

| Item | Value |
|---|---|
| CPU | Intel Core Ultra 7 255H, 16 logical CPUs, no SMT: CPUs 0–5 P-cores (max 5.1 GHz), 6–13 E-cores (4.4 GHz), 14–15 low-power E-cores (2.5 GHz); 24 MiB L3, one NUMA node |
| Memory | 15 GiB (≈ 4–5 GiB free during the runs; a desktop session with Firefox/VS Code was active) |
| OS / governor | Linux 7.1.8 (Fedora 44); `intel_pstate`, governor `powersave`, EPP `balance_performance`, ACPI platform profile `balanced`, turbo on. No root access, so the governor could not be set to `performance` |
| Power | AC online, battery reported `Not charging` throughout |
| Julia | 1.13.1 (`-O2`, bounds checks default); 1 interactive thread is started by Julia ≥ 1.12 for `--threads=N>1` — the main thread lives in that pool and runs the serial parts |
| Pinning | ThreadPinning.jl: default-pool thread *i* → CPU *i−1*, interactive/main thread → CPU 0 (recorded per run as `thread_to_cpu` with core types). t ≤ 6: P-cores only; t = 8: 6 P + 2 E; t = 14: all P + E; t = 16: + 2 LP-E |
| BLAS | OpenBLAS (recorded per run); `dot` on dense columns is a BLAS level-1 call, below OpenBLAS' threading threshold |
| GPU | NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU. `harness/tools/harness/gpu_check.py probe` → `GPU_PROBE=FUNCTIONAL device=NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU` (exit 0). `nvidia-smi` still fails with "Failed to initialize NVML: Driver/library version mismatch (NVML library version: 615.71)", so CUDA.jl works but NVML (clock/power queries) does not. CUDA runtime/driver versions and the `CUDA.versioninfo()` digest are in `results/baseline-20fc03b-cuda*.json` |

### 1.1 The clock is not stable on this host (main noise source)

The first full baseline pass (without a clock guard) showed **bimodal** run-to-run differences: 69 of 131 entries at
1 thread differed by more than 5% between two runs, almost all by a factor ≈ 2.1 (e.g. `imdp-sparse-n10000-nnz10-a1/bellman`
2.89 ms vs 6.21 ms) while the memory-latency-bound dense cases did not move (0.1%). `scaling_cur_freq` showed CPU 0 pinned at
2.3 GHz under full single-thread load for hours, with no thermal reason (package 45 °C, PL1 62 W). The P-cores switch, for
minutes to hours at a time, between ≈ 5.1 GHz and a firmware cap of ≈ 2.3 GHz (likely power-adapter related; "Not charging").
A dependent integer chain (`lib/clock.jl`) measures it directly: ≈ 197 µs at 5.1 GHz, ≈ 436 µs at the 2.3 GHz cap, and
≈ 557 µs (≈ 1.8 GHz) when most cores are busy (t ≥ 12).

Consequences and mitigation:

* The first pass was discarded (summary kept here: t=1 spread median 7.5%, 90th percentile 106%, 69/131 > 5%).
* Every entry is now bracketed by clock probes; entries whose pre-trial probe deviates by > 5% from the run's reference are
  re-measured after a pause (up to 3×) and otherwise flagged `clock_state = "deviating"`. `compare.jl` refuses to call a
  difference a speedup/regression when the clock probes of the two sides differ (`CLOCK (not evidence)`).
* **The baseline was measured almost entirely in the capped state** (probe ≈ 436 µs at t ≤ 8, ≈ 557 µs at t = 16, see
  `clock_probe_before_ns`). Absolute numbers are therefore ≈ 2× slower than the CPU can do uncapped for compute-bound cases.
  The relative comparisons that matter for Phases 1–4 (interleaved A/B in the same clock state) are not affected, but
  **the operator should fix the power situation** (adapter, `performance` platform profile) before Phase 1, or
  candidates and baseline must be re-measured interleaved in the same state.
* At t = 16 almost every entry is flagged "deviating" because the reference probe is taken before the worker threads
  load the package (436 → 557 µs). These t = 16 results are internally consistent (both runs at 557 µs) but must not be
  compared with lower thread counts at face value.

## 2. Suite

Case matrix (stable names, seeded `StableRNG`), timed entries, timing protocol, output format and correctness check:
see `README.md`. `run.jl --list` prints all 53 registered cases: `full` 40 cases (131 CPU entries per run), `quick` 16 cases
(smoke test at 1 and 4 threads), 156 CUDA entries (Float64 + Float32), 4 strong-scaling and 13
size-scaling cases. Deviations from the spec's matrix (memory, run time, missing CUDA implementations) are listed in
`README.md` → "Case sizing and deviations".

Reference values: `reference/cpu-Float64/` (144 entries) and `reference/cuda-Float64/`, `reference/cuda-Float32/`
(156 entries), written from the base ref with `--write-reference --reference-only`. Every baseline and re-run entry passed
its check (`valid = true`, 0 invalid in every file). Size of `reference/`: ≈ 16 MB (vectors of length > 2¹⁷ are stored as a
stride subsample; only the n = 10⁶ size case).

## 3. Baseline files

| File | Content |
|---|---|
| `results/baseline-20fc03b-cpu-t{1,4,8,16}.json` | full suite, CPU, Float64 |
| `results/baseline-20fc03b-cpu-t{1,4,8,16}-rerun1.json` | separate-process re-run (noise bound) |
| `results/noisefix/*.json` | re-measurement of the noisy entries with the raised budgets (two rounds `a`, `b`) |
| `results/baseline-20fc03b-cuda.json`, `-cuda-rerun{1,2}.json` | CUDA, Float64 and Float32 |
| `results/baseline-20fc03b-cuda-run0-unchecked.json` | first CUDA run, made before the CUDA reference existed (correctness "no-reference"; timing only; contains the B-3 hang) |
| `results/reference-run-20fc03b-{cpu-t1,cuda}.json` | the runs that wrote the reference values |
| `results/scaling-20fc03b-cpu-t{1,2,4,6,8,10,12,14,16}.json` | strong scaling |
| `results/sizes-20fc03b-cpu-t{1,6,16}.json` | size scaling |
| `results/stream-t*.json` | machine roofs |
| `results/scaling-roofline-20fc03b.md` | tables generated by `analyze.jl` |

Selected medians (full table: generate with `analyze.jl`/`compare.jl`; † = clock deviating):

| case | entry | t=1 | t=4 | t=8 | t=16 | allocs/call (t=1) |
|---|---|---:|---:|---:|---:|---:|
| imdp-dense-n100-a4 | bellman | 138.1 µs | 33.2 µs† | 19.4 µs† | 36.2 µs† | 0 |
| imdp-dense-n4000-a4 | bellman | 312.53 ms | 93.68 ms | 66.68 ms | 41.18 ms† | 0 |
| imdp-sparse-n10000-nnz10-a4 | bellman | 25.22 ms | 8.30 ms | 2.29 ms | 3.16 ms† | 80 000 |
| imdp-sparse-n100000-nnz100-a1 | bellman | 432.05 ms | 112.19 ms | 81.63 ms | 49.57 ms | 200 000 |
| fimdp-dense-v2-d50-a1-omax | bellman | 151.41 ms | 44.34 ms | 31.99 ms | 19.64 ms† | 255 000 |
| fimdp-sparse-v2-d10-k4-a1-mccormick | bellman | 95.45 ms | 28.27 ms | 19.96 ms | 18.01 ms† | 411 000 |
| fimdp-sparse-v3-d10-k3-a1-vertex | bellman | 314.36 ms | 83.58 ms | 89.48 ms | 66.90 ms† | 223 609 |
| product-imdp-sparse-n10000-nnz10-a4-dfa4 | bellman | 88.89 ms | 31.15 ms | 19.16 ms | 10.41 ms† | 320 000 |
| real-multiObj_robotIMDP | bellman | 304.9 µs | 198.4 µs | 46.1 µs | 61.4 µs† | 1 656 |
| imdp-sparse-n100000-nnz100-a4 | solve_rvi (67 it.) | 109.1 s | – | – | 12.7 s | 53.6 M |

**VI-loop overhead** (t = 1, solve median / (iterations × `bellman!` median × calls per iteration)): RVI 0.94–1.03 for all
O-max IMDP/fIMDP cases, 1.06–1.12 for McCormick/vertex enumeration (GC from their allocations); IVI 0.56–0.64 (its second
`bellman!` evaluates only the fixed action, so a = 4 gives the expected (1 + ¼)/2); product 0.73–0.77 (the accepting DFA
state is skipped). The VI loop itself (residual, copies, termination check, strategy cache) costs ≤ 3% at 1 thread.

## 4. Noise

Spread = (max − min)/min of the per-run medians, separate processes, `compare.jl --spread`.

| threads | entries | spread > 5% (before fix) | median spread | 90th pct |
|---:|---:|---:|---:|---:|
| 1 | 131 | 9 | 0.82% | 3.5% |
| 4 | 131 | 22 | 1.59% | 8.7% |
| 8 | 131 | 36 | 1.48% | 65.0% |
| 16 | 131 | 45 | 2.06% | 17.2% |
| CUDA | 128 | 26 | 0.92% | 28.8% |

(First pass without clock guard, for comparison: t=1 69/131 > 5%, median 7.5%.)

Causes of the remaining > 5% entries (full list with values: `results/noise-before-fix.md`):

1. **Clock-state changes** (clock-probe spread ≈ 28% or 184%, or `clock_state = deviating`): most of the t = 8/16 entries.
2. **`workspace` entries** (µs-scale allocation of fresh arrays; first-touch page faults vary between processes).
3. **Threaded small/medium kernels at t = 8/16** with equal clock probes (e.g. `imdp-sparse-n10000-nnz10-a4/bellman` 118%
   at t = 8, `imdp-dense-n100-a1/bellman` 85% at t = 8): static equal chunking makes the slowest pinned core (E/LP-E, or a core
   shared with desktop work) determine the time, and that differs from process to process.
4. **CUDA**: `workspace` entries take 2–80 ns (timer resolution), and host-synchronisation quantisation (§ 6.3) makes
   short solves bimodal.

Fix applied: budgets for `workspace` (1 s/10 → 3 s/50 samples) and `bellman` (2 s/20 → 4 s/40) were raised in
`cases/registry.jl`, solves were sampled with `--budget-scale 2`, and every noisy entry was re-measured twice in separate
processes with the clock guard (`results/noisefix/`). Result:

<!--NOISEFIX-->

## 5. Profiles (`profiles/*.md`)

At least one per case-matrix row; each has a CPU sampling profile (self-time and inclusive top frames, pruned tree),
a `Profile.Allocs` summary, the steady-state allocation of one call and `JET.@report_opt` (target IntervalMDP).

| Profile | Where the time goes (share of samples, self time) | Steady-state `bellman!` allocation |
|---|---|---|
| `imdp-dense-n1000-a4` / `imdp-dense-n4000-a4` (t=1) | gap walk `gap_value` 79–81% (`min(budget, gap[i])` 42%, float add/mul 37%), `dot(V, lower)` (BLAS) 17–19% | 0 B ✓ |
| `imdp-dense-n4000-a4` (t=16) | idle/wait 59% (main thread waiting + workers waiting on the slowest chunk), gap walk 31% | 7.2 KB, 82 allocs (task spawn of `@threadstid`) |
| `imdp-dense-n100-a4` (t=16) | idle/wait 87%, `threading_run` scheduling | 7.5 KB, 82 allocs |
| `imdp-sparse-n10000-nnz100-a4`, `-n100000-nnz100-a1` (t=1) | building the (V, gap) tuples (`setindex!`, gather of `V[support]`) 25%, `sort!` 10–12%, dynamic `_by` ordering 8% | 64 B per column (2 allocs: `SubArray` 48 B + kw `NamedTuple` 16 B) — 6.4 MB per call for 10⁵ columns |
| `real-multiObj_robotIMDP`, `product-imdp-sparse-n10000-nnz10-a4-dfa4` (small supports) | **dynamic `Base.Order._by(by::Function, …)` 50–69%**, `sort!` 13–17% | 64 B per column |
| `fimdp-dense-v2-d50-a1-omax`, `fimdp-sparse-v3-d20-k5-a1-omax` | `_by` 21% / 58%, `sort!` 11–13%, tuple building 7–13% | 8.2 MB / 15.9 MB per call (32 B/alloc) |
| `fimdp-sparse-v2-d10-k4-a1-mccormick` | HiGHS 80% (`Highs_run`, `Highs_create`/`destroy`, `addRow`): one JuMP model rebuilt per state–action | 21.6 MB per call, 411 000 allocs |
| `fimdp-sparse-v2-d10-k4-a1-vertex`, `fimdp-sparse-v3-d10-k3-a1-vertex` | products in the vertex sum (`promotion.jl`) 36–38%, vertex iterator | 1.1 MB / 7.7 MB per call |
| `cs-imdp-dense-n1000-a4`, `cs-imdp-sparse-n10000-nnz100-a4` | same as the IMDP kernels; strategy extraction not visible (< 3%) | as IMDP |
| CUDA (`cuda-*.md`) | `CUDA.@profile` traces, see § 6.3 | n/a |

JET (`@report_opt`, `target_modules = (IntervalMDP,)`) reports **0** problems for every hot function except 1 report in
the fIMDP solve. This filter hides the main type instability: the runtime dispatch happens inside `Base.sort!` because
IntervalMDP passes `rev = upper_bound` (a runtime `Bool`) and `by = first` as keywords, so `Base.Order.ord` returns an
ordering whose type depends on a runtime value. `Profile.Allocs` and the self-time profile show it (H1).

## 6. Scaling and roofline

Full tables: `results/scaling-roofline-20fc03b.md` (`analyze.jl`). Measured roofs (`stream.jl`, capped clock): DRAM read
12.5 GB/s at 1 thread, 43–48 GB/s at 8–16 threads; in-L1 FMA ≈ 13 GFLOP/s per P-core at the 2.3 GHz cap.
Theoretical: 16 DP flop/cycle per P-core (81.6 GFLOP/s at 5.1 GHz), LPDDR5X ≈ 134 GB/s platform maximum.

### 6.1 Strong scaling (`bellman!`, fixed size)

| case | t=1 | t=4 | t=6 | t=8 | t=14 | t=16 |
|---|---:|---:|---:|---:|---:|---:|
| imdp-dense-n4000-a1 | 77.97 ms | ×3.05 | ×4.56 | ×4.51 | ×10.05 | ×7.99 |
| imdp-dense-n4000-a4 | 312.66 ms | ×3.38 | ×4.55 | ×5.74 | ×7.63 | ×7.45 |
| imdp-sparse-n100000-nnz10-a1 | 63.21 ms | ×3.36 | ×5.43 | ×4.94 | ×17.6* | ×6.46 |
| imdp-sparse-n100000-nnz100-a1 | 431.63 ms | ×3.84 | ×5.70 | ×5.20 | ×9.06 | ×9.28 |

Efficiency stays 76–96% on the P-cores (t ≤ 6), drops when the first E-cores join (t = 8: the 6→8 step gains nothing
or loses), and **t = 16 is slower than t = 14 for three of four cases** — the two LP-E cores (2.5 GHz) get an equal static
chunk and finish last. (*t = 12/14 runs of the sparse k = 10 case were measured in a different clock state; see the clock
columns in the files.)

### 6.2 Size scaling and roofline (model: 16 B/nnz dense, 20 B/nnz sparse; 4 flop/nnz)

* Dense O-max at 1 thread: 3.3 ns/nnz up to n = 1000 (data ≤ 15 MB, L3-resident), 4.6–4.9 ns/nnz from n = 2000; 3.3–4.6 GB/s
  = 26–37% of the 1-thread DRAM roof and ≈ 8% of L2 for the cached sizes → **latency/compute-bound** (dependent gap walk
  with a gather through the permutation). At t = 16 the large dense cases reach 25–26 GB/s = 57–61% of the 16-thread DRAM
  roof → **memory-bound** there.
* Sparse O-max: 54–65 ns/nnz (k = 10) and 40 ns/nnz (k = 100), 0.3–0.5 GB/s (1–4% of any roof) at 1 thread and ≤ 4.8 GB/s at
  16 threads → **compute/overhead-bound** (dynamic dispatch, allocation, sorting), not memory-bound at any size or thread count.
* Small dense (n = 100) and everything at t = 16 with < 100 µs per call: **overhead-bound** (87% wait in the profile).
* Arithmetic intensity 0.2–0.25 flop/B: no case can become FLOP-bound.

### 6.3 CUDA

`results/baseline-20fc03b-cuda.json` (Float64 and Float32; dense/sparse IMDP, dense-marginal fIMDP O-max, control synthesis,
real model; IVI and sparse-marginal fIMDP excluded, Findings B-2/B-3; product processes and McCormick/vertex enumeration
have no CUDA implementation).

* Large cases gain a lot: `imdp-sparse-n100000-nnz100-a4` `bellman!` 27.3 ms (F64) / 16.3 ms (F32) vs 1.74 s at 1 CPU thread
  and 193 ms at 16; `solve_rvi` 1.88 s vs 109 s / 12.7 s.
* **A ≈ 6 ms floor**: most `bellman!` medians are 5.9–6.1 ms independent of size (dense n = 1000 a = 1: median 5.94 ms, min
  0.29 ms; sparse n = 10⁴ k = 10: 6.04 ms vs min 0.56 ms) and most solves cost ≈ 6 ms per iteration (dense n = 1000 a = 1:
  470 ms / 81 iterations, kernel min 0.29 ms). Occasionally the same case runs at kernel speed (Float32 dense n = 1000 a = 1:
  7.3 ms for 82 iterations). The time is spent waiting in host synchronisation (`CUDA.synchronize` in the timed call, and the
  device→host transfer of the residual in every VI iteration), not in the kernels.

<!--CUDAPROF-->

## 7. Ranked hypotheses (by expected gain)

Each entry: target cases — measurement — proposed change — predicted effect — phase. None of these is implemented; each
must be accepted or rejected by the evidence protocol.

1. **Type-stable ordering in the sparse and factored O-max sort (H1).** *Cases:* all sparse IMDP, product (sparse), real
   model, all fIMDP O-max. *Measurement:* dynamic `Base.Order._by(by::Function, …)` is 67% of samples for the real model,
   50% for `product-…-n10000-nnz10-a4`, 58% for `fimdp-sparse-v3-d20-k5`, 8% at k = 100; every column allocates exactly
   2 objects / 64 B (6.4 MB per call at 10⁵ columns; 0.2–3.7 GB per solve). *Change:* build the ordering once per call with a
   concrete type (dispatch on `upper_bound` via `Val`/two methods, `Base.Order.By(first)` / `ReverseOrdering`), call the
   positional `sort!(v, alg, order)` form with the preallocated scratch, no keyword `NamedTuple`, no per-column `SubArray`
   allocation. Same stable ordering → identical results. *Prediction:* 2–3× on k ≤ 10 cases and the real model, 1.1–1.3× at
   k = 100, zero steady-state allocations. *Phase 1* (IMDP/product), shared with *Phase 4* (factored `orthogonal_inner_bellman!`).
2. **Use the LP-E/E-cores proportionally (H2).** *Cases:* every threaded case at t ≥ 8. *Measurement:* t = 16 slower than
   t = 14 for 3 of 4 strong-scaling cases (dense n4000 a1 9.75 vs 7.75 ms; sparse k10 9.78 vs 3.60 ms); efficiency 47–58%
   at t = 16 vs 76–96% at t ≤ 6; 59–78% idle samples in the t = 16 profiles. *Change:* replace the static equal chunking of
   `@threadstid` by dynamic chunked scheduling (e.g. `:greedy` or an atomic chunk counter, chunk ≈ 64–256 states) as a new
   threaded workspace/`_bellman_helper!` method. *Prediction:* t = 16 ≥ t = 14 throughput, 1.2–2.5× at t = 16, no change at
   t ≤ 6. *Phase 2.*
3. **Rebuild-free McCormick LP (H3).** *Case:* `fimdp-*-mccormick`. *Measurement:* HiGHS 80% of samples incl.
   `Highs_create`/`destroy`/`addRow`; 411 000 allocations and 21.6 MB per `bellman!`. *Change:* build the JuMP/HiGHS model
   once per support pattern and only update bounds/objective coefficients (`set_normalized_coefficient`, `set_lower_bound`)
   between state–action pairs. Same LP → same values. *Prediction:* 3–5×. *Phase 4.*
4. **Vertex enumeration over the support only (H4).** *Cases:* `fimdp-*-vertex`. *Measurement:* 36–38% of samples in the
   products of the vertex sum; the sum runs over the full dense target product (`CartesianIndices(num_target.(…))`) although
   only the support carries mass — 1000 target tuples vs 27 support tuples for v3-d10-k3. *Change:* iterate the support
   product. Zero terms are dropped, so only the summation order changes (within tolerance). *Prediction:* 5–30× on sparse
   marginals, unchanged on dense ones. *Phase 4.*
5. **CUDA host synchronisation (H5).** *Cases:* all CUDA solves and `bellman!`. *Measurement:* ≈ 6 ms floor per
   synchronised call independent of size (median ≈ 6 ms, min 0.3–1 ms); solves ≈ 6 ms × iterations. *Change:* avoid a blocking
   host round-trip per VI iteration (compute the termination check on the device and transfer every k iterations, or
   use a spin-wait/stream-ordered check); investigate the CUDA.jl synchronisation mode. *Prediction:* 5–20× for solves of
   models whose kernel takes < 1 ms. *Phase 3.*
6. **Dense gap walk (H6).** *Cases:* dense IMDP, product (dense), control synthesis (dense). *Measurement:* 79–81% of
   samples in `gap_value` (42% at `min(budget, gap[i])`), 3.3–4.9 ns/nnz, only 26–37% of the 1-thread DRAM roof.
   *Change:* make the gather cheaper: e.g. walk the shared permutation on a gap column that was just streamed into cache by the
   `dot` pass (fuse `dot` and a prefetching pass), `Int32` permutation already used — try SIMD-friendly blocking of the budget
   accumulation (prefix sums per block, then a scalar finish in the block where the budget runs out). *Prediction:*
   1.3–2× at 1 thread for n ≥ 1000; smaller at t = 16 where it is memory-bound. *Phase 1.*
7. **Sparse gather/tuple construction (H7).** *Cases:* sparse k = 100. *Measurement:* 25% of samples in `setindex!` building
   `(V[j], gap)` tuples. *Change:* sort indices by a precomputed global rank of `V` (one `sortperm` per call, as the dense
   path does) instead of copying values into tuples; partial selection up to the budget. *Prediction:* 1.2–1.5× for k = 100.
   *Phase 1.*
8. **Threaded-workspace threshold (H8).** *Cases:* small models (n = 100, real model). *Measurement:* `imdp-dense-n100-a4`
   87% idle samples at t = 16 and 36 µs vs 19 µs at t = 8; the threaded path allocates 82 objects/7 KB per call.
   *Change:* select the threaded workspace by work (columns × support) instead of `num_sets > 10`, with the constant set from
   a measured sweep. *Prediction:* removes the 1.5–2× slowdown of small models at high thread counts. *Phase 2.*
9. **fIMDP O-max tuple/expectation overhead (H9).** *Cases:* `fimdp-*-omax`. *Measurement:* 255 000 allocations (8.2 MB)
   per call for v2-d50, `setindex!` 7–13%. *Change:* after H1, remove the remaining per-state temporaries (`ssize[2:end]`,
   `support.(…)` tuples). *Prediction:* 1.2–1.5× on top of H1. *Phase 4.*
10. **VI loop (H10, low).** *Measurement:* loop overhead ≤ 3% (§ 3). No action unless a later phase makes `bellman!` ≥ 10×
    faster.

## 8. Findings

* **B-1 — stationary strategy cache keeps the previous action only for states whose index ≤ number of actions.**
  `extract_strategy!(::StationaryStrategyCache, …)` (`src/strategy_cache.jl`) tests `jₛ ∉ available_actions`, i.e. the
  *state* index against the *action* `CartesianIndices`, instead of the cached action. For every state with index > number
  of actions the cache is reset each iteration and ties are broken towards the lowest action index. Reproduction (6 states,
  4 actions; states 3 and 6 identical: action 2 goes to the goal, actions 1/3/4 self-loop; `InfiniteTimeReachability([1])`,
  Pessimistic/Maximize): synthesised strategy `[(1,), (1,), (2,), (1,), (1,), (1,)]` — state 3 keeps the goal action, state 6
  switches to a self-loop. Policy evaluation of the synthesised strategy gives value 0 for state 6 although the reported
  value is 1 (the returned stationary strategy is not optimal). Not fixed (Phase 0 may not touch `src/`); needs its own commit
  with a regression test.
* **B-2 — `IntervalValueIteration` fails on CUDA models.** `max_initial_gap(V_lower, V_upper, ::AllStates)`
  (`src/interval_value_iteration.jl`) indexes the CuArrays element by element → "Scalar indexing is disallowed". The CUDA
  suite records `solve_ivi` as `unsupported`.
* **B-3 — CUDA factored O-max returns zeros for sparse marginals.** For a 2-variable fIMDP with 3 non-zeros per marginal
  column, `bellman` on CUDA returns all zeros (max |CPU − CUDA| = 0.68; dense marginals agree to 2e-16), so RVI stops after
  1 iteration (CPU: 667). In the first CUDA run one `solve` sample of `fimdp-sparse-v3-d20-k5-a4-omax` (Float64) also took
  3 287 s. Sparse-marginal fIMDP cases are excluded from the CUDA suite.
* **Environment (not a code bug):** the clock cap of § 1.1; JET's `target_modules` filter hides the dynamic dispatch inside
  `Base.sort!` (§ 5).
* No `I-`, `S-` or `T-` findings in Phase 0.

## 9. Limitations

* Optimised code may differ from the baseline in the last floating-point bits (different summation/reduction order, e.g.
  SIMD blocking or dropped zero terms); the correctness check allows the spec's tolerances for this. The Lean proofs are at
  abstract scope (exact arithmetic) and are not affected; they say nothing about Julia floating point or GPU execution.
* Absolute times are for a power-capped laptop (§ 1.1); thread placement is fixed by compact pinning, so t = 8/16 always
  include E/LP-E cores.
* Roofline traffic is a model (upper bound for the dense gap walk), not a hardware-counter measurement (no `perf` access).
* CUDA: no Nsight; `CUDA.@profile` traces only; NVML unavailable, so GPU clocks/power were not recorded.
