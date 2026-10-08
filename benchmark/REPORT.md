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

Reference values (local only, gitignored): `reference/cpu-Float64/` holds 106 outcomes (93 of the 131 `full` CPU entries
plus 13 size-scaling entries; the other 38 `full` entries are `workspace` timings without an outcome, `not-applicable`),
written from the base ref sources (`src`/`ext` equal to `20fc03b`) with `run.jl --suite full,sizes --write-reference
--reference-only` (record: `results/reference-run-20fc03b-cpu-t1.json`). `--write-reference` refuses to run (exit 3) when
`src/` or `ext/` differ from the base ref, committed or uncommitted. Check of the stored reference on 2026-10-07 (`run.jl
--suite full --reference-only`, 1 thread): 131 entries, 93 pass with ‖ΔV‖∞ = 0 and identical iteration counts and
strategies, 38 not-applicable, 0 invalid. `--suite quick` at 1 thread (`results/quick-0b-cpu-t1.json`): 54 entries, all
`valid = true` (39 pass, 15 not-applicable). A reference value perturbed by 1e-9 was reported `fail` / `valid = false`
(‖ΔV‖∞ = 1.0e-9 > 1e-12; demonstration only, reference restored). `reference/cuda-Float64/`, `reference/cuda-Float32/`
(156 entries) belong to § 6.3. Size: `reference/cpu-Float64/` ≈ 16 MB, `reference/` ≈ 31 MB (vectors of length > 2¹⁷ are stored as a stride subsample;
only the n = 10⁶ size case).

## 3. Baseline files

| File | Content |
|---|---|
| `results/baseline-20fc03b-cpu-t{1,4,8,16}.json` | full suite, CPU, Float64 |
| `results/baseline-20fc03b-cpu-t{1,4,8,16}-rerun1.json` | separate-process re-run (noise bound) |
| `benchmark/baseline.sh rounds` | the driver that produced the eight CPU files above (one Julia process per file, rounds interleaved over t = 1, 4, 8, 16) |
| `results/noisefix/*.json` | re-measurement of the noisy entries with the raised budgets (two rounds `a`, `b`) |
| `results/baseline-20fc03b-cuda.json`, `-cuda-rerun{1,2}.json` | CUDA, Float64 and Float32 |
| `results/baseline-20fc03b-cuda-run0-unchecked.json` | first CUDA run, made before the CUDA reference existed (correctness "no-reference"; timing only; contains the B-3 hang) |
| `results/reference-run-20fc03b-{cpu-t1,cuda}.json` | the runs that wrote the reference values |
| `results/scaling-20fc03b-cpu-t{1,2,4,6,8,10,12,14,16}.json` | strong scaling |
| `results/sizes-20fc03b-cpu-t{1,6,16}.json` | size scaling |
| `results/stream-t*.json` | machine roofs |
| `results/scaling-roofline-20fc03b.md` | tables generated by `analyze.jl` |

CPU baseline validation (all eight files): complete environment block; checked-out commit `91bc0c8` with
`src_ext_equal_to_base_ref = true` (src tree `290f675`, ext tree `5a4cf21`, identical to `20fc03b`) and no uncommitted
changes under `src/`/`ext/`; `--suite full`, no filter, budget scale 1.0, `Threads.nthreads()` = 1/4/8/16 as named, pinning
compact (ThreadPinning.jl), governor `powersave` with EPP `balance_performance` (intel_pstate; no `performance` governor
used); 131 entries each (the same case/entry set as the current `run.jl --suite full --list`), **0 invalid** (93 pass, 38
workspace entries not applicable). Wall time per file: t=1 4904/4870 s, t=4 2127/2360 s, t=8 5014/5321 s, t=16 9893/9915 s
(baseline/re-run; 2026-10-05 19:31 to 2026-10-06 07:53 UTC, each run started after the previous one finished).

Clock flags: at t=1 and t=4, ≤ 9 of 131 entries are clock-deviating; at t=8 54/59; at t=16 128/129 (baseline/re-run). The
t=16 deviation is systematic, not random: the probe taken before each trial is ≈1.28× slower than the run-start reference in
both runs (all-core load lowers the clock), so t=16 medians are internally consistent but marked †.

Selected medians (baseline file; full table: generate with `analyze.jl`/`compare.jl`; † = clock deviating):

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
| imdp-sparse-n100000-nnz100-a4 | solve_rvi (67 it.) | 109.09 s | 28.10 s | 21.12 s | 12.71 s† | 53.6 M |

Re-run vs baseline for these cells: within ±3% except imdp-sparse-n10000-nnz10-a4 (t=4 −16%, t=8 +118%, t=16 −17%),
fimdp-sparse-v3-d10-k3-a1-vertex (t=1 +8%, t=4 +6%, t=16 +121%), product-imdp-sparse-n10000-nnz10-a4-dfa4 (t=4 −6%,
t=8 −44%, t=16 +7%), fimdp-dense-v2-d50-a1-omax (t=16 +9%), real-multiObj_robotIMDP (t=8 −7%) and imdp-dense-n4000-a4
(t=16 +5%); the spread over all entries is analysed in § 4.

**Steady-state allocations per `bellman!`** (pre-built workspace and strategy cache, BenchmarkTools count / bytes per call,
t=1 baseline): zero only for the dense IMDP (6 cases) and dense product (2 cases) entries; 30 of the 38 `bellman!` entries
allocate. Sparse IMDP and sparse product: 2 allocations (32 B each) per state–action column, independent of nnz/column
(e.g. imdp-sparse-n100000-*-a4: 800 000 / 25.6 MB; product-imdp-sparse-n10000-nnz10-a4-dfa4: 320 000 / 10.2 MB). fIMDP
O-max: 2 × (1 + Σ inner marginal sizes) per column (v2-d10: 22, v2-d50: 102, v3-d10: 222 per column; e.g.
fimdp-dense-v2-d50-a4-omax 1 020 000 / 32.6 MB). McCormick allocates most (fimdp-sparse-v3-d10-k3-a1-mccormick 8 609 000 /
471 MB per call), vertex enumeration less (fimdp-sparse-v3-d10-k3-a1-vertex 223 609 / 7.7 MB). Real model: 1 656 / 53 KB.
Threaded calls add a constant per call on top (t=16: +82 allocations / ≈7–13 KB for IMDP/fIMDP, +328 / ≈30–35 KB for
product), so dense IMDP is not allocation-free at t > 1.

**VI-loop overhead** (t = 1, solve median / (iterations × `bellman!` median × `bellman!` calls per iteration), baseline
[re-run]): RVI 0.94–1.03 [0.95–1.03] for all O-max IMDP/fIMDP cases, 1.03–1.12 [1.03–1.07] for McCormick/vertex
enumeration (GC from their allocations); IVI (2 calls per iteration) 0.89–1.01 [0.90–1.02] at a = 1 and 0.56–0.64
[0.56–0.65] at a = 4 (its second `bellman!` evaluates only the fixed action, so a = 4 gives the expected (1 + ¼)/2);
product DFA reachability 0.73–0.77 [0.73–0.77] (the accepting DFA state is skipped); real model 0.98 [0.97] (RVI) and 0.98
[0.97] (stationary control synthesis). The VI loop itself (residual, copies, termination check, strategy cache) costs ≤ 3%
at 1 thread.

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

Causes of the remaining > 5% entries (every entry with its values, clock probes and cause code:
`results/noise-before-fix.md`, section "Causes of the > 5% entries and re-measurement"). The cause is assigned from the
`clock_probe_before_ns` of the entry in the two runs:

1. **CLK — clock-state change between the two runs** (probes differ by > 5%: ≈ 197 µs at 5.1 GHz, ≈ 436 µs in the 2.3 GHz
   cap, ≈ 557 µs under all-core load; § 1.1). Before the fix: t=1 0, t=4 4, t=8 16, t=16 3 entries. (At t = 16 both
   baseline runs sat at ≈ 557 µs, so the systematic `deviating` flag there is not counted as a cause.)
2. **WS — `workspace` entries** with equal clocks (µs-scale allocation of fresh arrays; first-touch page faults and GC
   state vary between processes): t=1 8, t=4 7, t=8 12, t=16 11.
3. **THR — threaded kernels/solves at t ≥ 4** with equal clock probes (e.g. `imdp-sparse-n10000-nnz10-a4/bellman` 118%
   at t = 8, `imdp-dense-n100-a1/bellman` 85% at t = 8): static equal chunking makes the slowest pinned core (E/LP-E, or a core
   shared with desktop work) determine the time, and that differs from process to process: t=4 11, t=8 8, t=16 31.
4. **ST — single-threaded compute entry** with equal clocks: t=1 1 (`fimdp-sparse-v3-d10-k3-a1-vertex/bellman`, 8.1%).
5. **CUDA**: `workspace` entries take 2–80 ns (timer resolution), and host-synchronisation quantisation (§ 6.3) makes
   short solves bimodal.

Fix applied: budgets for `workspace` (1 s/≥10 → 3 s/≥50 samples) and `bellman` (2 s/≥20 → 4 s/≥40) were raised in
`cases/registry.jl` (`E_ws`, `E_bellman`; committed with 0a, `333f674` — the baseline files of § 3 record the old budgets
in `parameters.default_budgets`), solves were sampled with `--budget-scale 2`, and every noisy entry was re-measured twice
in separate processes with the clock guard (`results/noisefix/`). Sizes were not changed. Result:

Re-measured: every entry that was > 5% at its thread count (112 entries in 19 a/b file pairs
`results/noisefix/noisefix-20fc03b-cpu-t<T>-<entry>-{a,b}.json`; 0 invalid; src/ext trees equal to `20fc03b`; AC power,
EPP `balance_performance`, compact pinning). Run a: 2026-10-06 13:03–13:41 UTC, run b: 13:41–14:24 UTC, one process
per file, no overlap. Spread = run a vs run b, `julia --project=benchmark benchmark/compare.jl --spread <…-a.json> <…-b.json>`.

| threads | > 5% before | ≤ 5% after | still > 5% | of which CLK | WS | THR | ST | median spread after (re-measured entries) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 2 | 7 | 1 | 6 | 0 | 0 | 11.3% |
| 4 | 22 | 5 | 17 | 17 | 0 | 0 | 0 | 17.7% |
| 8 | 36 | 15 | 21 | 18 | 3 | 0 | 0 | 5.4% |
| 16 | 45 | 18 | 27 | 13 | 2 | 12 | 0 | 6.5% |
| **total** | 112 | 40 | 72 | 49 | 11 | 12 | 0 | |

Reading: the larger budgets fixed 40 of 112 entries. Of the 72 that are still > 5%, **49 (CLK) are not sampling noise**:
the CPU switched between the 5.1 GHz state and the 2.3 GHz cap between run a and run b (§ 1.1), so their spread is the
clock ratio; more samples cannot remove it, only measuring both sides in the same clock state can. The **11 WS** entries
(sub-µs to ms allocations, ≥ 50 samples each) and the **12 THR** entries at t = 16 differ between processes with equal
clocks; their per-process medians are stable within a run, so more samples per run do not help either, and larger sizes
would change the suite (out of scope for 0d). A re-measurement of the CLK entries in a clock-stable session was not
possible in 0d: the host was on battery (`online` 0, EPP `balance_power`) when 0d ran (2026-10-07).

**Entries still > 5% — not usable as evidence from a single pair of runs.** Any later claim on them needs ≥ 3 interleaved
A/B rounds in one session with matching clock probes (`common.md` § Evidence Protocol; `compare.jl` marks clock mismatches
`CLOCK (not evidence)`), and the `Na` noise re-check of each phase must re-measure them on AC in one clock state first.

| threads | case | entry | before | after | medians a / b | clock probes a / b (µs) | cause |
|---:|---|---|---:|---:|---|---|---|
| 1 | fimdp-sparse-v2-d10-k4-a1-vertex | workspace | 6.9% | 32.6% | 161 ns / 214 ns | 200 / 197 | WS |
| 1 | fimdp-sparse-v3-d10-k3-a1-vertex | bellman | 8.1% | 13.7% | 195.03 ms / 171.53 ms | 557 / 197 | CLK |
| 1 | fimdp-sparse-v3-d10-k3-a1-vertex | workspace | 27.1% | 11.3% | 256 ns / 230 ns | 197 / 197 | WS |
| 1 | fimdp-sparse-v3-d20-k5-a1-omax | workspace | 22.1% | 81.1% | 197.1 µs / 108.8 µs | 197 / 197 | WS |
| 1 | imdp-dense-n100-a1 | workspace | 32.3% | 41.1% | 3.7 µs / 5.2 µs | 197 / 197 | WS |
| 1 | imdp-sparse-n10000-nnz10-a1 | workspace | 31.8% | 9.1% | 88.9 µs / 97.1 µs | 197 / 197 | WS |
| 1 | product-imdp-sparse-n1000-nnz10-a1-dfa4 | workspace | 12.8% | 7.0% | 15.4 µs / 14.4 µs | 203 / 197 | WS |
| 4 | fimdp-dense-v2-d10-a1-omax | workspace | 6.1% | 70.8% | 5.4 µs / 3.2 µs | 436 / 209 | CLK |
| 4 | fimdp-dense-v3-d10-a1-omax | bellman | 10.4% | 72.0% | 17.08 ms / 9.93 ms | 559 / 199 | CLK |
| 4 | fimdp-sparse-v2-d10-k4-a1-vertex | bellman | 42.7% | 44.8% | 2.55 ms / 1.76 ms | 436 / 197 | CLK |
| 4 | fimdp-sparse-v2-d10-k4-a1-vertex | workspace | 5.9% | 96.1% | 372 ns / 190 ns | 436 / 197 | CLK |
| 4 | fimdp-sparse-v2-d10-k4-a4-vertex | workspace | 20.5% | 38.0% | 382 ns / 277 ns | 436 / 197 | CLK |
| 4 | fimdp-sparse-v3-d10-k3-a1-vertex | workspace | 17.0% | 9.4% | 576 ns / 527 ns | 437 / 197 | CLK |
| 4 | imdp-dense-n100-a1 | bellman | 118.8% | 5.7% | 11.4 µs / 10.8 µs | 436 / 197 | CLK |
| 4 | imdp-dense-n100-a1 | workspace | 12.2% | 12.8% | 2.6 µs / 2.3 µs | 436 / 197 | CLK |
| 4 | imdp-sparse-n10000-nnz10-a4 | bellman | 18.4% | 13.9% | 6.78 ms / 5.95 ms | 559 / 197 | CLK |
| 4 | imdp-sparse-n10000-nnz100-a1 | bellman | 22.4% | 90.5% | 10.93 ms / 5.74 ms | 559 / 197 | CLK |
| 4 | imdp-sparse-n10000-nnz100-a1 | solve_ivi | 5.1% | 83.7% | 1.53 s / 831.66 ms | 436 / 197 | CLK |
| 4 | imdp-sparse-n100000-nnz100-a1 | workspace | 7.7% | 34.0% | 32.86 ms / 24.53 ms | 436 / 197 | CLK |
| 4 | imdp-sparse-n100000-nnz100-a4 | bellman | 8.7% | 16.9% | 441.63 ms / 377.63 ms | 436 / 197 | CLK |
| 4 | product-imdp-sparse-n1000-nnz10-a1-dfa4 | bellman | 7.1% | 18.4% | 2.01 ms / 1.70 ms | 436 / 197 | CLK |
| 4 | product-imdp-sparse-n10000-nnz10-a1-dfa4 | bellman | 8.4% | 65.2% | 12.89 ms / 7.80 ms | 436 / 197 | CLK |
| 4 | product-imdp-sparse-n10000-nnz10-a4-dfa4 | bellman | 5.9% | 58.4% | 31.87 ms / 20.11 ms | 436 / 197 | CLK |
| 4 | product-imdp-sparse-n10000-nnz10-a4-dfa4 | solve_dfa | 10.4% | 6.1% | 1.07 s / 1.13 s | 436 / 197 | CLK |
| 8 | cs-imdp-dense-n1000-a4 | solve_cs_stationary | 7.2% | 52.7% | 305.30 ms / 199.90 ms | 436 / 557 | CLK |
| 8 | fimdp-dense-v2-d10-a1-mccormick | workspace | 114.7% | 5.4% | 1.51 ms / 1.59 ms | 559 / 559 | WS |
| 8 | fimdp-dense-v2-d10-a1-omax | bellman | 123.7% | 14.5% | 68.4 µs / 78.4 µs | 436 / 197 | CLK |
| 8 | fimdp-dense-v2-d10-a4-omax | workspace | 12.8% | 5.4% | 20.7 µs / 21.8 µs | 209 / 560 | CLK |
| 8 | fimdp-dense-v3-d10-a1-omax | bellman | 142.0% | 139.7% | 11.93 ms / 4.98 ms | 436 / 209 | CLK |
| 8 | fimdp-sparse-v2-d10-k4-a1-vertex | bellman | 14.0% | 6.5% | 1.15 ms / 1.08 ms | 438 / 197 | CLK |
| 8 | fimdp-sparse-v2-d50-k10-a1-omax | bellman | 95.0% | 96.3% | 3.81 ms / 1.94 ms | 436 / 559 | CLK |
| 8 | fimdp-sparse-v2-d50-k10-a4-omax | workspace | 65.0% | 6.0% | 586.9 µs / 553.7 µs | 560 / 197 | CLK |
| 8 | fimdp-sparse-v3-d10-k3-a1-vertex | workspace | 16.4% | 10.5% | 1.1 µs / 1.0 µs | 436 / 197 | CLK |
| 8 | fimdp-sparse-v3-d20-k5-a1-omax | workspace | 75.9% | 81.8% | 1.05 ms / 578.4 µs | 436 / 197 | CLK |
| 8 | imdp-dense-n100-a1 | bellman | 85.2% | 14.1% | 8.0 µs / 9.1 µs | 436 / 209 | CLK |
| 8 | imdp-dense-n100-a4 | workspace | 5.4% | 64.8% | 5.3 µs / 8.8 µs | 540 / 559 | WS |
| 8 | imdp-dense-n1000-a1 | bellman | 5.1% | 7.7% | 862.2 µs / 800.2 µs | 436 / 197 | CLK |
| 8 | imdp-sparse-n10000-nnz10-a4 | bellman | 118.0% | 5.2% | 2.29 ms / 2.41 ms | 436 / 197 | CLK |
| 8 | imdp-sparse-n100000-nnz10-a1 | workspace | 37.6% | 7.6% | 3.41 ms / 3.66 ms | 197 / 558 | CLK |
| 8 | imdp-sparse-n100000-nnz10-a4 | workspace | 18.2% | 31.0% | 33.49 ms / 43.88 ms | 559 / 560 | WS |
| 8 | product-imdp-sparse-n10000-nnz10-a1-dfa4 | bellman | 7.4% | 21.0% | 5.90 ms / 4.88 ms | 436 / 197 | CLK |
| 8 | product-imdp-sparse-n10000-nnz10-a1-dfa4 | workspace | 56.6% | 43.9% | 295.7 µs / 425.6 µs | 456 / 559 | CLK |
| 8 | product-imdp-sparse-n10000-nnz10-a4-dfa4 | bellman | 79.6% | 5.1% | 11.42 ms / 10.87 ms | 559 / 197 | CLK |
| 8 | product-imdp-sparse-n10000-nnz10-a4-dfa4 | workspace | 69.0% | 71.0% | 2.17 ms / 1.27 ms | 299 / 557 | CLK |
| 8 | real-multiObj_robotIMDP | bellman | 7.2% | 14.6% | 49.0 µs / 56.2 µs | 557 / 199 | CLK |
| 16 | fimdp-dense-v2-d50-a1-omax | bellman | 9.2% | 7.8% | 21.88 ms / 20.30 ms | 197 / 197 | THR |
| 16 | fimdp-dense-v2-d50-a1-omax | solve_rvi | 7.4% | 11.0% | 1.93 s / 1.74 s | 197 / 560 | CLK |
| 16 | fimdp-sparse-v2-d10-k4-a1-vertex | bellman | 8.7% | 6.6% | 1.94 ms / 1.82 ms | 201 / 197 | THR |
| 16 | fimdp-sparse-v2-d10-k4-a1-vertex | workspace | 15.1% | 20.2% | 1.5 µs / 1.2 µs | 197 / 559 | CLK |
| 16 | fimdp-sparse-v2-d10-k4-a4-vertex | solve_rvi | 5.8% | 10.7% | 449.32 ms / 497.19 ms | 197 / 559 | CLK |
| 16 | fimdp-sparse-v2-d10-k4-a4-vertex | workspace | 11.0% | 12.1% | 1.7 µs / 1.9 µs | 557 / 209 | CLK |
| 16 | fimdp-sparse-v2-d50-k10-a4-omax | solve_rvi | 6.0% | 5.6% | 599.86 ms / 568.19 ms | 197 / 559 | CLK |
| 16 | fimdp-sparse-v3-d10-k3-a1-vertex | bellman | 120.9% | 41.2% | 90.15 ms / 63.85 ms | 209 / 197 | CLK |
| 16 | fimdp-sparse-v3-d10-k3-a1-vertex | workspace | 38.8% | 15.4% | 2.0 µs / 2.3 µs | 563 / 557 | WS |
| 16 | fimdp-sparse-v3-d20-k5-a1-omax | bellman | 10.8% | 6.5% | 18.05 ms / 16.96 ms | 197 / 197 | THR |
| 16 | fimdp-sparse-v3-d20-k5-a4-omax | bellman | 14.1% | 5.1% | 121.08 ms / 115.15 ms | 197 / 198 | THR |
| 16 | fimdp-sparse-v3-d20-k5-a4-omax | solve_rvi | 5.9% | 23.7% | 10.53 s / 8.51 s | 197 / 197 | THR |
| 16 | imdp-dense-n100-a1 | workspace | 33.4% | 29.1% | 1.6 µs / 2.1 µs | 436 / 559 | CLK |
| 16 | imdp-dense-n1000-a4 | solve_rvi | 20.6% | 5.5% | 219.20 ms / 207.72 ms | 197 / 557 | CLK |
| 16 | imdp-dense-n4000-a1 | solve_ivi | 20.9% | 27.3% | 1.09 s / 1.39 s | 197 / 436 | CLK |
| 16 | imdp-dense-n4000-a1 | solve_rvi | 25.3% | 8.1% | 882.47 ms / 816.56 ms | 197 / 559 | CLK |
| 16 | imdp-dense-n4000-a4 | solve_ivi | 11.7% | 5.2% | 2.69 s / 2.56 s | 197 / 436 | CLK |
| 16 | imdp-sparse-n10000-nnz10-a4 | bellman | 20.9% | 13.7% | 2.91 ms / 2.55 ms | 197 / 197 | THR |
| 16 | imdp-sparse-n100000-nnz10-a4 | bellman | 12.0% | 19.5% | 42.08 ms / 35.21 ms | 197 / 197 | THR |
| 16 | imdp-sparse-n100000-nnz10-a4 | solve_ivi | 17.2% | 14.7% | 6.34 s / 5.53 s | 203 / 197 | THR |
| 16 | imdp-sparse-n100000-nnz10-a4 | workspace | 7.4% | 7.2% | 77.79 ms / 83.38 ms | 559 / 559 | WS |
| 16 | product-imdp-dense-n1000-a1-dfa4 | workspace | 15.4% | 7.4% | 180.7 µs / 168.3 µs | 197 / 559 | CLK |
| 16 | product-imdp-sparse-n1000-nnz10-a4-dfa4 | bellman | 32.1% | 13.4% | 2.85 ms / 2.51 ms | 197 / 197 | THR |
| 16 | product-imdp-sparse-n1000-nnz10-a4-dfa4 | solve_dfa | 36.6% | 7.7% | 156.73 ms / 168.83 ms | 557 / 557 | THR |
| 16 | product-imdp-sparse-n10000-nnz10-a1-dfa4 | bellman | 9.4% | 59.0% | 5.94 ms / 3.74 ms | 197 / 197 | THR |
| 16 | product-imdp-sparse-n10000-nnz10-a1-dfa4 | solve_dfa | 14.6% | 14.1% | 707.37 ms / 806.96 ms | 560 / 559 | THR |
| 16 | real-multiObj_robotIMDP | workspace | 66.2% | 15.9% | 41.2 µs / 47.7 µs | 209 / 559 | CLK |

## 5. Profiles (`profiles/*.md`)

One file per profiled case, at least one per case-matrix row, each at **1 and 16 threads** (CPU, Float64, compact
pinning). Each run section has a CPU sampling profile (self-time and inclusive top frames, pruned tree; 6 s per entry,
0.5 ms sampling), a `Profile.Allocs` summary, the steady-state allocation of one `bellman!` call (`@allocated` after
warm-up) and `JET.@report_opt` (`target_modules = (IntervalMDP,)`). Control-synthesis cases have no `bellman` entry in
the registry, so `profile.jl` adds the pseudo-entry `bellman_cs`: one `bellman!` with the strategy cache that
`solve(ControlSynthesisProblem)` builds (stationary and time-varying), checked against `bellman!` with the default cache.

Provenance and correctness: the 1-thread sections (and the 16-thread IMDP sections) were recorded at `91bc0c8`, the new
16-thread sections and `bellman_cs` at `4a42913`; both have `src/`/`ext/` trees identical to `20fc03b` (src
`290f675`, ext `5a4cf21`) and `src/ext dirty: false`. New sections record the reference check of the profiled call in
the file (all **pass**, max |ΔV| = 0). The older sections are covered by `run.jl --reference-only` over the 15 profiled
cases at t=1 and t=16 (48 entries each: 35 pass, 13 not-applicable `workspace`, 0 fail).

Shares are self-time samples / all samples of the main thread and the default-pool threads. At t=16 the sampler counts
idle threads, so `wait()` is mostly idle time (main thread blocked on `@threads`, workers waiting for the slowest
chunk); the remaining frames show the per-thread kernel mix.

| Row | Profile (threads) | Where the time goes (self time) | Steady-state `bellman!` allocation |
|---|---|---|---|
| IMDP dense | `imdp-dense-n1000-a4`, `imdp-dense-n4000-a4` (t=1) | gap walk `gap_value` 79–81% (`min(budget, gap[i])` 42%, float add/mul 37%), `dot(V, lower)` (BLAS) 17–19% | 0 B ✓ |
| IMDP dense | `imdp-dense-n4000-a4` (t=16) | idle/wait 59%, gap walk 31% | 7.2 KB, 82 allocs (task spawn of `@threadstid`) |
| IMDP dense | `imdp-dense-n100-a4` (t=16) | idle/wait 87%, `threading_run` scheduling | 7.5 KB, 82 allocs |
| IMDP sparse | `imdp-sparse-n10000-nnz100-a4`, `imdp-sparse-n100000-nnz100-a1` (t=1) | building the (V, gap) tuples (`setindex!`, gather of `V[support]`) 25%, `sort!` 10–12%, dynamic `_by` ordering 8% | 64 B per column (`SubArray` 48 B + kw `NamedTuple` 16 B): 2.56 MB / 6.4 MB per call |
| IMDP sparse | `imdp-sparse-n100000-nnz100-a1` (t=16) | wait 78%; `_setindex!` 5%, `getindex` 3%, `sort!` 2%, `_by` 2% | 6.41 MB (64 B/column + task spawn) |
| fIMDP O-max | `fimdp-dense-v2-d50-a1-omax`, `fimdp-sparse-v3-d20-k5-a1-omax` (t=1) | `_by` 21% / 58%, `sort!` 11–13%, tuple building 7–13% | 8.2 MB / 15.9 MB per call (32 B/alloc) |
| fIMDP O-max | `fimdp-dense-v2-d50-a1-omax` (t=16) | wait 77%; `sort!` 5%, `_by` 4%, `_setindex!` 3% | 8.17 MB |
| fIMDP McCormick | `fimdp-sparse-v2-d10-k4-a1-mccormick` (t=1) | HiGHS 80% (`Highs_run`, `Highs_create`/`destroy`, `addRow`): one JuMP model rebuilt per state–action | 21.6 MB, 411 000 allocs |
| fIMDP McCormick | same (t=16) | wait 73%; `Highs_run` 11.5%, `GenericMemory` 3.6%, `Highs_create` 2% | 21.6 MB |
| fIMDP vertex | `fimdp-sparse-v2-d10-k4-a1-vertex`, `fimdp-sparse-v3-d10-k3-a1-vertex` (t=1) | products in the vertex sum (`promotion.jl`) 36–38%, vertex iterator | 1.1 MB / 7.7 MB per call |
| fIMDP vertex | `fimdp-sparse-v2-d10-k4-a1-vertex` (t=16) | wait 74%; vertex iterator `iterate` 6.6%, `+`/`==`/`*` 6% | 1.13 MB |
| Product IMDP × DFA | `product-imdp-dense-n1000-a4-dfa4` (t=1) | dense gap walk: `min` 38%, `dot` 19%, `*`/`+`/`-` 38% | 0 B ✓ |
| Product IMDP × DFA | same (t=16) | wait 68–72%; `min` 6–8%, `+` 6–7%, `dot` 3–4% | 30.6 KB (task spawn) |
| Product IMDP × DFA | `product-imdp-sparse-n10000-nnz10-a4-dfa4` (t=1) | **dynamic `Base.Order._by(by::Function, …)` 50%**, `_setindex!` 11%, `sort!` 10% (small supports, nnz 10) | 10.2 MB (64 B per column) |
| Product IMDP × DFA | same (t=16) | wait 73%; `sort!` 11–13%, `_by` 7.5% | 10.3 MB |
| Real model | `real-multiObj_robotIMDP` (t=1) | **dynamic `_by` 67%**, `sort!` 13% (small supports) | 53 KB (64 B per column) |
| Real model | same (t=16) | wait 77%; `sort!` 4–5%, `enq_work` 4% (task scheduling; 0.1 ms calls), `_by` 2–3% | 61 KB |
| Control synthesis | `cs-imdp-dense-n1000-a4`, `cs-imdp-sparse-n10000-nnz100-a4` (t=1) | same as the IMDP kernels; strategy extraction not visible (< 3%) | `bellman_cs`: 0 B ✓ (dense) / 2.56 MB (sparse), both caches |
| Control synthesis | same (t=16) | wait 71–80%; dense: `min`/`+`/`dot` 3–6% each; sparse: `_setindex!` 4–5%, `_by` 2%, `sort!` 2% | `bellman_cs`: 7.4 KB (dense) / 2.57 MB (sparse) |
| CUDA | `cuda-*.md` | `CUDA.@profile` traces, see § 6.3 | n/a |

JET (`@report_opt`, `target_modules = (IntervalMDP,)`) reports **0** problems for every hot function at both thread
counts except 1 report in the fIMDP O-max solve (`fimdp-dense-v2-d50-a1-omax`/`solve_rvi`). This filter hides the main
type instability: the runtime dispatch happens inside `Base.sort!` because IntervalMDP passes `rev = upper_bound` (a
runtime `Bool`) and `by = first` as keywords, so `Base.Order.ord` returns an ordering whose type depends on a runtime
value. `Profile.Allocs` and the self-time profile show it (H1). At t=16 every threaded `bellman!` allocates a few KB for
task spawning even where t=1 allocates 0 B, and the large idle share means the per-call work is too small or too
unevenly split for 16 threads on these sizes (§ 6.1).

## 6. Scaling and roofline

Full tables: `results/scaling-roofline-20fc03b.md`, generated by `julia --project=benchmark benchmark/analyze.jl` from the
local files `results/scaling-20fc03b-cpu-t{1,2,4,6,8,10,12,14,16}.json` (suite `scaling`, `--entries bellman`),
`results/sizes-20fc03b-cpu-t{1,6,16}.json` (suite `sizes`) and `results/stream-t{1,…,16}.json` (`benchmark/stream.jl`).
All 36 + 39 measured entries are valid (correctness `pass` against the stored reference; recorded `src_tree` = `20fc03b:src`).
Measured read roofs (max over 3 rounds, see below): DRAM 12.3 GB/s at 1 thread (`stream-t1.json`), 37.1 at 6
(`stream-t6.json`), 41.9 at 8 (`stream-t8.json`), 53.1 at 16 (`stream-t16.json`); L3 50.8 (`stream-t1.json`) to 268.3 GB/s
(`stream-t16.json`); in-L1 FMA 28.6 GFLOP/s at 1 thread (`stream-t1.json`), 95.9 at 16 (`stream-t16.json`).
Theoretical: 16 DP flop/cycle per P-core (81.6 GFLOP/s at 5.1 GHz), LPDDR5X ≈ 134 GB/s platform maximum.

Model and rule (header of `analyze.jl`): dense 16 B/nnz, sparse 20 B/nnz + 4 B/column, 4 flop/nnz (AI ≈ 0.2–0.25 flop/B,
so no case can be FLOP-bound: every case below is ≤ 9% of the in-L1 FMA roof). "% of roof" = model bandwidth / measured
read roof at the same thread count (DRAM if model data > 24 MiB, else L3). **memory-bound** ≥ 50% of that roof;
**overhead-bound** median < 100 µs or parallel efficiency < 25%; **compute-bound** otherwise (in-core work: sorting, branchy
scalar gap walk, gathers, per-column allocation — not FMA throughput).

Measurement caveats: (1) the scaling/sizes files were measured on 2026-10-06 in the capped clock state (pre-trial probe
436 µs at t ≤ 10, 557–559 µs at t ≥ 12, § 1.1); the roofs were re-measured on 2026-10-08 starting in the same state (probe
at start 436–437 µs in every `stream-t*.json`), but the clock switched between rounds (round probes 197/436/557–563 µs,
stored under `rounds`), so the roofs are a max over states, i.e. an upper bound, and "% of roof" a lower bound. Single-round
roofs differ by up to 2× (e.g. L3 at t=8: 175 / 195 / 297 GB/s in `stream-t8.json`). (2) † in the generated tables =
CLOCK flag. At t ≥ 8 it is the known systematic flag (unloaded reference probe). At t = 1 (`scaling-20fc03b-cpu-t1.json`,
ratio 2.22) it is a reference artefact too: the pre- and post-trial probes are 436/437 µs, the same state as the nominal
t = 2–6 and `sizes-20fc03b-cpu-t1.json` runs; only the run-start reference (≈ 197 µs) was taken while the core still
boosted. The t = 1 medians agree with the nominal size-scaling run (dense n = 4000: 77.97 ms vs 77.10 ms in
`sizes-20fc03b-cpu-t1.json`), so the file is used as the t = 1 base.

### 6.1 Strong scaling (`bellman!`, fixed size)

Speed-up vs t = 1; the value in column t=T comes from `results/scaling-20fc03b-cpu-tT.json` (T = 1, 2, 4, 6, 8, 10, 12, 14, 16).

| case | t=1 | t=2 | t=4 | t=6 | t=8 | t=10 | t=12 | t=14 | t=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| imdp-dense-n4000-a1 (244 MiB) | 77.97 ms | ×1.77 | ×3.05 | ×4.56 | ×4.51 | ×6.15 | ×9.41 | ×10.05 | ×7.99 |
| imdp-dense-n4000-a4 (977 MiB) | 312.66 ms | ×1.86 | ×3.38 | ×4.55 | ×5.74 | ×6.24 | ×7.13 | ×7.63 | ×7.45 |
| imdp-sparse-n100000-nnz10-a1 (19.5 MiB) | 63.21 ms | ×1.79 | ×3.36 | ×5.43 | ×4.94 | ×6.53 | ×15.05 | ×17.57 | ×6.46 |
| imdp-sparse-n100000-nnz100-a1 (191 MiB) | 431.63 ms | ×1.97 | ×3.84 | ×5.70 | ×5.20 | ×6.53 | ×7.75 | ×9.06 | ×9.28 |

Classification. Cell = model GB/s, % of the roof (DRAM or L3) from `stream-tT.json`, parallel efficiency; median from
`scaling-20fc03b-cpu-tT.json` of the same column.

| case | t=1 | t=6 | t=8 | t=14 | t=16 | class |
|---|---|---|---|---|---|---|
| imdp-dense-n4000-a1 | 3.3 GB/s, 27% DRAM | 15.0, 40%, eff 76% | 14.8, 35%, 56% | 33.0, 71%, 72% | 26.2, 49%, 50% | **compute-bound** (latency: dependent gap walk + permutation gather) at t ≤ 8; **memory-bound** at t = 10–14 (59–71% of DRAM roof); borderline at t = 16 (49%, LP-E cores finish last) |
| imdp-dense-n4000-a4 | 3.3, 27% DRAM | 14.9, 40%, 76% | 18.8, 45%, 72% | 25.0, 54%, 55% | 24.4, 46%, 47% | **compute-bound** at t ≤ 8; near the memory roof at t ≥ 10 (46–60%: t = 10 and 14 classify memory-bound, t = 12 and 16 compute-bound — within the roof noise) |
| imdp-sparse-n100000-nnz10-a1 | 0.3, 1% L3 | 1.8, 1%, 90% | 1.6, 1%, 62% | 5.7, 2%, 125%* | 2.1, 1%, 40% | **compute-bound** at every T (≤ 3% of any roof, 0.06 GFLOP/s at t = 1); 2 allocations per column (200 052–200 082 per call), GC 8–11% of the mean at t ≥ 10 (`scaling-20fc03b-cpu-t10.json`–`-t16.json`) |
| imdp-sparse-n100000-nnz100-a1 | 0.5, 4% DRAM | 2.6, 7%, 95% | 2.4, 6%, 65% | 4.2, 9%, 65% | 4.3, 8%, 58% | **compute-bound** at every T (≤ 9% of the DRAM roof) |

No strong-scaling case is overhead-bound (every median ≥ 3.6 ms, efficiency ≥ 40%). Efficiency is 76–98% on the P-cores
(t ≤ 6); the 6 → 8 step (first 2 E-cores, equal static chunks) gains nothing or loses for 3 of 4 cases (dense a1 ×4.56 →
×4.51, sparse k = 10 ×5.43 → ×4.94, k = 100 ×5.70 → ×5.20); **t = 16 is slower than t = 14 for 3 of 4 cases** (the 2 LP-E
cores get an equal chunk and finish last). *Superlinear t = 12/14 for sparse k = 10 (×15.05/×17.57, efficiency 125%,
`scaling-20fc03b-cpu-t12.json`/`-t14.json`) is not plausible for a compute-bound kernel and is not used as evidence; it
coincides with the 557 µs probe state and needs a re-measurement in 0h.

### 6.2 Size scaling and roofline (fixed threads, increasing `n`, 1 action)

Cell = ns per nnz (t = 1 only), % of roof, parallel efficiency vs t = 1. Column t=1 from `results/sizes-20fc03b-cpu-t1.json`
+ `stream-t1.json`, t=6 from `sizes-20fc03b-cpu-t6.json` + `stream-t6.json`, t=16 from `sizes-20fc03b-cpu-t16.json` +
`stream-t16.json`. Roof = L3 for data ≤ 24 MiB, DRAM above.

| case (model data) | t=1 | t=6 | t=16 | class |
|---|---|---|---|---|
| dense n=250 (1.0 MiB) | 3.30 ns, 10% L3 | 9%, 92% (37.2 µs) | 9%, 30% (43.7 µs) | compute-bound at t = 1; **overhead-bound** at t ≥ 6 (< 100 µs per call, t = 16 slower than t = 6) |
| dense n=500 (3.8 MiB) | 3.31 ns, 10% L3 | 10%, 102% | 8%, 28% | compute-bound (t = 16 near the overhead limit: 28%) |
| dense n=1000 (15.3 MiB) | 3.47 ns, 9% L3 | 5%, 48% | 8%, 28% | compute-bound |
| dense n=2000 (61 MiB) | 4.64 ns, 28% DRAM | 42%, 75% | 40%, 39% | compute-bound (latency) |
| dense n=4000 (244 MiB) | 4.82 ns, 27% DRAM | 40%, 75% | 50%, 50% | compute-bound at t ≤ 6; **memory-bound** at t = 16 (26.6 GB/s = 50% of 53.1) |
| dense n=8000 (977 MiB) | 4.89 ns, 26% DRAM | 39%, 73% | 46%, 47% | compute-bound; borderline memory at t = 16 (46%) |
| sparse k=10 n=1000 (0.2 MiB) | 54.18 ns, 1% L3 | 0%, 56% | 1%, 36% (93.8 µs) | compute-bound at t ≤ 6; **overhead-bound** at t = 16 (< 100 µs) |
| sparse k=10 n=10⁴ (1.9 MiB) | 62.16 ns, 1% L3 | 1%, 95% | 1%, 49% | compute-bound |
| sparse k=10 n=10⁵ (19.5 MiB) | 62.73 ns, 1% L3 | 1%, 91% | 1%, 51% | compute-bound |
| sparse k=10 n=10⁶ (195 MiB) | 65.02 ns, 3% DRAM | 5%, 92% | 2%, 21% | compute-bound at t ≤ 6; **overhead-bound** at t = 16 (efficiency 21%: 189.81 ms vs 118.38 ms at t = 6; 2 000 082 allocations per call, GC 4.7%) |
| sparse k=100 n=1000 (1.9 MiB) | 40.52 ns, 1% L3 | 1%, 67% | 1%, 41% | compute-bound |
| sparse k=100 n=10⁴ (19.1 MiB) | 44.85 ns, 1% L3 | 1%, 100% | 2%, 66% | compute-bound |
| sparse k=100 n=10⁵ (191 MiB) | 46.76 ns, 3% DRAM | 7%, 104% | 8%, 60% | compute-bound |

* Dense O-max at 1 thread costs 3.3–3.5 ns/nnz while the data fit in L3 (n ≤ 1000) and 4.6–4.9 ns/nnz from n = 2000 (data in
  DRAM), i.e. 3.3–4.8 GB/s = 26–28% of the 1-thread DRAM roof and 9–10% of the L3 roof (`sizes-20fc03b-cpu-t1.json`,
  `stream-t1.json`): latency-/compute-bound, not bandwidth-bound. Only with many threads and large n does it reach the
  memory roof (46–71% at t ≥ 10, § 6.1; 50% for n = 4000 at t = 16).
* Sparse O-max costs 40–65 ns/nnz independent of n (k = 10: 54–65, k = 100: 41–47 ns, `sizes-20fc03b-cpu-t1.json`), ≤ 8% of any
  roof at any thread count: compute-bound by per-column work (sorting, 2 allocations per column, gathers), not by memory.
* Overhead-bound: the smallest cases at t ≥ 6/16 (< 100 µs per call) and sparse n = 10⁶ at t = 16 (efficiency 21%).

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
