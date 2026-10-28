# Performance of the VI/Bellman Algorithms — 0r: Re-baseline under the Performance profile — Specification (Julia + Lean)

> Sub-phase **0r** (re-baseline), run on the Phase 1 branch after `1a` and before `1b.1`. One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Committed vs local benchmark outputs, § Evidence Protocol, § CPU / GPU Matrix, § Benchmark Suite.
> Phase branch: `perf/phase-1` (exists, draft PR #127, 1a committed as `cdee291`).

**Active phase: 0r**

## Objective *(required)*

Re-measure every Phase 0 measurement whose result depends on the host power profile, under the **Performance** profile
(the reference environment since 1a, `REPORT.md` § 10.2), on the merge base `bcbf0df`, and record the results in a new
`REPORT.md` § 11 "Re-baseline under the Performance profile (`bcbf0df`)". The Phase 0 timings (§ 3–6) were recorded
under `balanced`/`balance_performance` at `20fc03b`; the operator only cares about maximum-performance numbers. Only
timings, scaling/roofs and profile shares are re-measured; reference values, correctness checks and allocation counts
do not depend on the profile (allocations are still recorded, as a cross-check).

What to measure (all on `bcbf0df`; `src/` and `ext/` are identical to it on `perf/phase-1`):

1. **CPU baseline (was 0c, 0d):** full suite at 1, 4, 8 and 16 threads, **3 separate-process rounds** (base, `-rerun1`,
   `-rerun2`), interleaved over thread counts:
   `BENCH_BASE_REF=bcbf0df bash benchmark/baseline.sh rounds "" -rerun1 -rerun2`
   → `benchmark/results/baseline-bcbf0df-cpu-t{1,4,8,16}{,-rerun1,-rerun2}.json`.
   Spread per entry over the 3 rounds (`compare.jl --spread`); list every entry above 5% (no budget fixes in this
   sub-phase — those belong to the phase that uses the entry; 1a already fixed the Phase 1 target cases).
2. **Strong and size scaling (was 0f):**
   `BENCH_BASE_REF=bcbf0df bash benchmark/baseline.sh scaling` and `… sizes`
   → `scaling-bcbf0df-cpu-t*.json`, `sizes-bcbf0df-cpu-t*.json`.
3. **Machine roofs (was 0f):** do **not** use `baseline.sh stream` — it overwrites the Phase 0 `stream-tT.json`. Run
   `julia --project=benchmark --threads=<T> benchmark/stream.jl --out benchmark/results/stream-bcbf0df-t<T>.json` for
   T = 1 2 4 6 8 10 12 14 16, then
   `julia --project=benchmark benchmark/analyze.jl --sha bcbf0df --stream <the nine stream-bcbf0df files>` (check the
   `--stream` option form in the header of `analyze.jl`) → `benchmark/results/scaling-roofline-bcbf0df.md` (committed).
4. **CPU profiles (was 0e):** re-profile every case that has a CPU summary in `benchmark/profiles/*.md` (15 files, not
   `cuda-*`), with the same entries and thread counts as the existing summary (read its header), with
   `benchmark/profile.jl --case <case> … --out benchmark/profiles/bcbf0df/<case>.md`. Never `--append`, never write over
   the Phase 0 summaries.
5. **CUDA (was 0g):** probe with `python3 harness/tools/harness/gpu_check.py probe /home/fresen/.julia/dev/IntervalMDP`.
   If FUNCTIONAL: three separate processes
   `julia --project=benchmark benchmark/run.jl --suite full --backend cuda --eltype all --out benchmark/results/baseline-bcbf0df-cuda{,-rerun1,-rerun2}.json`
   (checked against the **stored** CUDA references — do **not** pass `--write-reference`; do not use
   `baseline.sh cuda`, which rewrites them), and the 5 CUDA profiles (`benchmark/profiles/cuda-*.md`) re-profiled into
   `benchmark/profiles/bcbf0df/cuda-<case>.md`. CUDA entries are bimodal across processes (Phase 0 lesson): report both
   modes where they appear. If the probe is not FUNCTIONAL: record CUDA as UNAVAILABLE with the probe output.

Rules:
- **Environment, per run.** Before every timed process (a wrapper script is fine): `cpuset.cpus.isolated` empty,
  `platform_profile` = `performance`, EPP `performance` on CPUs 0/4/8/12/15, AC online, load and top processes; append
  them to `benchmark/results/logs/0r-env.log` with the run name. A run with a non-empty isolated set, a non-Performance
  profile, or AC offline is discarded and repeated.
- **Settled clock.** A run whose log contains `NOT settled` (the 1a range flag of `lib/clock.jl`) is repeated, at most
  twice; a run still unsettled is kept, marked "not settled" in § 11, and its entries are not used for spreads or
  comparisons. Report how many runs were repeated.
- **No code changes.** No file under `src/`, `ext/`, `test/`, `lean/` or `benchmark/**/*.jl`/`*.sh` is changed. If a tool
  does not do what this spec says, report it as a Finding (`0r F-n`) and use the nearest working form.
- **Long runs.** Start each block in the background with output redirected to a log; read only tails; resume after an
  interruption from the files that already exist and are valid (resume, don't redo). Expected wall time: ≈ 9–12 h.

## Commands / Toolchain *(required)*

- Target root: `/home/fresen/.julia/dev/IntervalMDP`
- Julia version / compat: `[compat] julia = "1.11"`; local `julia` is 1.13.1.
- Julia instantiate: `julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'`
- Baseline driver: `BENCH_BASE_REF=bcbf0df bash benchmark/baseline.sh rounds|scaling|sizes` (see its header). Runs
  directly: `julia --project=benchmark --threads=<T> benchmark/run.jl --suite <name> [--backend cuda --eltype all] --out benchmark/results/<file>.json`.
- Roofs: `benchmark/stream.jl` (header). Tables: `benchmark/analyze.jl --sha bcbf0df …` (header).
- Profiles: `julia --project=benchmark --threads=<T> benchmark/profile.jl --case <case> [--entries …] [--backend cuda] --out benchmark/profiles/bcbf0df/<case>.md`.
- Spread: `julia --project=benchmark benchmark/compare.jl --spread R1.json R2.json R3.json`. Comparison with Phase 0:
  `compare.jl <phase0.json> <0r.json>` (ratios are for information only: different code base `20fc03b` → `bcbf0df`
  (B-1 fix in `src/strategy_cache.jl`) and different profile).
- GPU probe: `python3 harness/tools/harness/gpu_check.py probe /home/fresen/.julia/dev/IntervalMDP`.
- Formatter (always-on check; no Julia file should change): `python3 harness/tools/harness/julia_format.py check /home/fresen/.julia/dev/IntervalMDP --base bcbf0df`.
- Branch: `perf/phase-1` (exists). Base ref: `bcbf0df`.

## Julia Behavior & Tests *(required)*

No package code changes. Every measured entry must pass the suite's correctness check against the stored references
(`valid = true`); an invalid entry is listed and not used. `Pkg.test()` is not required (no package or test change;
1a ran it on the same `src/`).

## Algorithm ↔ Theorem Mapping *(required)*

none.

## CPU / GPU Matrix *(required)*

| Check | Backend | Required? | Notes |
|---|---|---|---|
| Package tests | CPU | no | no package code changes |
| Benchmarks, profiles | CPU | yes | 1, 4, 8, 16 threads (baseline); 1–16 (scaling); profiles at Phase 0 thread counts |
| GPU probe | CUDA | yes | PASS → CUDA baseline + profiles; otherwise UNAVAILABLE recorded with the probe output |
| Benchmarks, profiles | CUDA | yes if the probe passes | Float64 + Float32, 3 processes |

## Performance Evidence *(required)*

Measurement only; no speedup is claimed. QE spot-check (**non-gating**, operator decision of 1a: the noise bound is
session-dependent): QE re-runs a few entries in 3 separate processes and reports the result for information. The gate
is the consistency of § 11 with the files (acceptance criteria below), not reproduction in QE's session.

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to `bcbf0df`; no file under `src/`, `ext/`, `test/`, `lean/`, `benchmark/**/*.jl` or `benchmark/**/*.sh` changed.
- [ ] Every timed run has an environment record in `benchmark/results/logs/0r-env.log` showing the Performance profile, an empty isolated set and AC online; runs that failed this were discarded and repeated.
- [ ] CPU baseline: 12 files (t1/4/8/16 × 3 rounds), every entry `valid`; § 11 has the per-entry spread summary (count and list of entries above 5% per thread count), selected medians next to the Phase 0 values, steady-state allocations per `bellman!` and the VI-loop overhead, each citing its file.
- [ ] Scaling, sizes and roofs re-measured (`scaling-bcbf0df-*`, `sizes-bcbf0df-*`, `stream-bcbf0df-t*`); `benchmark/results/scaling-roofline-bcbf0df.md` exists and § 11 gives the strong-scaling efficiencies, the roofs and the classification, compared with Phase 0 § 6.
- [ ] CPU profiles: one `benchmark/profiles/bcbf0df/<case>.md` per Phase 0 CPU profile, same entries and thread counts; § 11 lists, per hypothesis H1–H9 of § 7, the measurement it cites and the re-measured value, and flags every hypothesis whose cited measurement changed by more than 10 percentage points (shares) or ×1.25 (times), for the 1b/2b/3b/4b specs.
- [ ] CUDA: either the probe passes and 3 CUDA baseline files (Float64 + Float32, checked against the stored references, no `--write-reference`) and `profiles/bcbf0df/cuda-*.md` exist with § 11 reporting both modes of bimodal entries, or CUDA is recorded as UNAVAILABLE with the probe output.
- [ ] Clock: § 11 states how many runs were repeated because of `NOT settled`, lists any run kept as not settled, and uses no not-settled run for a spread or comparison.
- [ ] `REPORT.md` § 3, § 4, § 5 and § 6 each start with a one-line note that their timings were recorded under `balanced`/`balance_performance` at `20fc03b` and are superseded by § 11 (the Phase 0 text itself is not rewritten).
- [ ] Every number in § 11 names its source file, and a Dev self-check script recomputes every number and range in § 11 from the files before hand-off; every qualitative claim ("all", "every", "steady", "within") is checked against the data for counterexamples.
- [ ] The sub-phase commit contains only the files in § File List (staged by explicit path).

## File List

- `benchmark/REPORT.md` — new § 11; one-line notes at the top of § 3–6 (tracked, commit)
- `benchmark/profiles/bcbf0df/*.md` — new (commit)
- `benchmark/results/scaling-roofline-bcbf0df.md` — new (commit)
- `harness/specs/perf-vi-bellman/0r-rebaseline-performance.md` — this spec (new, commit)
- `benchmark/results/**/*.json`, `benchmark/results/logs/**` — local only (gitignored), never staged
- `benchmark/reference/**` — local only; must not change

## Out of Scope

- Any change to `src/`, `ext/`, `test/`, `lean/` or to the benchmark tools (`benchmark/**/*.jl`, `*.sh`).
- Budget/noise fixes for entries outside the Phase 1 target set (recorded in § 11, fixed by the phase that uses them).
- Rewriting the Phase 0 hypotheses (§ 7) or Findings (§ 8); § 11 only flags changed measurements.
- Regenerating reference values.
