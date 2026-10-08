# Performance of the VI/Bellman Algorithms — 0g: CUDA baseline and profiles — Specification (Julia + Lean)

> Sub-phase **0g** of the performance work (Phase 0, measurement only). One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Sub-phases (Phase 0), § CPU / GPU Matrix, § Benchmark Suite.
> Previous sub-phase must be committed and pushed to `perf/phase0` first.

**Active phase: 0g**

## Objective *(required)*

Probe the GPU. If it is functional: write the CUDA reference values (Float64 and Float32), record the CUDA baseline with re-runs, and save `CUDA.@profile` (or Nsight) summaries as `benchmark/profiles/cuda-*.md`. If not: record CUDA as UNAVAILABLE with the probe output. Fill `REPORT.md` § 6.3 (including the `<!--CUDAPROF-->` placeholder) and the CUDA rows of § 3/§ 4. CUDA cases that cannot run are listed with the Finding that explains why (draft the Finding text in § 8 for `0h`).

Rules (from `common.md` § Sub-phases (Phase 0)): **resume, don't redo** — first check which deliverables below already
exist in the working tree and are valid, and only fill the gaps; valid result files are not regenerated; the hand-off lists
reused / changed / created files. **Stay in scope** — edit only the files in § File List and the named `REPORT.md` section;
a defect in an earlier sub-phase's file is fixed only if this sub-phase cannot be done without it (minimal edit, noted in
the hand-off), otherwise it is noted for `0h`. Follow `common.md` § Context budget.

Adds or semantically changes a VI/Bellman algorithm: **no**. `src/` and `ext/` must stay byte-identical to the base ref
`20fc03b`. Lean dependencies touched: none.

## Commands / Toolchain *(required)*

- Target root: `/home/fresen/.julia/dev/IntervalMDP`
- Julia version / compat: `[compat] julia = "1.11"`; local `julia` is 1.13.1.
- Julia instantiate: `julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'`
- GPU probe: `python3 harness/tools/harness/gpu_check.py probe`. CUDA benchmark: `run.jl … --backend cuda --eltype all`.
- Benchmark: `julia --project=benchmark --threads=<T> benchmark/run.jl --suite <name> --out benchmark/results/<file>.json` (options: `--filter`, `--entries`, `--budget-scale`, `--list`; see the header of `run.jl`). Redirect output to a log file and read only its tail.
- Benchmark comparison: `julia --project=benchmark benchmark/compare.jl <a.json> <b.json>`; spread: `compare.jl --spread R1.json R2.json …`.
- Branch: `perf/phase0` (exists). Base ref: `20fc03b`.

## Julia Behavior & Tests *(required)*

No package code changes. The benchmark suite's correctness check (stored reference values, tolerances in `common.md`
§ Julia Behavior & Tests) must pass for every entry this sub-phase measures; an invalid entry may not be used.
`Pkg.test()` is not required in this sub-phase (no package code changes); it is run in `0h`.

## Algorithm ↔ Theorem Mapping *(required)*

none.

## CPU / GPU Matrix *(required)*

| Check | Backend | Required? | Notes |
|---|---|---|---|
| Package tests | CPU | no | no package code changes; run in `0h` |
| GPU probe | CUDA | yes | PASS → CUDA baseline; otherwise UNAVAILABLE recorded with the probe output (acceptable in Phase 0) |
| Benchmarks | CUDA | yes if the probe passes | Float64 + Float32, with re-run |

## Performance Evidence *(required)*

Spot-check only: QE re-runs a few CUDA cases (`--filter`/`--entries`), minutes not hours, and compares with `compare.jl` against the committed files; a difference is acceptable when within the noise bound or flagged CLOCK (`common.md` § Evidence Protocol).

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to the base ref: `git diff --quiet 20fc03b -- src ext` (and no uncommitted changes under them).
- [ ] The sub-phase commit contains only the files listed in § File List of this spec (staged by explicit path, never `git add -A`).
- [ ] Either the GPU probe passes and the CUDA baseline (Float64 and Float32) with a re-run, its reference values and `profiles/cuda-*.md` exist, or CUDA is recorded as UNAVAILABLE with the probe output.
- [ ] `REPORT.md` § 6.3 is complete (no `<!--CUDAPROF-->` placeholder); CUDA cases that cannot run are listed with the Finding that explains why.

## File List

- `benchmark/reference/cuda-Float64/**`, `benchmark/reference/cuda-Float32/**` — local only, not committed
- `benchmark/results/baseline-<sha>-cuda*.json`, `benchmark/results/reference-run-<sha>-cuda.json` — local only, not committed
- `benchmark/profiles/cuda-*.md`
- `benchmark/REPORT.md` § 6.3, CUDA rows of § 3 and § 4
- Raw `benchmark/results/**/*.json` and `benchmark/reference/**` are **local only** (gitignored; `common.md` § Committed vs local benchmark outputs): produce and use them, but do not stage them. Commit the Markdown summaries.

## Out of Scope

- Any change under `src/`, `ext/`, `test/` or `lean/`.
- Deliverables of other sub-phases (see `common.md` § Sub-phases (Phase 0)).
