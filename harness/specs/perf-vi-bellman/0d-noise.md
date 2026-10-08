# Performance of the VI/Bellman Algorithms — 0d: Noise bound — Specification (Julia + Lean)

> Sub-phase **0d** of the performance work (Phase 0, measurement only). One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Sub-phases (Phase 0), § Evidence Protocol (Noise bound, Statistics).
> Previous sub-phase must be committed and pushed to `perf/phase0` first.

**Active phase: 0d**

## Objective *(required)*

Measure the run-to-run spread of every baseline entry at every thread count, find the cause of every entry above 5%, fix it (budgets, sizes) and re-measure. Fill `REPORT.md` § 4, including the result table at the `<!--NOISEFIX-->` placeholder. Measurements in `benchmark/results/noisefix/` already exist; check them and summarise rather than re-run unless they are invalid.

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
| Benchmarks | CPU 1/4/8/16 threads | yes | noisy entries only |
| GPU tests / CUDA benchmarks | CUDA | no | not part of this sub-phase |

## Performance Evidence *(required)*

Spot-check only: QE re-runs a few cases per family (`--filter`/`--entries`), minutes not hours, and compares with `compare.jl` against the committed files; a difference is acceptable when within the noise bound or flagged CLOCK (`common.md` § Evidence Protocol). QE recomputes the spread table from the committed files with `compare.jl --spread`.

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to the base ref: `git diff --quiet 20fc03b -- src ext` (and no uncommitted changes under them).
- [ ] The sub-phase commit contains only the files listed in § File List of this spec (staged by explicit path, never `git add -A`).
- [ ] The spread of every entry at every thread count is computed; entries with spread > 5% are listed with their cause (`results/noise-before-fix.md`).
- [ ] Each of them is fixed (budgets, sizes) and re-measured, and the `<!--NOISEFIX-->` placeholder is replaced by the result table; any entry still > 5% is named as not usable as evidence, with the reason.

## File List

- `benchmark/cases/registry.jl` (sampling budgets only)
- `benchmark/results/noise-before-fix.md`, `benchmark/results/noisefix/*.json` — local only, not committed
- `benchmark/REPORT.md` § 4
- Raw `benchmark/results/**/*.json` and `benchmark/reference/**` are **local only** (gitignored; `common.md` § Committed vs local benchmark outputs): produce and use them, but do not stage them. Commit the Markdown summaries.

## Out of Scope

- Any change under `src/`, `ext/`, `test/` or `lean/`.
- Deliverables of other sub-phases (see `common.md` § Sub-phases (Phase 0)).
