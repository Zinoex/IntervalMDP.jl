# Performance of the VI/Bellman Algorithms — 0f: Scaling and roofline — Specification (Julia + Lean)

> Sub-phase **0f** of the performance work (Phase 0, measurement only). One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Sub-phases (Phase 0), § Phase 0 Deliverables (item 4).
> Previous sub-phase must be committed and pushed to `perf/phase0` first.

**Active phase: 0f**

## Objective *(required)*

Measure strong scaling (fixed size, threads 1→16) and size scaling (fixed threads, increasing `n`) for dense and sparse O-max, the machine roofs (`stream.jl`), and classify each case as memory-, compute- or overhead-bound. Fill `REPORT.md` § 6.1 and § 6.2.

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
- Benchmark: `julia --project=benchmark --threads=<T> benchmark/run.jl --suite <name> --out benchmark/results/<file>.json` (options: `--filter`, `--entries`, `--budget-scale`, `--list`; see the header of `run.jl`). Redirect output to a log file and read only its tail. Suites `scaling` and `sizes`.
- Roofs: `julia --project=benchmark --threads=<T> benchmark/stream.jl --out benchmark/results/stream-t<T>.json`.
- Tables: `julia --project=benchmark benchmark/analyze.jl` → `benchmark/results/scaling-roofline-<sha>.md`.
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
| Benchmarks | CPU 1–16 threads | yes | `scaling`, `sizes`, `stream.jl` |
| GPU tests / CUDA benchmarks | CUDA | no | not part of this sub-phase |

## Performance Evidence *(required)*

Spot-check only: QE re-runs a few cases per family (`--filter`/`--entries`), minutes not hours, and compares with `compare.jl` against the committed files; a difference is acceptable when within the noise bound or flagged CLOCK (`common.md` § Evidence Protocol).

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to the base ref: `git diff --quiet 20fc03b -- src ext` (and no uncommitted changes under them).
- [ ] The sub-phase commit contains only the files listed in § File List of this spec (staged by explicit path, never `git add -A`).
- [ ] Strong scaling (threads 1→16) and size scaling exist for dense and sparse O-max, with measured machine roofs (`stream.jl`).
- [ ] `REPORT.md` § 6.1–6.2 classify each case as memory-, compute- or overhead-bound, with the numbers.

## File List

- `benchmark/stream.jl`, `benchmark/analyze.jl`
- `benchmark/results/scaling-*.json`, `benchmark/results/sizes-*.json`, `benchmark/results/stream-*.json`, `benchmark/results/scaling-roofline-<sha>.md` — local only, not committed
- `benchmark/REPORT.md` § 6.1, § 6.2
- Raw `benchmark/results/**/*.json` and `benchmark/reference/**` are **local only** (gitignored; `common.md` § Committed vs local benchmark outputs): produce and use them, but do not stage them. Commit the Markdown summaries.

## Out of Scope

- Any change under `src/`, `ext/`, `test/` or `lean/`.
- Deliverables of other sub-phases (see `common.md` § Sub-phases (Phase 0)).
