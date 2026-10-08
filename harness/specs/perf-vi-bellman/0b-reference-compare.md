# Performance of the VI/Bellman Algorithms — 0b: Reference values and comparison tool — Specification (Julia + Lean)

> Sub-phase **0b** of the performance work (Phase 0, measurement only). One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Sub-phases (Phase 0), § Benchmark Suite (Correctness check), § Evidence Protocol, § Julia Behavior & Tests (tolerances).
> Previous sub-phase must be committed and pushed to `perf/phase0` first.

**Active phase: 0b**

## Objective *(required)*

Store reference results from the base ref so every benchmark run checks its own correctness, and provide the comparison tool that turns two result files into evidence verdicts. Fill the reference paragraph of `REPORT.md` § 2.

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
- Write reference: `run.jl --suite full --write-reference --reference-only` (refuses when `src/`/`ext/` differ from the base ref).
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
| Benchmarks | CPU 1 thread | yes | `--suite quick` with correctness check |
| GPU tests / CUDA benchmarks | CUDA | no | not part of this sub-phase |

## Performance Evidence *(required)*

Not in scope (tooling only).

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to the base ref: `git diff --quiet 20fc03b -- src ext` (and no uncommitted changes under them).
- [ ] The sub-phase commit contains only the files listed in § File List of this spec (staged by explicit path, never `git add -A`).
- [ ] `reference/cpu-Float64/` was written from the base ref by `run.jl --write-reference`, which refuses to run when `src/`/`ext/` differ from the base ref.
- [ ] A `--suite quick` run reports `valid = true` for every entry; a deliberately perturbed value is reported invalid (demonstrated once, not committed).
- [ ] `compare.jl A.json B.json` prints median, ratio and a verdict (speedup / regression / no change / CLOCK) per entry, using the thresholds of § Evidence Protocol; `compare.jl --spread` prints the per-entry spread.

## File List

- `benchmark/lib/reference.jl`, the reference options in `benchmark/run.jl`
- `benchmark/reference/cpu-Float64/**` — local only, not committed
- `benchmark/compare.jl`
- `benchmark/README.md` (reference and comparison paragraphs only)
- `benchmark/REPORT.md` § 2 (reference paragraph)
- Raw `benchmark/results/**/*.json` and `benchmark/reference/**` are **local only** (gitignored; `common.md` § Committed vs local benchmark outputs): produce and use them, but do not stage them. Commit the Markdown summaries.

## Out of Scope

- Any change under `src/`, `ext/`, `test/` or `lean/`.
- Deliverables of other sub-phases (see `common.md` § Sub-phases (Phase 0)).
