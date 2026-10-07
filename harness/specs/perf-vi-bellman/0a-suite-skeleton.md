# Performance of the VI/Bellman Algorithms — 0a: Suite skeleton — Specification (Julia + Lean)

> Sub-phase **0a** of the performance work (Phase 0, measurement only). One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Sub-phases (Phase 0), § Benchmark Suite, § Evidence Protocol (Environment bullet).
> Previous sub-phase must be committed and pushed to `perf/phase0` first.

**Active phase: 0a**

## Objective *(required)*

Provide a benchmark suite that runs on the current API: the benchmark environment, seeded case generators for the whole case matrix, the measurement and environment-capture library, the `run.jl` entry point and the README. Replace the stale `benchmark/bellman/` and `benchmark/imdp/` scripts (they use the removed `IntervalProbabilities`/`construct_ordering` API). Fill `REPORT.md` § 1 Environment and § 2 Suite. Ops creates/pushes `perf/phase0` and opens a **draft** PR.

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
| Benchmarks | CPU 1 and 4 threads | yes | `--suite quick` |
| GPU tests / CUDA benchmarks | CUDA | no | not part of this sub-phase |

## Performance Evidence *(required)*

Not in scope beyond the `--suite quick` smoke runs (no timings are used as evidence yet).

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to the base ref: `git diff --quiet 20fc03b -- src ext` (and no uncommitted changes under them).
- [ ] The sub-phase commit contains only the files listed in § File List of this spec (staged by explicit path, never `git add -A`).
- [ ] The stale scripts (`benchmark/bellman/`, `benchmark/imdp/`) are removed; `benchmark/Project.toml` has no Revise, Cthulhu or ProfileView, and `Manifest.toml` is committed.
- [ ] `run.jl --list` prints every case of the case matrix (§ Benchmark Suite) with its stable name; deviations from the matrix are listed in `README.md`.
- [ ] `run.jl --suite quick` runs on the current API at 1 and 4 threads and writes JSON with all statistics fields and the full environment block (§ Benchmark Suite → Output, incl. thread-to-core mapping).
- [ ] `README.md` explains how to run and read the suite.

## File List

- `benchmark/Project.toml`, `benchmark/Manifest.toml`
- removal of `benchmark/bellman/`, `benchmark/imdp/`
- `benchmark/cases/*.jl`, `benchmark/lib/measure.jl`, `benchmark/lib/environment.jl`, `benchmark/lib/clock.jl`
- `benchmark/run.jl`, `benchmark/README.md`
- `benchmark/REPORT.md` § 1, § 2 (create the file with the section skeleton §§ 1–9 if missing)

## Out of Scope

- Any change under `src/`, `ext/`, `test/` or `lean/`.
- Deliverables of other sub-phases (see `common.md` § Sub-phases (Phase 0)).
