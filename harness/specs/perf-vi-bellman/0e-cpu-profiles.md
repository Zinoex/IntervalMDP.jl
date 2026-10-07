# Performance of the VI/Bellman Algorithms — 0e: CPU profiles — Specification (Julia + Lean)

> Sub-phase **0e** of the performance work (Phase 0, measurement only). One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Sub-phases (Phase 0), § Phase 0 Deliverables (item 3), § Benchmark Suite (Case matrix).
> Previous sub-phase must be committed and pushed to `perf/phase0` first.

**Active phase: 0e**

## Objective *(required)*

Profile at least one case per case-matrix row on CPU: sampling profile, `Profile.Allocs`, steady-state allocation of one `bellman!` call and `JET.@report_opt` on the hot functions. Save summaries (not raw dumps) as `benchmark/profiles/<case>.md` and fill `REPORT.md` § 5.

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
- Profiling: `julia --project=benchmark --threads=<T> benchmark/profile.jl <case> …` (see its header). Write summaries to `benchmark/profiles/`; never print a raw profile into the conversation.
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
| Profiles | CPU 1 and 16 threads | yes | one per case-matrix row |
| GPU tests / CUDA benchmarks | CUDA | no | not part of this sub-phase |

## Performance Evidence *(required)*

Not in scope (profiles inform § 7, they are not evidence for a speedup). QE re-runs one profile and checks that its top frames match the committed summary.

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to the base ref: `git diff --quiet 20fc03b -- src ext` (and no uncommitted changes under them).
- [ ] The sub-phase commit contains only the files listed in § File List of this spec (staged by explicit path, never `git add -A`).
- [ ] `profiles/<case>.md` exists for at least one case per case-matrix row, each with a sampling-profile summary, a `Profile.Allocs` summary, the steady-state allocation of one `bellman!` call and `JET.@report_opt` on the hot functions.
- [ ] `REPORT.md` § 5 summarises where the time goes per profile.

## File List

- `benchmark/profile.jl`
- `benchmark/profiles/<case>.md` (CPU; not `cuda-*.md`)
- `benchmark/REPORT.md` § 5
- Raw `benchmark/results/**/*.json` and `benchmark/reference/**` are **local only** (gitignored; `common.md` § Committed vs local benchmark outputs): produce and use them, but do not stage them. Commit the Markdown summaries.

## Out of Scope

- Any change under `src/`, `ext/`, `test/` or `lean/`.
- Deliverables of other sub-phases (see `common.md` § Sub-phases (Phase 0)).
