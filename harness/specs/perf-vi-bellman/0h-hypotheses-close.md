# Performance of the VI/Bellman Algorithms — 0h: Hypotheses and Phase 0 close — Specification (Julia + Lean)

> Sub-phase **0h** of the performance work (Phase 0, measurement only). One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Sub-phases (Phase 0), § Phase 0 Deliverables, § Candidate Hypotheses, § Findings policy.
> Previous sub-phase must be committed and pushed to `perf/phase0` first.

**Active phase: 0h**

## Objective *(required)*

Close Phase 0: write the ranked hypothesis list (`REPORT.md` § 7), the Findings (§ 8) and Limitations (§ 9); make one consistency pass over `REPORT.md` (no placeholders, every number traceable to a file under `benchmark/results/` or `benchmark/profiles/`); run `Pkg.test()`; check that every Phase 0 deliverable exists on `perf/phase0`. Ops pushes and marks the PR ready (`gh pr ready`).

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
- Julia test: `julia --project=. -e 'using Pkg; Pkg.test()'`; multi-threaded: `julia --project=. --threads=auto -e 'using Pkg; Pkg.test()'`.
- After every test run: `git checkout -- test/data/multiObj_robotIMDP.nc` (see `harness/LEARNING.md`).
- Benchmark comparison: `julia --project=benchmark benchmark/compare.jl <a.json> <b.json>`; spread: `compare.jl --spread R1.json R2.json …`.
- Branch: `perf/phase0` (exists). Base ref: `20fc03b`.

## Julia Behavior & Tests *(required)*

No package code changes. The benchmark suite's correctness check (stored reference values, tolerances in `common.md`
§ Julia Behavior & Tests) must pass for every entry this sub-phase measures; an invalid entry may not be used.
`Pkg.test()` with 1 thread and with `--threads=auto` must pass.

## Algorithm ↔ Theorem Mapping *(required)*

none.

## CPU / GPU Matrix *(required)*

| Check | Backend | Required? | Notes |
|---|---|---|---|
| Package tests | CPU, 1 thread | yes | Julia test command |
| Package tests | CPU, `--threads=auto` | yes | multi-threaded Julia test command |
| GPU tests / CUDA benchmarks | CUDA | no | not part of this sub-phase |

## Performance Evidence *(required)*

Each hypothesis cites measurements from §§ 3–6; no new measurements are required. QE checks a sample of the cited numbers against the files.

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to the base ref: `git diff --quiet 20fc03b -- src ext` (and no uncommitted changes under them).
- [ ] The sub-phase commit contains only the files listed in § File List of this spec (staged by explicit path, never `git add -A`).
- [ ] `REPORT.md` § 7 has a ranked hypothesis list; each entry names its target cases, cites at least one measurement from §§ 3–6, the change, the predicted effect and the phase.
- [ ] Findings and limitations are recorded (§ 8, § 9); `REPORT.md` has no placeholders, and every number in it can be traced to a file in `benchmark/results/` or `benchmark/profiles/`.
- [ ] `Pkg.test()` passes with 1 thread and with `--threads=auto`.
- [ ] All Phase 0 Deliverables (§ Phase 0 Deliverables) exist on `perf/phase0`, and the PR is marked ready.

## File List

- `benchmark/REPORT.md` §§ 7–9 and consistency edits anywhere in it (wording and references only; numbers only to fix a mismatch with the files)
- Raw `benchmark/results/**/*.json` and `benchmark/reference/**` are **local only** (gitignored; `common.md` § Committed vs local benchmark outputs): produce and use them, but do not stage them. Commit the Markdown summaries.

## Out of Scope

- Any change under `src/`, `ext/`, `test/` or `lean/`.
- Deliverables of other sub-phases (see `common.md` § Sub-phases (Phase 0)).
