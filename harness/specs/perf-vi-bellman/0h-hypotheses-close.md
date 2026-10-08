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

## Carry-over from 0a–0g *(required — Dev resolves each item, QE checks each one)*

Items that earlier sub-phases noted "for 0h" (sources: the 0a–0g sections of PR #112 and the commit messages of
`a9dc02e`, `4a42913`, `5f25007`, `87f1cc8`, `7b28f14`). Line numbers are approximate; find the text by `grep`.
Every item ends in exactly one of two states, which the hand-off states per item:

- **Fixed** — for the `REPORT.md` items in group A (they are within this sub-phase's File List).
- **Recorded** — for groups B and C: a § 9 Limitation, a § 8 Finding, or a § 7 hypothesis/prerequisite naming the
  phase that will do it. Where a later spec already handles an item, it is named after the arrow (→); for items
  without an arrow (8, 9, 10, 18), 0h names the phase or records them as open limitations. Code and tooling changes (`compare.jl`, `lib/clock.jl`, the registry,
  re-measurements) are **not** in this sub-phase's File List and must not be made here.

**A. `REPORT.md` corrections (fix in 0h)**

1. § 6.2 sparse bullet says "≤ 8% of any roof"; § 6.1 shows sparse k100 at 9% (t10/t14) → "≤ 9%". *(0f)*
2. § 9 (~line 455) "26–37% of the 1-thread DRAM roof" is stale; the 0f roofs give 26–28%. *(0f)*
3. § 3 CUDA sentence "Float64 at the floor in all three, so the floor is reproducible per entry, not random" is
   overstated: a 0g QE process ran dense n1000 a1 Float64 `bellman` at 0.193 ms vs ≈ 6 ms. Reword as bimodal across
   processes. *(0g)*
4. § 5 gives `fimdp-sparse-v3-d20-k5` `sort!` as 11–13%; 0e QE recomputed 14.2%. Check against the profile and fix. *(0e)*
5. § 8 Finding C-1 is marked "draft" and has no category (the Findings policy defines B, I, S, T). Assign a category,
   renumber if needed, drop "draft", and keep the title "CUDA coverage gaps" (`3f.3-c1-cuda-coverage.md` finds it by
   that title). *(0g)*
6. § 8 "Environment" bullet: 0g added an NVML clause (`nvidia-smi` fails with a driver/library mismatch, so no GPU
   clock/power data). Keep it (and move it to § 9 if it is a limitation rather than a finding) or drop it. *(0g)*
7. `profiles/cuda-*.md` headers say "calling CUDA APIs took 4.27 ms (28.85%)" while the listed calls sum to ≈ 0.17 ms,
   and § 6.3 says the ≈ 3 ms floor "is not in any CUDA API call". Make the § 6.3 wording consistent with the trace
   headers (wording only; do not regenerate profiles). *(0g)*

**B. Measurement validity (record in § 9, and in § 7 where a hypothesis depends on it)**

8. Stored CPU baselines (§ 3), noisefix files (§ 4), scaling/sizes files (§ 6) and CUDA baselines (§ 3/§ 4/§ 6.3) were
   recorded at `91bc0c8` with an uncommitted benchmark harness (src/ext equal to `20fc03b`). The 0d spot-check against
   the CPU baseline failed systematically (e.g. robot bellman ×1.186, product bellman ×1.152) with sample counts very
   different from the committed harness. Re-baseline with the committed harness before Phase 1 A/B work. *(0c/0d/0f/0g)*
9. Sparse k10 superlinear strong scaling at t12/t14 (×15.05/×17.57, eff. 125%) coincides with the 557 µs clock state;
   0f QE saw sparse k10 ×1.6 faster in that state at t8. Not evidence; re-measure in one clock state. *(0f)*
10. Machine roofs (§ 6) are the maximum over clock states; single-round in-cache roofs vary ≈ ×2.2 between states, so
    "% of roof" is a lower bound. Per-state roofs would be better. *(0f)*
11. CUDA noise is not fixed: 47/128 entries spread > 5% over baseline/rerun1/rerun2, some flipping between kernel speed
    and the ≈ 6 ms floor; the floor process got only 10–12 samples vs 311. *(0g)* → noise check in `3a-prepare.md`;
    the floor itself is H5 (`3b.1`).
12. CUDA "cannot run" cases (C-1: 7 McCormick/vertex, 6 product/DFA; B-2: 28 `solve_ivi`, a hard-coded skip at
    `run.jl:177`) are asserted from registry exclusions and skips, not measured; whether they throw or fall back to the
    CPU is unknown. *(0g)* → measured and resolved in `3f.3-c1-cuda-coverage.md` (C-1); B-2 in `3f.1`.
13. `bellman_cs` (control-synthesis) correctness compares against the same kernel with the default cache — no stored
    reference. A registry `bellman` entry for `cs-*` cases is needed. The older t1 profile sections (from `91bc0c8`)
    have no inline correctness line. *(0e)* → registry entry added in `4a-prepare.md` (step 6).
14. 72 entries are too noisy for single-pair evidence (§ 4); a single a/b pair spread is a lower bound on
    process-to-process noise. *(0d)* → noise check in every `Na-prepare.md`.

19. CUDA times are host wall-clock times around a call that ends in `CUDA.synchronize()`; no CUDA event timing and no
    comparison with blocking synchronisation was made. The size-independent ≈ 6 ms floor, the bimodal entries and minima
    of 0.29–3.3 ms suggest the floor may come (partly) from CUDA.jl's non-blocking, yielding synchronisation rather than
    from the package. Record in § 9, and in § 7 that H5's prediction depends on it. *(0g)* → `3a-prepare.md` (steps 7, 9).

**C. Benchmark tooling defects (record in § 9 with the phase that fixes them)**

15. `compare.jl` gives speedup/regression verdicts from a single A/B round without warning; the Evidence Protocol needs
    ≥ 3 interleaved rounds. *(0b)* → `ab.jl` (first prepare sub-phase) and a round-count warning in `3a-prepare.md` (step 6).
16. `compare.jl`'s CLOCK verdict uses only the CPU probe: a ×30 GPU floor flip was labelled CLOCK. CUDA comparisons need
    a GPU-aware check. *(0g)* → `3a-prepare.md` (step 6).
17. `lib/clock.jl` takes the run-start reference unloaded (so nearly every t8/t16 entry is flagged CLOCK) and, at t1,
    while the core still boosts (≈ 197 µs vs 436 µs; false CLOCK flag on `scaling-…-t1.json`). Take the reference
    after warm-up and under the same thread load. *(0c/0f)* → whichever prepare sub-phase runs first (step 5 of every
    `Na-prepare.md`).
18. t16 CPU profiles are dominated by idle `wait()` (68–80%); a busy-only or per-thread view is needed. *(0e)*
20. CUDA profiles are `CUDA.@profile` trace summaries only: no Nsight Systems timeline, no NVTX ranges, no Nsight Compute
    kernel analysis (occupancy, achieved bandwidth vs peak, registers/shared memory, limiter), so there is no
    kernel-level CUDA counterpart of the § 6 roofline. `nsys` and `ncu` are installed. *(0g)* → `3a-prepare.md` (step 8).

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `src/` and `ext/` are byte-identical to the base ref: `git diff --quiet 20fc03b -- src ext` (and no uncommitted changes under them).
- [ ] The sub-phase commit contains only the files listed in § File List of this spec (staged by explicit path, never `git add -A`).
- [ ] `REPORT.md` § 7 has a ranked hypothesis list; each entry names its target cases, cites at least one measurement from §§ 3–6, the change, the predicted effect and the phase.
- [ ] Findings and limitations are recorded (§ 8, § 9); `REPORT.md` has no placeholders, and every number in it can be traced to a file in `benchmark/results/` or `benchmark/profiles/`.
- [ ] Every item in § Carry-over from 0a–0g is either fixed (group A) or recorded in `REPORT.md` § 7/§ 8/§ 9 with the phase that will handle it (groups B, C); the hand-off lists each item number with its state and location.
- [ ] `Pkg.test()` passes with 1 thread and with `--threads=auto`.
- [ ] All Phase 0 Deliverables (§ Phase 0 Deliverables) exist on `perf/phase0`, and the PR is marked ready.

## File List

- `benchmark/REPORT.md` §§ 7–9 and consistency edits anywhere in it (wording and references only; numbers only to fix a mismatch with the files)
- Raw `benchmark/results/**/*.json` and `benchmark/reference/**` are **local only** (gitignored; `common.md` § Committed vs local benchmark outputs): produce and use them, but do not stage them. Commit the Markdown summaries.

## Out of Scope

- Any change under `src/`, `ext/`, `test/` or `lean/`.
- Deliverables of other sub-phases (see `common.md` § Sub-phases (Phase 0)).
