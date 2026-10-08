# Lean Proofs for Existing VI/Bellman Algorithms — 1a: Index foundations — Specification (Julia + Lean)

> Sub-phase **1a** of Phase 1 (Indexing and O-maximization). One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Indexing Correctness, § Lean Model Structure & Readability (Style rules), § Proof Obligations & Limitations, § Findings policy.
> Branch `lean/phase-1`. First sub-phase of the phase: starts after the phases it depends on have merged.

**Active phase: 1a**

## Objective *(required)*

Model Julia's 1-based indices once (`Index/Julia.lean`: `toJulia : Fin n → ℕ := (· + 1)` and its inverse on `1..n`) and prove the index lemmas everything later builds on: column-major linear ↔ Cartesian indexing, sparse CSC support pairing, and the sort permutation used by O-max (the greedy loop visits every index once; stopping early skips only zero allocations). All index theorems are stated through `toJulia`.

The Julia code is not changed (`src/`, `ext/` byte-identical to the base ref). If a theorem as stated is false for the
Julia behaviour, follow `common.md` § Findings policy: minimal counterexample, record the Finding in the inventory, mark
the row `partial` with the strongest proved statement, do not weaken the theorem under its original name, end with QE
FAIL for that criterion (no loop). Follow `common.md` § Context budget. Ops: branch `lean/phase-1` from `origin/main`, commit, open a **draft** PR.

## Commands / Toolchain *(required)*

- Target root: `/home/fresen/.julia/dev/IntervalMDP`
- Julia version / compat: `[compat] julia = "1.11"`; local `julia` is 1.13.1 (satisfies compat).
- Julia instantiate: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- Julia test: `julia --project=. -e 'using Pkg; Pkg.test()'`; afterwards `git checkout -- test/data/multiObj_robotIMDP.nc` (see `harness/LEARNING.md`).
- Lean root: `<root>/lean` (lake package `IntervalMDPProofs`, library root `IntervalMDPProofs.lean`).
- Lean toolchain: pinned in Phase 0 to `leanprover/lean4:v4.33.0-rc2` (Mathlib v4.33.0-rc2). It must not change. Always call the toolchain binaries directly, never the elan proxy: `~/.elan/toolchains/leanprover--lean4---v4.33.0-rc2/bin/lake`.
- Lean build (from `lean/`): `~/.elan/toolchains/leanprover--lean4---v4.33.0-rc2/bin/lake build`
- Lean axiom check (from `lean/`): `~/.elan/toolchains/leanprover--lean4---v4.33.0-rc2/bin/lake env lean AxiomCheck.lean`
- Julia reference check: `python3 lean/scripts/check_julia_refs.py`
- GPU test: not applicable. Benchmark: not in scope.

## Julia Behavior & Tests *(required)*

No Julia source changes; `git diff <base> -- src/ ext/` is empty. `Pkg.test()` must pass (regression guard).
No cross-check test is due in this sub-phase.

## Algorithm ↔ Theorem Mapping *(required)*

The verifier checks exactly these theorem names (plus, via `AxiomCheck.lean`, that every theorem of earlier
sub-phases and phases is still proved). Names are fixed; to change one, amend this file and `common.md` before the run.

### Indexing

| Ph | Index computation | Julia (function — file) | Lean definition | Lean theorem(s) |
|---|---|---|---|---|
| 1 | Column-major linear index | `LinearIndices`, `CartesianIndices` (Base) as used for `V[I]` and states | `IntervalMDP.Index.linear` — `Index/Linear.lean` | `IntervalMDP.Index.linear_bijective`; `IntervalMDP.Index.linear_succ_first` (the first dimension varies fastest) |
| 1 | Sparse support pairing | `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` zip over `support`/`nonzeros` — `src/bellman.jl` | `IntervalMDP.Index.SparseCol` (CSC column invariant) — `Index/Sparse.lean` | `IntervalMDP.Index.sparse_zip_correct` |
| 1 | Sort permutation | `sortperm!(perm, V; rev = upper_bound)` — `src/bellman.jl` | `IntervalMDP.Index.SortedPerm` — `Index/Perm.lean` | `IntervalMDP.Index.sortedPerm_bijective`; `IntervalMDP.Index.greedy_visits_once` |

Theorems due in this sub-phase:

- `IntervalMDP.Index.linear_bijective`
- `IntervalMDP.Index.linear_succ_first`
- `IntervalMDP.Index.sparse_zip_correct`
- `IntervalMDP.Index.sortedPerm_bijective`
- `IntervalMDP.Index.greedy_visits_once`

## Proof Obligations & Limitations *(required)*

`common.md` § Proof Obligations & Limitations applies: domain = structure invariants only, plus the hypotheses the row
names; all four satisfaction × strategy modes or an explicit restriction; scope `abstract`; index theorems carry the
overflow bound as a hypothesis; the listed limitations are copied into the inventory for each row of this sub-phase.

## Proof Policy *(required)*

Approved axioms: `propext`, `Classical.choice`, `Quot.sound`. Forbidden: `sorry`, `admit`, `stop`; placeholder statements
(`: True`, conclusion = hypothesis); new `axiom`; `native_decide`/`decide := true` on non-trivial goals;
`@[implemented_by]`/`unsafe` in proof dependencies. No specialisation to concrete instances (only `Models/Examples.lean`).
Soundness theorems go through `IntervalMDP.Approx.iter_sound`. Mathlib lemmas count as proofs.

## CPU / GPU Matrix *(required)*

| Check | Backend | Required? | Notes |
|---|---|---|---|
| Package tests | CPU | yes | regression guard |
| Lean build + axiom check | — | yes | pinned toolchain |
| GPU tests | CUDA | no | no GPU code; CUDA kernels are a stated limitation |

## Performance Evidence

Not in scope.

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `Pkg.test()` passes on CPU; `git diff <base> -- src/ ext/` is empty.
- [ ] `lake build` in `lean/` succeeds with the pinned toolchain and emits no linter warnings in `IntervalMDPProofs/`; `lean-toolchain` is unchanged.
- [ ] Every theorem listed above exists under its exact name with a complete proof and no forbidden construct; its statement matches its row (structure invariants plus only the hypotheses the row names; all four modes or an explicit restriction; no vacuous hypotheses).
- [ ] `lean/AxiomCheck.lean` lists every theorem of this sub-phase and of all earlier ones; its output contains only approved axioms.
- [ ] Every new definition, structure and theorem has a docstring naming its Julia counterpart and file; names follow the Julia names; `python3 lean/scripts/check_julia_refs.py` passes. Lean functions that transcribe a Julia loop say so and keep its structure (loop order, `- 1`/`+ 1` steps).
- [ ] New files are imported from `IntervalMDPProofs.lean`; `lean/README.md` (glossary, file map) covers them.
- [ ] `harness/specs/inventory-intervalmdp.md` is updated for every row of this sub-phase: status, theorem links, scope `abstract`, limitations, Findings.

## File List

- `lean/IntervalMDPProofs/Index/{Julia,Linear,Sparse,Perm}.lean`
- `lean/AxiomCheck.lean`, `lean/IntervalMDPProofs.lean`, `lean/README.md`
- `harness/specs/inventory-intervalmdp.md` (rows of this sub-phase)

## Out of Scope

- Any change to `src/` or `ext/`, including fixes for Findings (follow-up specs).
- Floating-point / implementation-scope proofs; Lean→Julia extraction; path-measure semantics; MEC deflation for IVI.
- Rows of other sub-phases (see the run order in `common.md`).
