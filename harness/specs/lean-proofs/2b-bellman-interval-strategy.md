# Lean Proofs for Existing VI/Bellman Algorithms — 2b: Interval specialisation, strategies, policy evaluation; Phase 2 close — Specification (Julia + Lean)

> Sub-phase **2b** of Phase 2 (Robust Bellman operator). One `/harness` run.
> Shared rules: [`common.md`](common.md). Agents read **only** these sections of it: § Context budget, § Lean Model Structure & Readability (Style rules), § Approximation Soundness, § Proof Obligations & Limitations, § Findings policy.
> Branch `lean/phase-2`. The previous sub-phase of this phase must be pushed first.

**Active phase: 2b**

## Objective *(required)*

Specialise `T` to IMDPs (`T_interval_eq_omax`: for `IMDP.toRMDP`, `T` agrees with `omax` looked up via `marginalSub2ind`), prove strategy lookup returns an available action (`Index/Strategy.lean`), that `argoptAction` attains the opt, and that policy evaluation `Tπ` is exact for a given strategy and sound w.r.t. `T` in the strategy-mode direction (A8). Close Phase 2.

The Julia code is not changed (`src/`, `ext/` byte-identical to the base ref). If a theorem as stated is false for the
Julia behaviour, follow `common.md` § Findings policy: minimal counterexample, record the Finding in the inventory, mark
the row `partial` with the strongest proved statement, do not weaken the theorem under its original name, end with QE
FAIL for that criterion (no loop). Follow `common.md` § Context budget. Ops: commit and push to `lean/phase-2` (same PR), then mark the PR ready (`gh pr ready`).

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
| 2 | Strategy lookup | `CartesianIndex(strategy_cache[jₛ])` — `src/bellman.jl`, `src/strategy_cache.jl` | `IntervalMDP.Index.strategyAction` — `Index/Strategy.lean` | `IntervalMDP.Index.strategyAction_available` |

### Algorithms

| Ph | Algorithm | Julia entry point (function — file) | Lean definition (name — file) | Lean theorem(s) | Correspondence note |
|---|---|---|---|---|---|
| 2 | Robust Bellman operator | `bellman!` → `state_bellman!` with `OptimizingStrategyCache` — `src/bellman.jl` | `IntervalMDP.Bellman.T` — `Bellman.lean` (on `RMDP`) | `IntervalMDP.Bellman.T_interval_eq_omax` | Proved for general `RMDP`, then specialised. |
| 2 | Strategy extraction and evaluation | `extract_strategy!` — `src/strategy_cache.jl`; `NonOptimizingStrategyCache` path of `state_bellman!` | `IntervalMDP.Bellman.argoptAction`, `IntervalMDP.Bellman.Tπ` — `Bellman.lean` | `IntervalMDP.Bellman.argopt_attains`; `IntervalMDP.Bellman.policy_eval_sound` (A8) | — |

Theorems due in this sub-phase:

- `IntervalMDP.Index.strategyAction_available`
- `IntervalMDP.Bellman.T_interval_eq_omax`
- `IntervalMDP.Bellman.argopt_attains`
- `IntervalMDP.Bellman.policy_eval_sound`

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
- [ ] Phase close: every theorem of every Phase 2 row (all sub-phases) exists and is proved; `lean/README.md` is current for the whole phase; the inventory has no Phase 2 row left at `none` (each is `proved` or `partial` with a Finding).

## File List

- `lean/IntervalMDPProofs/Bellman.lean` (interval and strategy parts)
- `lean/IntervalMDPProofs/Index/Strategy.lean`
- `lean/AxiomCheck.lean`, `lean/IntervalMDPProofs.lean`, `lean/README.md`
- `harness/specs/inventory-intervalmdp.md` (rows of this sub-phase)

## Out of Scope

- Any change to `src/` or `ext/`, including fixes for Findings (follow-up specs).
- Floating-point / implementation-scope proofs; Lean→Julia extraction; path-measure semantics; MEC deflation for IVI.
- Rows of other sub-phases (see the run order in `common.md`).
