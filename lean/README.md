# IntervalMDPProofs — Lean models and proofs for IntervalMDP.jl

This Lake package contains Lean 4 models of the objects IntervalMDP.jl operates on, and
machine-checked proofs about them. It is developed in phases (see
`harness/specs/lean-proofs-existing-algorithms.md`); the status of every row is tracked in
`harness/specs/inventory-intervalmdp.md`.

**Current phase: 0** — models with their well-formedness theorems, and the generic
approximation-soundness lift (A1). No algorithm (O-max, Bellman, VI, IVI) is proved yet.

**Proof scope: abstract.** Values are real numbers and the semantics is the dynamic-programming
recursion. The proofs do not cover floating-point rounding, overflow, CUDA kernels or threaded
execution, and the Lean ↔ Julia correspondence is argued by transcription and docstrings, not
proved.

## Build

The toolchain is pinned by `lean-toolchain` (`leanprover/lean4:v4.33.0-rc2`), copied verbatim from
the Mathlib release tag `v4.33.0-rc2` pinned in `lakefile.toml`. Do not change one without the other.

```sh
cd lean
lake exe cache get           # once: download Mathlib's build cache
lake build                   # builds the library; must finish with no warnings
lake env lean AxiomCheck.lean   # #print axioms for every mapped theorem
lake env lean DocLint.lean      # docBlame / docBlameThm: every declaration has a docstring
python3 scripts/check_julia_refs.py   # every declaration docstring names its Julia counterpart
```

`scripts/check_julia_refs.py` (no Lean needed; runs from any directory) checks every declaration
docstring (`/-- … -/` directly before a `def`, `theorem`, `structure`, `inductive`, `abbrev`,
`instance`, …) under `IntervalMDPProofs/`. Each one must contain a paragraph that starts with
`Julia counterpart:` and either cites the Julia source file, as in
``Julia counterpart: `budget` … (`src/workspace.jl`)``, or reads exactly
`Julia counterpart: none (Lean-side proof device).` for pure-Lean helpers. Every `src/….jl` path
cited in the Lean sources must exist. The script exits nonzero and lists the offenders otherwise;
`--verbose` prints every checked declaration and the files it cites.

Approved axioms: `propext`, `Classical.choice`, `Quot.sound`. On this host `~/.elan/bin` is not
on `PATH`; call the pinned toolchain directly
(`~/.elan/toolchains/leanprover--lean4---v4.33.0-rc2/bin/lake`) to avoid the elan proxy.

## Model hierarchy

Everything converts to `RMDP` through `toRMDP`; the general Bellman/VI theorems of later phases are
proved once, for `RMDP`.

```
ProbVec S                      distributions on a finite type S
 └─ AmbiguitySet S             a set of ProbVecs; WellFormed = nonempty ∧ closed
     │                         (IsConvex is a separate predicate, not part of WellFormed)
     ├─ IntervalAmbiguity.toSet       P(l, u) = {p : l ≤ p ≤ u}   (well formed and convex)
     ├─ AmbiguitySet.pi               literal product of marginals (well formed; not convex in general)
     └─ AmbiguitySet.map              image under a linear map (product with a DFA)

AvailableActions S A           nonempty available action sets
 └─ RMDP S A                   available + well-formed ambiguity s a
     ▲ IMDP.toRMDP             IMDP S A         (interval set per (s, a))
     ▲ FactoredIMDP.toRMDP     FactoredIMDP n m (Marginal per state variable; ambiguity = productSet)
     ▲ ProductProcess.toRMDP   ProductProcess S A Q Λ (RMDP × DFA × labelling) on S × Q

DFA Q Λ, Labelling S Λ, ProbLabelling S Λ, AbstractLabelling S Λ
StationaryStrategy S A, TimeVaryingStrategy S A (+ Valid w.r.t. available)
Property S Q, SatisfactionMode, StrategyMode, Specification S Q
Approx.Sound m W V, Approx.StepSound m T' T, Approx.iter_sound (A1)
```

## Julia ↔ Lean glossary

| Julia (file) | Lean (file) |
|---|---|
| probability column of `IntervalAmbiguitySets` | `IntervalMDP.ProbVec` (`Models/Distribution.lean`) |
| `AbstractAmbiguitySet` (`src/probabilities/probabilities.jl`) | `IntervalMDP.AmbiguitySet`, `AmbiguitySet.WellFormed` (nonempty, closed) (`Models/AmbiguitySet.lean`) |
| `PolytopicAmbiguitySet` (convex polytope) | `AmbiguitySet.IsConvex`; for intervals `IntervalAmbiguity.toSet_convex` |
| `IntervalAmbiguitySet` (`src/probabilities/IntervalAmbiguitySets.jl`) | `IntervalMDP.IntervalAmbiguity` (`Models/IntervalAmbiguity.lean`) |
| `lower`, `upper`, `gap` | `IntervalAmbiguity.lower`, `.upper`, `.gap` |
| `budget = 1 .- sum(lower; dims = 1)` (`src/workspace.jl`) | `IntervalAmbiguity.budget` |
| `checkprobabilities` | the fields of `IntervalAmbiguity` |
| feasible set of an `IntervalAmbiguitySet` | `IntervalAmbiguity.toSet` |
| `IntervalAmbiguitySets(; lower, upper)` (matrix of columns) | `Examples.IntervalAmbiguitySets` (`Models/Examples.lean`) |
| `AbstractAvailableActions`, `AllAvailableActions`, `ListAvailableActions` (`src/available_actions.jl`) | `IntervalMDP.AvailableActions`, `AvailableActions.all` (`Models/RMDP.lean`) |
| `TimeVaryingAvailableActions` | `IntervalMDP.TimeVaryingAvailableActions`, `RMDP.atTime` |
| `FactoredRobustMarkovDecisionProcess` as a flat RMDP (`IsRMDP`) | `IntervalMDP.RMDP` (`Models/RMDP.lean`) |
| `IntervalMarkovDecisionProcess(...)` (`src/models/IntervalMarkovDecisionProcess.jl`) | `IntervalMDP.IMDP`, `IMDP.toRMDP` (`Models/IMDP.lean`) |
| `IntervalMarkovChain(...)` (`src/models/IntervalMarkovChain.jl`) | `IntervalMDP.IMC` (= `IMDP S Unit`) |
| `state_vars`, `action_vars` | `IntervalMDP.StateVars`, `IntervalMDP.ActionVars` (`Models/Factored.lean`) |
| `Marginal` (`src/probabilities/Marginal.jl`): `state_indices`, `action_indices`, `ambiguity_sets` | `IntervalMDP.Marginal`: `stateIndices`, `actionIndices`, `sets` |
| `marginal[a, s]` (`getindex`) | `Marginal.get s a` |
| `FactoredRobustMarkovDecisionProcess` (`IsFIMDP`): `transition` / `marginals(mdp)` | `IntervalMDP.FactoredIMDP`: `marginals`; `FactoredIMDP.marginalSet`, `.toRMDP` |
| `Γ_{s,a} = ⨂ᵢ Γⁱ` (factored ambiguity set; not convex) | `FactoredIMDP.productSet` = `AmbiguitySet.pi`, used literally by `toRMDP`; `FactoredIMDP.productSet_not_convex` |
| `DFA` (`src/models/DFA.jl`): `transition`, `initial_state` | `IntervalMDP.DFA`: `δ`, `q₀`, `accepting` (`Models/DFA.lean`) |
| `DeterministicLabelling`, `ProbabilisticLabelling`, `AbstractLabelling` | `Labelling`, `ProbLabelling`, `AbstractLabelling` |
| `ProductProcess` (`src/models/ProductProcess.jl`): `mdp`, `dfa`, `labelling_func` | `IntervalMDP.ProductProcess`: `mdp`, `dfa`, `labelling`; `.toRMDP` (`Models/Product.lean`) |
| `V[idx, dfa[state, lf[idx]]]` (`src/bellman.jl`) | `ProductProcess.nextDFA`, `ProductProcess.lift_deterministic` |
| `StationaryStrategy`, `TimeVaryingStrategy` (`src/strategy.jl`) | `StationaryStrategy`, `TimeVaryingStrategy`, `.Valid` (`Models/Strategy.lean`) |
| property structs (`src/specification.jl`) | constructors of `IntervalMDP.Property` (`Models/Specification.lean`) |
| `time_horizon`, `convergence_eps`, `avoid_states` | `timeHorizon`, `convergenceEps`, `avoidStates` |
| `isfinitetime` | `Property.isFiniteTime` |
| `SatisfactionMode` (`Pessimistic`/`Optimistic`), `StrategyMode` (`Maximize`/`Minimize`) | `SatisfactionMode.pessimistic`/`.optimistic`, `StrategyMode.maximize`/`.minimize` |
| `Specification` (`prop`, `satisfaction`, `strategy`) | `IntervalMDP.Specification` |
| conservative approximation | `IntervalMDP.Approx.Sound` (`Approx/Sound.lean`) |
| repeated `bellman!` in `_value_iteration!` | `IntervalMDP.Approx.iter_sound`, `fixedPoint_sound` (`Approx/Lift.lean`) |

Indices: Julia is 1-based, Lean's `Fin n` is 0-based. Concrete examples and factored variable
values use `Fin` with Julia index `k` ↦ Lean `k - 1`; the formal conversion and the index
theorems come in Phase 1 (`Index/Julia.lean`).

## File map

| File | Contents |
|---|---|
| `IntervalMDPProofs.lean` | library root, imports every module |
| `IntervalMDPProofs/Models/Distribution.lean` | `ProbVec`, `dirac`, `map`, `pi`, `prodKernel` |
| `IntervalMDPProofs/Models/AmbiguitySet.lean` | `AmbiguitySet`, `WellFormed` (nonempty, closed), `IsConvex`, `map`, `pi`; `WellFormed.map`, `WellFormed.pi` |
| `IntervalMDPProofs/Models/IntervalAmbiguity.lean` | `IntervalAmbiguity`, `gap`, `budget`, `toSet`; `toSet_wellFormed`, `toSet_convex`, `budget_eq` |
| `IntervalMDPProofs/Models/RMDP.lean` | `AvailableActions`, `TimeVaryingAvailableActions`, `RMDP` |
| `IntervalMDPProofs/Models/IMDP.lean` | `IMDP`, `IMC`, `IMDP.toRMDP`; `IMDP.toRMDP_wellFormed` |
| `IntervalMDPProofs/Models/Factored.lean` | `StateVars`, `ActionVars`, `Marginal`, `FactoredIMDP`, `marginalSet`, `productSet`, `toRMDP`; `FactoredIMDP.toRMDP_wellFormed` |
| `IntervalMDPProofs/Models/DFA.lean` | `DFA`, `Labelling`, `ProbLabelling`, `AbstractLabelling` |
| `IntervalMDPProofs/Models/Product.lean` | `ProductProcess`, `toRMDP`; `ProductProcess.toRMDP_wellFormed`, `lift_deterministic` |
| `IntervalMDPProofs/Models/Strategy.lean` | `StationaryStrategy`, `TimeVaryingStrategy`, validity |
| `IntervalMDPProofs/Models/Specification.lean` | `Property`, `SatisfactionMode`, `StrategyMode`, `Specification` |
| `IntervalMDPProofs/Models/Examples.lean` | `docIMDP`, `docFIMDP`; `binaryFIMDP` and `FactoredIMDP.productSet_not_convex` |
| `IntervalMDPProofs/Approx/Sound.lean` | `Approx.Sound`, `Approx.StepSound` |
| `IntervalMDPProofs/Approx/Lift.lean` | `Approx.iter_sound` (A1), `Approx.fixedPoint_sound` |
| `AxiomCheck.lean` | `#print axioms` for every mapped theorem |
| `DocLint.lean` | `#lint only docBlame docBlameThm` |

## Modelling notes

* **Factored IMDPs.** The ambiguity set of a factored IMDP is the literal product of the marginal
  sets, `FactoredIMDP.productSet`, and `FactoredIMDP.toRMDP` uses exactly that set (no convex hull).
  It is not convex in general (`FactoredIMDP.productSet_not_convex`; arXiv:2411.11803,
  arXiv:2508.00707), so `AmbiguitySet.WellFormed` requires only nonempty and closed, and the general
  `RMDP` results of later phases must not assume convexity. The factored algorithms (Phase 5) follow
  the cited papers. The model assumes `source_dims = state_vars`.
* **Product process.** The DFA step uses the label of the *successor* state, as `bellman.jl` does
  (the `ProductProcess` docstring writes the source label).
* **Available actions** must be nonempty in Lean; Julia's `ListAvailableActions` does not check it.
* **Strategy validity** in Lean requires availability; Julia's `checkstrategy` checks only the
  action range.
