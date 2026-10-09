# IntervalMDPProofs — Lean models and proofs for IntervalMDP.jl

This Lake package contains Lean 4 models of the objects IntervalMDP.jl operates on, and
machine-checked proofs about them. It is developed in phases (see
`harness/specs/lean-proofs/`, shared rules in `common.md`, one spec per sub-phase); the status of every row is tracked in
`harness/specs/inventory-intervalmdp.md`.

**Current phase: 2 (sub-phase 2a)** — Phase 0 (models with their well-formedness
theorems, and the generic approximation-soundness lift A1), all of Phase 1 (index foundations
(1a: 1-based conversion, column-major linear index, sparse support pairing, O-max sort
permutation), marginal indexing (1b: `sub2ind` of `Marginal` and `IntervalAmbiguitySets`), dense
O-maximization (1c: exact `sSup` / `sInf`, tie invariance) and sparse O-maximization (1d: equal to
dense O-max for every sort of the support pairs)), and sub-phase 2a: the robust Bellman operator
`Bellman.T`, defined once on `RMDP`, is monotone, translation equivariant and sup-norm
non-expansive, for all four satisfaction × strategy modes and without any convexity assumption
(so the results apply to the non-convex factored product sets). The interval specialisation (2b),
strategies, VI and IVI (later sub-phases and phases) are not proved yet.

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
OMax.omax A s                  dense O-max on IntervalAmbiguity (Fin n) × SortedPerm n (exact: sSup / sInf)
OMax.omaxSparse A σ            sparse O-max on SparseIntervalAmbiguity n (= IntervalAmbiguity + SparseCol gap)
                               × SortedValuesGaps s A.gapCol (= omax: omaxSparse_eq_omax)
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
| 1-based index `i` of a Julia array | `IntervalMDP.Index.toJulia : Fin n → ℕ := (· + 1)`, inverse `ofJulia` on `juliaRange n = 1..n`, `juliaGet V k` = `V[k]` (`Index/Julia.lean`) |
| `Int32` / `Int` (`Int64`) index arithmetic (wrap-around) | `IntervalMDP.Index.machineInt N x` (= `Int.bmod x (2^N)`), `machineInt_eq_self` under `x < 2^(N-1)` |
| `CartesianIndex` in `CartesianIndices(dims)` | `IntervalMDP.Index.CartesianIndex dims` (`Index/Linear.lean`) |
| `LinearIndices(dims)[I]` (column-major) | `IntervalMDP.Index.linear`, `stride`, `linearInt`; `linear_bijective`, `linear_succ_first` |
| sparse column `gap(ambiguity_set)` (`SparseMatrixCSC` column: `rowvals` = `support`, `nonzeros`) | `IntervalMDP.Index.SparseCol` (`rowval`, `nzval`, CSC invariant as fields), `SparseCol.getindex`, `supportRows` (`Index/Sparse.lean`) |
| `zip(V[support(ambiguity_set)], nonzeros(gap(ambiguity_set)))` (`state_action_bellman(::SparseIntervalOMaxWorkspace, …)`) | `IntervalMDP.Index.valuesGaps`; `sparse_zip_correct` |
| `sortperm!(perm, V; rev = upper_bound)` (`bellman_precomputation!`) | `IntervalMDP.Index.SortedPerm` (`V`, `upperBound`), `.perm`, `.order` (`Index/Perm.lean`); `sortedPerm_bijective`, `sortedPerm_fits_int32` |
| loop of `gap_value(V, gap, budget, perm)` (`src/bellman.jl`) | `IntervalMDP.Index.gapValue` (literal transcription), `visited`, `allocation`; `greedy_visits_once`, `gapValue_eq_sum_allocation` |
| `sub2ind(p::Marginal, action, source)` loop (`src/probabilities/Marginal.jl`); fields `source_dims`, `action_vars` | `IntervalMDP.Index.marginalSub2ind` (literal transcription), `marginalSub2indInt` (`N`-bit value), `marginalSourceDims`, `marginalActionVars`, `marginalDims` (= `(action_vars…, source_dims…)`), `marginalCartesian` (= `(action[action_indices]…, source[state_indices]…)`), `juliaTuple` (= `Tuple(I)`), `hornerLoop` (= one `for i in StepRange(d, -1, 1)` loop of `sub2ind`), `marginalColumn` (= `N`-bit `sub2ind(p, a, s)` of a pair `(a, s)`) (`Index/Marginal.lean`); `foldl_horner`, `linear_eq_foldl`, `marginalSub2ind_eq_linear`, `marginalSub2ind_bijective`, `marginalSub2ind_depends_only` |
| `state_action_bellman(::DenseIntervalOMaxWorkspace, V, ambiguity_set, budget, upper_bound)` = `dot(V, lower) + gap_value(V, gap, budget, perm)` (`src/bellman.jl`) | `IntervalMDP.OMax.stateActionBellman` (transcription), `omax` (with the stable `sortperm!` of `bellman_precomputation!`), `dot` (= `LinearAlgebra.dot`), `greedy` (greedy distribution `lower + allocation`), `valueSet` (= `{⟨p, V⟩ : p ∈ P(l, u)}`), `IsThreshold` (proof device) (`OMax.lean`); `omax_mem`, `omax_eq_sSup` (`upper_bound = true`), `omax_eq_sInf` (`upper_bound = false`), `omax_tie_invariant`, `stateActionBellman_eq_dot`, `stateActionBellman_isGreatest`, `stateActionBellman_isLeast` |
| `permutation(workspace)` of `DenseIntervalOMaxWorkspace` (`src/workspace.jl`), any valid `sortperm!` output | `IntervalMDP.OMax.Permutation` (`perm`, invariants `perm_perm`, `sorted`), `stablePermutation` (= the stable `SortedPerm.perm`) (`OMax.lean`) |
| `IntervalAmbiguitySet` with a sparse column view (`SparseColumnView`, `src/probabilities/IntervalAmbiguitySets.jl`): `gap(ambiguity_set)` stored in CSC form, `support = rowvals(gap)` | `IntervalMDP.OMax.SparseIntervalAmbiguity` (extends `IntervalAmbiguity (Fin n)`; `gapCol : SparseCol n`, invariant `gap_eq : gap = gapCol.getindex`) (`OMax.lean`) |
| `Vp_workspace` = `workspace.values_gaps[1:supportsize]` after `sort!(Vp_workspace; rev = upper_bound, by = first)` (`src/bellman.jl`), any valid sort output | `IntervalMDP.OMax.SortedValuesGaps` (`Vp`, invariants `perm`, `sorted`), `stableValuesGaps` (= stable `mergeSort` with `pairLe`), `pairLe` (= `rev = upper_bound, by = first`) (`OMax.lean`) |
| `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` = `dot(V, lower) + gap_value(Vp, budget)`; loop of `gap_value(Vp, budget)` (`src/bellman.jl`) | `IntervalMDP.OMax.omaxSparse` (transcription), `gapValueSparse` (literal transcription of `gap_value(Vp, budget)`) (`OMax.lean`); `omaxSparse_eq_omax`, `omaxSparse_exact`, `gapValueSparse_map`, `gapValue_sublist`, `exists_permutation_sublist` |
| `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ) = jₛ[1]` (`src/probabilities/IntervalAmbiguitySets.jl`) | `IntervalMDP.Index.intervalSub2ind`; `intervalSub2ind_correct` (single-action layout), `intervalSub2ind_wrong_multiAction` (Observation O8) |
| `bellman!(workspace, strategy_cache, Vres, V, model; upper_bound, maximize)` with an `OptimizingStrategyCache` → `state_bellman!` (`src/bellman.jl`) | `IntervalMDP.Bellman.T M sat strat V` (`Bellman.lean`); `T_mono`, `T_monotone`, `T_add_const`, `T_nonexpansive`, `T_lipschitz`, `T_le_add` (all four modes, general `RMDP`, no convexity) |
| `state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)` (any workspace), `upper_bound = isoptimistic(spec)` | `IntervalMDP.Bellman.innerOpt sat Γ V` (`sSup` optimistic / `sInf` pessimistic of `expectations Γ V = {⟨p, V⟩ : p ∈ Γ}`), `stateActionBellman M sat V s a` (`Bellman.lean`); `innerOpt_mem` (attained), `innerOpt_le_add` |
| `extract_strategy!(…, values, available_actions, jₛ, maximize)` value, `maximize = ismaximize(spec)` (`src/strategy_cache.jl`) | `IntervalMDP.Bellman.extractValue strat acts h values` (`Finset.sup'` / `Finset.inf'`) (`Bellman.lean`); `extractValue_le_add` |

Indices: Julia is 1-based, Lean's `Fin n` is 0-based. The conversion is defined once, as
`Index.toJulia` (`Index/Julia.lean`), and every index theorem is stated through it. Concrete
examples and factored variable values use `Fin` with Julia index `k` ↦ Lean `k - 1`.

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
| `IntervalMDPProofs/Index/Julia.lean` | `Index.toJulia`, `juliaRange`, `ofJulia`, `juliaGet`, `machineInt`; round trips, `toJulia_bijOn`, `machineInt_eq_self` |
| `IntervalMDPProofs/Index/Linear.lean` | `Index.CartesianIndex`, `stride`, `linear`, `linearInt`, `incFirst`; `linear_bijective`, `linear_succ_first` |
| `IntervalMDPProofs/Index/Sparse.lean` | `Index.SparseCol`, `getindex`, `supportRows`, `valuesGaps`; `sparse_zip_correct` |
| `IntervalMDPProofs/Index/Perm.lean` | `Index.SortedPerm`, `perm`, `gapValue`, `visited`, `allocation`; `sortedPerm_bijective`, `sortedPerm_fits_int32`, `greedy_visits_once`, `gapValue_eq_sum_allocation` |
| `IntervalMDPProofs/Index/Marginal.lean` | `Index.marginalSub2ind` (transcription of `sub2ind(::Marginal, …)`), `marginalSub2indInt`, `marginalDims`, `marginalCartesian`, `juliaTuple`, `hornerLoop`, `marginalColumn`, `intervalSub2ind`; `foldl_horner`, `linear_eq_foldl`, `marginalSub2ind_eq_linear`, `marginalCartesian_surjective`, `marginalSub2ind_bijective`, `marginalSub2ind_depends_only`, `intervalSub2ind_correct`, `intervalSub2ind_wrong_multiAction` |
| `IntervalMDPProofs/OMax.lean` | `OMax.dot`, `valueSet`, `Permutation`, `stablePermutation`, `stateActionBellman` (transcription of dense `state_action_bellman`), `omax`, `IsThreshold`, `greedy`; `allocation_nonneg`, `allocation_le_gap`, `allocation_cons_of_ne`, `sum_allocation`, `exists_threshold`, `sum_perm_eq`, `Permutation.nodup`, `Permutation.mem`, `sum_allocation_eq_budget`, `greedy_apply`, `greedy_mem`, `stateActionBellman_eq_dot`, `juliaGet_neg`, `dot_neg`, `dot_le_greedy`, `stateActionBellman_isGreatest`, `stateActionBellman_isLeast`, `omax_mem`, `omax_eq_sSup`, `omax_eq_sInf`, `omax_tie_invariant`; sparse part: `SparseIntervalAmbiguity`, `SortedValuesGaps`, `pairLe`, `stableValuesGaps`, `gapValueSparse` (transcription of `gap_value(Vp, budget)`), `omaxSparse` (transcription of sparse `state_action_bellman`); `exists_perm_map_eq`, `gapValueSparse_map`, `gapValue_zero_budget`, `gapValue_sublist`, `sortedLe_trans`, `sortedLe_total`, `exists_permutation_sublist`, `omaxSparse_eq_omax`, `omaxSparse_exact` |
| `IntervalMDPProofs/Bellman.lean` | `Bellman.expectations`, `innerOpt`, `extractValue`, `stateActionBellman`, `T` (robust Bellman operator on `RMDP`); `continuous_dot`, `expectations_isCompact`, `expectations_nonempty`, `innerOpt_mem`, `dot_le_dot_add`, `innerOpt_le_add`, `extractValue_le_add`, `T_le_add`, `T_mono`, `T_monotone`, `T_add_const`, `T_nonexpansive`, `T_lipschitz` |
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
