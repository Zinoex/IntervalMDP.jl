# IntervalMDPProofs — Lean models and proofs for IntervalMDP.jl

This Lake package contains Lean 4 models of the objects IntervalMDP.jl operates on, and
machine-checked proofs about them. It is developed in phases (see
`harness/specs/lean-proofs/`, shared rules in `common.md`, one spec per sub-phase); the status of every row is tracked in
`harness/specs/inventory-intervalmdp.md`.

**Phase 3 complete (sub-phase 3d)** (Phases 0–3 complete) — Phase 0 (models with their well-formedness
theorems, and the generic approximation-soundness lift A1), all of Phase 1 (index foundations
(1a: 1-based conversion, column-major linear index, sparse support pairing, O-max sort
permutation), marginal indexing (1b: `sub2ind` of `Marginal` and `IntervalAmbiguitySets`), dense
O-maximization (1c: exact `sSup` / `sInf`, tie invariance) and sparse O-maximization (1d: equal to
dense O-max for every sort of the support pairs)), and all of Phase 2: the robust Bellman operator
`Bellman.T`, defined once on `RMDP`, is monotone, translation equivariant and sup-norm
non-expansive, for all four satisfaction × strategy modes and without any convexity assumption
(2a); for interval MDPs it is the optimum over the available actions of the dense O-max values of
the columns `sub2ind(marginal, jₐ, jₛ)` (`T_interval_eq_omax`); the action selected by
`extract_strategy!` is available and attains the optimum (`argopt_attains`); and policy evaluation
for a given strategy is exact and sound for `T` in the strategy-mode direction through value
iteration (`policy_eval_sound`, proved for valid strategies; A8 partial (F2)) (2b). The strategy-lookup row
and A8 are `partial`: Julia's
`checkstrategy` does not check availability, so "every strategy that passes validation looks up an
available action" is false (Finding F2, `Index.checkStrategy_admits_unavailable`); it is proved for
`AllAvailableActions` and for valid strategies. Phase 3a: value iteration for reachability and
reach-avoid (`VI.reachIter`, a transcription of `_value_iteration!` with the reachability
`initialize!` / `step_postprocess_value_function!`) stays in `[0, 1]`, is monotone in `k`, converges
to the least fixed point `VI.reachLfp` of the one-step map, and every iterate is below it (A4,
`reachIter_sound`, via `Approx.iter_sound`), all four modes; monotonicity, convergence and A4 are
stated for the non-exact-time properties. `V_k ≤ V*` is the conservative direction only in
pessimistic mode; in optimistic mode it is a lower bound, not `Sound .optimistic`. The rest of
Phase 3b: safety (`VI.safetyIter`, Julia's shifted iteration with `avoid` reset to `-1` and a final
`+ 1`) equals, after the `+ 1`, the reachability iteration of the exact-time reach-avoid property
`reach = avoidᶜ` (`safety_shift_eq`, via `T_add_const`, all four modes, no mode dualisation);
expected exit time (`VI.exitIter`) satisfies the one-step recursion `V_{k+1} = T V_k + 1` off
`avoid`, `0` on `avoid` (`exitIter_succ`), is monotone in `k` (`exitIter_mono`), and every iterate is
below every nonnegative real super-solution and below the `ℝ≥0∞` limit `VI.exitValue` (A4,
`exitIter_sound`, via `Approx.iter_sound`), all four modes; no convergence is claimed (values may
diverge), and in optimistic mode the lower bound is not `Sound .optimistic`. Phase 3c: discounted
reward (`VI.rewardIter`, Julia's `V₀ = r`, `V_{k+1} = ν · T V_k + r`) satisfies the one-step
recursion (`rewardIter_succ`, any `ν > 0`); for `0 < ν < 1` the one-step map is a Mathlib
`ContractingWith ν` (`reward_contracting`), the iterates converge to its unique fixed point
`VI.rewardValue`, and the A5 error bound `‖V_{k+1} - V*‖∞ ≤ ν / (1 - ν) · ‖V_{k+1} - V_k‖∞` holds
(`reward_error_bound`), so Julia's stop `‖V_k - V_{k-1}‖∞ < ε` gives `‖V_k - V*‖∞ < ν / (1 - ν) · ε`
(`reward_stop_bound`), all four modes. Phase 3d: synthesized strategies (A7, `VI/Strategy.lean`).
`timeVarying_attains`: evaluating the returned time-varying strategy (Julia's
`strategy[time_length - k]` at call `k`) gives exactly the computed `V_K`, every property type, all
four modes. `stationary_sound`: for infinite-time reachability / reach-avoid the returned stationary
strategy achieves at least the returned value, `V_{K+1} ≤ V^σ` (`Sound .pessimistic`), all four
modes; it relies on the documented cache keeping its action on ties (a state switches only on a
strict value increase). Exit time (`stationary_sound_exitTime`) and discounted reward
(`stationary_reward_error_bound`, the A5 interval contains `V^σ`) are covered too. Julia's guard
`jₛ ∉ available_actions` breaks this for states whose index exceeds the number of actions
(inventory Finding F3, benchmark B-1; witness `Examples.b1_stationary_unsound`). Phase 4
(complete, with open Finding F4): interval value iteration (`IVI.lean`). `IVI.iviIter` transcribes
`_interval_value_iteration!` / `ivi_step!` (strategy synthesized on the primary bound, applied to
the other); `lower_le_upper` (`V_lower_k ≤ V_upper_k`, all four modes, all reach-avoid properties)
and `primary_sound` (the returned bound is `Sound sat` for `V*`, all four modes, non-exact-time
reach-avoid properties `prop.toReachProperty.isExactTime = false`, via `Approx.iter_sound`) hold.
`V_lower_k ≤ V*` holds for `sat = pessimistic` or `strat = maximize` on non-exact-time properties
(`lower_le_reachLfp`), `V* ≤ V_upper_k` for `sat = optimistic` or `strat = minimize` on all
reach-avoid properties (`reachLfp_le_upper`), so the bracket holds for `(Pessimistic, Minimize)` and
`(Optimistic, Maximize)` on non-exact-time properties (`bracket_aligned`, 4a). Stopping on
`IVIInitialGapCriteria` (4b; `maxInitialGap`, `iviInitialGapCriteria`, `stopIndex`,
`iviValueFunction`), assuming the loop exits (hypothesis `h : Terminates …`; termination is not
proved, and Julia notes the gap may not close when an end component lies in the don't-care region):
in those two aligned modes the returned value is within `tol` of `V*` on the initial states, for
non-exact-time reach-avoid properties (`gap_stop_sound_aligned`); in all four modes it is sound,
`Sound sat`, for non-exact-time reach-avoid properties (`gap_stop_primary_sound`).
For `(Pessimistic, Maximize)` and `(Optimistic, Minimize)` the bracket is false under Julia's
strategy coupling (inventory Finding F4; witnesses `Examples.ivi_upper_lt_reachLfp`,
`Examples.ivi_reachLfp_lt_lower`), and Julia's gap test stops at `k = 1` with a value `½` away from
`V*` (`Examples.not_gap_stop_sound_pessimistic_maximize`,
`Examples.not_gap_stop_sound_optimistic_minimize`), so neither `IVI.bracket` nor
`IVI.gap_stop_sound` (all four modes) is stated.
Phase 5 (in progress; 5a done): factored IMDPs (`Factored.lean`). `Index.factored_successor_eq`:
under the overflow bound `∏ state_vars < 2 ^ (N - 1)` (`hBound`) and for a value array storing `W`
(`hV : StoresValues`), the index `I ∈ CartesianIndices(num_target.(ambiguity_sets))` of vertex
enumeration and the factored O-max loops is in bijection with the joint successor states, `V[I]`
reads `W (successorState I)`, and `prod(r -> γ[r][I[r]], …)` is the product distribution there.
`Factored.vertices_complete` (no hypotheses beyond the model invariants; dense marginal storage
only, sparse marginals not modelled, Observation O16): the loop over
`Iterators.product` of the marginal `IntervalAmbiguitySetVertexIterator`s, including the
iterator's permutation skipping (`nextPermutation`), visits exactly the tuples of extreme points
(vertices) of the marginal sets. `Factored.vertexValue_eq_opt` and its corollary
`vertexValue_eq_stateActionBellman` (no hypotheses beyond the model invariants; dense marginal
storage only, O16; all four modes, `upper_bound = isoptimistic(sat)`): the vertex-enumeration value
`vertexValue` equals the exact inner optimum over the literal, non-convex `productSet`
(arXiv:2508.00707, Theorem 1); this is the exact reference value for A2/A3. Both theorems cover
dense marginal storage only (`vertexValue` is built on the dense `vertices` transcription). Recursive O-max (5b) and McCormick (5c) are not proved yet.

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
Factored.vertexValue F s a W ub  vertex enumeration over Factored.vertexEnumeration (exact: = innerOpt
                               over FactoredIMDP.productSet, vertexValue_eq_opt; dense marginals only)
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
| `isoptimistic(spec)` passed as `upper_bound` (`src/specification.jl`, `src/robust_value_iteration.jl`) | `IntervalMDP.Bellman.isOptimistic` (`Bellman.lean`) |
| `IntervalMarkovDecisionProcess(ambiguity_sets, num_actions)` storage: columns `ambiguity_sets[j]` of `Marginal(ambiguity_sets, (num_states,), (num_actions,))` (`src/models/IntervalMarkovDecisionProcess.jl`) | `IntervalMDP.Bellman.IntervalMDPLayout` (`ambiguitySets`, available actions), `.marginal`, `.column` (= `sub2ind(marginal, jₐ, jₛ)`), `.toIMDP` (`Bellman.lean`); `IntervalMDPLayout.column_eq` (`(jₛ - 1) * num_actions + jₐ`), `columnInt_eq` |
| `state_action_bellman(::DenseIntervalOMaxWorkspace, V, marginal[jₐ, jₛ], budget[sub2ind(marginal, jₐ, jₛ)], upper_bound)` in `state_bellman!` | `IntervalMDP.Bellman.intervalStateActionBellman` (`Bellman.lean`); `innerOpt_interval_eq_omax`, `stateActionBellman_interval_eq_omax`, `T_interval_eq_omax` (all four modes), `expectations_toSet` |
| loop of `_extract_strategy!` (`gt = maximize ? (>) : (<)`, strict, first optimum kept) and its `neutral` seed (`src/strategy_cache.jl`) | `IntervalMDP.Bellman.argoptStep` (loop body), `argoptAction` (the loop, a `foldl` over the available actions from the seed), `optLE`, `stationarySeed` (seed of a `StationaryStrategyCache` across calls) (`Bellman.lean`); `argoptStep_spec`, `argoptAction_spec`, `extractValue_eq_of_opt`, `argopt_attains` (all four modes), `stationarySeed_available` |
| `state_bellman!` with a `NonOptimizingStrategyCache` (`GivenStrategyCache`, `ActiveGivenStrategyCache`): `Vres[jₛ] = state_action_bellman(…, marginal[jₐ, jₛ], …)` for `jₐ = CartesianIndex(strategy_cache[jₛ])` (`src/bellman.jl`) | `IntervalMDP.Bellman.Tπ M sat π V` (policy evaluation), `strategyAvailable π` (`available s = {π(s)}`), `policyEvalMode` (soundness direction: `maximize` ↦ `Tπ ≤ T`, `minimize` ↦ `Tπ ≥ T`) (`Bellman.lean`); `Tπ_eq_T_strategyAvailable`, `Tπ_stepSound`, `policy_eval_sound` (A8, via `Approx.iter_sound`), `Tπ_interval_eq_omax` |
| strategy array `AbstractArray{NTuple{M, Int32}}` and lookup `CartesianIndex(strategy_cache[jₛ])` (`src/strategy.jl`, `src/strategy_cache.jl`, `src/bellman.jl`) | `IntervalMDP.Index.StrategyArray`, `strategyAction` (lookup at `linearInt N`), `actionTuple` (= `Tuple(jₐ)`), `juliaAvailable` (= `Tuple.(available(model, jₛ))`), `Stores` (`Index/Strategy.lean`); `strategyAction_eq_linear`, `strategyAction_available_of_all`, `strategyAction_available_of_valid`, `actionTuple_injective` |
| `checkstrategy(strategy::AbstractArray, system::FactoredRMDP)` (`src/strategy.jl`): only `1 ≤ s[i] ≤ action_vars[i]` | `IntervalMDP.Index.checkStrategy` (`Index/Strategy.lean`); `checkStrategy_admits_unavailable` (Finding F2) |
| `_value_iteration!` loop (`initialize!`, `nextiteration!`, `step!`, `k += 1`, `term_criteria`) with an `AbstractReachability` specification (`src/robust_value_iteration.jl`) | `IntervalMDP.VI.reachIter M sat strat prop k` (`VI/Reach.lean`); `reachIter_eq_iterate`, `reachIter_mem_unit`, `reachIter_mono`, `reachIter_tendsto_lfp`, `reachIter_sound` (A4) |
| `step!(workspace, strategy_cache, value_function, k, mp, spec)` = `bellman!` + `step_postprocess_value_function!` (`src/robust_value_iteration.jl`) | `IntervalMDP.VI.step M sat strat prop V` (`VI/Reach.lean`); `step_mono`, `step_mem_unit`, `continuous_step` |
| `FiniteTimeReachability`, `InfiniteTimeReachability`, `ExactTimeReachability`, `FiniteTimeReachAvoid`, `InfiniteTimeReachAvoid`, `ExactTimeReachAvoid` (`src/specification.jl`) | `IntervalMDP.VI.ReachProperty` (`.reachability`, `.reachAvoid`, `.exactTimeReachability`, `.exactTimeReachAvoid`), `ReachProperty.reach`, `isExactTime`, `Property.toReachProperty` (`VI/Reach.lean`) |
| `initialize!(value_function, ::AbstractReachability)`, `step_postprocess_value_function!` for `AbstractReachability`, `AbstractReachAvoid`, `ExactTimeReachability`, `ExactTimeReachAvoid` (`src/specification.jl`) | `IntervalMDP.VI.initializeValueFunction`, `stepPostprocessValueFunction` (`VI/Reach.lean`) |
| exact infinite-horizon reachability value approximated by `_value_iteration!` | `IntervalMDP.VI.reachLfp` (`OrderHom.lfp` of `stepHom` on `S → Set.Icc 0 1`) (`VI/Reach.lean`); `step_reachLfp`, `reachLfp_mem_unit`, `reachLfp_le_of_step_le`, `reachLfp_le_of_fixedPt` |
| `_value_iteration!` loop with an `AbstractSafety` specification (`FiniteTimeSafety`, `InfiniteTimeSafety`; `src/robust_value_iteration.jl`, `src/specification.jl`) | `IntervalMDP.VI.safetyIter M sat strat prop k` (shifted iterate, before the final `+ 1`), `safetyStep` (`VI/Safety.lean`); `safety_shift_eq`, `safetyIter_eq_sub_one`, `safetyIter_postprocess_mem_unit` |
| `AbstractSafety`, `avoid(prop)`; `initialize!` (`current[avoid] .= -1`), `step_postprocess_value_function!` (`current[avoid] .= -1`), `postprocess_value_function!` (`current .+= 1`) (`src/specification.jl`) | `IntervalMDP.VI.SafetyProperty` (`avoid`), `Property.toSafetyProperty`, `SafetyProperty.initializeValueFunction`, `.stepPostprocessValueFunction`, `.postprocessValueFunction`, `.toReachProperty` (exact-time reach-avoid with `reach = avoidᶜ`) (`VI/Safety.lean`) |
| `_value_iteration!` loop with an `ExpectedExitTime` specification (`src/robust_value_iteration.jl`, `src/specification.jl`) | `IntervalMDP.VI.exitIter M sat strat prop k`, `exitStep` (`VI/ExitTime.lean`); `exitIter_eq_iterate`, `exitStep_mono`, `exitIter_succ`, `exitIter_nonneg`, `exitIter_mono`, `exitIter_sound` (A4) |
| `ExpectedExitTime(avoid_states, convergence_eps)`, `avoid(prop)`; `initialize!` (`current .= 1; current[avoid] .= 0`), `step_postprocess_value_function!` (`current .+= 1; current[avoid] .= 0`), `postprocess_value_function!(…, ::AbstractHittingTime)` (identity) (`src/specification.jl`) | `IntervalMDP.VI.ExpectedExitTime` (`avoidStates`), `Property.toExpectedExitTime`, `ExpectedExitTime.initializeValueFunction`, `.stepPostprocessValueFunction` (`VI/ExitTime.lean`); `initializeValueFunction_eq_exitStep_zero`, `ExpectedExitTime.initializeValueFunction_nonneg`, `ExpectedExitTime.stepPostprocessValueFunction_mono` |
| exact robust expected exit time `𝔼^{π,η}_exit(O)` approximated by `_value_iteration!` (possibly `∞`) | `IntervalMDP.VI.exitValue` (`⨆ k`, in `ℝ≥0∞`) (`VI/ExitTime.lean`); `exitIter_le_exitValue`, `exitValue_le_of_step_le` |
| `_value_iteration!` loop with an `AbstractReward` specification (`FiniteTimeReward`, `InfiniteTimeReward`; `src/robust_value_iteration.jl`, `src/specification.jl`) | `IntervalMDP.VI.rewardIter M sat strat prop k`, `rewardStep` (`VI/Reward.lean`); `rewardIter_eq_iterate`, `rewardIter_succ`, `rewardStep_dist_le`, `reward_contracting` (`0 < ν < 1`) |
| `AbstractReward`, `reward(prop)`, `discount(prop)`; `checkreward` (`discount > 0`), `checkdiscountupperbound` (`discount < 1`, infinite time); `initialize!` (`current .= reward`), `step_postprocess_value_function!` (`rmul!(current, discount); current .+= reward`), `postprocess_value_function!` (identity) (`src/specification.jl`) | `IntervalMDP.VI.RewardProperty` (`reward`, `discount`, `discount_pos`), `Property.toRewardProperty`, `RewardProperty.discountNNReal`, `.initializeValueFunction`, `.stepPostprocessValueFunction`, `.postprocessValueFunction` (`VI/Reward.lean`); `Property.toRewardProperty_discount_lt_one` |
| `CovergenceCriteria(convergence_eps)`: `maximum(abs, current - previous) < tol` (`src/robust_value_iteration.jl`) | `IntervalMDP.VI.convergenceCriteria ε V Vprev` (`dist V Vprev < ε`) (`VI/Reward.lean`) |
| exact robust discounted reward (`0 < ν < 1`) approximated by `_value_iteration!` | `IntervalMDP.VI.rewardValue` (Mathlib `ContractingWith.fixedPoint`) (`VI/Reward.lean`); `rewardValue_isFixedPt`, `rewardIter_tendsto`, `reward_error_bound` (A5), `reward_stop_bound`, `reward_error_interval` |
| `_value_iteration!` loop, any specification (`step!` = `bellman!` + `step_postprocess_value_function!`) | `IntervalMDP.VI.viIter M sat strat post V₀ k` (`VI/Strategy.lean`); `reachIter_eq_viIter`, `safetyIter_eq_viIter`, `exitIter_eq_viIter`, `rewardIter_eq_viIter` |
| `for jₐ in available_actions`, `first(available_actions)` (`available(model, jₛ)`, `src/available_actions.jl`) | `IntervalMDP.VI.ActionOrder M` (`acts`, `toFinset_acts`), `ActionOrder.first` (`VI/Strategy.lean`) |
| `extract_strategy!(::TimeVaryingStrategyCache \| ::StationaryStrategyCache, …)`: `cur_strategy[jₛ]` / `strategy[jₛ]` after call `k` (`src/strategy_cache.jl`) | `IntervalMDP.VI.synthesizedStrategy M sat strat o V seed k` with `timeVaryingSeed o` (neutral `first(available_actions)`) or `stationaryCacheSeed` (previous action, documented guard) (`VI/Strategy.lean`) |
| `cachetostrategy(::TimeVaryingStrategyCache)` = `TimeVaryingStrategy(reverse(strategy))`; `cachetostrategy(::StationaryStrategyCache)` (`src/strategy_cache.jl`) | `IntervalMDP.VI.timeVaryingCacheStrategy M sat strat o V K` (index `j` = call `Fin.rev j`), `stationaryCacheStrategy M sat strat o V K` (cache after call `K`) (`VI/Strategy.lean`) |
| `select_strategy_cache(cache, k) = cache[time_length(cache) - k]` (`src/robust_value_iteration.jl`); `_value_iteration!` of a `VerificationProblem` with a time-varying strategy | `IntervalMDP.VI.selectStrategy π k`, `policyEvalIter M sat post V₀ π k` (`VI/Strategy.lean`); `timeVarying_attains` (A7) |
| value of a given stationary strategy (`solve(VerificationProblem(mdp, spec, π))`, infinite horizon) | `IntervalMDP.VI.strategyReachLfp`, `strategyRewardValue` (`VI/Strategy.lean`); `stationary_sound` (A7), `stationary_sound_exitTime`, `stationary_reward_error_bound` |
| `step_postprocess_value_function!` of reset/shift shape (reachability types, `ExpectedExitTime`) | `IntervalMDP.VI.StepPostprocess` (`reset`, `resetValue`, `shift`, `apply`), `ReachProperty.stepPostprocess`, `ExpectedExitTime.stepPostprocess` (`VI/Strategy.lean`) |
| `AbstractReachAvoid` (`FiniteTimeReachAvoid`, `InfiniteTimeReachAvoid`, `ExactTimeReachAvoid`; `checkivisupported`, `src/specification.jl`) | `IntervalMDP.IVI.ReachAvoidProperty` (`reach`, `avoid`, `toReachProperty`) (`IVI.lean`) |
| `V_lower.current`, `V_upper.current`, strategy cache content in `_interval_value_iteration!` (`src/interval_value_iteration.jl`) | `IntervalMDP.IVI.Bounds` (`lower`, `upper`, `strategy`) (`IVI.lean`) |
| `construct_ivi_strategy_cache` (`TimeVaryingStrategyCache` / `StationaryStrategyCache`, `src/strategy_cache.jl`) | `IntervalMDP.IVI.StrategyCacheKind`, `cacheSeed` (`IVI.lean`) |
| `primary_current` / `secondary_current` in `ivi_step!` | `IntervalMDP.IVI.primary sat B`, `secondary sat B`, `ofPrimary` (`IVI.lean`) |
| `initialize_ivi!(V_lower, V_upper, prop)` (`src/specification.jl`) | `IntervalMDP.IVI.initializeIvi o prop` (`lower = 𝟙_reach`, `upper = initializeUpper prop = 1 - 𝟙_avoid`) (`IVI.lean`) |
| `ivi_step!` (`src/interval_value_iteration.jl`): `bellman!` on the primary bound with the optimizing cache, then on the secondary bound with `applied_strategy_cache` | `IntervalMDP.IVI.step M sat strat prop o kind B`, `iviStrategy` (the synthesized strategy) (`IVI.lean`) |
| `_interval_value_iteration!` loop (`src/interval_value_iteration.jl`) | `IntervalMDP.IVI.iviIter M sat strat prop o kind k`; `lower_le_upper`, `primary_sound`, `lower_le_reachLfp`, `reachLfp_le_upper`, `bracket_aligned` (A6) |
| `max_initial_gap(V_lower, V_upper, initial_states(mp))` (`src/interval_value_iteration.jl`); `gap = V_upper - V_lower` | `IntervalMDP.IVI.maxInitialGap B initial` (loop transcription, step `maxGapStep`), `IVI.gap B s` (`IVI.lean`) |
| `IVIInitialGapCriteria(convergence_eps)` (`src/interval_value_iteration.jl`) | `IntervalMDP.IVI.iviInitialGapCriteria tol initial B` (`IVI.lean`) |
| loop exit of `_interval_value_iteration!` (returned `k`, `num_iterations`) and the returned `value_function` | `IntervalMDP.IVI.Terminates`, `stopIndex h`, `iviValueFunction h`; `WithinOn initial tol V W`; `gap_stop_sound_aligned` (`(Pessimistic, Minimize)`, `(Optimistic, Maximize)`), `gap_stop_primary_sound` (all four modes), both non-exact-time reach-avoid only and both assuming the loop exits (`h : Terminates …`, termination not proved) (A6, 4b; Finding F4) (`IVI.lean`) |
| `getindex.(marginals(model), jₐ, jₛ)`, `num_target.(ambiguity_sets)`, `CartesianIndices(num_target.(…))` and `V[I]` in `state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`) | `IntervalMDP.Factored.ambiguitySets`, `numTarget`, `numTargets`, `successorState`, `StoresValues`, `readValue`; `Index.factored_successor_eq` (overflow bound `∏ state_vars < 2 ^ (N - 1)`, value array storing `W`) (`Factored.lean`) |
| `IntervalAmbiguitySetVertexIterator`, `Base.iterate` (both methods), `vertex_generator`, `vertices` (`src/probabilities/IntervalAmbiguitySets.jl`) | `IntervalMDP.Factored.addAt`, `vertexLoop` (greedy loop, `(v, break_idx)`), `vertexOf`, `permGet`, `suffixStep`, `nextInSuffix`, `findSwap`, `swapSort`, `nextPermutation` (permutation skip), `vertexRun`, `vertices`; `mem_vertices_iff`, `mem_vertices_iff_extremePoints` (`Factored.lean`) |
| `Iterators.product(iterators...)`, the `sum(V[I] * prod(…))` and `optval = optfunc(optval, v)` of `state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`) | `IntervalMDP.Factored.productList`, `vertexEnumeration`, `vertexSum`, `optStep`, `maxStep`, `minStep`, `vertexValue`; `vertices_complete` (no extra hypotheses; dense marginals only, O16), `vertexValue_eq_opt` (no extra hypotheses, all four modes; dense marginals only, O16; arXiv:2508.00707 Theorem 1) (`Factored.lean`) |

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
| `IntervalMDPProofs/Models/Examples.lean` | `docIMDP`, `docFIMDP`; `binaryFIMDP` and `FactoredIMDP.productSet_not_convex`; `selfLoopTarget`, `selfLoopRMDP`, `selfLoopOrder`, `selfLoopProp`, `b1Strategy`, `selfLoop_stateActionBellman`, `b1_stationary_unsound` (witness of Finding F3); `halfGoalAvoid`, `iviDist`, `iviRMDP`, `iviOrder`, `iviProp`, `iviRMDP_stateActionBellman`, `dot_dirac_fin5`, `dot_halfGoalAvoid`, `iviProp_post`, `iviOrder_cacheSeed`, `ivi_upper_lt_reachLfp`, `ivi_reachLfp_lt_lower`, `not_bracket_pessimistic_maximize`, `not_bracket_optimistic_minimize` (witnesses of Finding F4), `iviInitLower_apply`, `iviInitUpper_apply`, `iviStrategy_pessimistic_maximize`, `iviStrategy_optimistic_minimize`, `iviIter_one_pessimistic_maximize`, `iviIter_one_optimistic_minimize`, `one_le_reachLfp_pessimistic_maximize`, `reachLfp_optimistic_minimize_nonpos`, `iviInitialGapCriteria_one_pessimistic_maximize`, `iviInitialGapCriteria_one_optimistic_minimize`, `not_gap_stop_sound_pessimistic_maximize`, `not_gap_stop_sound_optimistic_minimize` (Finding F4 witnesses against `IVI.gap_stop_sound`) |
| `IntervalMDPProofs/Approx/Sound.lean` | `Approx.Sound`, `Approx.StepSound` |
| `IntervalMDPProofs/Approx/Lift.lean` | `Approx.iter_sound` (A1), `Approx.fixedPoint_sound` |
| `IntervalMDPProofs/Index/Julia.lean` | `Index.toJulia`, `juliaRange`, `ofJulia`, `juliaGet`, `machineInt`; round trips, `toJulia_bijOn`, `machineInt_eq_self` |
| `IntervalMDPProofs/Index/Linear.lean` | `Index.CartesianIndex`, `stride`, `linear`, `linearInt`, `incFirst`; `linear_bijective`, `linear_succ_first` |
| `IntervalMDPProofs/Index/Sparse.lean` | `Index.SparseCol`, `getindex`, `supportRows`, `valuesGaps`; `sparse_zip_correct` |
| `IntervalMDPProofs/Index/Perm.lean` | `Index.SortedPerm`, `perm`, `gapValue`, `visited`, `allocation`; `sortedPerm_bijective`, `sortedPerm_fits_int32`, `greedy_visits_once`, `gapValue_eq_sum_allocation` |
| `IntervalMDPProofs/Index/Marginal.lean` | `Index.marginalSub2ind` (transcription of `sub2ind(::Marginal, …)`), `marginalSub2indInt`, `marginalDims`, `marginalCartesian`, `juliaTuple`, `hornerLoop`, `marginalColumn`, `intervalSub2ind`; `foldl_horner`, `linear_eq_foldl`, `marginalSub2ind_eq_linear`, `marginalCartesian_surjective`, `marginalSub2ind_bijective`, `marginalSub2ind_depends_only`, `intervalSub2ind_correct`, `intervalSub2ind_wrong_multiAction` |
| `IntervalMDPProofs/Index/Strategy.lean` | `Index.StrategyArray`, `actionTuple`, `strategyAction` (strategy lookup), `juliaAvailable`, `checkStrategy` (Julia `checkstrategy`), `Stores`; `actionTuple_injective`, `strategyAction_eq_linear`, `strategyAction_available_of_all`, `strategyAction_available_of_valid`, `checkStrategy_admits_unavailable` (Finding F2) |
| `IntervalMDPProofs/OMax.lean` | `OMax.dot`, `valueSet`, `Permutation`, `stablePermutation`, `stateActionBellman` (transcription of dense `state_action_bellman`), `omax`, `IsThreshold`, `greedy`; `allocation_nonneg`, `allocation_le_gap`, `allocation_cons_of_ne`, `sum_allocation`, `exists_threshold`, `sum_perm_eq`, `Permutation.nodup`, `Permutation.mem`, `sum_allocation_eq_budget`, `greedy_apply`, `greedy_mem`, `stateActionBellman_eq_dot`, `juliaGet_neg`, `dot_neg`, `dot_le_greedy`, `stateActionBellman_isGreatest`, `stateActionBellman_isLeast`, `omax_mem`, `omax_eq_sSup`, `omax_eq_sInf`, `omax_tie_invariant`; sparse part: `SparseIntervalAmbiguity`, `SortedValuesGaps`, `pairLe`, `stableValuesGaps`, `gapValueSparse` (transcription of `gap_value(Vp, budget)`), `omaxSparse` (transcription of sparse `state_action_bellman`); `exists_perm_map_eq`, `gapValueSparse_map`, `gapValue_zero_budget`, `gapValue_sublist`, `sortedLe_trans`, `sortedLe_total`, `exists_permutation_sublist`, `omaxSparse_eq_omax`, `omaxSparse_exact` |
| `IntervalMDPProofs/Bellman.lean` | `Bellman.expectations`, `innerOpt`, `extractValue`, `stateActionBellman`, `T` (robust Bellman operator on `RMDP`); `continuous_dot`, `expectations_isCompact`, `expectations_nonempty`, `innerOpt_mem`, `dot_le_dot_add`, `innerOpt_le_add`, `extractValue_le_add`, `T_le_add`, `T_mono`, `T_monotone`, `T_add_const`, `T_nonexpansive`, `T_lipschitz`; interval part (2b): `isOptimistic`, `IntervalMDPLayout` (`stateVars`, `actionVars`, `marginal`, `jointState`, `jointAction`, `column`, `toIMDP`), `intervalStateActionBellman`; `expectations_toSet`, `innerOpt_interval_eq_omax`, `IntervalMDPLayout.column_eq`, `IntervalMDPLayout.columnInt_eq`, `stateActionBellman_interval_eq_omax`, `T_interval_eq_omax`; strategy part (2b): `optLE`, `argoptStep`, `argoptAction`, `stationarySeed`, `strategyAvailable`, `Tπ`, `policyEvalMode`; `optLE_refl`, `optLE_trans`, `argoptStep_spec`, `argoptAction_spec`, `extractValue_eq_of_opt`, `argopt_attains`, `stationarySeed_available`, `Tπ_eq_T_strategyAvailable`, `Tπ_stepSound`, `policy_eval_sound`, `Tπ_interval_eq_omax` |
| `IntervalMDPProofs/VI/Reach.lean` | `VI.ReachProperty`, `reach`, `isExactTime`, `Property.toReachProperty`, `initializeValueFunction`, `stepPostprocessValueFunction`, `step`, `reachIter` (transcription of `_value_iteration!` for reachability / reach-avoid), `stepHom`, `reachLfp`; `reachIter_eq_iterate`, `T_zero`, `T_const`, `T_mem_unit`, `stepPostprocessValueFunction_mono`, `stepPostprocessValueFunction_mem_unit`, `continuous_stepPostprocessValueFunction`, `step_mono`, `step_mem_unit`, `continuous_step`, `initializeValueFunction_le_step`, `step_reachLfp`, `reachLfp_mem_unit`, `reachLfp_le_of_step_le`, `reachIter_mem_unit`, `reachIter_mono`, `reachIter_sound` (A4), `reachIter_tendsto_lfp`, `reachLfp_le_of_fixedPt` |
| `IntervalMDPProofs/VI/Safety.lean` | `VI.SafetyProperty`, `Property.toSafetyProperty`, `SafetyProperty.initializeValueFunction`, `stepPostprocessValueFunction`, `postprocessValueFunction`, `toReachProperty`, `safetyStep`, `safetyIter` (transcription of `_value_iteration!` for safety, with the −1/+1 shift); `safety_shift_eq`, `safetyIter_eq_sub_one`, `safetyIter_postprocess_mem_unit` |
| `IntervalMDPProofs/VI/ExitTime.lean` | `VI.ExpectedExitTime`, `Property.toExpectedExitTime`, `ExpectedExitTime.initializeValueFunction`, `stepPostprocessValueFunction`, `exitStep`, `exitIter` (transcription of `_value_iteration!` for expected exit time), `exitValue`; `exitIter_eq_iterate`, `initializeValueFunction_eq_exitStep_zero`, `ExpectedExitTime.stepPostprocessValueFunction_mono`, `exitStep_mono`, `exitIter_succ`, `ExpectedExitTime.initializeValueFunction_nonneg`, `exitIter_nonneg`, `exitIter_mono`, `exitIter_sound` (A4), `exitIter_le_exitValue`, `exitValue_le_of_step_le` |
| `IntervalMDPProofs/VI/Reward.lean` | `VI.RewardProperty`, `Property.toRewardProperty`, `RewardProperty.discountNNReal`, `initializeValueFunction`, `stepPostprocessValueFunction`, `postprocessValueFunction`, `convergenceCriteria`, `rewardStep`, `rewardIter` (transcription of `_value_iteration!` for discounted reward), `rewardValue`; `Property.toRewardProperty_discount_lt_one`, `rewardIter_eq_iterate`, `rewardIter_succ`, `rewardStep_dist_le`, `reward_contracting`, `rewardValue_isFixedPt`, `rewardIter_tendsto`, `reward_error_bound` (A5), `reward_stop_bound`, `reward_error_interval` |
| `IntervalMDPProofs/VI/Strategy.lean` | `VI.ActionOrder`, `ActionOrder.first`, `viIter`, `synthesizedStrategy`, `timeVaryingSeed`, `stationaryCacheSeed`, `timeVaryingCacheStrategy`, `stationaryCacheStrategy`, `selectStrategy`, `policyEvalIter`, `StepPostprocess`, `ReachProperty.stepPostprocess`, `ExpectedExitTime.stepPostprocess`, `strategyReachLfp`, `strategyRewardValue`; `timeVarying_attains`, `timeVarying_attains_reach`, `timeVarying_attains_safety`, `timeVarying_attains_reward`, `policyEvalIter_timeVaryingCacheStrategy`, `stationary_sound`, `stationary_sound_exitTime`, `stationary_reward_error_bound`, `stationary_le_superSolution`, `stationary_backward_step`, `viIter_le_superSolution_minimize`, `T_lt_of_switch`, `strategy_eq_of_T_eq`, `argoptAction_eq_or_lt`, `synthesizedStrategy_spec`, `stationaryCacheSeed_mem`, `stationaryCacheSeed_succ`, `timeVaryingCacheStrategy_valid`, `stationaryCacheStrategy_valid`, `selectStrategy_timeVaryingCacheStrategy`, `exists_dot_bracket`, `exists_eq_zero_of_dot_nonpos`, `stateActionBellman_mono`, `reachIter_eq_viIter`, `safetyIter_eq_viIter`, `exitIter_eq_viIter`, `rewardIter_eq_viIter`, `StepPostprocess.apply_mono`, `ReachProperty.stepPostprocess_apply`, `ExpectedExitTime.stepPostprocess_apply` |
| `IntervalMDPProofs/IVI.lean` | `IVI.ReachAvoidProperty`, `ReachAvoidProperty.reach`, `avoid`, `toReachProperty`, `Bounds`, `StrategyCacheKind`, `cacheSeed`, `primary`, `secondary`, `ofPrimary`, `iviStrategy`, `step` (transcription of `ivi_step!`), `initializeUpper`, `initializeIvi` (`initialize_ivi!`), `iviIter` (transcription of `_interval_value_iteration!`); `lower_le_upper`, `primary_sound`, `lower_le_reachLfp`, `reachLfp_le_upper`, `bracket_aligned` (A6, restricted; Finding F4), `ReachAvoidProperty.disjoint_reach_avoid`, `reach_toReachProperty`, `stepPostprocess_avoid`, `step_lower`, `step_upper`, `step_strategy`, `primary_step_eq`, `cacheSeed_mem`, `iviStrategy_spec`, `iviIter_strategy_mem`, `Tπ_mono`, `primary_step`, `primary_iviIter`, `initializeIvi_lower_le_upper`, `initializeValueFunction_le_reachLfp`, `reachLfp_le_initializeUpper`, `iterate_sound`, `Tπ_iviStrategy_lower_le`, `T_le_Tπ_iviStrategy_upper`, `lower_le_iterate`, `iterate_le_upper`; stopping (4b): `gap`, `maxGapStep`, `maxInitialGap` (transcription of `max_initial_gap`), `iviInitialGapCriteria` (`IVIInitialGapCriteria`), `Terminates`, `stopIndex` (loop exit `k`), `iviValueFunction` (returned `value_function`), `WithinOn`; `gap_stop_sound_aligned` (A6, `(Pessimistic, Minimize)` / `(Optimistic, Maximize)`, non-exact-time, assumes `Terminates` (not proved); Finding F4), `gap_stop_primary_sound` (all four modes, non-exact-time, assumes `Terminates` (not proved)), `withinOn_of_iviInitialGapCriteria_aligned`, `withinOn_of_bracket`, `le_foldl_maxGapStep`, `gap_le_maxInitialGap`, `one_le_stopIndex`, `iviInitialGapCriteria_stopIndex`, `not_iviInitialGapCriteria_of_lt_stopIndex` |
| `IntervalMDPProofs/Factored.lean` | Phase 5a. Index: `Factored.ambiguitySets`, `numTarget`, `numTargets`, `successorState`, `StoresValues`, `readValue`; `successorState_bijective`, `Index.factored_successor_eq` (hypotheses: overflow bound `∏ state_vars < 2 ^ (N - 1)`, `StoresValues`). Marginal iterator: `addAt`, `vertexLoop`, `vertexOf`, `permGet`, `suffixStep`, `nextInSuffix`, `findSwap`, `swapSort`, `nextPermutation`, `vertexRun`, `vertices`, `permsFrom`; `vertexLoop_fst`, `vertexLoop_prefix`, `vertexOf_snd_ne_zero`, `nextPermutation_some`, `nextPermutation_spec`, `vertexOf_mem_vertexRun`, `mem_vertices_iff`, `vertexOf_mem_extremePoints`, `boundary_of_mem_extremePoints`, `exists_perm_vertexOf_eq`, `mem_vertices_iff_extremePoints`. Product and value: `productList`, `vertexEnumeration`, `vertexSum`, `optStep`, `maxStep`, `minStep`, `vertexValue`, `vertexFinset`; `mem_productList`, `vertices_complete` (no extra hypotheses, dense marginals only, O16), `vecs_eq_convexHull_vertices`, `dot_piVec_eq_sum`, `exists_vertex_expansion`, `vertexValue_eq_opt` (all four modes, no extra hypotheses, dense marginals only, O16), `vertexValue_eq_stateActionBellman` (same scope) |
| `AxiomCheck.lean` | `#print axioms` for every mapped theorem |
| `DocLint.lean` | `#lint only docBlame docBlameThm` |

## Modelling notes

* **Factored IMDPs.** The ambiguity set of a factored IMDP is the literal product of the marginal
  sets, `FactoredIMDP.productSet`, and `FactoredIMDP.toRMDP` uses exactly that set (no convex hull).
  It is not convex in general (`FactoredIMDP.productSet_not_convex`; arXiv:2411.11803,
  arXiv:2508.00707), so `AmbiguitySet.WellFormed` requires only nonempty and closed, and the general
  `RMDP` results of later phases must not assume convexity. The factored algorithms (Phase 5) follow
  the cited papers. The model assumes `source_dims = state_vars`. Vertex enumeration (5a) is
  modelled for dense marginals (`support(p) = 1:d`); for sparse marginals Julia permutes only the
  support, which is not modelled. Its exactness proof follows arXiv:2508.00707, Theorem 1 (convex
  hull of the product set = convex hull of products of marginal vertices), without convexifying
  `productSet`.
* **Product process.** The DFA step uses the label of the *successor* state, as `bellman.jl` does
  (the `ProductProcess` docstring writes the source label).
* **Available actions** must be nonempty in Lean; Julia's `ListAvailableActions` does not check it.
* **Strategy validity** in Lean requires availability; Julia's `checkstrategy` checks only the
  action range. With `ListAvailableActions` a `VerificationProblem` therefore accepts strategies
  with unavailable actions, and the verified value can exceed the optimum (Finding F2 in the
  inventory; `Index.checkStrategy_admits_unavailable`). `policy_eval_sound` assumes a valid strategy.
* **Strategy extraction seed.** `argoptAction` starts from an available seed. Julia's
  `TimeVaryingStrategyCache` seeds with `first(available_actions)` (in `ℝ`, `typemin` loses the first
  comparison); `StationaryStrategyCache` seeds with the previous action unless the cache entry is
  zero or the guard `jₛ ∉ available_actions` (state index compared with the action list) holds
  (Observation O11). On a model with fixed available actions the seed is always available
  (`stationarySeed_available`); with `TimeVaryingAvailableActions` it need not be. The strategy
  theorems of 3d (`stationaryCacheSeed`) model the **documented** stationary cache (the previous
  action is always the seed); `stationary_sound` depends on it. Julia's guard resets the seed for
  every state whose index exceeds the number of actions, and then the returned stationary strategy
  can be strictly worse than the reported value for `maximize` (Finding F3, benchmark B-1; witness
  `Examples.b1_stationary_unsound`). For `minimize` every valid strategy is sound
  (`viIter_le_superSolution_minimize`), so the guard cannot break it.
* **IVI strategy coupling.** `ivi_step!` synthesizes the strategy on the primary bound (`V_lower`
  for `Pessimistic`, `V_upper` for `Optimistic`) and applies it, with the same nature, to the other
  bound. The primary bound is plain robust value iteration (`primary_iviIter`) and is sound; the
  secondary bound is sound only when the strategy-mode direction agrees (a fixed strategy is never
  better than the optimum). For `(Pessimistic, Maximize)` the upper bound can fall below `V*`, for
  `(Optimistic, Minimize)` the lower bound can exceed `V*` (Finding F4), and Julia's gap test can
  then stop with gap `0` at a value that is not `V*` (`½` away in the F4 model,
  `Examples.not_gap_stop_sound_*`). `V*` is `reachLfp` (the infinite-horizon value); the
  finite-horizon value (`IVIFixedIterationsCriteria`) is not compared (Observation O15).
