# IntervalMDP.jl — Verification Inventory

> Created in Phase 0 of `harness/specs/lean-proofs-existing-algorithms.md` (now split into `harness/specs/lean-proofs/`) from
> `TEMPLATE-onboarding-inventory.md`. Dev keeps it current whenever a proof is added or an
> algorithm changes. This inventory is a status report, **not** a verification claim.

- Target: `git@github.com:Zinoex/IntervalMDP.jl.git` @ `20fc03b90d3a3dbe96cfe8335a47052e99ffa1f5` (base of Phase 0; the Lean project is uncommitted work on top of it)
- Target root: `/home/fresen/.julia/dev/IntervalMDP`
- Julia compat: `julia = "1.11"`; Lean root / pinned toolchain: `/home/fresen/.julia/dev/IntervalMDP/lean` / `leanprover/lean4:v4.33.0-rc2` (Mathlib tag `v4.33.0-rc2`)
- Inventory date: `2026-10-05`
- Phase completed: **0** (models, well-formedness, A1); Phase **1a** (index foundations: linear index, sparse support pairing, sort permutation); Phase **1b** (marginal indexing: `Marginal` and `IntervalAmbiguitySets` `sub2ind`); Phase **1c** (dense O-maximization); Phase **1d** (sparse O-maximization). **Phase 1 is complete.** Phases 2–6: not started.

## Standard limitations (apply to every row)

Proof scope is `abstract` for every row (values in `ℝ`, semantics = dynamic-programming
recursion). Not covered:

- L1 floating-point rounding, including the early-exit test `budget ≤ 0` and accumulated error in `budget -= p`;
- L2 overflow and underflow (beyond the index bound stated per index theorem, `Π dims < 2^31` for `Int32` paths);
- L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`);
- L4 the Lean↔Julia correspondence (argued via literal transcription, docstrings and cross-check tests, not proved);
- L5 DP value ↔ path-measure semantics (`ℙ^{π,η}[…]`);
- L6 external LP solver correctness (McCormick).

## Models (Phase 0)

| # | Model | Julia (type — file) | Lean definition — file | Lean theorem(s) | Status | Scope | Limitations |
|---|---|---|---|---|---|---|---|
| M1 | Interval ambiguity set | `IntervalAmbiguitySet` — `src/probabilities/IntervalAmbiguitySets.jl` | `IntervalMDP.IntervalAmbiguity.toSet` — `lean/IntervalMDPProofs/Models/IntervalAmbiguity.lean` | `IntervalMDP.IntervalAmbiguity.toSet_wellFormed` (nonempty, closed), `IntervalMDP.IntervalAmbiguity.toSet_convex`, `IntervalMDP.IntervalAmbiguity.budget_eq` | `proved` | `abstract` | L1–L4 |
| M2 | IMDP | `IntervalMarkovDecisionProcess` — `src/models/IntervalMarkovDecisionProcess.jl` | `IntervalMDP.IMDP.toRMDP` — `lean/IntervalMDPProofs/Models/IMDP.lean` | `IntervalMDP.IMDP.toRMDP_wellFormed` | `proved` | `abstract` | L1–L4; column lookup (`sub2ind`) abstracted as `ambiguitySets s a` until Phase 1 |
| M3 | Factored IMDP | `FactoredRobustMarkovDecisionProcess`, `Marginal` — `src/models/FactoredRobustMarkovDecisionProcess.jl`, `src/probabilities/Marginal.jl` | `IntervalMDP.FactoredIMDP.toRMDP` — `lean/IntervalMDPProofs/Models/Factored.lean` | `IntervalMDP.FactoredIMDP.toRMDP_wellFormed` (every product of marginal distributions is a `ProbVec` in `productSet`; `productSet` nonempty and closed), `IntervalMDP.FactoredIMDP.productSet_not_convex` (in `Models/Examples.lean`) | `proved` | `abstract` | L1–L4; RMDP ambiguity set is the literal product set `FactoredIMDP.productSet` (no hull; convexity not assumed, see F1); `source_dims = state_vars` assumed (terminal slices not modeled); marginal indices assumed strictly increasing (spec invariant; Julia does not check it) |
| M4 | Product process | `ProductProcess` — `src/models/ProductProcess.jl` | `IntervalMDP.ProductProcess.toRMDP` — `lean/IntervalMDPProofs/Models/Product.lean` | `IntervalMDP.ProductProcess.toRMDP_wellFormed` (deterministic and probabilistic labelling, via `AbstractLabelling`); supporting `IntervalMDP.ProductProcess.lift_deterministic` | `proved` | `abstract` | L1–L4; successor-label convention of `src/bellman.jl` (the docstring states the source label, see Observation O3) |
| M5 | Examples | docstrings of `RobustValueIteration` (`src/robust_value_iteration.jl`), `FactoredRobustMarkovDecisionProcess` | `IntervalMDP.Examples.docIMDP`, `IntervalMDP.Examples.docFIMDP` — `lean/IntervalMDPProofs/Models/Examples.lean` | (definitions; all invariants discharged by `norm_num`) | `proved` (typechecks) | `abstract` | decimal literals read as exact reals |
| M6 | Other models (no theorem required) | `DFA`, labellings, strategies, `Property`, `Specification` | `Models/{DFA,Strategy,Specification,RMDP,AmbiguitySet,Distribution}.lean` | — | defined | `abstract` | — |

## Approximation soundness

| # | Approximation | Phase | Lean theorem(s) | Status | Scope | Limitations |
|---|---|---|---|---|---|---|
| A1 | Generic lift: a sound one-step operator stays sound through VI | 0 | `IntervalMDP.Approx.iter_sound` (iterates and limits), `IntervalMDP.Approx.fixedPoint_sound` — `lean/IntervalMDPProofs/Approx/Lift.lean`; definition `IntervalMDP.Approx.Sound` — `Approx/Sound.lean` | `proved` | `abstract` | L1–L5 |
| A2 | Recursive O-max (fIMDP) | 5 | `IntervalMDP.Factored.recursiveOMax_sound`, `recursiveOMax_vi_sound` | `none` | — | L1–L5 |
| A3 | LP McCormick relaxation (fIMDP) | 5 | `IntervalMDP.Factored.mcCormick_sound`, `mcCormick_vi_sound` | `none` | — | L1–L6 |
| A4 | Stopping infinite-horizon reachability / reach-avoid / exit time at finite k | 3 | `IntervalMDP.VI.reachIter_sound`, `IntervalMDP.VI.exitIter_sound` | `none` | — | L1–L5 |
| A5 | Stopping infinite-horizon reward at the ε-criterion | 3 | `IntervalMDP.VI.reward_error_bound` | `none` | — | L1–L5 |
| A6 | IVI bounds | 4 | `IntervalMDP.IVI.bracket`, `IntervalMDP.IVI.gap_stop_sound` | `none` | — | L1–L5 |
| A7 | Synthesized strategy | 3 | `IntervalMDP.VI.timeVarying_attains`, `IntervalMDP.VI.stationary_sound` | `none` | — | L1–L5 |
| A8 | Given-strategy verification | 2 | `IntervalMDP.Bellman.policy_eval_sound` | `none` | — | L1–L5; see Observation O2 |

## Indexing

| Ph | Index computation | Julia (function — file) | Lean theorem(s) | Status | Scope | Limitations |
|---|---|---|---|---|---|---|
| 1 | Column-major linear index | `LinearIndices`, `CartesianIndices` (Base), as used for `V[I]` and `FullUpdateSequence` (`src/bellman.jl`, `src/update_sequence.jl`) | `IntervalMDP.Index.linear_bijective` (bijection onto `1..∏ dims`, and the `N`-bit value is exact), `IntervalMDP.Index.linear_succ_first` (first dimension fastest) — `lean/IntervalMDPProofs/Index/Linear.lean` (definition `IntervalMDP.Index.linear`; 1-based conversion `toJulia`, overflow model `machineInt` in `Index/Julia.lean`) | `proved` (Phase 1a) | `abstract` | L1–L6; index bound `∏ dims < 2^(N-1)` is a hypothesis of both theorems (`N = 64` for `Int`, `N = 32` for `Int32` paths) |
| 1 | Marginal → ambiguity-set column | `sub2ind(::Marginal, action, source)` — `src/probabilities/Marginal.jl` | `IntervalMDP.Index.marginalSub2ind_eq_linear` (equals `linear` over `(action_vars…, source_dims…)` on `(action[action_indices]…, source[state_indices]…)`, actions first; the `N`-bit value is the same), `IntervalMDP.Index.marginalSub2ind_bijective` (values = `1..∏ action_vars · ∏ source_dims`, equal values ⇒ equal conditioning tuples), `IntervalMDP.Index.marginalSub2ind_depends_only` — `lean/IntervalMDPProofs/Index/Marginal.lean` (literal loop transcription `IntervalMDP.Index.marginalSub2ind`, `N`-bit value `marginalSub2indInt`; supporting `marginalCartesian_surjective`); cross-check `test/base/indexing_reference.jl` | `proved` (Phase 1b) | `abstract` | L1–L6; index bound `∏ dims < 2^(N-1)` is a hypothesis of `_eq_linear` and `_bijective` (`N = 64` for `Int` tuples on the CPU, `N = 32` for `Int32` on CUDA); `_depends_only` needs no bound. Mode-independent (pure index arithmetic: no satisfaction/strategy mode involved). Marginal indices strictly increasing (O4) and `source_dims = sv.dims ∘ state_indices` (`check_transition`) are structure invariants; for terminal slices (O5) instantiate `StateVars` with `source_dims` |
| 1 | Non-factored set lookup | `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ) = jₛ[1]` — `src/probabilities/IntervalAmbiguitySets.jl` | `IntervalMDP.Index.intervalSub2ind_correct` (equals `sub2ind(::Marginal, …)` when the marginal conditions on state variable 1 only and every conditioning action variable has one value, i.e. a single-action `IntervalMarkovDecisionProcess` layout), `IntervalMDP.Index.intervalSub2ind_wrong_multiAction` (with two actions differing on a conditioning variable it cannot match; Observation O8) — `lean/IntervalMDPProofs/Index/Marginal.lean` (model `IntervalMDP.Index.intervalSub2ind`); cross-check `test/base/indexing_reference.jl` | `proved` (Phase 1b; see O8) | `abstract` | L1–L6; mode-independent (pure index arithmetic). Reachability is argued by call-site inspection (L4), not proved: no call site in `src/` or `ext/` reaches it (see O8) |
| 1 | Sparse support pairing | `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` — `src/bellman.jl` | `IntervalMDP.Index.sparse_zip_correct` — `lean/IntervalMDPProofs/Index/Sparse.lean` (structure `IntervalMDP.Index.SparseCol` = CSC column invariant) | `proved` (Phase 1a) | `abstract` | L1–L6; the CSC invariant (strictly increasing, in-range `rowval`, `nzval` aligned) is assumed as structure fields — Julia's `checkprobabilities` does not re-check it (Observation O6); no integer products, so no index bound |
| 1 | Sort permutation | `sortperm!(perm, V; rev = upper_bound)`, loop of `gap_value(V, gap, budget, perm)` — `src/bellman.jl` | `IntervalMDP.Index.sortedPerm_bijective` (both `rev = true/false`), `IntervalMDP.Index.greedy_visits_once` — `lean/IntervalMDPProofs/Index/Perm.lean` (structure `IntervalMDP.Index.SortedPerm`, loop transcription `gapValue`); supporting `IntervalMDP.Index.sortedPerm_fits_int32` (`n < 2^31`), `IntervalMDP.Index.gapValue_eq_sum_allocation` (early exit does not change the result) | `proved` (Phase 1a) | `abstract` | L1–L6; in particular L1: the early exit `budget <= 0` is exact in `ℝ` (the loop breaks only at budget `0`); in floating point `budget -= p` can leave a residual, so the loop may continue; `perm` is modeled by a stable merge sort (Julia documents `sortperm` as stable; equality with Julia's order on ties is argued, L4) |
| 2 | Strategy lookup | `CartesianIndex(strategy_cache[jₛ])` — `src/bellman.jl`, `src/strategy_cache.jl` | `IntervalMDP.Index.strategyAction_available` | `none` | — | L2–L4; see Observation O2 |
| 5 | Factored successor index | `CartesianIndices(num_target.(ambiguity_sets))` — `src/bellman.jl` | `IntervalMDP.Index.factored_successor_eq` | `none` | — | L2–L4 |
| 6 | Product state | `V[idx, dfa[state, lf[idx]]]`, `selectdim(Vres, ndims(Vres), state)` — `src/bellman.jl` | `IntervalMDP.Index.productIndex_bijective`, `IntervalMDP.Index.product_read_successor` | `none` | — | L2–L4 |

## VI / Bellman algorithms

| # | Algorithm | Model family | Julia entry point (function — file) | CPU / GPU paths | Lean theorem(s) (name — file) | Proof status | Proof scope | Limitations |
|---|---|---|---|---|---|---|---|---|
| 1 | O-maximization, dense | interval | `state_action_bellman(::DenseIntervalOMaxWorkspace, …)`, `gap_value`, `bellman_precomputation!` — `src/bellman.jl` | dense CPU, CUDA (`ext/`) | `IntervalMDP.OMax.omax_mem` (the greedy distribution `p = lower + allocation` is in `P(l, u)` and `omax = ⟨p, V⟩`), `IntervalMDP.OMax.omax_eq_sSup` (`upper_bound = true`, descending `perm`: `omax = sSup {⟨p, V⟩ : p ∈ P(l, u)}`), `IntervalMDP.OMax.omax_eq_sInf` (`upper_bound = false`, ascending: `sInf`), `IntervalMDP.OMax.omax_tie_invariant` (any two permutation vectors sorting `V` in the same direction, stable or not, give the same value) — `lean/IntervalMDPProofs/OMax.lean` (definitions `IntervalMDP.OMax.omax` = `stateActionBellman A s.V s.perm` = `dot(V, lower) + gapValue(V, gap, perm, budget)`, transcription reusing `Index.gapValue`; `greedy`; `Permutation`); supporting `stateActionBellman_isGreatest`, `stateActionBellman_isLeast`, `stateActionBellman_eq_dot`; cross-check `test/base/omax_reference.jl` (Phase 1d: `IntervalMDP.bellman` on small dense IMDPs vs. a brute-force HiGHS LP over `P(l, u)`, both `upper_bound` and both `maximize` values, `Float64`, tolerance `1e-9`) | `proved` (Phase 1c) | `abstract` | L1–L6. Hypotheses: structure invariants of `IntervalAmbiguity` only, plus `upper_bound = true/false` for `_eq_sSup`/`_eq_sInf`. Mode coverage: O-max is mode-agnostic; modes reach it only through `upper_bound = isoptimistic(spec)` (`src/robust_value_iteration.jl`), the strategy mode acts afterwards across actions. Both directions are proved, so all four satisfaction × strategy modes are covered. L1 in particular: the early exit `budget <= 0` and the accumulated error of `budget -= p` are exact only in `ℝ` (Lean replaces the early exit by `p_i = min(budget_i, gap_i) = 0` via `greedy_visits_once`/`gapValue_eq_sum_allocation`); tie invariance holds in `ℝ` only (different tie orders can round differently, O9). L2: overflow/underflow and `NaN` in `V` not modeled. L3: CUDA kernels and `@threadstid` not modeled. L4: Lean↔Julia by literal transcription (`gap_value` loop, `dot(V, lower)`). L5: DP value ↔ path measure not proved. L6: no LP solver in the algorithm; the cross-check's LP reference uses HiGHS, which is not verified |
| 2 | O-maximization, sparse | interval | `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` (`src/bellman.jl:536–553`: `zip(V[support], nonzeros(gap))`, `sort!(Vp_workspace; rev = upper_bound, by = first)`, `dot(V, lower) + gap_value(Vp, budget)`), `gap_value(Vp, budget)` (`src/bellman.jl:555–571`) — `src/bellman.jl` | sparse CPU, CUDA (`ext/`) | `IntervalMDP.OMax.omaxSparse_eq_omax` (for every `SparseIntervalAmbiguity` and every sort result `SortedValuesGaps` — stable or not, ties in any order — `omaxSparse = omax` on the same ambiguity set, the dense gap being the sparse column's `getindex`, `0` off the support) — `lean/IntervalMDPProofs/OMax.lean` (definitions `IntervalMDP.OMax.omaxSparse` = `dot(V, lower) + gapValueSparse(Vp, budget, 0)`, literal loop transcription `gapValueSparse` of `gap_value(Vp, budget)`; structures `SparseIntervalAmbiguity` (= `IntervalAmbiguity` + CSC gap column `gapCol : Index.SparseCol` + invariant `gap_eq`) and `SortedValuesGaps` (any permutation of `Index.valuesGaps` sorted by first component in direction `rev = upper_bound`), stable instance `stableValuesGaps`; uses `Index.sparse_zip_correct` and `omax_tie_invariant`); supporting `IntervalMDP.OMax.omaxSparse_exact` (`= sSup` for `upper_bound = true`, `= sInf` for `false`), `gapValue_sublist` (zero-gap rows inserted in the loop order change nothing), `exists_permutation_sublist` (a sorted support order extends to a full sorted `perm`); cross-check `test/base/omax_reference.jl` (`IntervalMDP.bellman` on small sparse IMDPs vs. a brute-force HiGHS LP, both `upper_bound`/`maximize` values, `Float64`, tolerance `1e-9`; cases: degenerate `l = u` with budget 0, ties in `V`, all budget in one successor, sparse column with an all-zero stored gap, 40 seeded random dense+sparse IMDPs) | `proved` (Phase 1d) | `abstract` | L1–L6. Hypotheses: structure invariants only (`IntervalAmbiguity`, CSC invariant of `SparseCol`, `gap_eq` linking the stored column to `gap`). Mode coverage: modes reach O-max only through `upper_bound = isoptimistic(spec)` (`src/robust_value_iteration.jl`); `omaxSparse_eq_omax` holds for both `upper_bound` values, so all four satisfaction × strategy modes are covered (the strategy mode acts afterwards across actions). L1: the early exit `budget <= 0` and the accumulated error of `budget -= p` are exact only in `ℝ`; the sparse and dense paths (and different tie orders of `sort!`) can round differently in floating point (O10). L2: overflow/underflow and `NaN` in `V` not modeled. L3: CUDA kernels (`ext/`) and `@threadstid` (`ThreadedSparseIntervalOMaxWorkspace`) not modeled. L4: Lean↔Julia by literal transcription (`zip` loop, `sort!` by first, `gap_value(Vp, budget)` loop, `dot(V, lower)`) plus the cross-check test, not proved; `lower` is modeled as a dense function (a sparse `dot` gives the same sum). L5: DP value ↔ path measure not proved. L6: no LP solver in the algorithm; the cross-check's LP reference uses HiGHS, which is not verified |
| 3 | Robust Bellman operator | RMDP (general) | `bellman!` → `state_bellman!` (`OptimizingStrategyCache`) — `src/bellman.jl` | dense, sparse, CUDA | `IntervalMDP.Bellman.T_mono`, `T_add_const`, `T_nonexpansive`, `T_interval_eq_omax` — `Bellman.lean` | `none` | — | L1–L4 |
| 4 | Strategy extraction and evaluation | RMDP | `extract_strategy!` — `src/strategy_cache.jl`; `NonOptimizingStrategyCache` path | CPU, CUDA | `IntervalMDP.Bellman.argopt_attains`, `IntervalMDP.Bellman.policy_eval_sound` — `Bellman.lean` | `none` | — | L1–L4 |
| 5 | Robust VI: reachability / reach-avoid | interval, factored, product | `_value_iteration!`, `step!`, `initialize!`/`step_postprocess_value_function!` — `src/robust_value_iteration.jl`, `src/specification.jl` | CPU, CUDA | `IntervalMDP.VI.reachIter_mem_unit`, `reachIter_mono`, `reachIter_tendsto_lfp`, `reachIter_sound` — `VI/Reach.lean` | `none` | — | L1–L5 |
| 6 | Robust VI: safety | interval, factored | `AbstractSafety` initialize/postprocess — `src/specification.jl` | CPU, CUDA | `IntervalMDP.VI.safety_shift_eq` — `VI/Safety.lean` | `none` | — | L1–L5 |
| 7 | Robust VI: reward | interval, factored | `AbstractReward` initialize/postprocess — `src/specification.jl` | CPU, CUDA | `IntervalMDP.VI.rewardIter_succ`, `reward_contracting`, `reward_error_bound` — `VI/Reward.lean` | `none` | — | L1–L5 |
| 8 | Robust VI: expected exit time | interval, factored | `ExpectedExitTime` initialize/postprocess — `src/specification.jl` | CPU, CUDA | `IntervalMDP.VI.exitIter_mono`, `exitIter_succ`, `exitIter_sound` — `VI/ExitTime.lean` | `none` | — | L1–L5 |
| 9 | Synthesized strategy | interval, factored, product | `TimeVaryingStrategyCache`, `StationaryStrategyCache` — `src/strategy_cache.jl` | CPU, CUDA | `IntervalMDP.VI.timeVarying_attains`, `stationary_sound` — `VI/Strategy.lean` | `none` | — | L1–L5 |
| 10 | Interval value iteration | interval, factored | `ivi_step!`, `initialize_ivi!`, `IVIInitialGapCriteria` — `src/interval_value_iteration.jl`, `src/specification.jl` | CPU | `IntervalMDP.IVI.lower_le_upper`, `bracket`, `gap_stop_sound` — `IVI.lean` | `none` | — | L1–L5 |
| 11 | Vertex enumeration | factored | `state_action_bellman(::FactoredVertexIteratorWorkspace, …)` — `src/bellman.jl` | CPU | `IntervalMDP.Factored.vertices_complete`, `vertexValue_eq_opt` — `Factored.lean` | `none` | — | L1–L4 |
| 12 | Recursive O-max | factored | `state_action_bellman(::FactoredIntervalOMaxWorkspace, …)` — `src/bellman.jl` | CPU, CUDA | `IntervalMDP.Factored.recursiveOMax_sound`, `recursiveOMax_vi_sound` — `Factored.lean` | `none` | — | L1–L5 |
| 13 | LP McCormick relaxation | factored | `state_action_bellman(::FactoredIntervalMcCormickWorkspace, …)` — `src/bellman.jl` | CPU | `IntervalMDP.Factored.mcCormick_sound`, `mcCormick_vi_sound` — `Factored.lean` | `none` | — | L1–L6 |
| 14 | Product process (IMDP × DFA) | product | `bellman!` for `ProductProcess`, DFA reach/safety postprocess — `src/bellman.jl`, `src/specification.jl` | CPU, CUDA | `IntervalMDP.Product.T_eq_flat`, `IntervalMDP.Product.dfaReach_eq` — `Product.lean` | `none` | — | L1–L5 |

## Findings

- **Phase 1d: no findings.** `omaxSparse_eq_omax` holds as stated for the transcribed sparse
  `state_action_bellman`/`gap_value(Vp, budget)`, for every sort of the support pairs; the LP
  cross-check `test/base/omax_reference.jl` agrees with `IntervalMDP.bellman` within `1e-9` on all
  cases (see Observation O10).
- **Phase 1c: no findings.** `omax_mem`, `omax_eq_sSup`, `omax_eq_sInf`, `omax_tie_invariant`
  hold as stated for the transcribed `state_action_bellman`/`gap_value` (see Observation O9).
- **Phase 1b: no findings.** `marginalSub2ind_eq_linear`, `_bijective`, `_depends_only` hold as
  stated for the Julia loop, and `intervalSub2ind_correct` holds for the single-action condition of
  § Indexing Correctness (see Observation O8 for the latent multi-action defect).
- **Phase 1a: no findings.** All five index theorems hold as stated for the Julia behaviour (see
  Observations O6, O7 for assumptions recorded alongside).
- **F1 (Phase 0, models — design finding, no Julia defect) — RESOLVED by spec amendment.** The
  ambiguity set of a factored IMDP, `Γ_{s,a} = ⨂ᵢ Γⁱ` (products of marginal distributions;
  `FactoredRobustMarkovDecisionProcess` docstring), is **not convex** in general. The first Phase 0
  run found that this contradicts the original spec's RMDP invariant (nonempty, convex, closed) and
  worked around it with a closed convex hull; the user was asked to decide. The spec was amended
  (§ "Factored ambiguity is not convex"): `AmbiguitySet.WellFormed` is now nonempty + closed only,
  convexity is the separate predicate `AmbiguitySet.IsConvex`, and `FactoredIMDP.toRMDP` uses the
  literal `FactoredIMDP.productSet` (the hull, `AmbiguitySet.hull`, has been removed). The fact is
  recorded as `IntervalMDP.FactoredIMDP.productSet_not_convex` (witness `Examples.binaryFIMDP`: two
  binary variables, unconstrained marginals; the point masses at `(0,0)` and `(1,1)` are products,
  their midpoint is not), citing arXiv:2411.11803 and arXiv:2508.00707. Julia setting (not run):
  two binary state variables whose selected marginal columns are `lower = [0, 0]`,
  `upper = [1, 1]`; the joint distributions `[1 0; 0 0]` and `[0 0; 0 1]` are feasible, their
  average `[0.5 0; 0 0.5]` is not a product. Consequences for later phases: the general `RMDP`
  results (Phase 2 onward) must hold without convexity, the exact value `V*` of a factored IMDP is
  defined over `productSet`, and the factored algorithm proofs (Phase 5) follow the cited papers.

## Blockers

- **B1 (Phase 0 run 1, Julia tests) — RESOLVED.** The first Phase 0 run saw 11 failures in
  `test/base/specification.jl`: its expected `show` strings depended on the Julia version (local
  `julia` 1.13.1). The operator fixed the expected strings in a separate, approved change
  (version-independent strings in `test/base/specification.jl`), which is not part of this phase.
  No `src/` change was involved.

## Observations (not findings; recorded for later phases)

- **O1.** `ListAvailableActions` does not check that each state's list is nonempty; the Lean
  `AvailableActions` requires it (otherwise the Bellman optimum is undefined).
- **O2.** `checkstrategy` (`src/strategy.jl`) checks only `1 ≤ a ≤ action_vars`, not membership in
  `available(model, s)`. Phase 2's `strategyAction_available` ("for strategies that pass
  validation") is likely to need availability as an extra hypothesis or to become a Finding.
- **O3.** The `ProductProcess` docstring writes `δ_{q, L(s)}` (source label); `src/bellman.jl` uses
  the successor's label (`dfa[state, lf[idx]]`). The Lean model follows the code.
- **O4.** The spec lists "strictly increasing" marginal indices as matching `checkindices`, but
  `checkindices` (`src/probabilities/Marginal.jl`) checks only positivity and the column count.
  The Lean `Marginal` requires strict monotonicity (spec invariant), so it is narrower than Julia.
- **O5.** The Lean factored model assumes `source_dims = state_vars`; Julia's terminal slices
  (`source_dims < state_vars`) are not modeled yet.

- **O6 (Phase 1a).** `sparse_zip_correct` assumes the `SparseMatrixCSC` column invariant (row
  indices strictly increasing and in range, `nzval` aligned to `rowval`). SparseArrays maintains it
  for matrices built through its constructors; `checkprobabilities` checks only that `lower` and
  `gap` have the same `rowvals`, not the invariant itself.
- **O7 (Phase 1a).** The O-max permutation is a `Vector{Int32}` (`src/workspace.jl`). For
  `n ≥ 2^31` targets `sortperm!` cannot store the indices (Julia raises `InexactError`, no silent
  wrap); `sortedPerm_fits_int32` proves the entries fit for `n < 2^31`.
- **O8 (Phase 1b).** Latent defect in the exported `IntervalAmbiguitySets` method, reachable only
  by user code calling it directly (not a Finding: § Indexing Correctness counts it only if package
  call sites reach it with more than one action, and none do).
  `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ) = jₛ[1]` and the public method
  `getindex(p::IntervalAmbiguitySets, jₐ, jₛ) = p[sub2ind(p, jₐ, jₛ)]`
  (`src/probabilities/IntervalAmbiguitySets.jl:284–295`) ignore the action (and every source
  coordinate after the first). For a set of columns laid out as in `Marginal` /
  `IntervalMarkovDecisionProcess` (column `(s - 1) * num_actions + a`) they return the wrong
  column whenever there is more than one action. Lean: `intervalSub2ind_wrong_multiAction` (for any
  two actions differing on a conditioning action variable, `intervalSub2ind` disagrees with
  `marginalSub2ind` on one of them); `intervalSub2ind_correct` proves agreement on single-action
  layouts. Julia (run on 1.13.1):

      sets = IntervalAmbiguitySets(; lower = zeros(2, 4), upper = ones(2, 4))
      marginal = Marginal(sets, (2,), (2,))
      IntervalMDP.sub2ind(marginal, (2,), (1,))  # 2
      IntervalMDP.sub2ind(sets, (2,), (1,))      # 1  (wrong column for action 2)
      sets[(2,), (1,)] == sets[1]                # true; marginal[(2,), (1,)] == sets[2]

  Reachability (call-site inspection at `origin/main`): every `sub2ind` / two-argument lookup goes
  through `Marginal` — `src/bellman.jl:471–472, 499–500, 830–831, 858–859` (`marginals(model)[…]`),
  `src/Data/bmdp-tool.jl:213`, `src/Data/prism.jl:105` (`marginal[jₐ, jₛ]`),
  `ext/cuda/bellman/dense.jl:260, 294`, `ext/cuda/bellman/sparse.jl:339, 380, 762, 800`,
  `ext/cuda/bellman/factored.jl:451–848` (`model[k][jₐ, jₛ]` with `getindex(::FactoredRMDP, r) =
  transition[r]`, a `Marginal`); `FactoredRobustMarkovDecisionProcess.transition` is typed
  `NTuple{N, Marginal}`, and `IntervalMarkovDecisionProcess`/`IntervalMarkovChain` wrap their sets
  in `Marginal(sets, source_dims, action_vars)`. `Marginal.getindex` calls the one-argument
  `ambiguity_sets[j]`. So the package's algorithms never reach the defective method; only user
  code calling `sets[jₐ, jₛ]` / `sub2ind(sets, …)` directly on an exported `IntervalAmbiguitySets`
  can. Fix (out of scope, no `src/` change): remove the method or make it error for more than one
  action.
- **O9 (Phase 1c).** No defect found in dense O-maximization. Notes for later phases: (a)
  `omax_tie_invariant` and exactness hold in exact arithmetic; in `Float64`/`Float32` two tie
  orders (e.g. CPU `sortperm!` vs. a CUDA sort) can accumulate `res += p * V[i]` and
  `budget -= p` in a different order and differ by rounding (L1, L3). (b) `sortperm!` orders `NaN`
  entries of `V` by `isless` (largest); values in `ℝ` have no `NaN`, so this is not modeled (L2).
- **O10 (Phase 1d).** No defect found in sparse O-maximization. Notes: (a) in exact arithmetic the
  sparse path equals the dense path for every sort of `Vp` (`omaxSparse_eq_omax`), so Julia's
  stable `sort!` (default since Julia 1.9) is not needed for correctness; in floating point, the
  sparse loop (support only) and the dense loop (all rows, zero gaps included) and different tie
  orders can accumulate `res`/`budget` differently and differ by rounding (L1). (b) A sparse column
  with no stored entries is not a valid ambiguity set (`∑ u = 0 < 1`), and the positional
  `IntervalAmbiguitySets(lower, gap)` requires `lower` and `gap` to share their column structure, so
  the "empty gap" case is a column whose stored gaps are all `0` (budget `0`); the cross-check
  covers it with explicit stored zeros. (c) The keyword constructor takes the support from
  `upper`'s stored entries (`compute_gap`), so stored rows may have gap `0`; the Lean model allows
  this (`gap_eq` via `getindex`, stored zeros included). (d) The cross-check reference is an LP
  solved by HiGHS (L6: unverified); it agrees with `bellman` within `1e-9` on all cases.

## Legacy verification gaps

Every algorithm row (1–14) except rows 1 and 2 (dense and sparse O-maximization, `proved` in Phases 1c and 1d), A2–A8 and every index row except the three Phase 1a rows (linear index, sparse support pairing, sort permutation) and the two Phase 1b rows (`Marginal` and `IntervalAmbiguitySets` `sub2ind`; `proved`; see Observation O8) are `none`: no Lean theorem yet. A task
touching one of these must supply its theorem and proof before it can pass the Formal Verification
gate.

## Summary

- Algorithms: 14; proved: 2 (dense O-maximization, Phase 1c; sparse O-maximization, Phase 1d); partial: 0; none: 12.
- Models: 5 mapped rows proved (M1–M5), including `toSet_convex` and `productSet_not_convex`; approximation: A1 proved, A2–A8 none; indexing: 5 of 8 rows proved (Phase 1a: linear index, sparse support pairing, sort permutation; Phase 1b: `Marginal` and `IntervalAmbiguitySets` `sub2ind`, with Observation O8 recorded).
- Statement: the package is **not** verified. Phase 0 proves only model well-formedness and the
  generic soundness lift; Phase 1 (complete) adds the index theorems and dense and sparse O-max exactness, all at abstract scope.
- Phase 1 close: every Phase 1 row is `proved` — index rows linear index, `Marginal` `sub2ind`, `IntervalAmbiguitySets` `sub2ind` (Observation O8), sparse support pairing, sort permutation; algorithm rows 1 (dense O-max) and 2 (sparse O-max). No Phase 1 row is `none` or `partial`; no open Findings.
