# IntervalMDP.jl — Verification Inventory

> Created in Phase 0 of `harness/specs/lean-proofs-existing-algorithms.md` from
> `TEMPLATE-onboarding-inventory.md`. Dev keeps it current whenever a proof is added or an
> algorithm changes. This inventory is a status report, **not** a verification claim.

- Target: `git@github.com:Zinoex/IntervalMDP.jl.git` @ `20fc03b90d3a3dbe96cfe8335a47052e99ffa1f5` (base of Phase 0; the Lean project is uncommitted work on top of it)
- Target root: `/home/fresen/.julia/dev/IntervalMDP`
- Julia compat: `julia = "1.11"`; Lean root / pinned toolchain: `/home/fresen/.julia/dev/IntervalMDP/lean` / `leanprover/lean4:v4.33.0-rc2` (Mathlib tag `v4.33.0-rc2`)
- Inventory date: `2026-10-05`
- Phase completed: **0** (models, well-formedness, A1). Phases 1–6: not started.

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
| 1 | Column-major linear index | `LinearIndices`, `CartesianIndices` (Base) | `IntervalMDP.Index.linear_bijective`, `IntervalMDP.Index.linear_succ_first` | `none` | — | L2–L4 |
| 1 | Marginal → ambiguity-set column | `sub2ind(::Marginal, …)` — `src/probabilities/Marginal.jl` | `IntervalMDP.Index.marginalSub2ind_eq_linear`, `_bijective`, `_depends_only` | `none` | — | L2–L4 |
| 1 | Non-factored set lookup | `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ) = jₛ[1]` — `src/probabilities/IntervalAmbiguitySets.jl` | `IntervalMDP.Index.intervalSub2ind_correct` | `none` | — | L2–L4 |
| 1 | Sparse support pairing | `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` — `src/bellman.jl` | `IntervalMDP.Index.sparse_zip_correct` | `none` | — | L2–L4 |
| 1 | Sort permutation | `sortperm!(perm, V; rev = upper_bound)` — `src/bellman.jl` | `IntervalMDP.Index.sortedPerm_bijective`, `IntervalMDP.Index.greedy_visits_once` | `none` | — | L1–L4 |
| 2 | Strategy lookup | `CartesianIndex(strategy_cache[jₛ])` — `src/bellman.jl`, `src/strategy_cache.jl` | `IntervalMDP.Index.strategyAction_available` | `none` | — | L2–L4; see Observation O2 |
| 5 | Factored successor index | `CartesianIndices(num_target.(ambiguity_sets))` — `src/bellman.jl` | `IntervalMDP.Index.factored_successor_eq` | `none` | — | L2–L4 |
| 6 | Product state | `V[idx, dfa[state, lf[idx]]]`, `selectdim(Vres, ndims(Vres), state)` — `src/bellman.jl` | `IntervalMDP.Index.productIndex_bijective`, `IntervalMDP.Index.product_read_successor` | `none` | — | L2–L4 |

## VI / Bellman algorithms

| # | Algorithm | Model family | Julia entry point (function — file) | CPU / GPU paths | Lean theorem(s) (name — file) | Proof status | Proof scope | Limitations |
|---|---|---|---|---|---|---|---|---|
| 1 | O-maximization, dense | interval | `state_action_bellman(::DenseIntervalOMaxWorkspace, …)`, `gap_value`, `bellman_precomputation!` — `src/bellman.jl` | dense CPU, CUDA (`ext/`) | `IntervalMDP.OMax.omax_mem`, `omax_eq_sSup`, `omax_eq_sInf`, `omax_tie_invariant` — `lean/IntervalMDPProofs/OMax.lean` | `none` | — | L1–L4 |
| 2 | O-maximization, sparse | interval | `state_action_bellman(::SparseIntervalOMaxWorkspace, …)`, `gap_value(Vp, budget)` — `src/bellman.jl` | sparse CPU, CUDA | `IntervalMDP.OMax.omaxSparse_eq_omax` — `OMax.lean` | `none` | — | L1–L4 |
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

## Legacy verification gaps

Every algorithm row (1–14), every index row and A2–A8 are `none`: no Lean theorem yet. A task
touching one of these must supply its theorem and proof before it can pass the Formal Verification
gate.

## Summary

- Algorithms: 14; proved: 0; partial: 0; none: 14.
- Models: 5 mapped rows proved (M1–M5), including `toSet_convex` and `productSet_not_convex`; approximation: A1 proved, A2–A8 none; indexing: 0 of 8 rows.
- Statement: the package is **not** verified. Phase 0 proves only model well-formedness and the
  generic soundness lift, at abstract scope.
