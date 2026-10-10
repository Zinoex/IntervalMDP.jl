# IntervalMDP.jl — Verification Inventory

> Created in Phase 0 of `harness/specs/lean-proofs-existing-algorithms.md` (now split into `harness/specs/lean-proofs/`) from
> `TEMPLATE-onboarding-inventory.md`. Dev keeps it current whenever a proof is added or an
> algorithm changes. This inventory is a status report, **not** a verification claim.

- Target: `git@github.com:Zinoex/IntervalMDP.jl.git` @ `20fc03b90d3a3dbe96cfe8335a47052e99ffa1f5` (base of Phase 0; the Lean project is uncommitted work on top of it)
- Target root: `/home/fresen/.julia/dev/IntervalMDP`
- Julia compat: `julia = "1.11"`; Lean root / pinned toolchain: `/home/fresen/.julia/dev/IntervalMDP/lean` / `leanprover/lean4:v4.33.0-rc2` (Mathlib tag `v4.33.0-rc2`)
- Inventory date: `2026-10-05`
- Phase completed: **0** (models, well-formedness, A1); Phase **1a** (index foundations: linear index, sparse support pairing, sort permutation); Phase **1b** (marginal indexing: `Marginal` and `IntervalAmbiguitySets` `sub2ind`); Phase **1c** (dense O-maximization); Phase **1d** (sparse O-maximization). **Phase 1 is complete.** Phase **2a** (robust Bellman operator on general RMDPs: `T_mono`, `T_add_const`, `T_nonexpansive`); Phase **2b** (interval specialisation `T_interval_eq_omax`, strategy extraction `argopt_attains`, policy evaluation `policy_eval_sound` (proved for valid strategies; A8 `partial`, Finding F2); strategy lookup `partial`, Finding F2). **Phase 2 is complete** (with open Finding F2). Phase **3a** (value iteration for reachability / reach-avoid: `reachIter_mem_unit`, `reachIter_mono`, `reachIter_tendsto_lfp`, `reachIter_sound`; A4 `partial` until `exitIter_sound`, 3b). Phase **3b** (safety −1/+1 shift `safety_shift_eq`; expected exit time `exitIter_succ`, `exitIter_mono`, `exitIter_sound`; A4 `proved`). Phase **3c** (discounted reward: `rewardIter_succ`, `reward_contracting` for `0 < ν < 1`, A5 error bound `reward_error_bound`; row 7 and A5 `proved`). Phase **3d** (synthesized strategies: `timeVarying_attains`, `stationary_sound` (proved for the documented stationary cache); row 9 and A7 `partial`, Finding F3 = benchmark B-1). **Phase 3 is complete** (with open Findings F2, F3). Phase **4a** (interval value iteration: `lower_le_upper`, `primary_sound`, `bracket_aligned`, `lower_le_reachLfp`, `reachLfp_le_upper`; `IVI.bracket` false for two modes, Finding F4; row 10 and A6 `partial`). Phase **4b** (gap stopping: `gap_stop_sound_aligned` for `(Pessimistic, Minimize)` / `(Optimistic, Maximize)` and `gap_stop_primary_sound` for all four modes, both for non-exact-time reach-avoid (`isExactTime = false`) and assuming the loop exits (`Terminates`, not proved); `IVI.gap_stop_sound` (all four modes) false for `(Pessimistic, Maximize)` / `(Optimistic, Minimize)`, Finding F4; row 10 and A6 stay `partial`). **Phase 4 is complete** (with open Finding F4). Phases 5–6: not started.

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
| A4 | Stopping infinite-horizon reachability / reach-avoid / exit time at finite k | 3 | Reachability / reach-avoid (3a): `IntervalMDP.VI.reachIter_sound` — `lean/IntervalMDPProofs/VI/Reach.lean` (for `FiniteTime`/`InfiniteTime` `Reachability`/`ReachAvoid`, i.e. `prop.isExactTime = false`: `Sound .pessimistic (reachIter M sat strat prop k) (reachLfp M sat strat prop)`, i.e. `V_k ≤ V*` for every `k`, including the `k` at which `FixedIterationsCriteria` / `CovergenceCriteria` stops; proved via `Approx.iter_sound` with exact operator `step` started from its fixed point `reachLfp`); supporting `reachIter_tendsto_lfp` (`V_k → V*`), `reachLfp_le_of_fixedPt`. Exit time (3b): `IntervalMDP.VI.exitIter_sound` — `lean/IntervalMDPProofs/VI/ExitTime.lean` (`Sound .pessimistic (exitIter M sat strat prop k) W` for every nonnegative real super-solution `W` of `exitStep`, every `k`, via `Approx.iter_sound`; with `exitIter_le_exitValue` / `exitValue_le_of_step_le` against the exact value `exitValue` in `ℝ≥0∞`; values may diverge, no convergence claimed) | `proved` (Phase 3a: reachability / reach-avoid; Phase 3b: exit time) | `abstract` | L1–L6 (L1 floating-point rounding, including the residual test `maximum(abs, u) < tol`; L2 overflow/underflow not modeled; L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription and docstrings, not proved; L5 DP value ↔ path measure not proved: `V*` is `reachLfp`, the least fixed point of the DP map, resp. `exitValue` for exit time; L6 no LP solver involved). Mode coverage: all four satisfaction × strategy modes for `V_k ≤ V*` (reachability / reach-avoid and exit time). **Direction limitation:** `V_k ≤ V*` is the conservative direction (`Sound sat`) only for `Pessimistic`; for `Optimistic` the iterates are lower bounds of the optimistic value, which is **not** conservative (`Sound .optimistic` would need `V* ≤ V_k`) — not claimed. Without A4/A6, no error bound is claimed for the reachability ε-criterion. Exact-time properties excluded (no stopping approximation: they run exactly `time_horizon` steps). Exit time: no convergence claimed; when the exact value is infinite Julia does not terminate (Observation O13). Findings: none; see Observations O12, O13 |
| A5 | Stopping infinite-horizon reward at the ε-criterion | 3 | `IntervalMDP.VI.reward_error_bound` — `lean/IntervalMDPProofs/VI/Reward.lean` (for `0 < ν < 1`, all four modes: `‖V_{k+1} − V*‖∞ ≤ ν/(1−ν)·‖V_{k+1} − V_k‖∞` with `V* = rewardValue`, the unique fixed point of `rewardStep`; Julia's `k ≥ 1` is the Lean `k + 1`, so no truncated `k − 1`); supporting `reward_stop_bound` (if Julia's test `convergenceCriteria ε V_{k+1} V_k`, i.e. `maximum(abs, V_{k+1} − V_k) < ε`, holds, then `‖V_{k+1} − V*‖∞ < ν/(1−ν)·ε`), `reward_error_interval` (pointwise `abs (V_{k+1} s − V* s) ≤ ν/(1−ν)·‖V_{k+1} − V_k‖∞`), `reward_contracting`, `rewardIter_tendsto`, `Property.toRewardProperty_discount_lt_one` | `proved` (Phase 3c) | `abstract` | L1–L6 (L1 floating-point rounding, incl. `rmul!`/`.+=` in each step and the residual test `maximum(abs, u) < tol` of `CovergenceCriteria` (the A5 bound holds for exact reals; a rounded residual can understate `‖V_k − V_{k−1}‖∞` by rounding error); L2 overflow/underflow, `NaN`/`Inf` not modeled (e.g. `FiniteTimeReward` with `ν ≥ 1` and large `K`); L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription and docstrings (plus the scratch cross-check in Observation O14), not proved; L5 DP value ↔ path measure not proved: `V*` is `rewardValue`, the fixed point of the DP map `V ↦ ν·T V + r`, not shown equal to `𝔼^{π,η}[∑ νᵏ r(ω[k])]`; L6 no LP solver involved). **What Julia's stop guarantees:** `solve` with `InfiniteTimeReward(r, ν, ε)` returns `V_k` with `V* ∈ [V_k − c, V_k + c]` statewise, `c = ν/(1−ν)·ε` (strict, in exact arithmetic); for `ν` close to `1` this interval is much wider than `ε` (e.g. `ν = 0.99`, `c = 99·ε`), so `convergence_eps` is not the accuracy of the result. Hypothesis `ν < 1` holds for every `InfiniteTimeReward` Julia accepts (`checkdiscountupperbound`). Mode coverage: all four satisfaction × strategy modes; the bound is two-sided, so it gives the conservative direction for both `Pessimistic` (`V_k − c ≤ V*`) and `Optimistic` (`V* ≤ V_k + c`). Not modeled: time-varying models/strategies. Findings: none; see Observation O14 |
| A6 | IVI bounds | 4 | 4a: `IntervalMDP.IVI.bracket` (all four modes) is **not stated**: it is false for Julia's strategy coupling in `(Pessimistic, Maximize)` and `(Optimistic, Minimize)` (Finding F4; Lean witnesses `IntervalMDP.Examples.ivi_upper_lt_reachLfp`, `ivi_reachLfp_lt_lower`, `not_bracket_pessimistic_maximize`, `not_bracket_optimistic_minimize` — `lean/IntervalMDPProofs/Models/Examples.lean`). Strongest proved statements — `lean/IntervalMDPProofs/IVI.lean`: `IntervalMDP.IVI.bracket_aligned` (`V_lower_k ≤ V* ≤ V_upper_k` for every `k`, modes `(Pessimistic, Minimize)` and `(Optimistic, Maximize)`, non-exact-time reach-avoid only); `IntervalMDP.IVI.lower_le_reachLfp` (`V_lower_k ≤ V*` when `sat = pessimistic` or `strat = maximize`, non-exact-time reach-avoid only); `IntervalMDP.IVI.reachLfp_le_upper` (`V* ≤ V_upper_k` when `sat = optimistic` or `strat = minimize`, all reach-avoid properties); `IntervalMDP.IVI.primary_sound` (`Sound sat V_primary_k V*`, i.e. the returned `value_function` is sound, all four modes, non-exact-time reach-avoid only, via `Approx.iter_sound`); `IntervalMDP.IVI.lower_le_upper` (all four modes, all reach-avoid properties). `V* = reachLfp M sat strat prop.toReachProperty`; hypotheses: `RMDP` invariants, `ActionOrder` (Julia's iteration order), non-exact-time reach-avoid (`isExactTime = false`) where `V*` is the infinite-horizon value, the explicit mode restriction, and, for the 4b stopping theorems `gap_stop_sound_aligned` and `gap_stop_primary_sound`, loop termination `h : Terminates M sat strat prop o kind tol initial` (assumed, **not proved**: the gap need not close, e.g. when an end component lies in the don't-care region, design note in `src/interval_value_iteration.jl`). 4b: `IntervalMDP.IVI.gap_stop_sound` (all four modes: stopping on `IVIInitialGapCriteria(tol)` returns a value within `tol` of `V*` on the initial states) is **not stated**: it is false for `(Pessimistic, Maximize)` and `(Optimistic, Minimize)` (Finding F4: gap `0` at `k = 1`, `abs (V_primary − V*) = 1/2` at the initial state; Lean witnesses `IntervalMDP.Examples.not_gap_stop_sound_pessimistic_maximize`, `not_gap_stop_sound_optimistic_minimize` for every `0 < tol ≤ 1/2`, both caches). Strongest proved — `lean/IntervalMDPProofs/IVI.lean`: `IntervalMDP.IVI.gap_stop_sound_aligned` (assuming the loop exits, `Terminates`, not proved: `WithinOn initial tol (iviValueFunction h) V*`, i.e. `abs (V_returned(s) − V*(s)) < tol` for every initial state `s`, modes `(Pessimistic, Minimize)` and `(Optimistic, Maximize)`, non-exact-time reach-avoid only (`isExactTime = false`), via `bracket_aligned`); `IntervalMDP.IVI.gap_stop_primary_sound` (assuming the loop exits, `Terminates`, not proved: `Sound sat (iviValueFunction h) V*`, all four modes, non-exact-time reach-avoid only, via `primary_sound`); supporting `withinOn_of_iviInitialGapCriteria_aligned` (any call `k` satisfying the criterion), `withinOn_of_bracket`, `gap_le_maxInitialGap`, `iviInitialGapCriteria_stopIndex`, `not_iviInitialGapCriteria_of_lt_stopIndex`, `one_le_stopIndex`. Gap as in Julia: absolute, `max(0, max_{s ∈ initial_states} (V_upper(s) − V_lower(s)))`, tested after each call `k ≥ 1` (never on the initialised bounds), strict `< tol`; returned value = primary bound at the first such `k` | `partial` (Phases 4a, 4b; Finding F4) | `abstract` | L1–L6 (L1 floating-point rounding, including the gap test `max_initial_gap(…) < convergence_eps` and the early exit `budget ≤ 0` / `budget -= p` inside each `bellman!`; L2 overflow/underflow not modeled; L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription of `ivi_step!` / `initialize_ivi!` / `_interval_value_iteration!` and docstrings, not proved; L5 DP value ↔ path measure not proved: `V*` is `reachLfp`, the least fixed point of the reach-avoid DP map; L6 no LP solver involved). Mode coverage: `primary_sound`, `lower_le_upper` all four modes (`primary_sound` for non-exact-time properties only); bracket (non-exact-time properties) only `(Pessimistic, Minimize)`, `(Optimistic, Maximize)` (explicit restriction; the other two are refuted, F4). Gap stopping (4b): `gap_stop_sound_aligned` only `(Pessimistic, Minimize)`, `(Optimistic, Maximize)`; `gap_stop_primary_sound` all four modes (both non-exact-time only, both assuming `Terminates`, not proved). Not modelled: time-varying models (`select_model(mp, k)`), `IVIFixedIterationsCriteria` beyond `iviIter k` at `k = time_horizon`, MEC deflation |
| A7 | Synthesized strategy | 3 | Finite horizon: `IntervalMDP.VI.timeVarying_attains` — `lean/IntervalMDPProofs/VI/Strategy.lean` (for every step postprocessing `post` and start `V₀`, i.e. every property type: `policyEvalIter M sat post V₀ (timeVaryingCacheStrategy M sat strat o (viIter M sat strat post V₀) K) K = viIter M sat strat post V₀ K` — policy evaluation of the returned `TimeVaryingStrategy`, with Julia's `strategy[time_length − k]` at call `k`, gives exactly the computed `V_K`); instances `timeVarying_attains_reach` (all six reachability types), `timeVarying_attains_safety` (shifted iterates), `timeVarying_attains_reward`. Infinite horizon: `IntervalMDP.VI.stationary_sound` (for `prop.isExactTime = false` and every `K`: `Sound .pessimistic (reachIter M sat strat prop (K + 1)) (strategyReachLfp M sat strat prop σ)` with `σ = stationaryCacheStrategy M sat strat o (reachIter M sat strat prop) K`, i.e. `V_{K+1} ≤ V^σ`, the least fixed point of the returned strategy's policy-evaluation step, in particular at the `K` where `CovergenceCriteria` stops); instances `stationary_sound_exitTime` (`V_{K+1} ≤ W` for every nonnegative real super-solution `W` of `σ`'s exit-time step), `stationary_reward_error_bound` (`ν < 1`: `‖V_{K+1} − V^σ‖∞ ≤ ν/(1−ν)·‖V_{K+1} − V_K‖∞`, the A5 interval contains `V^σ`); supporting `stationary_le_superSolution`, `stationary_backward_step`, `strategy_eq_of_T_eq` (ties keep the action, so a state switches only on a strict value increase, `T_lt_of_switch`), `viIter_le_superSolution_minimize` (`minimize`: every valid strategy), `synthesizedStrategy_spec`, `policyEvalIter_timeVaryingCacheStrategy`. Model: `synthesizedStrategy` (decision rule of call `k`: `argoptAction` from a seed over `ActionOrder`), seeds `timeVaryingSeed` (`first(available_actions)`) and `stationaryCacheSeed` (documented stationary cache: the previous action), `timeVaryingCacheStrategy` (`reverse`), `stationaryCacheStrategy`, `policyEvalIter`, `viIter`. Counterexample for Julia's cache: `IntervalMDP.Examples.b1_stationary_unsound` — `lean/IntervalMDPProofs/Models/Examples.lean` (Finding F3) | `partial` (Phase 3d; Finding F3) | `abstract` | L1–L6 (L1 floating-point rounding: Lean decides ties between action values by exact real comparison, Julia by `Float64` `>`/`<`, so on near-ties the selected action can differ; also the residual test of `CovergenceCriteria`; L2 overflow/underflow, `NaN`/`±Inf` (the `typemin`/`typemax` neutral) not modeled; L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription and docstrings (plus the scratch reproductions of Finding F3), not proved; L5 DP value ↔ path measure not proved: a strategy's value is the DP fixed point of its policy-evaluation step (`strategyReachLfp`, `strategyRewardValue`, super-solutions for exit time); L6 no LP solver involved). Hypotheses: `RMDP` structure invariants, `ActionOrder` (Julia's iteration order of `available(model, jₛ)`), time-invariant model; `stationary_sound` takes `isExactTime = false` (infinite-time reachability / reach-avoid), the exit-time instance a nonnegative real super-solution, the reward instance `ν < 1` (holds for every Julia `InfiniteTimeReward`). Mode coverage: all four satisfaction × strategy modes. **Direction:** `V_{K+1} ≤ V^σ` in every mode, as A4: conservative (`Sound sat`) for `Pessimistic`; for `Optimistic` a lower bound of the strategy's optimistic value, not `Sound .optimistic` — not claimed. Not covered: infinite-time safety (its iterates decrease, so `V_K` is not a lower bound; no A4 for safety); `TimeVaryingAvailableActions` (Observation O11). **Julia deviates (Finding F3, benchmark B-1):** the guard `jₛ ∉ available_actions` resets the stationary seed for every state whose index exceeds the number of actions; then the returned strategy can be strictly worse than the reported value for `Maximize` (both satisfaction modes). `stationary_sound` is proved for the documented cache only, hence `partial`. `timeVarying_attains` is not affected. |
| A8 | Given-strategy verification (`NonOptimizingStrategyCache`) | 2 | `IntervalMDP.Bellman.policy_eval_sound` — `lean/IntervalMDPProofs/Bellman.lean` (for a strategy `π` valid for the model, `π.Valid M.toAvailableActions`: (1) exact: `Tπ M sat π = T (M.withAvailable (strategyAvailable π)) sat strat`, the Bellman operator of the model restricted to `available s = {π(s)}`; (2) `Approx.StepSound (policyEvalMode strat) (Tπ M sat π) (T M sat strat)`, i.e. `Tπ V ≤ T V` for `maximize` (`Sound .pessimistic` direction) and `Tπ V ≥ T V` for `minimize` (`Sound .optimistic` direction); (3) every value-iteration iterate from a common `V₀` is sound, via `Approx.iter_sound` with `T_monotone`); supporting `Tπ_eq_T_strategyAvailable`, `Tπ_stepSound`, `Tπ_interval_eq_omax`. `policy_eval_sound` is proved for valid strategies (`π.Valid M.toAvailableActions`); Julia accepts strategies that play unavailable actions (`checkstrategy`, `src/strategy.jl:35–65`, does not check availability), and for those A8's soundness fails (Finding F2 counterexample: `Pessimistic`/`Maximize` verified `V = 1.0` > optimum `0.0`; `Minimize`, with the availability mirrored (only action 2 available in state 1, strategy plays action 1): verified `0.0` < optimum `1.0`) | `partial` (Phase 2b; Finding F2) | `abstract` | L1–L6 (L1 floating-point rounding; L2 overflow/underflow beyond the stated index bound, `NaN`/`±Inf` (Julia's `typemin`/`typemax` seed) not modeled; L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription and docstrings, not proved; L5 DP value ↔ path measure not proved; L6 no LP solver involved). Mode coverage: all four satisfaction × strategy modes (`sat`, `strat` arbitrary; the direction convention is `policyEvalMode`). Hypothesis `π.Valid` (availability) is **not checked by Julia** (Finding F2): for a strategy with an unavailable action the Julia result can exceed the optimum (`maximize`) or fall below it (`minimize`); the theorem covers valid strategies only, hence `partial`. |

## Indexing

| Ph | Index computation | Julia (function — file) | Lean theorem(s) | Status | Scope | Limitations |
|---|---|---|---|---|---|---|
| 1 | Column-major linear index | `LinearIndices`, `CartesianIndices` (Base), as used for `V[I]` and `FullUpdateSequence` (`src/bellman.jl`, `src/update_sequence.jl`) | `IntervalMDP.Index.linear_bijective` (bijection onto `1..∏ dims`, and the `N`-bit value is exact), `IntervalMDP.Index.linear_succ_first` (first dimension fastest) — `lean/IntervalMDPProofs/Index/Linear.lean` (definition `IntervalMDP.Index.linear`; 1-based conversion `toJulia`, overflow model `machineInt` in `Index/Julia.lean`) | `proved` (Phase 1a) | `abstract` | L1–L6; index bound `∏ dims < 2^(N-1)` is a hypothesis of both theorems (`N = 64` for `Int`, `N = 32` for `Int32` paths) |
| 1 | Marginal → ambiguity-set column | `sub2ind(::Marginal, action, source)` — `src/probabilities/Marginal.jl` | `IntervalMDP.Index.marginalSub2ind_eq_linear` (equals `linear` over `(action_vars…, source_dims…)` on `(action[action_indices]…, source[state_indices]…)`, actions first; the `N`-bit value is the same), `IntervalMDP.Index.marginalSub2ind_bijective` (values = `1..∏ action_vars · ∏ source_dims`, equal values ⇒ equal conditioning tuples), `IntervalMDP.Index.marginalSub2ind_depends_only` — `lean/IntervalMDPProofs/Index/Marginal.lean` (literal loop transcription `IntervalMDP.Index.marginalSub2ind`, `N`-bit value `marginalSub2indInt`; supporting `marginalCartesian_surjective`); cross-check `test/base/indexing_reference.jl` | `proved` (Phase 1b) | `abstract` | L1–L6; index bound `∏ dims < 2^(N-1)` is a hypothesis of `_eq_linear` and `_bijective` (`N = 64` for `Int` tuples on the CPU, `N = 32` for `Int32` on CUDA); `_depends_only` needs no bound. Mode-independent (pure index arithmetic: no satisfaction/strategy mode involved). Marginal indices strictly increasing (O4) and `source_dims = sv.dims ∘ state_indices` (`check_transition`) are structure invariants; for terminal slices (O5) instantiate `StateVars` with `source_dims` |
| 1 | Non-factored set lookup | `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ) = jₛ[1]` — `src/probabilities/IntervalAmbiguitySets.jl` | `IntervalMDP.Index.intervalSub2ind_correct` (equals `sub2ind(::Marginal, …)` when the marginal conditions on state variable 1 only and every conditioning action variable has one value, i.e. a single-action `IntervalMarkovDecisionProcess` layout), `IntervalMDP.Index.intervalSub2ind_wrong_multiAction` (with two actions differing on a conditioning variable it cannot match; Observation O8) — `lean/IntervalMDPProofs/Index/Marginal.lean` (model `IntervalMDP.Index.intervalSub2ind`); cross-check `test/base/indexing_reference.jl` | `proved` (Phase 1b; see O8) | `abstract` | L1–L6; mode-independent (pure index arithmetic). Reachability is argued by call-site inspection (L4), not proved: no call site in `src/` or `ext/` reaches it (see O8) |
| 1 | Sparse support pairing | `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` — `src/bellman.jl` | `IntervalMDP.Index.sparse_zip_correct` — `lean/IntervalMDPProofs/Index/Sparse.lean` (structure `IntervalMDP.Index.SparseCol` = CSC column invariant) | `proved` (Phase 1a) | `abstract` | L1–L6; the CSC invariant (strictly increasing, in-range `rowval`, `nzval` aligned) is assumed as structure fields — Julia's `checkprobabilities` does not re-check it (Observation O6); no integer products, so no index bound |
| 1 | Sort permutation | `sortperm!(perm, V; rev = upper_bound)`, loop of `gap_value(V, gap, budget, perm)` — `src/bellman.jl` | `IntervalMDP.Index.sortedPerm_bijective` (both `rev = true/false`), `IntervalMDP.Index.greedy_visits_once` — `lean/IntervalMDPProofs/Index/Perm.lean` (structure `IntervalMDP.Index.SortedPerm`, loop transcription `gapValue`); supporting `IntervalMDP.Index.sortedPerm_fits_int32` (`n < 2^31`), `IntervalMDP.Index.gapValue_eq_sum_allocation` (early exit does not change the result) | `proved` (Phase 1a) | `abstract` | L1–L6; in particular L1: the early exit `budget <= 0` is exact in `ℝ` (the loop breaks only at budget `0`); in floating point `budget -= p` can leave a residual, so the loop may continue; `perm` is modeled by a stable merge sort (Julia documents `sortperm` as stable; equality with Julia's order on ties is argued, L4) |
| 2 | Strategy lookup | `CartesianIndex(strategy_cache[jₛ])` — `src/bellman.jl`, `src/strategy_cache.jl`; validation `checkstrategy` — `src/strategy.jl` | `IntervalMDP.Index.strategyAction_available` (spec: "for strategies that pass validation, the looked-up action is in `available(model, jₛ)`") is **false** for Julia's validation and is not declared (Finding F2; witness `IntervalMDP.Index.checkStrategy_admits_unavailable`). Strongest proved statements — `lean/IntervalMDPProofs/Index/Strategy.lean` (definitions `strategyAction N sv strategy jₛ = strategy (linearInt N sv.dims jₛ)`, `checkStrategy` = `checkstrategy`, `juliaAvailable`, `Stores`): `IntervalMDP.Index.strategyAction_available_of_all` (every strategy passing `checkStrategy` looks up an available action under `AllAvailableActions`, the default and the only option of `IntervalMarkovDecisionProcess`), `IntervalMDP.Index.strategyAction_available_of_valid` (a strategy array storing a valid strategy, `π.Valid aa`, looks up an available action, any available actions); supporting `strategyAction_eq_linear`, `actionTuple_injective`. All carry the overflow bound `∏ source_shape < 2 ^ (N - 1)` | `partial` (Finding F2) | `abstract` | L1–L6 (L1 floating-point: not applicable, integer index arithmetic only; L2 overflow beyond the stated bound ∏ source_shape < 2^(N-1); L3 CUDA kernels and threads not modeled; L4 correspondence by literal transcription, not proved; L5 not applicable; L6 no LP solver). See Finding F2 (supersedes Observation O2) |
| 5 | Factored successor index | `CartesianIndices(num_target.(ambiguity_sets))` — `src/bellman.jl` | `IntervalMDP.Index.factored_successor_eq` | `none` | — | L2–L4 |
| 6 | Product state | `V[idx, dfa[state, lf[idx]]]`, `selectdim(Vres, ndims(Vres), state)` — `src/bellman.jl` | `IntervalMDP.Index.productIndex_bijective`, `IntervalMDP.Index.product_read_successor` | `none` | — | L2–L4 |

## VI / Bellman algorithms

| # | Algorithm | Model family | Julia entry point (function — file) | CPU / GPU paths | Lean theorem(s) (name — file) | Proof status | Proof scope | Limitations |
|---|---|---|---|---|---|---|---|---|
| 1 | O-maximization, dense | interval | `state_action_bellman(::DenseIntervalOMaxWorkspace, …)`, `gap_value`, `bellman_precomputation!` — `src/bellman.jl` | dense CPU, CUDA (`ext/`) | `IntervalMDP.OMax.omax_mem` (the greedy distribution `p = lower + allocation` is in `P(l, u)` and `omax = ⟨p, V⟩`), `IntervalMDP.OMax.omax_eq_sSup` (`upper_bound = true`, descending `perm`: `omax = sSup {⟨p, V⟩ : p ∈ P(l, u)}`), `IntervalMDP.OMax.omax_eq_sInf` (`upper_bound = false`, ascending: `sInf`), `IntervalMDP.OMax.omax_tie_invariant` (any two permutation vectors sorting `V` in the same direction, stable or not, give the same value) — `lean/IntervalMDPProofs/OMax.lean` (definitions `IntervalMDP.OMax.omax` = `stateActionBellman A s.V s.perm` = `dot(V, lower) + gapValue(V, gap, perm, budget)`, transcription reusing `Index.gapValue`; `greedy`; `Permutation`); supporting `stateActionBellman_isGreatest`, `stateActionBellman_isLeast`, `stateActionBellman_eq_dot`; cross-check `test/base/omax_reference.jl` (Phase 1d: `IntervalMDP.bellman` on small dense IMDPs vs. a brute-force HiGHS LP over `P(l, u)`, both `upper_bound` and both `maximize` values, `Float64`, tolerance `1e-9`) | `proved` (Phase 1c) | `abstract` | L1–L6. Hypotheses: structure invariants of `IntervalAmbiguity` only, plus `upper_bound = true/false` for `_eq_sSup`/`_eq_sInf`. Mode coverage: O-max is mode-agnostic; modes reach it only through `upper_bound = isoptimistic(spec)` (`src/robust_value_iteration.jl`), the strategy mode acts afterwards across actions. Both directions are proved, so all four satisfaction × strategy modes are covered. L1 in particular: the early exit `budget <= 0` and the accumulated error of `budget -= p` are exact only in `ℝ` (Lean replaces the early exit by `p_i = min(budget_i, gap_i) = 0` via `greedy_visits_once`/`gapValue_eq_sum_allocation`); tie invariance holds in `ℝ` only (different tie orders can round differently, O9). L2: overflow/underflow and `NaN` in `V` not modeled. L3: CUDA kernels and `@threadstid` not modeled. L4: Lean↔Julia by literal transcription (`gap_value` loop, `dot(V, lower)`). L5: DP value ↔ path measure not proved. L6: no LP solver in the algorithm; the cross-check's LP reference uses HiGHS, which is not verified |
| 2 | O-maximization, sparse | interval | `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` (`src/bellman.jl:536–553`: `zip(V[support], nonzeros(gap))`, `sort!(Vp_workspace; rev = upper_bound, by = first)`, `dot(V, lower) + gap_value(Vp, budget)`), `gap_value(Vp, budget)` (`src/bellman.jl:555–571`) — `src/bellman.jl` | sparse CPU, CUDA (`ext/`) | `IntervalMDP.OMax.omaxSparse_eq_omax` (for every `SparseIntervalAmbiguity` and every sort result `SortedValuesGaps` — stable or not, ties in any order — `omaxSparse = omax` on the same ambiguity set, the dense gap being the sparse column's `getindex`, `0` off the support) — `lean/IntervalMDPProofs/OMax.lean` (definitions `IntervalMDP.OMax.omaxSparse` = `dot(V, lower) + gapValueSparse(Vp, budget, 0)`, literal loop transcription `gapValueSparse` of `gap_value(Vp, budget)`; structures `SparseIntervalAmbiguity` (= `IntervalAmbiguity` + CSC gap column `gapCol : Index.SparseCol` + invariant `gap_eq`) and `SortedValuesGaps` (any permutation of `Index.valuesGaps` sorted by first component in direction `rev = upper_bound`), stable instance `stableValuesGaps`; uses `Index.sparse_zip_correct` and `omax_tie_invariant`); supporting `IntervalMDP.OMax.omaxSparse_exact` (`= sSup` for `upper_bound = true`, `= sInf` for `false`), `gapValue_sublist` (zero-gap rows inserted in the loop order change nothing), `exists_permutation_sublist` (a sorted support order extends to a full sorted `perm`); cross-check `test/base/omax_reference.jl` (`IntervalMDP.bellman` on small sparse IMDPs vs. a brute-force HiGHS LP, both `upper_bound`/`maximize` values, `Float64`, tolerance `1e-9`; cases: degenerate `l = u` with budget 0, ties in `V`, all budget in one successor, sparse column with an all-zero stored gap, 40 seeded random dense+sparse IMDPs) | `proved` (Phase 1d) | `abstract` | L1–L6. Hypotheses: structure invariants only (`IntervalAmbiguity`, CSC invariant of `SparseCol`, `gap_eq` linking the stored column to `gap`). Mode coverage: modes reach O-max only through `upper_bound = isoptimistic(spec)` (`src/robust_value_iteration.jl`); `omaxSparse_eq_omax` holds for both `upper_bound` values, so all four satisfaction × strategy modes are covered (the strategy mode acts afterwards across actions). L1: the early exit `budget <= 0` and the accumulated error of `budget -= p` are exact only in `ℝ`; the sparse and dense paths (and different tie orders of `sort!`) can round differently in floating point (O10). L2: overflow/underflow and `NaN` in `V` not modeled. L3: CUDA kernels (`ext/`) and `@threadstid` (`ThreadedSparseIntervalOMaxWorkspace`) not modeled. L4: Lean↔Julia by literal transcription (`zip` loop, `sort!` by first, `gap_value(Vp, budget)` loop, `dot(V, lower)`) plus the cross-check test, not proved; `lower` is modeled as a dense function (a sparse `dot` gives the same sum). L5: DP value ↔ path measure not proved. L6: no LP solver in the algorithm; the cross-check's LP reference uses HiGHS, which is not verified |
| 3 | Robust Bellman operator | RMDP (general); interval specialisation | `bellman!` → `_bellman_helper!` → `state_bellman!` (`OptimizingStrategyCache`; `state_action_bellman` per available action, then `extract_strategy!`) — `src/bellman.jl`, `src/strategy_cache.jl`; modes from `step!` (`upper_bound = isoptimistic(spec)`, `maximize = ismaximize(spec)`, `src/robust_value_iteration.jl`) | dense, sparse, CUDA | Phase 2a (general `RMDP`): `IntervalMDP.Bellman.T_mono`, `T_add_const`, `T_nonexpansive` — `lean/IntervalMDPProofs/Bellman.lean` (definition `T M sat strat V s` = `extractValue strat (available s)` of `stateActionBellman` = `innerOpt sat (ambiguity s a) V`; `innerOpt` = `sSup`/`sInf` of `{⟨p, V⟩ : p ∈ Γ}`, `extractValue` = `Finset.sup'`/`Finset.inf'`); supporting `T_le_add`, `T_monotone`, `T_lipschitz`, `innerOpt_mem`, `innerOpt_le_add`, `extractValue_le_add`. Phase 2b (interval): `IntervalMDP.Bellman.T_interval_eq_omax` — for an IMDP in Julia's layout (`IntervalMDPLayout`: columns `ambiguity_sets[j]` of `Marginal(ambiguity_sets, (num_states,), (num_actions,))`, `toIMDP`), `T C.toIMDP.toRMDP sat strat V s = extractValue strat (available s)` of `intervalStateActionBellman C sat V s a = omax (ambiguitySets (column s a)) ⟨V, isOptimistic sat⟩`, with `column s a = marginalSub2ind marginal …` (1b) and `omax` the dense O-max (1c), sort direction `upper_bound = isoptimistic(sat)`; supporting `innerOpt_interval_eq_omax`, `stateActionBellman_interval_eq_omax`, `expectations_toSet`, `IntervalMDPLayout.column_eq` (`column = (jₛ - 1) * num_actions + jₐ`), `IntervalMDPLayout.columnInt_eq` (`N`-bit value exact under `num_states * num_actions < 2 ^ (N - 1)`) | `proved` (Phases 2a, 2b) | `abstract` | L1–L6 (L1 floating-point rounding; L2 overflow/underflow beyond the stated index bound, `NaN`/`±Inf` (Julia's `typemin`/`typemax` seed) not modeled; L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription and docstrings, not proved; L5 DP value ↔ path measure not proved; L6 no LP solver involved). Hypotheses: `RMDP` structure invariants only (ambiguity sets `WellFormed` = nonempty + closed; nonempty finite `available s`); **no convexity** in the 2a theorems (they hold for `FactoredIMDP.toRMDP`); `T_interval_eq_omax` uses only `omax_eq_sSup`/`omax_eq_sInf` (interval sets are convex, `toSet_convex`, but convexity is not needed); `IntervalMDPLayout` fields `numStates_pos`, `numActions_pos` are the Julia `num_target ≥ 1`, `num_actions ≥ 1`. Mode coverage: all four satisfaction × strategy modes (every theorem stated for arbitrary `sat`, `strat`). Scope of the interval theorem: dense O-max workspace (`DenseIntervalOMaxWorkspace`); the sparse workspace equals it by `omaxSparse_eq_omax` (1d). See Observation O11 |
| 4 | Strategy extraction and evaluation | RMDP | `extract_strategy!` / `_extract_strategy!` (`TimeVaryingStrategyCache`, `StationaryStrategyCache`; strict `>`/`<`, first optimum kept) — `src/strategy_cache.jl`; `NonOptimizingStrategyCache` path of `state_bellman!` — `src/bellman.jl` | CPU, CUDA | `IntervalMDP.Bellman.argopt_attains` (for any iteration order `acts` of `available s` (`acts.toFinset = M.available s`) and any available seed, the action `argoptAction strat (stateActionBellman M sat V s) seed acts` — a `foldl` transcription of the `_extract_strategy!` loop with its tie-breaking — is available and `stateActionBellman M sat V s (argoptAction …) = T M sat strat V s`), `IntervalMDP.Bellman.policy_eval_sound` (proved for valid strategies; A8 is `partial`, Finding F2, see the approximation table) — `lean/IntervalMDPProofs/Bellman.lean`; supporting `argoptStep_spec`, `argoptAction_spec`, `extractValue_eq_of_opt`, `stationarySeed_available` (the seed of a `StationaryStrategyCache` — `first(available_actions)` or the previous selection, for any outcome of the O11 guard — is available at every call on a model with fixed available actions, so `argopt_attains` applies to every call of value iteration), `Tπ_eq_T_strategyAvailable`, `Tπ_stepSound` | `proved` (Phase 2b) | `abstract` | L1–L6 (L1 floating-point rounding; L2 overflow/underflow beyond the stated index bound, `NaN`/`±Inf` (Julia's `typemin`/`typemax` seed) not modeled; L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription and docstrings, not proved; L5 DP value ↔ path measure not proved; L6 no LP solver involved). Mode coverage: all four satisfaction × strategy modes. The `typemin`/`typemax` seed of Julia is modeled as `first(available_actions)` (equivalent in `ℝ`; `-Inf`/`NaN` values are L2). Not covered: available actions that change between calls with a `StationaryStrategyCache` (`TimeVaryingAvailableActions` with an infinite-horizon property), where the seed can be unavailable — Observation O11. `policy_eval_sound` assumes a valid strategy, which Julia does not check (Finding F2) |
| 5 | Robust VI: reachability / reach-avoid | interval, factored, product | `_value_iteration!`, `step!`, `initialize!`/`step_postprocess_value_function!` — `src/robust_value_iteration.jl`, `src/specification.jl` | CPU, CUDA | `IntervalMDP.VI.reachIter_mem_unit` (every iterate in `[0, 1]`; all six reachability types incl. exact time), `IntervalMDP.VI.reachIter_mono` (`Monotone (reachIter …)`), `IntervalMDP.VI.reachIter_tendsto_lfp` (`Tendsto (reachIter …) atTop (𝓝 (reachLfp …))`), `IntervalMDP.VI.reachIter_sound` (A4: `Sound .pessimistic V_k V*` for every `k`, via `Approx.iter_sound`) — `lean/IntervalMDPProofs/VI/Reach.lean`. Definitions: `reachIter M sat strat prop k` (transcription of the `_value_iteration!` loop: `reachIter 0 = initializeValueFunction prop` = `𝟙_reach` (`ValueFunction` zeros + `initialize!`, `src/specification.jl:352–354`), `reachIter (k + 1) = step (reachIter k)`; Lean index = Julia counter `k`), `step M sat strat prop V = stepPostprocessValueFunction prop (T M sat strat V)` (`step!`, `src/robust_value_iteration.jl:236–249`), `stepPostprocessValueFunction` per `ReachProperty` constructor (`.reachability`: `V[reach] .= 1`, `src/specification.jl:356–358`; `.reachAvoid`: `V[reach] .= 1; V[avoid] .= 0`, 532–535; `.exactTimeReachability`: unchanged, 497–499; `.exactTimeReachAvoid`: `V[avoid] .= 0`, 760–762), `Property.toReachProperty` (the six Julia types onto the four constructors), `reachLfp` = `OrderHom.lfp` of `stepHom` on the complete lattice `S → Set.Icc 0 1`. Supporting: `reachIter_eq_iterate`, `T_zero`, `T_const`, `T_mem_unit`, `step_mono`, `step_mem_unit`, `continuous_step`, `initializeValueFunction_le_step`, `step_reachLfp`, `reachLfp_mem_unit`, `reachLfp_le_of_step_le`, `reachLfp_le_of_fixedPt` (`reachLfp` is below every nonnegative real fixed point of `step`) | `proved` (Phase 3a) | `abstract` | L1–L6 (L1 floating-point rounding, incl. the residual test of `CovergenceCriteria`; L2 overflow/underflow, `NaN` not modeled; L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription and docstrings, not proved; L5 DP value ↔ path measure not proved: `V*` is `reachLfp`; L6 no LP solver involved). Hypotheses: `RMDP` structure invariants (no convexity, so interval and factored models via `toRMDP`), `ReachProperty` invariant `Disjoint reach avoid` (Julia `checkdisjoint`); `reachIter_mono`, `reachIter_tendsto_lfp`, `reachIter_sound` additionally take `prop.isExactTime = false` (justified restriction: for `ExactTimeReachability`/`ExactTimeReachAvoid` the postprocessing does not reset `reach`, the `K`-step value `ℙ[ω[K] ∈ G]` is not monotone in `K` and not a fixed point, and Julia runs exactly `time_horizon` steps, so there is no stopping approximation). Mode coverage: all four satisfaction × strategy modes (`sat`, `strat` arbitrary). **Optimistic direction:** `reachIter_sound` gives `V_k ≤ V*` in every mode; this is conservative (`Sound sat`) only for `Pessimistic`; for `Optimistic` it is a lower bound, not the conservative `Sound .optimistic` (`V* ≤ V_k`), which is not claimed. Without A4/A6, no error bound is claimed for the reachability ε-criterion. Not modeled: time-varying models (`select_model(mp, k)` with `TimeVaryingAvailableActions`; Observation O11), time-varying given strategies (`NonOptimizingStrategyCache` indexed by `k`; a stationary given strategy is covered via `M.withAvailable (strategyAvailable π)`), DFA properties on product processes (row 14). Findings: none; see Observation O12 |
| 6 | Robust VI: safety | interval, factored | `AbstractSafety` initialize/postprocess (`initialize!`: `current[avoid] .= -1.0`, `step_postprocess_value_function!`: `current[avoid] .= -1.0`, `postprocess_value_function!`: `current .+= 1.0` — `src/specification.jl:803–813`) inside `_value_iteration!`/`step!` — `src/robust_value_iteration.jl:170–206, 236–249` | CPU, CUDA | `IntervalMDP.VI.safety_shift_eq` — `lean/IntervalMDPProofs/VI/Safety.lean` (`prop.postprocessValueFunction (safetyIter M sat strat prop k) = reachIter M sat strat prop.toReachProperty k`: undoing the shift (`+ 1`) on the `k`-th shifted safety iterate gives the `k`-th iterate of the unshifted safety recursion, i.e. `reachIter` (3a) of the exact-time reach-avoid property `reach = avoidᶜ`, `avoid`; proved by induction with `Bellman.T_add_const`). Definitions: `SafetyProperty` (`avoid`), `Property.toSafetyProperty` (`FiniteTimeSafety`, `InfiniteTimeSafety`), `SafetyProperty.initializeValueFunction` (`-1` on `avoid`, `0` elsewhere), `.stepPostprocessValueFunction` (`V[avoid] .= -1`), `.postprocessValueFunction` (`V .+ 1`), `.toReachProperty`, `safetyStep`, `safetyIter` (transcription of the loop; Lean index = Julia counter `k`). Supporting: `safetyIter_eq_sub_one` (shifted iterate = unshifted − 1), `safetyIter_postprocess_mem_unit` (returned value in `[0, 1]`) | `proved` (Phase 3b) | `abstract` | L1–L6 (L1 floating-point rounding, incl. the residual test of `CovergenceCriteria`; the `-1`/`+1` shift itself rounds (cross-check difference `2.2e-16`, Observation O13); L2 overflow/underflow, `NaN`/`Inf` not modeled; L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription and docstrings (plus the scratch cross-check in Observation O13), not proved; L5 DP value ↔ path measure not proved: the unshifted recursion is the DP value of `k`-step safety; L6 no LP solver involved). Hypotheses: `RMDP` structure invariants only (no convexity). Mode coverage: all four satisfaction × strategy modes; Julia passes `isoptimistic(spec)`/`ismaximize(spec)` to `bellman!` unchanged, so no dualisation is involved (the dual characterisation `1 − reach(avoid)` with swapped modes is not used by Julia and not stated). No convergence or stopping (A4) claim is made for infinite-time safety (not required by this row). Not modeled: time-varying models/strategies, DFA safety (row 14). Findings: none; see Observation O13 |
| 7 | Robust VI: reward | interval, factored | `AbstractReward` initialize/postprocess (`initialize!`: `current .= reward(prop)`, `step_postprocess_value_function!`: `rmul!(current, discount(prop)); current .+= reward(prop)`, `postprocess_value_function!`: identity — `src/specification.jl:927–935`; validation `checkreward` (`discount > 0`, lines 937–953) and `checkdiscountupperbound` (`discount < 1` for `InfiniteTimeReward` only, lines 1035–1044)) inside `_value_iteration!`/`step!` — `src/robust_value_iteration.jl:170–206, 236–249`; termination `FixedIterationsCriteria` / `CovergenceCriteria` (lines 1–20) | CPU, CUDA | `IntervalMDP.VI.rewardIter_succ` (`rewardIter (k + 1) s = prop.discount * T M sat strat (rewardIter k) s + prop.reward s`, as Julia computes it: Bellman update, then discount, then add the reward; `V₀ = r`), `IntervalMDP.VI.reward_contracting` (hypothesis `prop.discount < 1`; `ContractingWith prop.discountNNReal (rewardStep M sat strat prop)` on `S → ℝ` with the sup distance, from `Bellman.T_nonexpansive`), `IntervalMDP.VI.reward_error_bound` (A5, hypothesis `prop.discount < 1`: `dist (rewardIter (k + 1)) (rewardValue …) ≤ ν / (1 − ν) * dist (rewardIter (k + 1)) (rewardIter k)`) — `lean/IntervalMDPProofs/VI/Reward.lean`. Definitions: `RewardProperty` (`reward`, `discount`, invariant `discount_pos`), `Property.toRewardProperty` (`FiniteTimeReward`, `InfiniteTimeReward`), `RewardProperty.discountNNReal`, `.initializeValueFunction`, `.stepPostprocessValueFunction` (`ν • V + r`), `.postprocessValueFunction` (identity), `convergenceCriteria` (`dist V Vprev < ε`), `rewardStep`, `rewardIter` (transcription of the loop; Lean index = Julia counter `k`), `rewardValue` (Mathlib `ContractingWith.fixedPoint`, the unique fixed point `V* = ν • T V* + r`). Supporting: `Property.toRewardProperty_discount_lt_one` (every `InfiniteTimeReward` has `ν < 1`, so the hypothesis holds for every infinite-horizon reward Julia accepts), `rewardIter_eq_iterate`, `rewardStep_dist_le`, `rewardValue_isFixedPt`, `rewardIter_tendsto` (`V_k → V*`), `reward_stop_bound`, `reward_error_interval` | `proved` (Phase 3c) | `abstract` | L1–L6 (L1 floating-point rounding, incl. `rmul!`/`.+=` in each step and the residual test `maximum(abs, u) < tol` of `CovergenceCriteria` (the A5 bound holds for exact reals; a rounded residual can understate `‖V_k − V_{k−1}‖∞` by rounding error); L2 overflow/underflow, `NaN`/`Inf` not modeled (e.g. `FiniteTimeReward` with `ν ≥ 1` and large `K`); L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription and docstrings (plus the scratch cross-check in Observation O14), not proved; L5 DP value ↔ path measure not proved: `V*` is `rewardValue`, the fixed point of the DP map `V ↦ ν·T V + r`, not shown equal to `𝔼^{π,η}[∑ νᵏ r(ω[k])]`; L6 no LP solver involved). Hypotheses: `RMDP` structure invariants and `0 < ν` (structure invariant, `checkreward`); `ν < 1` only for `reward_contracting` / `reward_error_bound` (the row's named hypothesis). Mode coverage: all four satisfaction × strategy modes. **Restriction (ν range):** `FiniteTimeReward` accepts `ν ≥ 1` (validated only `ν > 0`; scratch check: `FiniteTimeReward(r, 1.5, 3)` is accepted, `InfiniteTimeReward(r, 1.0, 1e-3)` throws `DomainError`); for `ν ≥ 1` only `rewardIter_succ` applies (the map is not a contraction, no fixed point or error bound is claimed; the finite horizon needs none). Not modeled: time-varying models/strategies; a given strategy (`NonOptimizingStrategyCache`) is covered through `Bellman.Tπ_eq_T_strategyAvailable` only informally (not stated). Findings: none; see Observation O14 |
| 8 | Robust VI: expected exit time | interval, factored | `ExpectedExitTime` initialize/postprocess (`initialize!`: `current .= 1.0; current[avoid] .= 0.0`, `step_postprocess_value_function!`: `current .+= 1.0; current[avoid] .= 0.0` — `src/specification.jl:1126–1134`; `postprocess_value_function!(…, ::AbstractHittingTime)` identity, line 1092) inside `_value_iteration!`/`step!` — `src/robust_value_iteration.jl:170–206, 236–249` | CPU, CUDA | `IntervalMDP.VI.exitIter_succ` (`exitIter (k + 1) s = if s ∈ prop.avoidStates then 0 else T M sat strat (exitIter k) s + 1`, as Julia computes it), `IntervalMDP.VI.exitIter_mono` (`Monotone (exitIter …)`), `IntervalMDP.VI.exitIter_sound` (A4: for every `W` with `0 ≤ W` and `exitStep W ≤ W`, `Sound .pessimistic (exitIter M sat strat prop k) W`, via `Approx.iter_sound`) — `lean/IntervalMDPProofs/VI/ExitTime.lean`. Definitions: `ExpectedExitTime` (`avoidStates`), `Property.toExpectedExitTime`, `ExpectedExitTime.initializeValueFunction`, `.stepPostprocessValueFunction`, `exitStep`, `exitIter` (transcription of the loop; Lean index = Julia counter `k`), `exitValue` (`⨆ k, ofReal (exitIter k s)` in `ℝ≥0∞`, the exact value, possibly `∞`). Supporting: `exitIter_eq_iterate`, `initializeValueFunction_eq_exitStep_zero` (`V₀ = exitStep 0`), `ExpectedExitTime.stepPostprocessValueFunction_mono`, `ExpectedExitTime.initializeValueFunction_nonneg`, `exitStep_mono`, `exitIter_nonneg`, `exitIter_le_exitValue`, `exitValue_le_of_step_le` | `proved` (Phase 3b) | `abstract` | L1–L6 (L1 floating-point rounding, incl. the residual test of `CovergenceCriteria`; L2 overflow/underflow, `NaN`/`Inf` not modeled; diverging values (`exitValue = ∞`) grow without bound in Julia (Observation O13); L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription and docstrings (plus the scratch cross-check in Observation O13), not proved; L5 DP value ↔ path measure not proved: the target is `exitValue`, the Kleene limit of the DP recursion in `ℝ≥0∞`; L6 no LP solver involved). Hypotheses: `RMDP` structure invariants only; `exitIter_sound` takes a nonnegative real super-solution `W` (satisfiable, e.g. by the exact value whenever it is finite; `exitValue_le_of_step_le` then gives `exitIter k ≤ exitValue ≤ W`). Mode coverage: all four satisfaction × strategy modes. **No convergence is claimed** (values may diverge; no finiteness of `exitValue`, no limit in `ℝ`, no error bound for the `convergence_eps` stop). **Optimistic direction:** `exitIter_sound` gives `V_k ≤ W` in every mode; conservative (`Sound sat`) only for `Pessimistic`; for `Optimistic` it is a lower bound, not `Sound .optimistic` — not claimed. Not modeled: time-varying models/strategies. Findings: none; see Observation O13 |
| 9 | Synthesized strategy | interval, factored, product | `TimeVaryingStrategyCache`, `StationaryStrategyCache` — `src/strategy_cache.jl` (`extract_strategy!`, `_extract_strategy!`, `step_postprocess_strategy_cache!`, `cachetostrategy`); policy evaluation of the returned strategy: `select_strategy_cache` — `src/robust_value_iteration.jl` | CPU, CUDA | `IntervalMDP.VI.timeVarying_attains` (finite horizon: evaluating the returned time-varying strategy gives exactly `V_K`, every property type; instances `timeVarying_attains_reach`, `timeVarying_attains_safety`, `timeVarying_attains_reward`), `IntervalMDP.VI.stationary_sound` (infinite-time reachability / reach-avoid: `Sound .pessimistic V_{K+1} (strategyReachLfp … σ)`, i.e. the returned stationary strategy `σ` achieves at least `V_{K+1}`; instances `stationary_sound_exitTime`, `stationary_reward_error_bound`) — `lean/IntervalMDPProofs/VI/Strategy.lean`; definitions `synthesizedStrategy`, `timeVaryingSeed`, `stationaryCacheSeed`, `timeVaryingCacheStrategy`, `stationaryCacheStrategy`, `selectStrategy`, `policyEvalIter`, `viIter`, `ActionOrder`, `StepPostprocess`, `strategyReachLfp`, `strategyRewardValue`; Finding F3 witness `IntervalMDP.Examples.b1_stationary_unsound` — `lean/IntervalMDPProofs/Models/Examples.lean`. Details: approximation row A7 | `partial` (Phase 3d; Finding F3) | `abstract` | L1–L6 (L1 floating-point rounding: Lean decides ties between action values by exact real comparison, Julia by `Float64` `>`/`<`, so on near-ties the selected action can differ; also the residual test of `CovergenceCriteria`; L2 overflow/underflow, `NaN`/`±Inf` (the `typemin`/`typemax` neutral) not modeled; L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription and docstrings (plus the scratch reproductions of Finding F3), not proved; L5 DP value ↔ path measure not proved: a strategy's value is the DP fixed point of its policy-evaluation step (`strategyReachLfp`, `strategyRewardValue`, super-solutions for exit time); L6 no LP solver involved). Hypotheses: `RMDP` structure invariants, `ActionOrder`, time-invariant model; `isExactTime = false` / super-solution / `ν < 1` for the infinite-horizon theorems (see A7). Mode coverage: all four satisfaction × strategy modes; optimistic-direction limitation as A4. Not covered: infinite-time safety, `TimeVaryingAvailableActions`. Findings: F3 (Julia's stationary-cache guard, benchmark B-1; `stationary_sound` holds for the documented cache only) |
| 10 | Interval value iteration | interval, factored | `ivi_step!`, `initialize_ivi!`, `IVIInitialGapCriteria` — `src/interval_value_iteration.jl`, `src/specification.jl` | CPU | Model: `IntervalMDP.IVI.step` (transcription of `ivi_step!`: strategy `iviStrategy` synthesized by `argoptAction` on the primary bound, applied by `Tπ` to the secondary bound, same nature, then `step_postprocess_value_function!` on both), `IntervalMDP.IVI.initializeIvi` (`initialize_ivi!`), `IntervalMDP.IVI.iviIter` (loop of `_interval_value_iteration!`), for a general `RMDP` (interval and factored IMDPs via `toRMDP`), both IVI strategy caches (`StrategyCacheKind`) — `lean/IntervalMDPProofs/IVI.lean`. Theorems: `IntervalMDP.IVI.lower_le_upper` (`V_lower_k ≤ V_upper_k`, every `k`, all four modes, all reach-avoid properties incl. exact time) — **proved**; `IntervalMDP.IVI.bracket` — **not stated, false** for `(Pessimistic, Maximize)` and `(Optimistic, Minimize)` (Finding F4); strongest proved: `IntervalMDP.IVI.bracket_aligned`, `lower_le_reachLfp`, `reachLfp_le_upper`, `primary_sound` (see A6); supporting `primary_iviIter` (the primary bound is robust VI `VI.step^[k]`), `iviStrategy_spec`, `iviIter_strategy_mem`, `iterate_sound`. Stopping (4b): `IntervalMDP.IVI.maxInitialGap` (transcription of the loop of `max_initial_gap`), `iviInitialGapCriteria` (`IVIInitialGapCriteria`), `stopIndex` (loop exit `k`), `iviValueFunction` (returned `value_function`); `IntervalMDP.IVI.gap_stop_sound` — **not stated, false** for `(Pessimistic, Maximize)` and `(Optimistic, Minimize)` (Finding F4; witnesses `IntervalMDP.Examples.not_gap_stop_sound_pessimistic_maximize`, `not_gap_stop_sound_optimistic_minimize`); strongest proved: `IntervalMDP.IVI.gap_stop_sound_aligned` (`(Pessimistic, Minimize)`, `(Optimistic, Maximize)`, non-exact-time reach-avoid only) and `IntervalMDP.IVI.gap_stop_primary_sound` (all four modes, non-exact-time reach-avoid only), both assuming the loop exits (hypothesis `h : Terminates …`; termination **not proved**) (see A6) | `partial` (Phases 4a, 4b; Finding F4) | `abstract` | L1–L6 (L1 floating-point rounding, including the gap test `max_initial_gap(…) < convergence_eps` and the early exit `budget ≤ 0` / `budget -= p` inside each `bellman!`; L2 overflow/underflow not modeled; L3 CUDA kernels (`ext/`) and threaded execution (`@threadstid`) not modeled; L4 Lean↔Julia correspondence by literal transcription of `ivi_step!` / `initialize_ivi!` / `_interval_value_iteration!` and docstrings, not proved; L5 DP value ↔ path measure not proved: `V*` is `reachLfp`, the least fixed point of the reach-avoid DP map; L6 no LP solver involved). Mode coverage: see A6 (bracket and gap stopping within `tol` restricted to `(Pessimistic, Minimize)`, `(Optimistic, Maximize)`, non-exact-time; `gap_stop_primary_sound` all four modes, non-exact-time; gap stopping assumes `Terminates`, not proved). Time-invariant model only |
| 11 | Vertex enumeration | factored | `state_action_bellman(::FactoredVertexIteratorWorkspace, …)` — `src/bellman.jl` | CPU | `IntervalMDP.Factored.vertices_complete`, `vertexValue_eq_opt` — `Factored.lean` | `none` | — | L1–L4 |
| 12 | Recursive O-max | factored | `state_action_bellman(::FactoredIntervalOMaxWorkspace, …)` — `src/bellman.jl` | CPU, CUDA | `IntervalMDP.Factored.recursiveOMax_sound`, `recursiveOMax_vi_sound` — `Factored.lean` | `none` | — | L1–L5 |
| 13 | LP McCormick relaxation | factored | `state_action_bellman(::FactoredIntervalMcCormickWorkspace, …)` — `src/bellman.jl` | CPU | `IntervalMDP.Factored.mcCormick_sound`, `mcCormick_vi_sound` — `Factored.lean` | `none` | — | L1–L6 |
| 14 | Product process (IMDP × DFA) | product | `bellman!` for `ProductProcess`, DFA reach/safety postprocess — `src/bellman.jl`, `src/specification.jl` | CPU, CUDA | `IntervalMDP.Product.T_eq_flat`, `IntervalMDP.Product.dfaReach_eq` — `Product.lean` | `none` | — | L1–L5 |

## Findings

- **F4 (Phase 4a, interval value iteration; A6 bracket and, Phase 4b, gap stopping) — OPEN.** Affected rows: algorithm row 10
  "Interval value iteration" and approximation row A6 (both `partial`). `IntervalMDP.IVI.bracket`
  (`V_lower_k ≤ V* ≤ V_upper_k` for every `k`, all four satisfaction × strategy modes) is false for
  Julia's strategy coupling in `ivi_step!` (`src/interval_value_iteration.jl`): the strategy is
  synthesized on the primary bound and applied unchanged to the secondary bound. For
  `(Pessimistic, Maximize)` the strategy that is best for `V_lower` can be worse than optimal for
  `V_upper`, so `V_upper_{k+1} = T_σ V_upper_k` can drop below `V*`; dually for
  `(Optimistic, Minimize)` `V_lower` can exceed `V*`. The other two modes are fine: there the
  strategy-mode direction keeps the secondary bound sound (`bracket_aligned`). The theorem is
  therefore not stated under its name; the strongest proved statements are `bracket_aligned`,
  `lower_le_reachLfp`, `reachLfp_le_upper`, `primary_sound` and `lower_le_upper`. Julia
  (1.13.1, `src/` = `origin/main` `4d7943c`; reproduction script below): five states, three actions, point masses (`lower = upper`); state 1 = `s` (action 1:
  `0.5·goal + 0.5·avoid`, action 2: `→ 2`, action 3: `→ 3`), state 2 `→ goal`, state 3 `→ avoid`,
  state 4 = goal, state 5 = avoid (absorbing); `InfiniteTimeReachAvoid([4], [5], 1e-6)`, initial
  states `[1]`, `VerificationProblem`, `IntervalValueIteration`, `V*` from
  `RobustValueIteration`, iterates recorded by `callback`:

  ```
  mode = (Pessimistic, Maximize)
    V* (RVI)        = [1.0, 1.0, 0.0, 1.0, 0.0]
    k=1  V_lower = [0.5, 1.0, 0.0, 1.0, 0.0]  V_upper = [0.5, 1.0, 0.0, 1.0, 0.0]
    IVI returned value_function = [0.5, 1.0, 0.0, 1.0, 0.0]  iterations = 1
    bracket: VIOLATED: k=1 upper<V* at [1]
  mode = (Pessimistic, Minimize)    ... bracket: HOLDS (stops at k = 2, value [0.0, …])
  mode = (Optimistic, Maximize)     ... bracket: HOLDS (stops at k = 2, value [1.0, …])
  mode = (Optimistic, Minimize)
    V* (RVI)        = [0.0, 1.0, 0.0, 1.0, 0.0]
    k=1  V_lower = [0.5, 1.0, 0.0, 1.0, 0.0]  V_upper = [0.5, 1.0, 0.0, 1.0, 0.0]
    IVI returned value_function = [0.5, 1.0, 0.0, 1.0, 0.0]  iterations = 1
    bracket: VIOLATED: k=1 lower>V* at [1]
  ```

  ```julia
  using IntervalMDP
  pointsets(cols) = (M = hcat(cols...); IntervalAmbiguitySets(; lower = M, upper = M))
  e(i) = (v = zeros(5); v[i] = 1.0; v)
  half = [0, 0, 0, 0.5, 0.5]
  ps = [pointsets([half, e(2), e(3)]), pointsets([e(4), e(4), e(4)]), pointsets([e(5), e(5), e(5)]),
      pointsets([e(4), e(4), e(4)]), pointsets([e(5), e(5), e(5)])]
  mdp = IntervalMarkovDecisionProcess(ps, [1])
  alg = IntervalMDP.default_bellman_algorithm(mdp)
  for sat in (Pessimistic, Optimistic), strat in (Maximize, Minimize)
      spec = Specification(InfiniteTimeReachAvoid([4], [5], 1e-6), sat, strat)
      vstar = value_function(solve(VerificationProblem(mdp, spec), RobustValueIteration(alg)))
      iters = []
      sol = solve(VerificationProblem(mdp, spec), IntervalValueIteration(alg);
          callback = (Vl, Vu, k) -> push!(iters, (copy(Vl), copy(Vu))))
      # compare every (Vl, Vu) in iters with vstar; print value_function(sol), num_iterations(sol)
  end
  ```

  One counterexample per direction (upper side for `Maximize`, lower side for `Minimize`), the
  same model. In both failing modes the gap at the initial state is `0` after call `1`, so
  `IVIInitialGapCriteria` stops at `k = 1` and returns `0.5` while `V* = 1` resp. `0`: the returned
  value is still sound for the satisfaction mode (`primary_sound`: `0.5 ≤ 1` pessimistic,
  `0.5 ≥ 0` optimistic) but not within `convergence_eps` of `V*`, so the second half of A6
  (`gap_stop_sound`, 4b) cannot hold as stated for these modes. Lean witnesses (Julia's 1-based
  states/actions are Lean's 0-based): `IntervalMDP.Examples.ivi_upper_lt_reachLfp` (`V_upper_1(s) =
  1/2 < 1 ≤ V*(s)`), `IntervalMDP.Examples.ivi_reachLfp_lt_lower` (`V*(s) ≤ 0 < 1/2 = V_lower_1(s)`),
  packaged as `not_bracket_pessimistic_maximize` / `not_bracket_optimistic_minimize`, for both IVI
  strategy caches — `lean/IntervalMDPProofs/Models/Examples.lean`. Possible fixes (follow-up spec,
  no `src/` change here): synthesize a separate strategy on each bound (optimize both bounds with
  their own `bellman!` call), or apply the primary strategy only in the aligned modes. No `src/`
  change was made.

  **Phase 4b consequence (`gap_stop_sound`).** The same defect refutes `IntervalMDP.IVI.gap_stop_sound`
  (all four modes: stopping on `IVIInitialGapCriteria(tol)` returns a value within `tol` of `V*` on
  the initial states). It is recorded here and not as a new Finding because it has the same root cause
  (the secondary bound is not a bound in the two misaligned modes, so the gap certifies nothing) and
  the same model and fix. Julia check (1.13.1, `src/` = `origin/main` `4d7943c`; script
  `ivi_gap_stop_4b.jl`, the model above, initial states `[1]`, `VerificationProblem` and
  `ControlSynthesisProblem`, `convergence_eps ∈ {1e-6, 0.25}`), identical for both problem types
  and both tolerances:

  ```
  criterion type: IntervalMDP.IVIInitialGapCriteria{Float64}
  mode=(Pessimistic,Maximize): k=1 gap_init=0.0 V*[1]=1.0 IVI[1]=0.5 err=0.5 -> NOT within eps
  mode=(Pessimistic,Minimize): k=2 gap_init=0.0 V*[1]=0.0 IVI[1]=0.0 err=0.0 -> WITHIN eps
  mode=(Optimistic,Maximize):  k=2 gap_init=0.0 V*[1]=1.0 IVI[1]=1.0 err=0.0 -> WITHIN eps
  mode=(Optimistic,Minimize):  k=1 gap_init=0.0 V*[1]=0.0 IVI[1]=0.5 err=0.5 -> NOT within eps
  aligned random sweep: 400/400 within eps=1e-4 (V* from RVI at 1e-14); worst err = 5.66e-5
  ```

  ```julia
  # model `mdp`, `alg` as in the reproduction above
  for eps in (1e-6, 0.25), sat in (Pessimistic, Optimistic), strat in (Maximize, Minimize)
      spec = Specification(InfiniteTimeReachAvoid([4], [5], eps), sat, strat)
      vstar = value_function(solve(VerificationProblem(mdp, spec), RobustValueIteration(alg)))
      sol = solve(VerificationProblem(mdp, spec), IntervalValueIteration(alg))
      err = abs(value_function(sol)[1] - vstar[1])   # initial state 1
      println((sat, strat), " k=", num_iterations(sol), " gap=", residual(sol)[1], " err=", err)
  end
  ```

  One counterexample per direction (`Maximize`: returned value below `V*`; `Minimize`: above). The
  random sweep (200 random 6-state, 2-action IMDPs per aligned mode, initial states `[1, 2]`) is
  evidence only. Lean witnesses (`lean/IntervalMDPProofs/Models/Examples.lean`):
  `IntervalMDP.Examples.not_gap_stop_sound_pessimistic_maximize` and
  `not_gap_stop_sound_optimistic_minimize` (for every `0 < tol ≤ 1/2` and both strategy caches, the
  loop exits at `stopIndex h = 1` and `¬ WithinOn [s] tol (iviValueFunction h) V*`). Strongest
  proved: `gap_stop_sound_aligned` (aligned modes) and `gap_stop_primary_sound` (all four modes,
  one-sided `Sound sat`), both for non-exact-time reach-avoid (`isExactTime = false`) and both
  assuming the loop exits (`h : Terminates …`; termination is not proved). Any fix of F4
  that restores the bracket in all modes makes `gap_stop_sound` provable from the same argument
  (`withinOn_of_bracket`).

- **F3 (Phase 3d, synthesized stationary strategy; = benchmark Finding B-1) — OPEN (Julia defect fixed, Lean re-check pending; see "Fix status" at the end of this entry).** Affected rows:
  algorithm row 9 "Synthesized strategy" and approximation row A7 (both `partial`). `stationary_sound`
  (A7) is proved for the **documented** `StationaryStrategyCache` (the previous action seeds every
  call, so ties keep it). Julia's `extract_strategy!(::StationaryStrategyCache, …)`
  (`src/strategy_cache.jl:127`) instead tests `all(iszero.(strategy[jₛ])) || jₛ ∉ available_actions`,
  comparing the *state* index `jₛ` with the action indices. For every state whose index exceeds the
  number of actions the seed is reset to `first(available_actions)` at each call, the tie between a
  self-loop and an exit then goes to the lower action index, and the returned strategy can be
  strictly worse than the reported value. So `stationary_sound` as stated is false for the Julia
  code. Julia (1.13.1, branch `lean/phase-3`, `src/` = `origin/main` `7b4f2a0`; 3 states, 2 actions,
  point masses; state 1 = goal, state 3 = `s`: action 1 self-loop, action 2 to the goal; state 2 = an
  absorbing filler):

      pt(ns, tgts) = (L = zeros(ns, length(tgts)); for (j, t) in enumerate(tgts); L[t, j] = 1.0; end; L)
      mdp_from(tg, ns) = IntervalMarkovDecisionProcess(
          [IntervalAmbiguitySets(; lower = pt(ns, t), upper = pt(ns, t)) for t in tg])
      mdp = mdp_from([[1, 1], [2, 2], [3, 1]], 3)
      spec = Specification(InfiniteTimeReachability([1], 1e-6), Pessimistic, Maximize)
      pol, V, k, _ = solve(ControlSynthesisProblem(mdp, spec))   # pol[1] = [(1,), (1,), (1,)], V = [1.0, 0.0, 1.0], k = 2
      Vσ, _, _ = solve(VerificationProblem(mdp, spec, pol))      # Vσ = [1.0, 0.0, 0.0]

  The reported value `1.0` at state 3 exceeds the value `0.0` of the returned strategy (self-loop),
  so `V_K ≤ V^σ` fails; `Optimistic`/`Maximize` gives the same numbers. With `s` moved to index 2
  (≤ 2 actions, guard passes; `mdp_from([[1, 1], [2, 1], [3, 3]], 3)`) the returned strategy keeps
  the goal action and `V = Vσ = [1.0, 1.0, 0.0]` in both modes. Random cross-check (scratch, 3000
  random 3–6-state IMDPs with point-mass, `[0, 1]` and `[0.25, 0.75]` columns,
  `InfiniteTimeReachability` / `InfiniteTimeReachAvoid`, `Maximize`, both modes): 0 violations of
  `V_K ≤ V^σ + 1e-6` in the 1500 models with `#states ≤ #actions` (guard inactive), 250 violations in
  the 1500 models with `#states > #actions`. **Minimize:** no counterexample exists — the
  `minimize` case of `stationary_le_superSolution` holds for every valid strategy
  (`viIter_le_superSolution_minimize`), and Julia's seeds are always available
  (`stationarySeed_available`, any guard outcome); the same model with `Minimize` gives
  `V = Vσ = [1.0, 0.0, 0.0]` in both modes. Lean witness: `IntervalMDP.Examples.b1_stationary_unsound`
  (`Models/Examples.lean`): on a two-state self-loop model, the strategy of a cache whose guard resets
  the seed (`stationarySeed … (keep := false)`) has `¬ Sound .pessimistic V₂ V^σ`, in both
  satisfaction modes (`V₂(s) = 1`, `V^σ(s) ≤ 0`). Strongest proved statement: `stationary_sound`
  (and `stationary_sound_exitTime`, `stationary_reward_error_bound`) for the documented cache, all
  four modes; `timeVarying_attains` (finite horizon) is unaffected. Fix (out of scope, no `src/`
  change; follow-up spec `4f.1-b1-strategy-cache.md` of the benchmark report): test
  `s ∉ available_actions` (the cached action) instead of `jₛ ∉ available_actions`.
  **Fix status.** The underlying Julia defect (benchmark B-1, GitHub issue #119) is fixed by the
  standalone fix change of `4f.1-b1-strategy-cache.md` (commit `1fc44d3`): `extract_strategy!(::StationaryStrategyCache, …)`
  now tests `all(iszero.(s)) || CartesianIndex(s) ∉ available_actions` with
  `s = strategy_cache.strategy[jₛ]`, so an available cached action seeds every call. The description
  above is the pre-fix (merge base `7498d0f`) behaviour. Regression test items in
  `test/base/synthesis.jl` ("base/synthesis: stationary strategy cache, …"): the 3-state, 2-action
  case above (both satisfaction modes, now `V = Vσ = [1.0, 0.0, 1.0]`, strategy `(2,)` at state 3),
  the 6-state, 4-action reproduction of benchmark B-1, and a Minimize case. Rows "Synthesized
  strategy" (row 9) and A7 stay `partial` and F3 stays open until `stationary_sound` (and
  `stationary_sound_exitTime`, `stationary_reward_error_bound`) has been re-checked against the
  fixed code. The fix changed no row status and no Lean statement or proof; only the Lean docstrings
  that described the pre-fix guard in the present tense (`VI/Strategy.lean` module docstring,
  `Bellman.stationarySeed`, `VI.stationaryCacheSeed`, `VI.stationary_sound`, the `Examples` module
  docstring, its "Finding F3" section docstring, `b1Strategy`, `Examples.b1_stationary_unsound`)
  were reworded to the past tense.
- **Phase 3d: other theorems.** `timeVarying_attains` (and its instances) holds for the Julia
  code as stated (time-varying caches seed with `first(available_actions)`, no guard involved).
- **F2 (Phase 2b, strategy lookup and given-strategy verification) — OPEN.** Affected rows: index row "Strategy lookup" (`partial`) and approximation row A8 "Given-strategy verification" (`partial`). Operator decision: kept open; fixing `checkstrategy` is a follow-up spec, not part of sub-phase 2b. `checkstrategy(strategy::AbstractArray,
  system::FactoredRMDP)` (`src/strategy.jl:35–65`, called by the `VerificationProblem` constructor,
  `src/problem.jl:34`) checks only the shape and `1 ≤ s[i] ≤ action_vars[i]`, not
  `isavailable(model, jₛ, jₐ)`. With `ListAvailableActions` (exported, `src/available_actions.jl`)
  a strategy that plays an unavailable action passes validation, `state_bellman!` with a
  `NonOptimizingStrategyCache` evaluates that action (`jₐ = CartesianIndex(strategy_cache[jₛ])`,
  `src/bellman.jl:498`), and the "verified" value can exceed the optimal value. So the spec
  theorem `IntervalMDP.Index.strategyAction_available` ("for strategies that pass validation, the
  looked-up action is in `available(model, jₛ)`") is false for the Julia code; it is not declared
  under that name. Lean counterexample (general, any model with an unavailable action):
  `IntervalMDP.Index.checkStrategy_admits_unavailable` — for every state `jₛ` and action
  `a ∉ available jₛ`, the array storing `Tuple(a)` everywhere passes `checkStrategy` and its lookup
  at `jₛ` is not in `juliaAvailable aa jₛ`. Julia (run on 1.13.1 at `origin/main` `a734447`):

      lower = [1.0 0.0 0.0 0.0; 0.0 1.0 1.0 1.0]   # columns (a, s): s1,a1 -> s1; s1,a2 -> s2; s2 -> s2
      sets = IntervalAmbiguitySets(; lower = lower, upper = copy(lower))
      marginal = Marginal(sets, (2,), (2,))
      aa = ListAvailableActions([[CartesianIndex(1)], [CartesianIndex(1), CartesianIndex(2)]])
      mdp = FactoredRobustMarkovDecisionProcess((2,), (2,), (2,), (marginal,), aa)
      spec = Specification(FiniteTimeReachability([2], 1), Pessimistic, Maximize)
      vp = VerificationProblem(mdp, spec, StationaryStrategy([(Int32(2),), (Int32(1),)]))  # accepted
      value_function(solve(vp))[1]                               # 1.0
      value_function(solve(ControlSynthesisProblem(mdp, spec)))[1]  # 0.0 (optimal)
      IntervalMDP.isavailable(mdp, CartesianIndex(1), CartesianIndex(2))  # false

  The verified value `1.0` of the given strategy exceeds the maximal value `0.0`, so the
  verification result is not sound (A8 direction `Tπ ≤ T` for `maximize` fails). Mirrored for `minimize`: with
  `aa = ListAvailableActions([[CartesianIndex(2)], [CartesianIndex(1), CartesianIndex(2)]])`,
  `Pessimistic`/`Minimize` and `StationaryStrategy([(Int32(1),), (Int32(1),)])` (action 1 unavailable
  in state 1), the verified value is `0.0` and the minimal value is `1.0`, so `Tπ ≥ T` fails. Strongest proved
  statements: `strategyAction_available_of_all` (validation suffices under `AllAvailableActions`,
  the only option of `IntervalMarkovDecisionProcess`), `strategyAction_available_of_valid` (any
  available actions, for a strategy valid in the Lean sense); `policy_eval_sound` (A8) is stated
  for valid strategies. Strategy-lookup row and A8: `partial`. Fix (out of scope, no `src/` change;
  follow-up spec): make `checkstrategy` check `isavailable(system, jₛ, Tuple(s))` for every source state.
- **Phase 2b: other theorems.** `T_interval_eq_omax`, `argopt_attains` and `policy_eval_sound`
  (for valid strategies) hold as stated; Observation O11 is extended below (not a Finding: no stated theorem is
  contradicted).
- **Phase 2a: no findings.** `T_mono`, `T_add_const`, `T_nonexpansive` hold as stated for every
  `RMDP` and all four modes, without convexity; no stated theorem is contradicted by the Julia
  `bellman!`/`state_bellman!`/`extract_strategy!` code (see Observation O11).
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
  **Phase 2b: became Finding F2.**
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
  `ext/cuda/bellman/dense.jl:260, 294`, `ext/cuda/bellman/sparse.jl:339, 380, 764, 802`,
  `ext/cuda/bellman/factored.jl:449–846` (`model[k][jₐ, jₛ]` with `getindex(::FactoredRMDP, r) =
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
- **O11 (Phase 2a).** No defect found in the Bellman operator for the value it returns under
  static available actions. Notes for later phases: (a) `extract_strategy!` for
  `StationaryStrategyCache` (`src/strategy_cache.jl`) seeds the loop with the previous action `s`
  and `values[s]` unless `all(iszero.(s)) || jₛ ∉ available_actions`; the second test compares
  the *state* index `jₛ` with the action list (presumably `s ∉ available_actions` was meant). With
  static available actions the previous `s` is always available, so the result is still the
  `max`/`min` over the available actions (= `extractValue`; `_extract_strategy!` uses strict
  `>`/`<`, which only affects which action is kept on ties). If the available actions changed
  between steps while a stationary cache is used, `values[s]` could be a stale entry of the shared
  `workspace.actions` from another state. Stationary caches are built only for infinite-horizon
  properties, and `TimeVaryingAvailableActions` is selected per step `k`; whether that combination
  is reachable is not established here. Follow-up: Phase 2b/3 (`argopt_attains`,
  `stationary_sound`). (b) `T` is the exact operator; for factored workspaces Julia's
  `state_action_bellman` is a sound approximation of `innerOpt` (Phase 5, A2/A3 via
  `Approx.iter_sound` with `T_monotone`). (c) Only nonemptiness and boundedness of the expected
  values are needed for the three theorems; closedness (`WellFormed`) additionally makes each inner
  optimum attained (`innerOpt_mem`), which strategy extraction (2b) will need.

- **O11, continued (Phase 2b).** Re-examined against `argopt_attains`, `stationarySeed_available`
  and `strategyAction_available_of_*`. (a) **No stated theorem is contradicted.** On a model with
  fixed available actions, the seed of `extract_strategy!(::StationaryStrategyCache, …)` is either
  `first(available_actions)` or the action the same cache stored for the same state at the previous
  call, which `argopt_attains` proves available; so whatever the guard
  `all(iszero.(s)) || jₛ ∉ available_actions` (`src/strategy_cache.jl:127`) evaluates to, every
  seed is available (`stationarySeed_available`, which leaves the guard outcome arbitrary) and
  every selected action is available and attains `T` (`argopt_attains`). The misguard only
  changes which available action is kept on ties. It does not affect `Index.strategyAction`
  (a lookup, not a selection). (b) **Reachable defect outside the theorems' model.** If the
  available actions change between calls (`TimeVaryingAvailableActions` with an infinite-horizon
  property, which selects a `StationaryStrategyCache` and steps through
  `select_available_actions(aa, k) = aa.actions[time_length(aa) - k]`,
  `src/robust_value_iteration.jl:266–267`; not rejected by any check), the previous action can be
  unavailable at the current step; when the guard passes (`jₛ` equal to an available action tuple)
  Julia seeds with it and with a stale `workspace.actions` entry. Trace (Julia 1.13.1,
  `origin/main` `a734447`; three states, `s1,a1 → s3` (sink), `s1,a2 → s2` (goal),
  `available(s1)` alternating `[a1]` / `[a2]`, internal `bellman!` calls in value-iteration order
  with `select_model(mdp, k)`): at `k = 3`, `available(s1) = [a1]`, the stationary cache returns
  `Vres[1] = 1.0` and keeps `a2`, while `NoStrategyCache` (the exact max) gives `0.0`; through
  `solve`, `ControlSynthesisProblem` converges to `[1.0, 1.0, 0.0]` after 2 iterations while the
  no-strategy `VerificationProblem` of the same spec runs out of time steps (`BoundsError`). The
  Phase 2 theorems are about a fixed `RMDP` (and `RMDP.atTime` for one step), so this is recorded
  here, not as a Finding; Phase 3 (`stationary_sound`, time-varying available actions) must decide
  whether the infinite-horizon + `TimeVaryingAvailableActions` combination is in scope (and, if
  so, raise it as a Finding). Fix (out of scope): test `s ∉ available_actions` instead of
  `jₛ ∉ available_actions`.
- **O11, concluded (Phase 3d).** (a) The guard `jₛ ∉ available_actions` (part (a) above) does
  contradict a stated theorem once strategies are synthesized for infinite horizons: it is Finding F3
  (benchmark B-1), with Julia and Lean counterexamples. (b) The infinite-horizon +
  `TimeVaryingAvailableActions` combination stays out of scope: all Phase 3 theorems are about a
  time-invariant `RMDP`, and `stationarySeed_available` (hence `stationary_sound`) needs fixed
  available actions; the stale-seed trace of part (b) above is unchanged and remains an Observation
  for a follow-up spec (fixing F3 by testing `s ∉ available_actions` also removes the stale seed).
- **O12 (Phase 3a).** No defect found in value iteration for reachability / reach-avoid: the
  transcription (`initialize!` on the zero `ValueFunction`, `step!` = `bellman!` +
  `step_postprocess_value_function!`, loop counter `k`) satisfies all four stated theorems for
  a time-invariant model. Scope notes: (a) the `CovergenceCriteria` residual `maximum(abs, u) < tol`
  gives no bound on `V* - V_k` (only `V_k ≤ V*`, A4); no error bound is claimed; (b) in
  `Optimistic` mode `V_k ≤ V*` is not the conservative direction (see A4); (c) exact-time
  properties are covered by `reachIter_mem_unit` only (their value is not monotone in the
  horizon); (d) time-varying available actions (`select_model(mp, k)`) are not modeled; the
  infinite-horizon + `TimeVaryingAvailableActions` question of O11(b) stays with the strategy
  sub-phase (`stationary_sound`).
- **O13 (Phase 3b).** No defect found in value iteration for safety or expected exit time: the
  transcriptions satisfy all stated theorems for a time-invariant model. Scratch cross-check
  (Julia 1.13.1, Float64, the 3-state test IMDP of `test/base/imdp.jl`): `FiniteTimeSafety([3], K)`
  and `ExactTimeReachAvoid([1, 2], [3], K)` agree to `2.2e-16` for `K = 1:6` in all four modes,
  the difference being the rounding of the `-1`/`+1` shift (L1). Scope note: when the exact
  expected exit time is infinite (e.g. an `IntervalMarkovChain` whose safe state 1 is a sure
  self-loop, `ExpectedExitTime([2], 1e-6)`), the iterates grow by `1` per step (`V = [1001.0, 0.0]`
  at `k = 1000`, stopped by a throwing `callback`), so the `CovergenceCriteria` residual stays `1`
  and `solve` never terminates for `convergence_eps ≤ 1`. This contradicts no stated theorem (no
  convergence or termination is claimed; `exitIter_mono`, `exitIter_succ` and `exitValue = ∞` match
  it); a follow-up spec may add a divergence check or an iteration cap.
- **O14 (Phase 3c).** No defect found in value iteration for discounted reward: the transcription
  satisfies all stated theorems for a time-invariant model. Scratch cross-check (Julia 1.13.1,
  Float64, the 3-state IMDP of `test/base/imdp.jl`, `InfiniteTimeReward([2, 1, -1], 0.9, 1e-3)`,
  all four modes, iterates recorded by `callback`): `V_1 = ν·bellman(r) + r` and every
  `V_{k+1} = ν·bellman(V_k) + r` hold exactly (difference `0.0`); `solve` stops at `k = 66` with
  residual `9.55e-4 < ε`, and `‖V_66 − V*‖∞ = 8.595044556349e-3 ≤ ν/(1−ν)·residual =
  8.595044557168e-3` (`V*` from `convergence_eps = 1e-13`), so the A5 bound is nearly tight here.
  `FiniteTimeReward(r, 1.5, 3)` is accepted and `InfiniteTimeReward(r, 1.0, 1e-3)` throws a
  `DomainError`, as `checkreward` / `checkdiscountupperbound` state. Scope notes: (a)
  `convergence_eps` is a bound on the last step, not on the error; the returned value is within
  `ν/(1−ν)·convergence_eps` of `V*` (A5), which exceeds `convergence_eps` when `ν > 1/2`; the
  `InfiniteTimeReward` docstring does not say so (documentation follow-up, no stated theorem
  contradicted); (b) for `FiniteTimeReward` with `ν ≥ 1` only the recursion is claimed.
- **O15 (Phase 4a).** Notes on interval value iteration beyond Finding F4. (a) For
  finite-horizon optimistic specifications IVI returns `V_upper_K`, which starts from `1 − 𝟙_avoid`
  instead of `𝟙_reach`, so it is not the `K`-step value that `RobustValueIteration` returns. Julia
  (the F4 model and script with `FiniteTimeReachAvoid([4], [5], 1)`, comparing the
  `value_function` of `RobustValueIteration` and `IntervalValueIteration`): `(Optimistic, Maximize)` RVI `[0.5, 1, 0, 1, 0]`, IVI `value_function`
  `[1.0, 1, 0, 1, 0]`; the pessimistic modes agree with RVI (`V_lower_K` is RVI, `primary_iviIter`).
  The value is still an upper bound of the infinite-horizon `V*` (`primary_sound`), and no stated
  theorem compares IVI with the finite-horizon value, so this is an Observation (documentation /
  follow-up spec). (b) The `upper` bound iterates from above towards the greatest fixed point; on end
  components in the don't-care region it does not reach `V*` and the gap may never close (the design
  note in `src/interval_value_iteration.jl` says so; MEC deflation is out of scope). This does not
  contradict any 4a theorem (`V* ≤ V_upper_k` still holds where proved). (c) `lower_le_upper` holds
  in every mode, so the returned `residual` `V_upper − V_lower` is nonnegative in exact arithmetic.

## Legacy verification gaps

Every algorithm row (1–14) except rows 1 and 2 (dense and sparse O-maximization, `proved` in Phases 1c and 1d), row 3 (robust Bellman operator, `proved` in Phases 2a and 2b), row 4 (strategy extraction and evaluation, `proved` in Phase 2b) row 5 (robust VI: reachability / reach-avoid, `proved` in Phase 3a), rows 6 and 8 (robust VI: safety and expected exit time, `proved` in Phase 3b), row 7 (robust VI: reward, `proved` in Phase 3c), row 9 (synthesized strategy, `partial` in Phase 3d, Finding F3), row 10 (interval value iteration, `partial` in Phases 4a and 4b, Finding F4), A2 and A3 (A1, A4 and A5 are `proved`; A6 is `partial`, Finding F4; A7 is `partial`, Finding F3; A8 is `partial`, Finding F2) and every index row except the three Phase 1a rows (linear index, sparse support pairing, sort permutation), the two Phase 1b rows (`Marginal` and `IntervalAmbiguitySets` `sub2ind`; `proved`; see Observation O8) and the Phase 2b strategy-lookup row (`partial`, Finding F2) are `none`: no Lean theorem yet. A task
touching one of these must supply its theorem and proof before it can pass the Formal Verification
gate.

## Summary

- Algorithms: 14; proved: 8 (dense O-maximization, Phase 1c; sparse O-maximization, Phase 1d; robust Bellman operator, Phases 2a + 2b: `T_mono`, `T_add_const`, `T_nonexpansive` for general `RMDP`, no convexity, and `T_interval_eq_omax`; strategy extraction and evaluation, Phase 2b: `argopt_attains`, `policy_eval_sound`; robust VI reachability / reach-avoid, Phase 3a: `reachIter_mem_unit`, `reachIter_mono`, `reachIter_tendsto_lfp`, `reachIter_sound`; robust VI safety, Phase 3b: `safety_shift_eq`; robust VI expected exit time, Phase 3b: `exitIter_succ`, `exitIter_mono`, `exitIter_sound`; robust VI reward, Phase 3c: `rewardIter_succ`, `reward_contracting`, `reward_error_bound`), all four modes; partial: 2 (row 9 synthesized strategy, Phase 3d: `timeVarying_attains`, `stationary_sound` for the documented cache; Finding F3; row 10 interval value iteration, Phases 4a and 4b: `lower_le_upper`, `primary_sound`, `bracket_aligned`, `lower_le_reachLfp`, `reachLfp_le_upper`, `gap_stop_sound_aligned`, `gap_stop_primary_sound`; Finding F4); none: 4 (rows 11–14).
- Models: 5 mapped rows proved (M1–M5), including `toSet_convex` and `productSet_not_convex`; approximation: A1 proved, A4 proved (reachability / reach-avoid in 3a, exit time in 3b; optimistic-direction limitation), A5 proved (3c, `0 < ν < 1`), A6 partial (4a, 4b, Finding F4), A7 partial (3d, Finding F3), A8 partial (Finding F2), A2, A3 none; indexing: 5 of 8 rows proved (Phase 1a: linear index, sparse support pairing, sort permutation; Phase 1b: `Marginal` and `IntervalAmbiguitySets` `sub2ind`, with Observation O8), 1 partial (Phase 2b strategy lookup, Finding F2), 2 none.
- Statement: the package is **not** verified. Phase 0 proves only model well-formedness and the
  generic soundness lift; Phase 1 (complete) adds the index theorems and dense and sparse O-max exactness; Phase 2 (complete) adds monotonicity, translation equivariance and sup-norm non-expansiveness of the robust Bellman operator on general `RMDP`s, its O-max form for interval MDPs, strategy extraction and given-strategy evaluation (for valid strategies); Phase 3a adds value iteration for reachability / reach-avoid (iterates in `[0, 1]`, monotone, convergent to the least fixed point, `V_k ≤ V*` — conservative only in pessimistic mode); Phase 3b adds the safety −1/+1 shift identity and, for expected exit time, the one-step recursion, monotone iterates and `V_k` below every finite super-solution (no convergence claimed); Phase 3c adds, for discounted reward, the one-step recursion, contraction for `0 < ν < 1` and the A5 stopping error bound `ν/(1−ν)·‖V_k − V_{k−1}‖∞`; Phase 3d adds synthesized strategies: the returned time-varying strategy attains `V_K` exactly, and the returned stationary strategy achieves at least `V_{K+1}` for infinite-time reachability / reach-avoid (exit time: below every super-solution; reward: inside the A5 interval) — for the documented stationary cache; all at abstract scope. Open Findings: F2 (Julia's `checkstrategy` does not check availability; affects the strategy-lookup row and A8, both `partial`); F3 (Julia's stationary-cache guard, benchmark B-1; affects row 9 and A7, both `partial`).
- Phase 1 close: every Phase 1 row is `proved` — index rows linear index, `Marginal` `sub2ind`, `IntervalAmbiguitySets` `sub2ind` (Observation O8), sparse support pairing, sort permutation; algorithm rows 1 (dense O-max) and 2 (sparse O-max). No Phase 1 row is `none` or `partial`; no open Findings.
- Phase 2 close: algorithm rows 3 (robust Bellman operator) and 4 (strategy extraction and evaluation) are `proved`; approximation row A8 (given-strategy verification) is `partial` with open Finding F2 (`policy_eval_sound` proved for valid strategies; Julia accepts strategies with unavailable actions); index row "Strategy lookup" is `partial` with open Finding F2 (`strategyAction_available` is false for Julia's validation; `strategyAction_available_of_all` and `strategyAction_available_of_valid` proved). No Phase 2 row is `none`. Observation O11 extended (stationary seed with time-varying available actions; Phase 3 follow-up).
- Phase 3a: algorithm row 5 (robust VI: reachability / reach-avoid) is `proved` (all four modes; monotonicity, convergence and A4 for non-exact-time properties); approximation row A4 is `partial` (reachability / reach-avoid part proved by `reachIter_sound`; `exitIter_sound` due in 3b), with the optimistic-direction limitation. No new Finding; Observation O12.
- Phase 3b: algorithm rows 6 (robust VI: safety, `safety_shift_eq`) and 8 (robust VI: expected exit time, `exitIter_succ`, `exitIter_mono`, `exitIter_sound`) are `proved` (all four modes; no convergence claimed for exit time); approximation row A4 is `proved` (reachability / reach-avoid and exit time), with the optimistic-direction limitation. No new Finding; Observation O13.
- Phase 3c: algorithm row 7 (robust VI: reward, `rewardIter_succ`, `reward_contracting`, `reward_error_bound`) is `proved` (all four modes; contraction and error bound for `0 < ν < 1`, restriction: `FiniteTimeReward` with `ν ≥ 1` has only the recursion); approximation row A5 is `proved` (Julia's stop at `‖V_k − V_{k−1}‖∞ < ε` gives `‖V_k − V*‖∞ < ν/(1−ν)·ε`). No new Finding; Observation O14.
- Phase 3d: algorithm row 9 (synthesized strategy) and approximation row A7 are `partial` with open Finding F3 (`timeVarying_attains` proved and holds for Julia; `stationary_sound`, `stationary_sound_exitTime`, `stationary_reward_error_bound` proved for the documented stationary cache, all four modes; Julia's guard `jₛ ∉ available_actions` (benchmark B-1) makes the returned stationary strategy unsound for `Maximize`, witnessed in Julia and by `Examples.b1_stationary_unsound`). Observation O11 concluded.
- Phase 3 close: algorithm rows 5 (reachability / reach-avoid), 6 (safety), 7 (reward) and 8 (expected exit time) are `proved`; row 9 (synthesized strategy) is `partial` with open Finding F3; approximation rows A4 and A5 are `proved`, A7 is `partial` with open Finding F3. No Phase 3 row is `none`. Every Phase 3 theorem (3a–3d) is listed in `lean/AxiomCheck.lean` and depends only on `propext`, `Classical.choice`, `Quot.sound`.
- Phase 4a: algorithm row 10 (interval value iteration) and approximation row A6 are `partial` with open Finding F4: `IVI.bracket` (all four modes) is false for Julia's strategy coupling in `(Pessimistic, Maximize)` (upper bound below `V*`) and `(Optimistic, Minimize)` (lower bound above `V*`), Lean witnesses in `Models/Examples.lean`; proved: `lower_le_upper` and `primary_sound` (all four modes; `primary_sound` for non-exact-time properties only), `lower_le_reachLfp` (three modes, non-exact-time properties only), `reachLfp_le_upper` (three modes, all reach-avoid properties) and `bracket_aligned` (`(Pessimistic, Minimize)`, `(Optimistic, Maximize)`, non-exact-time properties only). All listed in `lean/AxiomCheck.lean`, only `propext`, `Classical.choice`, `Quot.sound`. Observation O15.
- Phase 4b: algorithm row 10 and approximation row A6 stay `partial` with open Finding F4, extended: `IVI.gap_stop_sound` (all four modes) is false for `(Pessimistic, Maximize)` and `(Optimistic, Minimize)` (Julia stops at `k = 1` with gap `0` and returns `0.5` where `V* = 1` resp. `0`; Lean witnesses `Examples.not_gap_stop_sound_pessimistic_maximize`, `not_gap_stop_sound_optimistic_minimize`); proved: `gap_stop_sound_aligned` (returned value within `tol` of `V*` on the initial states, `(Pessimistic, Minimize)`, `(Optimistic, Maximize)`, non-exact-time reach-avoid only) and `gap_stop_primary_sound` (returned value `Sound sat`, all four modes, non-exact-time reach-avoid only), both assuming the loop exits (`Terminates`, not proved). All listed in `lean/AxiomCheck.lean`, only `propext`, `Classical.choice`, `Quot.sound`.
- Phase 4 close: algorithm row 10 (interval value iteration) and approximation row A6 are `partial` with open Finding F4 (`IVI.bracket` and `IVI.gap_stop_sound` false for two of four modes; `bracket_aligned` and `gap_stop_sound_aligned` proved for the other two, `lower_le_upper`, `primary_sound`, `gap_stop_primary_sound` for all four; `primary_sound`, `bracket_aligned`, `gap_stop_*` for non-exact-time reach-avoid only; `gap_stop_*` assume loop termination `Terminates`, which is not proved). No Phase 4 row is `none`. Every Phase 4 theorem (4a, 4b) is listed in `lean/AxiomCheck.lean`.
