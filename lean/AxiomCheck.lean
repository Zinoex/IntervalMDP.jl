import IntervalMDPProofs

/-!
# Axiom check

One `#print axioms` line per theorem in the spec's mapping tables, for the current and all earlier
phases, and (since Phase 5c) for every other `theorem` declared in `IntervalMDPProofs/**`. Run from
`lean/`:

    lake env lean AxiomCheck.lean

Completeness check (every declared theorem has an entry):

    python3 lean/scripts/check_axiom_check.py

Approved axioms: `propext`, `Classical.choice`, `Quot.sound`. Anything else (in particular
`sorryAx`) fails the formal gate.
-/

-- Phase 0: models
#print axioms IntervalMDP.IntervalAmbiguity.toSet_wellFormed
#print axioms IntervalMDP.IntervalAmbiguity.toSet_convex
#print axioms IntervalMDP.IntervalAmbiguity.budget_eq
#print axioms IntervalMDP.IMDP.toRMDP_wellFormed
#print axioms IntervalMDP.FactoredIMDP.toRMDP_wellFormed
#print axioms IntervalMDP.FactoredIMDP.productSet_not_convex
#print axioms IntervalMDP.ProductProcess.toRMDP_wellFormed
-- Phase 0: generic approximation soundness (A1)
#print axioms IntervalMDP.Approx.iter_sound
#print axioms IntervalMDP.Approx.fixedPoint_sound
-- Phase 0: supporting results referenced by the inventory
#print axioms IntervalMDP.AmbiguitySet.WellFormed.map
#print axioms IntervalMDP.AmbiguitySet.WellFormed.pi
#print axioms IntervalMDP.ProductProcess.lift_deterministic
#print axioms IntervalMDP.Examples.pi_not_convex_of_dirac_mem
#print axioms IntervalMDP.Examples.binaryFIMDP_productSet_not_convex
-- Phase 0: further theorems declared in this phase (completeness: every `theorem` of
-- `IntervalMDPProofs/**` is listed; hypotheses and scope as in each docstring)
#print axioms IntervalMDP.Approx.sound_pessimistic
#print axioms IntervalMDP.Approx.sound_optimistic
#print axioms IntervalMDP.Approx.Sound.refl
#print axioms IntervalMDP.Approx.Sound.trans
#print axioms IntervalMDP.Approx.Sound.apply_mono
#print axioms IntervalMDP.Approx.Sound.of_tendsto
#print axioms IntervalMDP.AmbiguitySet.mem_vecs
#print axioms IntervalMDP.AmbiguitySet.coe_mem_vecs
#print axioms IntervalMDP.AmbiguitySet.vecs_subset_stdSimplex
#print axioms IntervalMDP.AmbiguitySet.WellFormed.isCompact
#print axioms IntervalMDP.AmbiguitySet.vecs_ofVecs
#print axioms IntervalMDP.AmbiguitySet.ofVecs_vecs
#print axioms IntervalMDP.AmbiguitySet.vecs_map
#print axioms IntervalMDP.AmbiguitySet.IsConvex.map
#print axioms IntervalMDP.AmbiguitySet.continuous_piVec
#print axioms IntervalMDP.AmbiguitySet.vecs_pi
#print axioms IntervalMDP.AmbiguitySet.pi_nonempty
#print axioms IntervalMDP.AbstractLabelling.dist_deterministic
#print axioms IntervalMDP.AbstractLabelling.dist_probabilistic
#print axioms IntervalMDP.ProbVec.toFun_eq_coe
#print axioms IntervalMDP.ProbVec.ext
#print axioms IntervalMDP.ProbVec.ext_iff
#print axioms IntervalMDP.ProbVec.coe_injective
#print axioms IntervalMDP.ProbVec.coe_mem_stdSimplex
#print axioms IntervalMDP.ProbVec.coe_ofStdSimplex
#print axioms IntervalMDP.ProbVec.range_coe
#print axioms IntervalMDP.ProbVec.le_one
#print axioms IntervalMDP.ProbVec.dirac_apply
#print axioms IntervalMDP.ProbVec.map_apply
#print axioms IntervalMDP.ProbVec.map_dirac
#print axioms IntervalMDP.ProbVec.pi_apply
#print axioms IntervalMDP.ProbVec.prodKernel_apply
#print axioms IntervalMDP.Examples.mem_fullBinary_toSet
#print axioms IntervalMDP.Marginal.get_congr
#print axioms IntervalMDP.FactoredIMDP.toRMDP_ambiguity
#print axioms IntervalMDP.IMDP.toRMDP_ambiguity
#print axioms IntervalMDP.ProductProcess.toRMDP_ambiguity
#print axioms IntervalMDP.IMDP.toRMDP_available
#print axioms IntervalMDP.IntervalAmbiguity.lower_add_gap
#print axioms IntervalMDP.IntervalAmbiguity.gap_nonneg
#print axioms IntervalMDP.IntervalAmbiguity.budget_le_sum_gap
#print axioms IntervalMDP.IntervalAmbiguity.feasiblePoint_mem
#print axioms IntervalMDP.IntervalAmbiguity.toSet_eq_ofVecs
#print axioms IntervalMDP.IntervalAmbiguity.vecs_toSet
#print axioms IntervalMDP.ProductProcess.coe_lift
#print axioms IntervalMDP.AvailableActions.mem_all
#print axioms IntervalMDP.RMDP.withAvailable_ambiguity
#print axioms IntervalMDP.StationaryStrategy.Valid.toTimeVarying
#print axioms IntervalMDP.AvailableActions.exists_valid
-- Phase 1a: index foundations
#print axioms IntervalMDP.Index.linear_bijective
#print axioms IntervalMDP.Index.linear_succ_first
#print axioms IntervalMDP.Index.sparse_zip_correct
#print axioms IntervalMDP.Index.sortedPerm_bijective
#print axioms IntervalMDP.Index.greedy_visits_once
-- Phase 1a: supporting results referenced by the inventory
#print axioms IntervalMDP.Index.machineInt_eq_self
#print axioms IntervalMDP.Index.toJulia_bijOn
#print axioms IntervalMDP.Index.sortedPerm_fits_int32
#print axioms IntervalMDP.Index.gapValue_eq_sum_allocation
-- Phase 1a: further theorems declared in this phase (completeness: every `theorem` of
-- `IntervalMDPProofs/**` is listed; hypotheses and scope as in each docstring)
#print axioms IntervalMDP.Index.toJulia_mem
#print axioms IntervalMDP.Index.ofJulia_toJulia
#print axioms IntervalMDP.Index.toJulia_ofJulia
#print axioms IntervalMDP.Index.toJulia_injective
#print axioms IntervalMDP.Index.juliaGet_toJulia
#print axioms IntervalMDP.Index.map_toJulia_finRange
#print axioms IntervalMDP.Index.linear_eq_toJulia
#print axioms IntervalMDP.Index.juliaGet_nonneg
#print axioms IntervalMDP.Index.allocation_zero
#print axioms IntervalMDP.Index.budget_sub_min_nonneg
#print axioms IntervalMDP.Index.allocation_eq_zero_of_not_visited
#print axioms IntervalMDP.Index.gapValue_eq_res_add_sum
#print axioms IntervalMDP.Index.lookup_zip_eq_none
#print axioms IntervalMDP.Index.zip_map_eq
#print axioms IntervalMDP.Index.SparseCol.rowval_eq
-- Phase 1b: marginal indexing
#print axioms IntervalMDP.Index.marginalSub2ind_eq_linear
#print axioms IntervalMDP.Index.marginalSub2ind_bijective
#print axioms IntervalMDP.Index.marginalSub2ind_depends_only
#print axioms IntervalMDP.Index.intervalSub2ind_correct
-- Phase 1b: supporting results referenced by the inventory
#print axioms IntervalMDP.Index.intervalSub2ind_wrong_multiAction
#print axioms IntervalMDP.Index.marginalCartesian_surjective
-- Phase 1b: further theorems declared in this phase (completeness: every `theorem` of
-- `IntervalMDPProofs/**` is listed; hypotheses and scope as in each docstring)
#print axioms IntervalMDP.Index.foldl_horner
#print axioms IntervalMDP.Index.linear_eq_foldl
#print axioms IntervalMDP.Index.finRange_add
-- Phase 1c: dense O-maximization
#print axioms IntervalMDP.OMax.omax_mem
#print axioms IntervalMDP.OMax.omax_eq_sSup
#print axioms IntervalMDP.OMax.omax_eq_sInf
#print axioms IntervalMDP.OMax.omax_tie_invariant
-- Phase 1c: supporting results referenced by the inventory
#print axioms IntervalMDP.OMax.stateActionBellman_isGreatest
#print axioms IntervalMDP.OMax.stateActionBellman_isLeast
#print axioms IntervalMDP.OMax.stateActionBellman_eq_dot
-- Phase 1c: further theorems declared in this phase (completeness: every `theorem` of
-- `IntervalMDPProofs/**` is listed; hypotheses and scope as in each docstring)
#print axioms IntervalMDP.OMax.allocation_nonneg
#print axioms IntervalMDP.OMax.allocation_le_gap
#print axioms IntervalMDP.OMax.allocation_cons_of_ne
#print axioms IntervalMDP.OMax.sum_allocation
#print axioms IntervalMDP.OMax.exists_threshold
#print axioms IntervalMDP.OMax.sum_perm_eq
#print axioms IntervalMDP.OMax.Permutation.nodup
#print axioms IntervalMDP.OMax.Permutation.mem
#print axioms IntervalMDP.OMax.sum_allocation_eq_budget
#print axioms IntervalMDP.OMax.greedy_apply
#print axioms IntervalMDP.OMax.greedy_mem
#print axioms IntervalMDP.OMax.juliaGet_neg
#print axioms IntervalMDP.OMax.dot_neg
#print axioms IntervalMDP.OMax.dot_le_greedy
-- Phase 1d: sparse O-maximization
#print axioms IntervalMDP.OMax.omaxSparse_eq_omax
-- Phase 1d: supporting results referenced by the inventory
#print axioms IntervalMDP.OMax.omaxSparse_exact
#print axioms IntervalMDP.OMax.gapValue_sublist
#print axioms IntervalMDP.OMax.exists_permutation_sublist
-- Phase 1d: further theorems declared in this phase (completeness: every `theorem` of
-- `IntervalMDPProofs/**` is listed; hypotheses and scope as in each docstring)
#print axioms IntervalMDP.OMax.exists_perm_map_eq
#print axioms IntervalMDP.OMax.gapValueSparse_map
#print axioms IntervalMDP.OMax.gapValue_zero_budget
#print axioms IntervalMDP.OMax.sortedLe_trans
#print axioms IntervalMDP.OMax.sortedLe_total
-- Phase 2a: robust Bellman operator on general RMDPs (all four modes, no convexity)
#print axioms IntervalMDP.Bellman.T_mono
#print axioms IntervalMDP.Bellman.T_add_const
#print axioms IntervalMDP.Bellman.T_nonexpansive
-- Phase 2a: supporting results referenced by the inventory
#print axioms IntervalMDP.Bellman.T_le_add
#print axioms IntervalMDP.Bellman.T_monotone
#print axioms IntervalMDP.Bellman.T_lipschitz
#print axioms IntervalMDP.Bellman.innerOpt_mem
-- Phase 2a: further theorems declared in this phase (completeness: every `theorem` of
-- `IntervalMDPProofs/**` is listed; hypotheses and scope as in each docstring)
#print axioms IntervalMDP.Bellman.continuous_dot
#print axioms IntervalMDP.Bellman.expectations_isCompact
#print axioms IntervalMDP.Bellman.expectations_nonempty
#print axioms IntervalMDP.Bellman.dot_le_dot_add
#print axioms IntervalMDP.Bellman.innerOpt_le_add
#print axioms IntervalMDP.Bellman.extractValue_le_add

-- Phase 2b: interval specialisation, strategy extraction, policy evaluation (spec theorems)
#print axioms IntervalMDP.Bellman.T_interval_eq_omax
#print axioms IntervalMDP.Bellman.argopt_attains
#print axioms IntervalMDP.Bellman.policy_eval_sound
-- Phase 2b: `IntervalMDP.Index.strategyAction_available` is not proved: it is false for Julia's
-- `checkstrategy` (Finding F2). The strongest proved statements and the Finding witness:
#print axioms IntervalMDP.Index.strategyAction_available_of_all
#print axioms IntervalMDP.Index.strategyAction_available_of_valid
#print axioms IntervalMDP.Index.checkStrategy_admits_unavailable
-- Phase 2b: supporting results referenced by the inventory
#print axioms IntervalMDP.Bellman.stateActionBellman_interval_eq_omax
#print axioms IntervalMDP.Bellman.IntervalMDPLayout.column_eq
#print axioms IntervalMDP.Bellman.IntervalMDPLayout.columnInt_eq
#print axioms IntervalMDP.Bellman.stationarySeed_available
#print axioms IntervalMDP.Bellman.Tπ_eq_T_strategyAvailable
#print axioms IntervalMDP.Bellman.Tπ_stepSound
#print axioms IntervalMDP.Bellman.Tπ_interval_eq_omax
-- Phase 2b: further theorems declared in this phase (completeness: every `theorem` of
-- `IntervalMDPProofs/**` is listed; hypotheses and scope as in each docstring)
#print axioms IntervalMDP.Bellman.expectations_toSet
#print axioms IntervalMDP.Bellman.innerOpt_interval_eq_omax
#print axioms IntervalMDP.Bellman.optLE_refl
#print axioms IntervalMDP.Bellman.optLE_trans
#print axioms IntervalMDP.Bellman.argoptStep_spec
#print axioms IntervalMDP.Bellman.argoptAction_spec
#print axioms IntervalMDP.Bellman.extractValue_eq_of_opt
#print axioms IntervalMDP.Index.actionTuple_injective
#print axioms IntervalMDP.Index.strategyAction_eq_linear
-- Phase 3a: value iteration, reachability / reach-avoid
#print axioms IntervalMDP.VI.reachIter_mem_unit
#print axioms IntervalMDP.VI.reachIter_mono
#print axioms IntervalMDP.VI.reachIter_tendsto_lfp
#print axioms IntervalMDP.VI.reachIter_sound
-- Phase 3a: supporting results referenced by the inventory
#print axioms IntervalMDP.VI.reachIter_eq_iterate
#print axioms IntervalMDP.VI.T_zero
#print axioms IntervalMDP.VI.T_const
#print axioms IntervalMDP.VI.T_mem_unit
#print axioms IntervalMDP.VI.stepPostprocessValueFunction_mono
#print axioms IntervalMDP.VI.stepPostprocessValueFunction_mem_unit
#print axioms IntervalMDP.VI.continuous_stepPostprocessValueFunction
#print axioms IntervalMDP.VI.step_mono
#print axioms IntervalMDP.VI.step_mem_unit
#print axioms IntervalMDP.VI.continuous_step
#print axioms IntervalMDP.VI.initializeValueFunction_le_step
#print axioms IntervalMDP.VI.step_reachLfp
#print axioms IntervalMDP.VI.reachLfp_mem_unit
#print axioms IntervalMDP.VI.reachLfp_le_of_step_le
#print axioms IntervalMDP.VI.reachLfp_le_of_fixedPt

-- Phase 3b: value iteration for safety (−1/+1 shift) and expected exit time
#print axioms IntervalMDP.VI.safety_shift_eq
#print axioms IntervalMDP.VI.safetyIter_eq_sub_one
#print axioms IntervalMDP.VI.safetyIter_postprocess_mem_unit
#print axioms IntervalMDP.VI.exitIter_eq_iterate
#print axioms IntervalMDP.VI.initializeValueFunction_eq_exitStep_zero
#print axioms IntervalMDP.VI.ExpectedExitTime.stepPostprocessValueFunction_mono
#print axioms IntervalMDP.VI.ExpectedExitTime.initializeValueFunction_nonneg
#print axioms IntervalMDP.VI.exitStep_mono
#print axioms IntervalMDP.VI.exitIter_succ
#print axioms IntervalMDP.VI.exitIter_nonneg
#print axioms IntervalMDP.VI.exitIter_mono
#print axioms IntervalMDP.VI.exitIter_sound
#print axioms IntervalMDP.VI.exitIter_le_exitValue
#print axioms IntervalMDP.VI.exitValue_le_of_step_le

-- Phase 3c: value iteration for discounted reward (contraction, A5 error bound)
#print axioms IntervalMDP.VI.rewardIter_succ
#print axioms IntervalMDP.VI.reward_contracting
#print axioms IntervalMDP.VI.reward_error_bound

-- Phase 3c: supporting results referenced by the inventory
#print axioms IntervalMDP.Property.toRewardProperty_discount_lt_one
#print axioms IntervalMDP.VI.rewardIter_eq_iterate
#print axioms IntervalMDP.VI.rewardStep_dist_le
#print axioms IntervalMDP.VI.rewardValue_isFixedPt
#print axioms IntervalMDP.VI.rewardIter_tendsto
#print axioms IntervalMDP.VI.reward_stop_bound
#print axioms IntervalMDP.VI.reward_error_interval

-- Phase 3d: synthesized strategies (A7)
#print axioms IntervalMDP.VI.timeVarying_attains
#print axioms IntervalMDP.VI.stationary_sound
#print axioms IntervalMDP.VI.timeVarying_attains_reach
#print axioms IntervalMDP.VI.timeVarying_attains_safety
#print axioms IntervalMDP.VI.timeVarying_attains_reward
#print axioms IntervalMDP.VI.policyEvalIter_timeVaryingCacheStrategy
#print axioms IntervalMDP.VI.synthesizedStrategy_spec
#print axioms IntervalMDP.VI.timeVaryingCacheStrategy_valid
#print axioms IntervalMDP.VI.stationaryCacheStrategy_valid
#print axioms IntervalMDP.VI.T_lt_of_switch
#print axioms IntervalMDP.VI.strategy_eq_of_T_eq
#print axioms IntervalMDP.VI.viIter_le_superSolution_minimize
#print axioms IntervalMDP.VI.stationary_backward_step
#print axioms IntervalMDP.VI.stationary_le_superSolution
#print axioms IntervalMDP.VI.stationary_sound_exitTime
#print axioms IntervalMDP.VI.stationary_reward_error_bound
-- Phase 3d: Finding F3 witness (benchmark B-1)
#print axioms IntervalMDP.Examples.b1_stationary_unsound
-- Phase 3d: further theorems declared in this phase (completeness: every `theorem` of
-- `IntervalMDPProofs/**` is listed; hypotheses and scope as in each docstring)
#print axioms IntervalMDP.VI.StepPostprocess.apply_mono
#print axioms IntervalMDP.Examples.selfLoop_stateActionBellman
#print axioms IntervalMDP.VI.ActionOrder.acts_ne_nil
#print axioms IntervalMDP.VI.ActionOrder.first_mem
#print axioms IntervalMDP.VI.stationaryCacheSeed_mem
#print axioms IntervalMDP.VI.stationaryCacheSeed_succ
#print axioms IntervalMDP.VI.selectStrategy_timeVaryingCacheStrategy
#print axioms IntervalMDP.VI.reachIter_eq_viIter
#print axioms IntervalMDP.VI.safetyIter_eq_viIter
#print axioms IntervalMDP.VI.exitIter_eq_viIter
#print axioms IntervalMDP.VI.rewardIter_eq_viIter
#print axioms IntervalMDP.VI.ReachProperty.stepPostprocess_apply
#print axioms IntervalMDP.VI.ExpectedExitTime.stepPostprocess_apply
#print axioms IntervalMDP.VI.argoptAction_eq_or_lt
#print axioms IntervalMDP.VI.stateActionBellman_mono
#print axioms IntervalMDP.VI.exists_dot_bracket
#print axioms IntervalMDP.VI.exists_eq_zero_of_dot_nonpos
-- Phase 4a: interval value iteration (IVI): bounds and bracket (A6)
-- `IntervalMDP.IVI.bracket` (all four modes) is false for Julia's coupling (Finding F4) and is not stated;
-- `bracket_aligned` is the proved restriction.
#print axioms IntervalMDP.IVI.lower_le_upper
#print axioms IntervalMDP.IVI.primary_sound
#print axioms IntervalMDP.IVI.lower_le_reachLfp
#print axioms IntervalMDP.IVI.reachLfp_le_upper
#print axioms IntervalMDP.IVI.bracket_aligned
#print axioms IntervalMDP.IVI.primary_iviIter
#print axioms IntervalMDP.IVI.primary_step
#print axioms IntervalMDP.IVI.iviStrategy_spec
#print axioms IntervalMDP.IVI.iviIter_strategy_mem
#print axioms IntervalMDP.IVI.iterate_sound
#print axioms IntervalMDP.IVI.initializeIvi_lower_le_upper
#print axioms IntervalMDP.IVI.initializeValueFunction_le_reachLfp
#print axioms IntervalMDP.IVI.reachLfp_le_initializeUpper
#print axioms IntervalMDP.IVI.lower_le_iterate
#print axioms IntervalMDP.IVI.iterate_le_upper
#print axioms IntervalMDP.IVI.Tπ_iviStrategy_lower_le
#print axioms IntervalMDP.IVI.T_le_Tπ_iviStrategy_upper
-- Phase 4a: Finding F4 witnesses
#print axioms IntervalMDP.Examples.ivi_upper_lt_reachLfp
#print axioms IntervalMDP.Examples.ivi_reachLfp_lt_lower
#print axioms IntervalMDP.Examples.not_bracket_pessimistic_maximize
#print axioms IntervalMDP.Examples.not_bracket_optimistic_minimize
-- Phase 4a: further theorems declared in this phase (completeness: every `theorem` of
-- `IntervalMDPProofs/**` is listed; hypotheses and scope as in each docstring)
#print axioms IntervalMDP.IVI.ReachAvoidProperty.disjoint_reach_avoid
#print axioms IntervalMDP.IVI.ReachAvoidProperty.reach_toReachProperty
#print axioms IntervalMDP.IVI.ReachAvoidProperty.stepPostprocess_avoid
#print axioms IntervalMDP.IVI.step_lower
#print axioms IntervalMDP.IVI.step_upper
#print axioms IntervalMDP.IVI.step_strategy
#print axioms IntervalMDP.IVI.primary_step_eq
#print axioms IntervalMDP.IVI.cacheSeed_mem
#print axioms IntervalMDP.IVI.Tπ_mono
#print axioms IntervalMDP.Examples.halfGoalAvoid_apply
#print axioms IntervalMDP.Examples.iviRMDP_stateActionBellman
#print axioms IntervalMDP.Examples.dot_dirac_fin5
#print axioms IntervalMDP.Examples.dot_halfGoalAvoid
#print axioms IntervalMDP.Examples.iviProp_post
#print axioms IntervalMDP.Examples.iviOrder_cacheSeed
-- Phase 4b: interval value iteration (IVI): stopping on the initial-state gap (A6)
-- `IntervalMDP.IVI.gap_stop_sound` (all four modes) is false for Julia's coupling (Finding F4) and is
-- not stated; `gap_stop_sound_aligned` is the proved restriction, `gap_stop_primary_sound` the
-- all-mode one-sided part.
#print axioms IntervalMDP.IVI.gap_stop_sound_aligned
#print axioms IntervalMDP.IVI.gap_stop_primary_sound
#print axioms IntervalMDP.IVI.withinOn_of_iviInitialGapCriteria_aligned
#print axioms IntervalMDP.IVI.withinOn_of_bracket
#print axioms IntervalMDP.IVI.gap_le_maxInitialGap
#print axioms IntervalMDP.IVI.le_foldl_maxGapStep
#print axioms IntervalMDP.IVI.one_le_stopIndex
#print axioms IntervalMDP.IVI.iviInitialGapCriteria_stopIndex
#print axioms IntervalMDP.IVI.not_iviInitialGapCriteria_of_lt_stopIndex
-- Phase 4b: Finding F4 witnesses for `gap_stop_sound`
#print axioms IntervalMDP.Examples.not_gap_stop_sound_pessimistic_maximize
#print axioms IntervalMDP.Examples.not_gap_stop_sound_optimistic_minimize
#print axioms IntervalMDP.Examples.iviInitialGapCriteria_one_pessimistic_maximize
#print axioms IntervalMDP.Examples.iviInitialGapCriteria_one_optimistic_minimize
#print axioms IntervalMDP.Examples.iviIter_one_pessimistic_maximize
#print axioms IntervalMDP.Examples.iviIter_one_optimistic_minimize
#print axioms IntervalMDP.Examples.one_le_reachLfp_pessimistic_maximize
#print axioms IntervalMDP.Examples.reachLfp_optimistic_minimize_nonpos
-- Phase 4b: further theorems declared in this phase (completeness: every `theorem` of
-- `IntervalMDPProofs/**` is listed; hypotheses and scope as in each docstring)
#print axioms IntervalMDP.Examples.iviInitLower_apply
#print axioms IntervalMDP.Examples.iviInitUpper_apply
#print axioms IntervalMDP.Examples.iviStrategy_pessimistic_maximize
#print axioms IntervalMDP.Examples.iviStrategy_optimistic_minimize
-- Phase 5a: factored successor index and vertex enumeration (arXiv:2508.00707, Theorem 1)
#print axioms IntervalMDP.Index.factored_successor_eq
#print axioms IntervalMDP.Factored.vertices_complete
#print axioms IntervalMDP.Factored.vertexValue_eq_opt
#print axioms IntervalMDP.Factored.vertexValue_eq_stateActionBellman
#print axioms IntervalMDP.Factored.mem_vertices_iff
#print axioms IntervalMDP.Factored.mem_vertices_iff_extremePoints
#print axioms IntervalMDP.Factored.vertexLoop_prefix
#print axioms IntervalMDP.Factored.nextPermutation_spec
#print axioms IntervalMDP.Factored.vertexOf_mem_vertexRun
#print axioms IntervalMDP.Factored.vecs_eq_convexHull_vertices
#print axioms IntervalMDP.Factored.exists_vertex_expansion
-- Phase 5a: further theorems declared in this phase (completeness: every `theorem` of
-- `IntervalMDPProofs/**` is listed; hypotheses and scope as in each docstring)
#print axioms IntervalMDP.Factored.successorState_bijective
#print axioms IntervalMDP.Factored.allocation_of_not_mem
#print axioms IntervalMDP.Factored.vertexLoop_fst
#print axioms IntervalMDP.Factored.vertexLoop_snd
#print axioms IntervalMDP.Factored.vertexLoop_take
#print axioms IntervalMDP.Factored.vertexLoop_append
#print axioms IntervalMDP.Factored.sum_lt_of_vertexLoop_snd_eq_zero
#print axioms IntervalMDP.Factored.sum_map_of_perm
#print axioms IntervalMDP.Factored.sum_map_juliaGet
#print axioms IntervalMDP.Factored.vertexOf_snd_ne_zero
#print axioms IntervalMDP.Factored.vertexOf_fst
#print axioms IntervalMDP.Factored.vertexOf_fst_of_budget_eq_zero
#print axioms IntervalMDP.Factored.suffixStep_spec
#print axioms IntervalMDP.Factored.foldl_suffixStep
#print axioms IntervalMDP.Factored.nextInSuffix_some
#print axioms IntervalMDP.Factored.nextInSuffix_none
#print axioms IntervalMDP.Factored.findSwap_some
#print axioms IntervalMDP.Factored.findSwap_none
#print axioms IntervalMDP.Factored.permGet_eq
#print axioms IntervalMDP.Factored.swapSort_eq
#print axioms IntervalMDP.Factored.lt_or_eq_of_sorted_perm
#print axioms IntervalMDP.Factored.nextPermutation_some
#print axioms IntervalMDP.Factored.mem_vertexRun
#print axioms IntervalMDP.Factored.permsFrom_pos
#print axioms IntervalMDP.Factored.permsFrom_le
#print axioms IntervalMDP.Factored.permsFrom_lt
#print axioms IntervalMDP.Factored.range_le_of_perm
#print axioms IntervalMDP.Factored.allocation_boundary
#print axioms IntervalMDP.Factored.vertexOf_mem_vecs
#print axioms IntervalMDP.Factored.mem_extremePoints_of_boundary
#print axioms IntervalMDP.Factored.vertexOf_mem_extremePoints
#print axioms IntervalMDP.Factored.boundary_of_mem_extremePoints
#print axioms IntervalMDP.Factored.allocation_full_prefix
#print axioms IntervalMDP.Factored.exists_toJulia_of_mem_range
#print axioms IntervalMDP.Factored.exists_perm_vertexOf_eq
#print axioms IntervalMDP.Factored.mem_productList
#print axioms IntervalMDP.Factored.foldl_maxStep
#print axioms IntervalMDP.Factored.foldl_minStep
#print axioms IntervalMDP.Factored.vertexSum_eq_dot
#print axioms IntervalMDP.Factored.dot_piVec_eq_sum
#print axioms IntervalMDP.Factored.sum_prod_weights
#print axioms IntervalMDP.Factored.vertexEnumeration_ne_nil
#print axioms IntervalMDP.Factored.mem_vertexEnumeration_iff
#print axioms IntervalMDP.Factored.vertexSum_mem_expectations

-- Phase 5b: recursive O-max soundness (A2; arXiv:2411.11803, Theorem 6.1).
-- The six theorem entries directly below assume two or more state variables
-- (`FactoredIMDP (n + 2) m`) and dense marginals.
-- `recursiveOMax_eq_nestedOMax`: `n + 2` state variables, dense (no mode).
-- `recursiveOMax_sound` (`n + 2`, dense, all four modes), `recursiveOMax_vi_sound` (`n + 2`, dense,
-- all four modes, takes `h₀ : Sound sat W₀ V₀`) and `recursiveOMax_vi_sound_post` (`n + 2`, dense,
-- takes `hpost : Monotone post` and `h₀`; all four modes).
-- `recursiveOMax_le_innerOpt`: `n + 2`, dense, pessimistic only (`upper_bound = false`).
-- `innerOpt_le_recursiveOMax`: `n + 2`, dense, optimistic only (`upper_bound = true`).
#print axioms IntervalMDP.Factored.recursiveOMax_sound
#print axioms IntervalMDP.Factored.recursiveOMax_vi_sound
#print axioms IntervalMDP.Factored.recursiveOMax_vi_sound_post
#print axioms IntervalMDP.Factored.recursiveOMax_le_innerOpt
#print axioms IntervalMDP.Factored.innerOpt_le_recursiveOMax
#print axioms IntervalMDP.Factored.recursiveOMax_eq_nestedOMax
-- Phase 5b helper entries. These take `FactoredIMDP (n + 2) m`, except seven generic ones:
-- `lookup_zip_map_toJulia`, `denseCol_getindex` and `orthogonalInnerBellman_eq_innerOpt` (a single
-- marginal over `Fin d`; the last covers both directions), and `foldl_finRange_last`,
-- `foldl_writeAt_apply`, `foldl_writeAt_finRange` and `dot_piVec_update_sum` (list/fold/sum
-- lemmas); see their docstrings.
#print axioms IntervalMDP.Factored.lookup_zip_map_toJulia
#print axioms IntervalMDP.Factored.denseCol_getindex
#print axioms IntervalMDP.Factored.orthogonalInnerBellman_eq_innerOpt
#print axioms IntervalMDP.Factored.lastOf_apply
#print axioms IntervalMDP.Factored.nestedOMax_congr
#print axioms IntervalMDP.Factored.foldl_finRange_last
#print axioms IntervalMDP.Factored.foldl_writeAt_apply
#print axioms IntervalMDP.Factored.foldl_writeAt_finRange
#print axioms IntervalMDP.Factored.cascadeLoop_tail
#print axioms IntervalMDP.Factored.firstInnerBellman_eq
#print axioms IntervalMDP.Factored.innerBellmanAt_cacheValue
#print axioms IntervalMDP.Factored.supportSize_succ
#print axioms IntervalMDP.Factored.foldl_cartesianBlock
#print axioms IntervalMDP.Factored.expectationCache_fill
#print axioms IntervalMDP.Factored.dot_piVec_update_sum
#print axioms IntervalMDP.Factored.dot_piVec_partialProduct_zero
#print axioms IntervalMDP.Factored.partialProduct_update
#print axioms IntervalMDP.Factored.partialProduct_succ
#print axioms IntervalMDP.Factored.partialProduct_of_le
#print axioms IntervalMDP.Factored.nestedOMax_le_dot
#print axioms IntervalMDP.Factored.dot_le_nestedOMax

-- Phase 5c: McCormick relaxation soundness (A3; arXiv:2508.00707, Section 3.2 and Appendix B,
-- no theorem number in the paper). Entries grouped by their actual scope.
-- `mcCormick_sound`: one or more state variables (`n + 1`, `FactoredIMDP (n + 1) m`), dense, all four modes.
#print axioms IntervalMDP.Factored.mcCormick_sound
-- `mcCormick_vi_sound`: `n + 1`, dense, all four modes, takes `h₀ : Sound sat W₀ V₀`.
#print axioms IntervalMDP.Factored.mcCormick_vi_sound
-- `mcCormick_vi_sound_post`: `n + 1`, dense, all four modes, takes `hpost : Monotone post` and `h₀`.
#print axioms IntervalMDP.Factored.mcCormick_vi_sound_post
-- `mcCormickLP_le_innerOpt`: `n + 1`, dense, pessimistic only (`upper_bound = false`).
#print axioms IntervalMDP.Factored.mcCormickLP_le_innerOpt
-- `innerOpt_le_mcCormickLP`: `n + 1`, dense, optimistic only (`upper_bound = true`).
#print axioms IntervalMDP.Factored.innerOpt_le_mcCormickLP
-- Phase 5c helpers taking `FactoredIMDP (n + 1) m` (dense; no mode):
#print axioms IntervalMDP.Factored.dot_mem_mcCormickValues
#print axioms IntervalMDP.Factored.mcCormickValues_bddBelow
#print axioms IntervalMDP.Factored.mcCormickValues_bddAbove
#print axioms IntervalMDP.Factored.mcCormickObjective_bounds
#print axioms IntervalMDP.Factored.mcCormickTree_branchIndex_bijective
-- Phase 5c helpers on the tree shape only (lists of `Fin (n + 1)`, no model):
#print axioms IntervalMDP.Factored.mccormickBranch_leaves
#print axioms IntervalMDP.Factored.mcCormickTree_leaves
-- Phase 5c generic helpers (any `McCormickTree N` and any interval marginals; the envelope lemma is
-- about reals only):
#print axioms IntervalMDP.Factored.mcCormickEnvelope_mul
#print axioms IntervalMDP.Factored.branchLower_nonneg
#print axioms IntervalMDP.Factored.branchFeasible_bounds
#print axioms IntervalMDP.Factored.branchFeasible_sum
#print axioms IntervalMDP.Factored.branchFeasible_mem_unit
#print axioms IntervalMDP.Factored.branchFeasible_branchProduct
#print axioms IntervalMDP.Factored.branchVars_branchProduct
#print axioms IntervalMDP.Factored.branchIndex_congr
#print axioms IntervalMDP.Factored.eq_of_branchIndex_eq
#print axioms IntervalMDP.Factored.branchFill_of_not_mem
#print axioms IntervalMDP.Factored.branchIndex_branchFill
-- Phase 6a: product-state indexing (scope `abstract`; mode-independent: indexing is the same for all
-- four satisfaction × strategy modes). `productIndex_bijective`: hypothesis `hBound`
-- (`(∏ state_values(mp)) * nQ < 2 ^ (N - 1)`), any `StateVars n` and `nQ`.
#print axioms IntervalMDP.Index.productIndex_bijective
-- `product_read_successor`: hypothesis `hBound`, deterministic labelling, mode-independent.
#print axioms IntervalMDP.Index.product_read_successor
-- Phase 6a helpers with hypothesis `hBound`, mode-independent: `productIndexInt_eq` and
-- `productGet_productArray` (`hBound` only) and `product_read_nextDFA` (`hBound`, `hL`, deterministic
-- labelling).
#print axioms IntervalMDP.Index.productIndexInt_eq
#print axioms IntervalMDP.Index.productGet_productArray
#print axioms IntervalMDP.Index.product_read_nextDFA
-- Phase 6a helpers on exact natural-number indices (no overflow bound needed, no model invariant
-- beyond `StateVars n`):
#print axioms IntervalMDP.Index.productDims_castSucc
#print axioms IntervalMDP.Index.productDims_last
#print axioms IntervalMDP.Index.productCartesian_castSucc_val
#print axioms IntervalMDP.Index.productCartesian_last_val
#print axioms IntervalMDP.Index.prod_productDims
#print axioms IntervalMDP.Index.productIndex_eq_toJulia
#print axioms IntervalMDP.Index.stride_productDims_castSucc
#print axioms IntervalMDP.Index.stride_productDims_last
#print axioms IntervalMDP.Index.productIndex_eq
#print axioms IntervalMDP.Index.productIndex_bijOn
#print axioms IntervalMDP.Index.selectdimIndex_bijOn
-- Phase 6b: product Bellman and DFA reachability / safety (scope `abstract`). `T_eq_flat`:
-- hypothesis `hBound` (`(∏ state_values(mp)) * nQ < 2 ^ (N - 1)`), deterministic labelling (`hL`),
-- all four satisfaction × strategy modes; uses `product_read_successor`.
#print axioms IntervalMDP.Product.T_eq_flat
-- `dfaReach_eq`: hypothesis `hBound`, deterministic labelling (`hL`), all four modes, time-invariant
-- model; reachability and safety.
#print axioms IntervalMDP.Product.dfaReach_eq
-- Phase 6b helpers with hypothesis `hBound` and deterministic labelling (`hL`), all four modes:
#print axioms IntervalMDP.Product.T_eq_flat_empty
#print axioms IntervalMDP.Product.T_read
#print axioms IntervalMDP.Product.dfaReachIter_eq
#print axioms IntervalMDP.Product.dfaSafetyIter_eq
#print axioms IntervalMDP.Product.dfaSafety_eq
#print axioms IntervalMDP.Product.dfaReach_read
#print axioms IntervalMDP.Product.dfaSafety_read
-- Phase 6b helpers with hypothesis `hBound` only (index level; `bellmanLoop_apply` and
-- `intermediateValues_productArray` take a deterministic `Labelling`; mode-independent or all four
-- modes):
#print axioms IntervalMDP.Product.productIndexInt_toNat
#print axioms IntervalMDP.Product.writeSlice_apply
#print axioms IntervalMDP.Product.setDFAStates_apply
#print axioms IntervalMDP.Product.intermediateValues_productArray
#print axioms IntervalMDP.Product.bellmanLoop_apply
-- Phase 6b helper with deterministic labelling (`hL`) only, no index bound:
#print axioms IntervalMDP.Product.nextDFAValue_deterministic
-- Phase 6b helpers for any labelling (deterministic or probabilistic), no hypotheses beyond the
-- model invariants; `T_toRMDP_apply` for all four modes:
#print axioms IntervalMDP.Product.dot_liftLinear
#print axioms IntervalMDP.Product.expectations_toRMDP
#print axioms IntervalMDP.Product.T_toRMDP_apply
