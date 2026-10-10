import IntervalMDPProofs

/-!
# Axiom check

One `#print axioms` line per theorem in the spec's mapping tables, for the current and all earlier
phases. Run from `lean/`:

    lake env lean AxiomCheck.lean

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
-- Phase 1b: marginal indexing
#print axioms IntervalMDP.Index.marginalSub2ind_eq_linear
#print axioms IntervalMDP.Index.marginalSub2ind_bijective
#print axioms IntervalMDP.Index.marginalSub2ind_depends_only
#print axioms IntervalMDP.Index.intervalSub2ind_correct
-- Phase 1b: supporting results referenced by the inventory
#print axioms IntervalMDP.Index.intervalSub2ind_wrong_multiAction
#print axioms IntervalMDP.Index.marginalCartesian_surjective
-- Phase 1c: dense O-maximization
#print axioms IntervalMDP.OMax.omax_mem
#print axioms IntervalMDP.OMax.omax_eq_sSup
#print axioms IntervalMDP.OMax.omax_eq_sInf
#print axioms IntervalMDP.OMax.omax_tie_invariant
-- Phase 1c: supporting results referenced by the inventory
#print axioms IntervalMDP.OMax.stateActionBellman_isGreatest
#print axioms IntervalMDP.OMax.stateActionBellman_isLeast
#print axioms IntervalMDP.OMax.stateActionBellman_eq_dot
-- Phase 1d: sparse O-maximization
#print axioms IntervalMDP.OMax.omaxSparse_eq_omax
-- Phase 1d: supporting results referenced by the inventory
#print axioms IntervalMDP.OMax.omaxSparse_exact
#print axioms IntervalMDP.OMax.gapValue_sublist
#print axioms IntervalMDP.OMax.exists_permutation_sublist
-- Phase 2a: robust Bellman operator on general RMDPs (all four modes, no convexity)
#print axioms IntervalMDP.Bellman.T_mono
#print axioms IntervalMDP.Bellman.T_add_const
#print axioms IntervalMDP.Bellman.T_nonexpansive
-- Phase 2a: supporting results referenced by the inventory
#print axioms IntervalMDP.Bellman.T_le_add
#print axioms IntervalMDP.Bellman.T_monotone
#print axioms IntervalMDP.Bellman.T_lipschitz
#print axioms IntervalMDP.Bellman.innerOpt_mem

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
