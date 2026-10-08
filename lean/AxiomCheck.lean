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
