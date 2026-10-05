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
