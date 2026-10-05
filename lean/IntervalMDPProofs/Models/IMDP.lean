import IntervalMDPProofs.Models.IntervalAmbiguity
import IntervalMDPProofs.Models.RMDP

/-!
# Interval MDPs

An IMDP is an RMDP whose ambiguity sets are interval sets `P(l, u)`.

Julia counterpart: `IntervalMarkovDecisionProcess(...)`
(`src/models/IntervalMarkovDecisionProcess.jl`) and `IntervalMarkovChain(...)`
(`src/models/IntervalMarkovChain.jl`). Both are convenience constructors of a
`FactoredRobustMarkovDecisionProcess` with a single state variable, a single action variable and
one `Marginal` over `IntervalAmbiguitySets`; the column of state `s` and action `a` is selected by
`sub2ind` (the index layer of Phase 1). Here the column lookup is abstracted as the function
`ambiguitySets s a`.
-/

namespace IntervalMDP

/-- An interval MDP: available actions plus an interval ambiguity set for every state–action pair.

Julia counterpart: `IntervalMarkovDecisionProcess(ambiguity_sets, num_actions)` /
`IntervalMarkovDecisionProcess(ps::Vector{<:IntervalAmbiguitySets})`
(`src/models/IntervalMarkovDecisionProcess.jl`). -/
structure IMDP (S A : Type*) [Fintype S] extends AvailableActions S A where
  /-- The interval set of state `s` and action `a` (Julia: the column of the `IntervalAmbiguitySets`
  that `sub2ind(marginal, a, s)` selects). -/
  ambiguitySets : S → A → IntervalAmbiguity S

/-- An interval Markov chain: an IMDP with a single action.

Julia counterpart: `IntervalMarkovChain` (`src/models/IntervalMarkovChain.jl`), an fRMDP with
`action_vars = (1,)`. -/
abbrev IMC (S : Type*) [Fintype S] := IMDP S Unit

namespace IMDP

variable {S A : Type*} [Fintype S] (M : IMDP S A)

/-- The ambiguity set `Γ_{s,a} = P(l_{s,a}, u_{s,a})` of state `s` and action `a`.

Julia counterpart: `marginal[a, s]` of the single `Marginal` of an
`IntervalMarkovDecisionProcess` (`src/probabilities/Marginal.jl`,
`src/models/IntervalMarkovDecisionProcess.jl`). -/
def ambiguity (s : S) (a : A) : AmbiguitySet S := (M.ambiguitySets s a).toSet

/-- Every ambiguity set of an IMDP is well formed (nonempty and closed); this is what makes
`IMDP.toRMDP` an `RMDP`. (Interval sets are moreover convex, `IntervalAmbiguity.toSet_convex`.)

Julia counterpart: `checkprobabilities` on each column of the `IntervalAmbiguitySets` of an
`IntervalMarkovDecisionProcess` (`src/probabilities/IntervalAmbiguitySets.jl`,
`src/models/IntervalMarkovDecisionProcess.jl`). -/
theorem toRMDP_wellFormed (s : S) (a : A) : (M.ambiguity s a).WellFormed :=
  (M.ambiguitySets s a).toSet_wellFormed

/-- The IMDP as a robust MDP with ambiguity sets `P(l_{s,a}, u_{s,a})`.

Julia counterpart: the `FactoredRobustMarkovDecisionProcess` returned by
`IntervalMarkovDecisionProcess(...)` (`src/models/IntervalMarkovDecisionProcess.jl`). -/
def toRMDP : RMDP S A where
  toAvailableActions := M.toAvailableActions
  ambiguity := M.ambiguity
  ambiguity_wellFormed := M.toRMDP_wellFormed

/-- The ambiguity sets of `M.toRMDP` are the interval sets of `M`.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem toRMDP_ambiguity (s : S) (a : A) :
    M.toRMDP.ambiguity s a = (M.ambiguitySets s a).toSet := rfl

/-- `M.toRMDP` has the available actions of `M`.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem toRMDP_available (s : S) : M.toRMDP.available s = M.available s := rfl

end IMDP

end IntervalMDP
