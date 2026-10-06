import IntervalMDPProofs.Models.AmbiguitySet

/-!
# Interval ambiguity sets

An interval ambiguity set is given by elementwise bounds `lower ≤ p ≤ upper` on the successor
distribution `p`:

  `P(l, u) = {p ∈ 𝒟(S) : l(s) ≤ p(s) ≤ u(s) for all s}`.

Julia counterpart: `IntervalAmbiguitySet` (one column) and `IntervalAmbiguitySets` (a matrix of
columns) in `src/probabilities/IntervalAmbiguitySets.jl`. Julia stores `lower` and
`gap = upper - lower`; `upper(p)` is computed as `lower + gap`. The O-maximization workspaces
(`src/workspace.jl`) precompute `budget = 1 .- vec(sum(lower; dims = 1))`.

The structure invariants are exactly what `checkprobabilities` checks:
`0 ≤ lower ≤ upper ≤ 1` elementwise, `∑ lower ≤ 1` and `∑ upper ≥ 1`.
-/

namespace IntervalMDP

open Finset

/-- An interval ambiguity set over the finite state type `S`, given by its bounds.

Julia counterpart: one column of `IntervalAmbiguitySets` (`IntervalAmbiguitySet`,
`src/probabilities/IntervalAmbiguitySets.jl`). The invariants are those of `checkprobabilities`
in the same file. -/
structure IntervalAmbiguity (S : Type*) [Fintype S] where
  /-- Lower bounds `l(s)` (Julia `lower`). -/
  lower : S → ℝ
  /-- Upper bounds `u(s)` (Julia `upper(p) = lower + gap`). -/
  upper : S → ℝ
  /-- `0 ≤ l(s)` ("lower bound transition probabilities must be non-negative"). -/
  lower_nonneg : ∀ s, 0 ≤ lower s
  /-- `l(s) ≤ u(s)` (Julia: the gap is non-negative). -/
  lower_le_upper : ∀ s, lower s ≤ upper s
  /-- `u(s) ≤ 1` (Julia: "the sum of lower and gap … must be less than or equal to 1"). -/
  upper_le_one : ∀ s, upper s ≤ 1
  /-- `∑ₛ l(s) ≤ 1` (Julia: "joint lower bound transition probability per column … ≤ 1"). -/
  sum_lower_le_one : ∑ s, lower s ≤ 1
  /-- `∑ₛ u(s) ≥ 1` (Julia: "joint upper bound transition probability per column … ≥ 1"). -/
  one_le_sum_upper : 1 ≤ ∑ s, upper s

namespace IntervalAmbiguity

variable {S : Type*} [Fintype S] (I : IntervalAmbiguity S)

/-- The gap `g(s) = u(s) - l(s)` between the bounds.

Julia counterpart: `gap(p::IntervalAmbiguitySet)` / the stored field `gap`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
def gap (s : S) : ℝ := I.upper s - I.lower s

/-- The budget `1 - ∑ₛ l(s)`: the probability mass left to distribute on top of the lower bounds.

Julia counterpart: `budget = 1 .- vec(sum(ambiguity_set.lower; dims = 1))` in the O-maximization
workspaces (`src/workspace.jl`), consumed by `gap_value` in `src/bellman.jl`. -/
def budget : ℝ := 1 - ∑ s, I.lower s

/-- The interval ambiguity set `P(l, u) = {p ∈ 𝒟(S) : l ≤ p ≤ u}`.

Julia counterpart: the set of feasible distributions of an `IntervalAmbiguitySet`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
def toSet : AmbiguitySet S := {p | ∀ s, I.lower s ≤ p s ∧ p s ≤ I.upper s}

/-- Julia's representation is consistent: `upper = lower + gap`.

Julia counterpart: `upper(p::IntervalAmbiguitySet) = p.lower + p.gap`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem lower_add_gap (s : S) : I.lower s + I.gap s = I.upper s := by
  simp [gap]

/-- The gap is nonnegative.

Julia counterpart: `checkprobabilities` (`src/probabilities/IntervalAmbiguitySets.jl`), which
rejects negative gaps. -/
theorem gap_nonneg (s : S) : 0 ≤ I.gap s := sub_nonneg.mpr (I.lower_le_upper s)

/-- The budget never exceeds the total gap: `1 - ∑ l ≤ ∑ (u - l)`, because `∑ u ≥ 1`.

Julia counterpart: none (Lean-side proof device). -/
theorem budget_le_sum_gap : I.budget ≤ ∑ s, I.gap s := by
  simp only [budget, gap, Finset.sum_sub_distrib]
  linarith [I.one_le_sum_upper]

/-- The budget is `1 - ∑ₛ l(s)` and it is nonnegative (because `∑ l ≤ 1`).

Julia counterpart: `budget = 1 .- vec(sum(ambiguity_set.lower; dims = 1))` in the O-maximization
workspaces (`src/workspace.jl`), consumed by `gap_value` (`src/bellman.jl`). -/
theorem budget_eq : I.budget = 1 - ∑ s, I.lower s ∧ 0 ≤ I.budget :=
  ⟨rfl, sub_nonneg.mpr I.sum_lower_le_one⟩

/-- A distribution in `P(l, u)`: `p(s) = l(s) + (budget / ∑ gap) · g(s)`, i.e. the budget is spread
over the gaps proportionally. This is a proof device for nonemptiness (it is not the O-max
distribution).

Julia counterpart: none (Lean-side proof device). -/
noncomputable def feasiblePoint : ProbVec S where
  toFun s := I.lower s + I.budget / (∑ t, I.gap t) * I.gap s
  nonneg s := add_nonneg (I.lower_nonneg s)
    (mul_nonneg (div_nonneg I.budget_eq.2 (Finset.sum_nonneg fun t _ => I.gap_nonneg t))
      (I.gap_nonneg s))
  sum_eq_one := by
    rw [Finset.sum_add_distrib, ← Finset.mul_sum]
    have hG : 0 ≤ ∑ t, I.gap t := Finset.sum_nonneg fun t _ => I.gap_nonneg t
    have hb : I.budget / (∑ t, I.gap t) * ∑ t, I.gap t = I.budget := by
      rcases hG.eq_or_lt with h | h
      · rw [← h, mul_zero]
        exact le_antisymm (h ▸ I.budget_le_sum_gap) I.budget_eq.2 |>.symm
      · exact div_mul_cancel₀ _ h.ne'
    rw [hb, budget]
    ring

/-- The proportional point lies in `P(l, u)`.

Julia counterpart: none (Lean-side proof device). -/
theorem feasiblePoint_mem : I.feasiblePoint ∈ I.toSet := by
  intro s
  have hG : 0 ≤ ∑ t, I.gap t := Finset.sum_nonneg fun t _ => I.gap_nonneg t
  have ht0 : 0 ≤ I.budget / ∑ t, I.gap t := div_nonneg I.budget_eq.2 hG
  have ht1 : I.budget / ∑ t, I.gap t ≤ 1 := div_le_one_of_le₀ I.budget_le_sum_gap hG
  have hg := I.gap_nonneg s
  change I.lower s ≤ I.lower s + _ * I.gap s ∧ I.lower s + _ * I.gap s ≤ I.upper s
  constructor
  · nlinarith
  · have : I.budget / (∑ t, I.gap t) * I.gap s ≤ I.gap s := by nlinarith
    linarith [I.lower_add_gap s]

/-- `P(l, u)` written through vectors: the simplex intersected with the box `[l, u]`.

Julia counterpart: none (Lean-side proof device). -/
theorem toSet_eq_ofVecs : I.toSet = AmbiguitySet.ofVecs (stdSimplex ℝ S ∩ Set.Icc I.lower I.upper) := by
  ext p
  simp only [toSet, AmbiguitySet.ofVecs, Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_Icc,
    Pi.le_def, p.coe_mem_stdSimplex, true_and, forall_and]

/-- The vectors of `P(l, u)` are `stdSimplex ℝ S ∩ Set.Icc l u`.

Julia counterpart: none (Lean-side proof device). -/
theorem vecs_toSet : I.toSet.vecs = stdSimplex ℝ S ∩ Set.Icc I.lower I.upper := by
  rw [toSet_eq_ofVecs]
  exact AmbiguitySet.vecs_ofVecs Set.inter_subset_left

/-- `P(l, u)` is well formed: nonempty (because `∑ l ≤ 1 ≤ ∑ u`, witnessed by `feasiblePoint`)
and closed (an intersection of the simplex with a box).

Julia counterpart: the guarantees of `checkprobabilities` for an `IntervalAmbiguitySet`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem toSet_wellFormed : I.toSet.WellFormed where
  nonempty := ⟨I.feasiblePoint, I.feasiblePoint_mem⟩
  closed := by rw [vecs_toSet]; exact (isClosed_stdSimplex ℝ S).inter isClosed_Icc

/-- `P(l, u)` is convex: it is the intersection of the (convex) simplex with the (convex) box
`[l, u]`. Interval sets are convex polytopes, which the O-maximization results rely on; this is
not part of `WellFormed` because factored product sets are not convex.

Julia counterpart: `IntervalAmbiguitySet <: PolytopicAmbiguitySet`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem toSet_convex : I.toSet.IsConvex := by
  unfold AmbiguitySet.IsConvex
  rw [vecs_toSet]
  exact (convex_stdSimplex ℝ S).inter (convex_Icc _ _)

end IntervalAmbiguity

end IntervalMDP
