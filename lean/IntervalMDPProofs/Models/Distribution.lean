import Mathlib

/-!
# Probability vectors

`ProbVec S` is a probability distribution on a finite type `S`: a function `S → ℝ` that is
nonnegative and sums to one. In IntervalMDP.jl these are the probability columns that the
interval ambiguity sets of `src/probabilities/IntervalAmbiguitySets.jl` range over (each column
of `lower`/`upper` bounds one `ProbVec`).

This file also provides the four ways the models combine distributions:

* `ProbVec.dirac` — a point mass (deterministic labellings, `DeterministicLabelling`);
* `ProbVec.map` — push-forward along a function (DFA transitions in the product, `ProductProcess`);
* `ProbVec.pi` — the product of independent marginals (factored IMDPs, `Marginal`);
* `ProbVec.prodKernel` — a distribution on `S` followed by a kernel `S → ProbVec Q`
  (the product process `S × Q`).
-/

namespace IntervalMDP

open Finset

/-- A probability distribution `p` on a finite type `S`: `p(s) ≥ 0` for every `s` and
`∑ₛ p(s) = 1`.

Julia counterpart: a probability column of `IntervalAmbiguitySets`
(`src/probabilities/IntervalAmbiguitySets.jl`), i.e. a feasible distribution of one ambiguity set. -/
structure ProbVec (S : Type*) [Fintype S] where
  /-- The probability `p(s)` of each outcome `s`. -/
  toFun : S → ℝ
  /-- Every probability is nonnegative: `0 ≤ p(s)`. -/
  nonneg : ∀ s, 0 ≤ toFun s
  /-- The probabilities sum to one: `∑ₛ p(s) = 1`. -/
  sum_eq_one : ∑ s, toFun s = 1

namespace ProbVec

variable {S : Type*} [Fintype S]

/-- A `ProbVec` can be applied like the function `S → ℝ` it wraps (Julia: indexing `p[s]`).

Julia counterpart: indexing a probability column, e.g. `lower(p)[s]` of an
`IntervalAmbiguitySet` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
instance instCoeFun : CoeFun (ProbVec S) (fun _ => S → ℝ) := ⟨ProbVec.toFun⟩

/-- Unfolding lemma for the coercion to functions.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem toFun_eq_coe (p : ProbVec S) : p.toFun = (p : S → ℝ) := rfl

/-- Two probability vectors are equal when they agree on every outcome.

Julia counterpart: none (Lean-side proof device). -/
theorem ext {p q : ProbVec S} (h : ∀ s, p s = q s) : p = q := by
  cases p; cases q; congr; funext s; exact h s

/-- Two probability vectors are equal exactly when they agree on every outcome.

Julia counterpart: none (Lean-side proof device). -/
theorem ext_iff {p q : ProbVec S} : p = q ↔ ∀ s, p s = q s :=
  ⟨fun h _ => h ▸ rfl, ext⟩

/-- A probability vector is determined by its underlying function.

Julia counterpart: none (Lean-side proof device). -/
theorem coe_injective : Function.Injective (fun p : ProbVec S => (p : S → ℝ)) :=
  fun _ _ h => ext (fun s => congrFun h s)

/-- The underlying function of a `ProbVec` lies in Mathlib's standard simplex `stdSimplex ℝ S`.

Julia counterpart: none (Lean-side proof device). -/
theorem coe_mem_stdSimplex (p : ProbVec S) : (p : S → ℝ) ∈ stdSimplex ℝ S :=
  ⟨p.nonneg, p.sum_eq_one⟩

/-- Build a `ProbVec` from a point of Mathlib's standard simplex.

Julia counterpart: none (Lean-side proof device). -/
def ofStdSimplex (x : S → ℝ) (hx : x ∈ stdSimplex ℝ S) : ProbVec S :=
  ⟨x, hx.1, hx.2⟩

/-- `ofStdSimplex` keeps the underlying function.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem coe_ofStdSimplex (x : S → ℝ) (hx : x ∈ stdSimplex ℝ S) :
    (ofStdSimplex x hx : S → ℝ) = x := rfl

/-- The functions underlying probability vectors are exactly the points of `stdSimplex ℝ S`.

Julia counterpart: none (Lean-side proof device). -/
theorem range_coe : Set.range (fun p : ProbVec S => (p : S → ℝ)) = stdSimplex ℝ S := by
  ext x
  constructor
  · rintro ⟨p, rfl⟩; exact p.coe_mem_stdSimplex
  · intro hx; exact ⟨ofStdSimplex x hx, rfl⟩

/-- Every probability is at most one: `p(s) ≤ 1`.

Julia counterpart: none (Lean-side proof device). -/
theorem le_one (p : ProbVec S) (s : S) : p s ≤ 1 := by
  rw [← p.sum_eq_one]
  exact Finset.single_le_sum (fun t _ => p.nonneg t) (Finset.mem_univ s)

/-- The point mass `δₛ` at `s`: `δₛ(t) = 1` if `t = s` and `0` otherwise.

Julia counterpart: the distribution over labels that a `DeterministicLabelling`
(`src/probabilities/DeterministicLabelling.jl`) assigns to a state. -/
def dirac [DecidableEq S] (s : S) : ProbVec S where
  toFun t := if t = s then 1 else 0
  nonneg t := by split_ifs <;> norm_num
  sum_eq_one := by simp

/-- Evaluation of the point mass.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem dirac_apply [DecidableEq S] (s t : S) : dirac s t = if t = s then 1 else 0 := rfl

/-- The push-forward of `p` along `f : S → T`: `(map f p)(t) = ∑_{s : f s = t} p(s)`.

Julia counterpart: in the product Bellman operator (`src/bellman.jl`, `_bellman_helper!` for
`ProbabilisticLabelling`), the probability of moving to DFA state `q'` is the total probability of
the labels `l` with `dfa[q, l] = q'`. -/
def map {T : Type*} [Fintype T] [DecidableEq T] (f : S → T) (p : ProbVec S) : ProbVec T where
  toFun t := ∑ s ∈ univ.filter (fun s => f s = t), p s
  nonneg _ := Finset.sum_nonneg (fun s _ => p.nonneg s)
  sum_eq_one := by rw [Finset.sum_fiberwise]; exact p.sum_eq_one

/-- Evaluation of the push-forward.

Julia counterpart: none (Lean-side proof device). -/
theorem map_apply {T : Type*} [Fintype T] [DecidableEq T] (f : S → T) (p : ProbVec S) (t : T) :
    map f p t = ∑ s ∈ univ.filter (fun s => f s = t), p s := rfl

/-- The push-forward of a point mass is the point mass at the image: `map f δₛ = δ_{f s}`.

Julia counterpart: none (Lean-side proof device). -/
theorem map_dirac {T : Type*} [Fintype T] [DecidableEq T] [DecidableEq S] (f : S → T) (s : S) :
    map f (dirac s) = dirac (f s) := by
  refine ext fun t => ?_
  simp only [map_apply, dirac_apply, Finset.sum_ite_eq', Finset.mem_filter, Finset.mem_univ,
    true_and]
  by_cases h : f s = t
  · simp [h]
  · simp [h, Ne.symm h]

/-- The product of independent marginals `γᵢ`: `(pi γ)(t) = ∏ᵢ γᵢ(tᵢ)` on the joint type
`∀ i, X i`.

Julia counterpart: a feasible distribution of a factored ambiguity set
`Γ_{s,a} = ⨂ᵢ Γⁱ` (`FactoredRobustMarkovDecisionProcess` docstring,
`src/models/FactoredRobustMarkovDecisionProcess.jl`): `γ(t) = ∏ᵢ γⁱ(tᵢ | …)`. -/
def pi {ι : Type*} [Fintype ι] [DecidableEq ι] {X : ι → Type*} [∀ i, Fintype (X i)]
    (γ : ∀ i, ProbVec (X i)) : ProbVec (∀ i, X i) where
  toFun t := ∏ i, γ i (t i)
  nonneg t := Finset.prod_nonneg (fun i _ => (γ i).nonneg (t i))
  sum_eq_one := by
    rw [← Fintype.prod_sum (fun i x => γ i x)]
    simp [(γ _).sum_eq_one]

/-- Evaluation of the product distribution.

Julia counterpart: none (Lean-side proof device). -/
theorem pi_apply {ι : Type*} [Fintype ι] [DecidableEq ι] {X : ι → Type*} [∀ i, Fintype (X i)]
    (γ : ∀ i, ProbVec (X i)) (t : ∀ i, X i) : pi γ t = ∏ i, γ i (t i) := rfl

/-- A distribution `p` on `S` followed by a kernel `k : S → ProbVec Q`, as a distribution on
`S × Q`: `(prodKernel p k)(t, q) = p(t) · k(t)(q)`.

Julia counterpart: the product-process transition (`ProductProcess` docstring,
`src/models/ProductProcess.jl`), where `k(t)` is the distribution of the next DFA state given the
successor `t`. -/
def prodKernel {Q : Type*} [Fintype Q] (p : ProbVec S) (k : S → ProbVec Q) : ProbVec (S × Q) where
  toFun x := p x.1 * k x.1 x.2
  nonneg x := mul_nonneg (p.nonneg x.1) ((k x.1).nonneg x.2)
  sum_eq_one := by
    rw [Fintype.sum_prod_type]
    simp [← Finset.mul_sum, (k _).sum_eq_one, p.sum_eq_one]

/-- Evaluation of `prodKernel`.

Julia counterpart: none (Lean-side proof device). -/
theorem prodKernel_apply {Q : Type*} [Fintype Q] (p : ProbVec S) (k : S → ProbVec Q)
    (x : S × Q) : prodKernel p k x = p x.1 * k x.1 x.2 := rfl

end ProbVec

end IntervalMDP
