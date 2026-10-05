import IntervalMDPProofs.Models.Distribution

/-!
# Ambiguity sets

An ambiguity set `Γ` is a set of distributions over the successor states. The robust Bellman
operator optimises over it. Every algorithm theorem assumes `Γ` is *well formed*: nonempty and
closed (`AmbiguitySet.WellFormed`), so that every linear objective `⟨p, V⟩` attains its optimum
over `Γ` (a closed subset of the simplex is compact).

Well-formedness deliberately does **not** include convexity. The ambiguity set of a factored IMDP
is the set of products of marginal distributions, which is not convex in general
(`IntervalMDP.FactoredIMDP.productSet_not_convex`; arXiv:2411.11803, arXiv:2508.00707).
Convexity is the separate predicate `AmbiguitySet.IsConvex`, used only by results that need it
(for example the interval / O-maximization results).

Interval sets (`IntervalAmbiguity.toSet`), products of marginals (factored IMDPs) and
product-process sets are all instances; L1 balls and mixtures can be added later as further
instances.

Julia counterpart: `AbstractAmbiguitySet` / `PolytopicAmbiguitySet`
(`src/probabilities/probabilities.jl`, `src/probabilities/IntervalAmbiguitySets.jl`).

Closedness and convexity are stated for the image `Γ.vecs ⊆ S → ℝ` (the distributions as plain
vectors), because `ProbVec S` itself is not a vector space.
-/

namespace IntervalMDP

/-- An ambiguity set `Γ ⊆ 𝒟(S)`: a set of probability distributions on the finite state type `S`.

Julia counterpart: `AbstractAmbiguitySet` / `PolytopicAmbiguitySet`
(`src/probabilities/probabilities.jl`). -/
abbrev AmbiguitySet (S : Type*) [Fintype S] := Set (ProbVec S)

namespace AmbiguitySet

variable {S : Type*} [Fintype S]

/-- The distributions of `Γ` viewed as vectors in `S → ℝ` (the image of `Γ` under the coercion
`ProbVec S → (S → ℝ)`). Closedness and convexity are measured here.

Julia counterpart: none (Lean-side proof device). -/
def vecs (Γ : AmbiguitySet S) : Set (S → ℝ) := (fun p : ProbVec S => (p : S → ℝ)) '' Γ

/-- Membership in `Γ.vecs`.

Julia counterpart: none (Lean-side proof device). -/
theorem mem_vecs {Γ : AmbiguitySet S} {x : S → ℝ} :
    x ∈ Γ.vecs ↔ ∃ p ∈ Γ, (p : S → ℝ) = x := Iff.rfl

/-- A distribution is in `Γ` exactly when its vector is in `Γ.vecs`.

Julia counterpart: none (Lean-side proof device). -/
theorem coe_mem_vecs {Γ : AmbiguitySet S} {p : ProbVec S} : (p : S → ℝ) ∈ Γ.vecs ↔ p ∈ Γ :=
  ProbVec.coe_injective.mem_set_image

/-- The vectors of an ambiguity set lie in Mathlib's standard simplex.

Julia counterpart: none (Lean-side proof device). -/
theorem vecs_subset_stdSimplex (Γ : AmbiguitySet S) : Γ.vecs ⊆ stdSimplex ℝ S := by
  rintro _ ⟨p, _, rfl⟩
  exact p.coe_mem_stdSimplex

/-- A well-formed ambiguity set: nonempty and closed. These are the standing assumptions of the
robust Bellman operator: the optimum of a linear objective over `Γ` exists and is attained.
Convexity is **not** required (see `AmbiguitySet.IsConvex`).

Julia counterpart: the validity checks of the concrete ambiguity-set types, e.g.
`checkprobabilities` in `src/probabilities/IntervalAmbiguitySets.jl`. -/
structure WellFormed (Γ : AmbiguitySet S) : Prop where
  /-- `Γ` contains at least one distribution. -/
  nonempty : Γ.Nonempty
  /-- `Γ` is closed (as a subset of `S → ℝ`). -/
  closed : IsClosed Γ.vecs

/-- A well-formed ambiguity set is compact (closed and contained in the compact simplex).

Julia counterpart: none (Lean-side proof device). -/
theorem WellFormed.isCompact {Γ : AmbiguitySet S} (h : Γ.WellFormed) : IsCompact Γ.vecs :=
  (isCompact_stdSimplex ℝ S).of_isClosed_subset h.closed Γ.vecs_subset_stdSimplex

/-- `Γ` is convex (as a subset of `S → ℝ`). This is a separate property, not part of
`WellFormed`: interval sets are convex (`IntervalAmbiguity.toSet_convex`), products of marginal
sets are not in general (`FactoredIMDP.productSet_not_convex`).

Julia counterpart: `PolytopicAmbiguitySet` (`src/probabilities/probabilities.jl`) describes
convex polytopes; the factored product sets of `FactoredRobustMarkovDecisionProcess` are not. -/
def IsConvex (Γ : AmbiguitySet S) : Prop := Convex ℝ Γ.vecs

/-- The ambiguity set of all distributions whose vector lies in `C ⊆ S → ℝ`.

Julia counterpart: none (Lean-side proof device). -/
def ofVecs (C : Set (S → ℝ)) : AmbiguitySet S := {p | (p : S → ℝ) ∈ C}

/-- If `C` lies in the simplex, the vectors of `ofVecs C` are exactly `C`.

Julia counterpart: none (Lean-side proof device). -/
theorem vecs_ofVecs {C : Set (S → ℝ)} (hC : C ⊆ stdSimplex ℝ S) : (ofVecs C).vecs = C := by
  ext x
  constructor
  · rintro ⟨p, hp, rfl⟩; exact hp
  · intro hx; exact ⟨ProbVec.ofStdSimplex x (hC hx), hx, rfl⟩

/-- Rebuilding an ambiguity set from its vectors gives the same set.

Julia counterpart: none (Lean-side proof device). -/
theorem ofVecs_vecs (Γ : AmbiguitySet S) : ofVecs Γ.vecs = Γ := by
  ext p
  exact coe_mem_vecs

/-! ### Images under linear maps -/

/-- The image `f '' Γ` of an ambiguity set under a map of distributions `f`. Used for the product
process, where `f` attaches the DFA transition to each successor.

Julia counterpart: `ProductProcess` (`src/models/ProductProcess.jl`), whose product ambiguity
sets are built with this map (`ProductProcess.ambiguity`). -/
def map {T : Type*} [Fintype T] (f : ProbVec S → ProbVec T) (Γ : AmbiguitySet S) :
    AmbiguitySet T :=
  f '' Γ

/-- If `f` acts on vectors as the linear map `Φ`, then `(map f Γ).vecs = Φ '' Γ.vecs`.

Julia counterpart: none (Lean-side proof device). -/
theorem vecs_map {T : Type*} [Fintype T] {f : ProbVec S → ProbVec T}
    {Φ : (S → ℝ) →ₗ[ℝ] (T → ℝ)} (hf : ∀ p, (f p : T → ℝ) = Φ p) (Γ : AmbiguitySet S) :
    (map f Γ).vecs = Φ '' Γ.vecs := by
  simp only [vecs, map, Set.image_image, hf]

/-- Well-formedness is preserved by maps that act linearly on vectors: nonemptiness by taking
images, and closedness because the (continuous, linear) image of a compact set is compact. No
convexity is needed.

Julia counterpart: `ProductProcess` (`src/models/ProductProcess.jl`), through
`ProductProcess.toRMDP_wellFormed`. -/
theorem WellFormed.map {T : Type*} [Fintype T] {Γ : AmbiguitySet S} (h : Γ.WellFormed)
    {f : ProbVec S → ProbVec T} {Φ : (S → ℝ) →ₗ[ℝ] (T → ℝ)} (hf : ∀ p, (f p : T → ℝ) = Φ p) :
    (AmbiguitySet.map f Γ).WellFormed where
  nonempty := h.nonempty.image f
  closed := by
    rw [vecs_map hf]
    exact (h.isCompact.image Φ.continuous_of_finiteDimensional).isClosed

/-- Convexity is preserved by maps that act linearly on vectors.

Julia counterpart: none (Lean-side proof device). -/
theorem IsConvex.map {T : Type*} [Fintype T] {Γ : AmbiguitySet S} (h : Γ.IsConvex)
    {f : ProbVec S → ProbVec T} {Φ : (S → ℝ) →ₗ[ℝ] (T → ℝ)} (hf : ∀ p, (f p : T → ℝ) = Φ p) :
    (AmbiguitySet.map f Γ).IsConvex := by
  unfold IsConvex
  rw [vecs_map hf]
  exact Convex.linear_image h Φ

/-! ### Products of marginal sets -/

/-- The product `⨂ᵢ Γᵢ` of marginal ambiguity sets: all product distributions `∏ᵢ γᵢ(tᵢ)` with
`γᵢ ∈ Γᵢ` for every `i`. This is the literal set of products; it is **not** convexified.

Julia counterpart: `Γ_{s,a} = ⨂ᵢ Γⁱ_{Pa(S'ᵢ) ∩ (s,a)}` in the
`FactoredRobustMarkovDecisionProcess` docstring (`src/models/FactoredRobustMarkovDecisionProcess.jl`).
This set is in general **not convex**; see `FactoredIMDP.productSet_not_convex`. -/
def pi {ι : Type*} [Fintype ι] [DecidableEq ι] {X : ι → Type*} [∀ i, Fintype (X i)]
    (Γ : ∀ i, AmbiguitySet (X i)) : AmbiguitySet (∀ i, X i) :=
  ProbVec.pi '' Set.univ.pi Γ

/-- The multilinear product map on vectors, `x ↦ (t ↦ ∏ᵢ xᵢ(tᵢ))`; on distributions it is
`ProbVec.pi`.

Julia counterpart: none (Lean-side proof device). -/
def piVec {ι : Type*} [Fintype ι] {X : ι → Type*} (x : ∀ i, X i → ℝ) : (∀ i, X i) → ℝ :=
  fun t => ∏ i, x i (t i)

/-- `piVec` is continuous (each coordinate is a finite product of coordinate projections).

Julia counterpart: none (Lean-side proof device). -/
theorem continuous_piVec {ι : Type*} [Fintype ι] {X : ι → Type*} :
    Continuous (piVec : (∀ i, X i → ℝ) → (∀ i, X i) → ℝ) :=
  continuous_pi fun t =>
    continuous_finsetProd _ fun i _ => (continuous_apply (t i)).comp (continuous_apply i)

/-- The vectors of a product set are the image of the product of the marginal vector sets under
`piVec`.

Julia counterpart: none (Lean-side proof device). -/
theorem vecs_pi {ι : Type*} [Fintype ι] [DecidableEq ι] {X : ι → Type*} [∀ i, Fintype (X i)]
    (Γ : ∀ i, AmbiguitySet (X i)) :
    (pi Γ).vecs = piVec '' Set.univ.pi fun i => (Γ i).vecs := by
  ext x
  constructor
  · rintro ⟨_, ⟨γ, hγ, rfl⟩, rfl⟩
    exact ⟨fun i => (γ i : X i → ℝ), fun i _ => coe_mem_vecs.mpr (hγ i (Set.mem_univ i)), rfl⟩
  · rintro ⟨y, hy, rfl⟩
    choose γ hγ hγy using fun i => hy i (Set.mem_univ i)
    refine ⟨ProbVec.pi γ, ⟨γ, fun i _ => hγ i, rfl⟩, ?_⟩
    funext t
    simp only [ProbVec.pi_apply, piVec, hγy]

/-- A product of nonempty marginal sets is nonempty.

Julia counterpart: none (Lean-side proof device). -/
theorem pi_nonempty {ι : Type*} [Fintype ι] [DecidableEq ι] {X : ι → Type*}
    [∀ i, Fintype (X i)] {Γ : ∀ i, AmbiguitySet (X i)} (h : ∀ i, (Γ i).Nonempty) :
    (pi Γ).Nonempty :=
  (Set.univ_pi_nonempty_iff.mpr h).image _

/-- A product of well-formed marginal sets is well formed: nonempty, and closed because it is the
continuous image (`piVec`) of the compact product of the compact marginal sets. Convexity is
neither assumed nor concluded.

Julia counterpart: the factored set `Γ_{s,a} = ⨂ᵢ Γⁱ` of `FactoredRobustMarkovDecisionProcess`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`), through `FactoredIMDP.toRMDP_wellFormed`. -/
theorem WellFormed.pi {ι : Type*} [Fintype ι] [DecidableEq ι] {X : ι → Type*}
    [∀ i, Fintype (X i)] {Γ : ∀ i, AmbiguitySet (X i)} (h : ∀ i, (Γ i).WellFormed) :
    (AmbiguitySet.pi Γ).WellFormed where
  nonempty := pi_nonempty fun i => (h i).nonempty
  closed := by
    rw [vecs_pi]
    exact ((isCompact_univ_pi fun i => (h i).isCompact).image continuous_piVec).isClosed

end AmbiguitySet

end IntervalMDP
