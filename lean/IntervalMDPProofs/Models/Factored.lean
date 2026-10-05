import IntervalMDPProofs.Models.IMDP

/-!
# Factored interval MDPs

A factored IMDP has state variables `S = S₁ × ⋯ × Sₙ` and action variables `A = A₁ × ⋯ × Aₘ`.
Marginal `i` describes the next value of state variable `i`; it conditions on a subset of the
state variables (`stateIndices`) and of the action variables (`actionIndices`), and stores one
interval set per value of those variables. The joint ambiguity set of `(s, a)` is the product of
the selected marginal sets:

  `Γ_{s,a} = ⨂ᵢ Γⁱ_{Pa(S'ᵢ) ∩ (s,a)} = {γ : γ(t) = ∏ᵢ γⁱ(tᵢ), γⁱ ∈ Γⁱ}`.

Julia counterpart: `FactoredRobustMarkovDecisionProcess` (`IsFIMDP`) in
`src/models/FactoredRobustMarkovDecisionProcess.jl` and `Marginal` in
`src/probabilities/Marginal.jl`.

Variable values are `Fin (dims i)` (Julia's `1:state_vars[i]`, shifted to 0-based; the 1-based
conversion is `Index/Julia.lean`, Phase 1). The conditioning tuple is passed to `sets` as a
dependent function; the linear column index that Julia computes in `sub2ind` is Phase 1
(`Index/Marginal.lean`).

**Convexity.** The product set `⨂ᵢ Γⁱ` is in general **not convex**
(`IntervalMDP.FactoredIMDP.productSet_not_convex`, in `Models/Examples.lean`; known from
arXiv:2411.11803 and arXiv:2508.00707). `toRMDP` uses the literal `productSet`, not a convex
hull; it is well formed in the sense of `AmbiguitySet.WellFormed` (nonempty and closed), which is
all the general `RMDP` results require.

Scope: the model assumes `source_dims = state_vars` (no terminal slices).
-/

namespace IntervalMDP

/-- The number of values of each state variable, `|Sᵢ|`.

Julia counterpart: `state_vars::NTuple{N, Int32}` of `FactoredRobustMarkovDecisionProcess`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`); positivity is `check_state_values`. -/
structure StateVars (n : ℕ) where
  /-- `dims i = |Sᵢ|` (Julia `state_vars[i]`). -/
  dims : Fin n → ℕ
  /-- Every state variable has at least one value. -/
  dims_pos : ∀ i, 0 < dims i

/-- The number of values of each action variable, `|Aₖ|`.

Julia counterpart: `action_vars::NTuple{M, Int32}`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`); positivity is `check_action_values`. -/
structure ActionVars (m : ℕ) where
  /-- `dims k = |Aₖ|` (Julia `action_vars[k]`). -/
  dims : Fin m → ℕ
  /-- Every action variable has at least one value. -/
  dims_pos : ∀ k, 0 < dims k

/-- A joint state `s = (s₁, …, sₙ)` with `sᵢ ∈ Fin |Sᵢ|` (Julia: a `CartesianIndex` over
`state_vars`).

Julia counterpart: a `CartesianIndex` over `state_vars` of `FactoredRobustMarkovDecisionProcess`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
abbrev StateVars.State {n : ℕ} (sv : StateVars n) := (i : Fin n) → Fin (sv.dims i)

/-- A joint action `a = (a₁, …, aₘ)` with `aₖ ∈ Fin |Aₖ|` (Julia: a `CartesianIndex` over
`action_vars`).

Julia counterpart: a `CartesianIndex` over `action_vars` of
`FactoredRobustMarkovDecisionProcess` (`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
abbrev ActionVars.Action {m : ℕ} (av : ActionVars m) := (k : Fin m) → Fin (av.dims k)

/-- Every state variable has a value, so joint states exist (the all-first-values state).

Julia counterpart: `check_state_values` (`src/models/FactoredRobustMarkovDecisionProcess.jl`),
which requires every state variable to be positive. -/
instance StateVars.instNonemptyState {n : ℕ} (sv : StateVars n) : Nonempty sv.State :=
  ⟨fun i => ⟨0, sv.dims_pos i⟩⟩

/-- Every action variable has a value, so joint actions exist.

Julia counterpart: `check_action_values` (`src/models/FactoredRobustMarkovDecisionProcess.jl`),
which requires every action variable to be positive. -/
instance ActionVars.instNonemptyAction {m : ℕ} (av : ActionVars m) : Nonempty av.Action :=
  ⟨fun k => ⟨0, av.dims_pos k⟩⟩

/-- Marginal `i` of a factored IMDP: the interval sets for the next value of state variable `i`,
conditioned on the state variables `stateIndices` and the action variables `actionIndices`.

Julia counterpart: `Marginal` (`src/probabilities/Marginal.jl`) with fields `state_indices`,
`action_indices` and `ambiguity_sets::IntervalAmbiguitySets`. "In range" is enforced by the types
(`Fin n`, `Fin m`); "one interval set per conditioning tuple" by the type of `sets`. -/
structure Marginal {n m : ℕ} (sv : StateVars n) (av : ActionVars m) (i : Fin n) where
  /-- The number of conditioning state variables (Julia: `N` in `NTuple{N, Int32}`). -/
  numStateIndices : ℕ
  /-- The number of conditioning action variables (Julia: `M` in `NTuple{M, Int32}`). -/
  numActionIndices : ℕ
  /-- The conditioning state variables (Julia `state_indices`). -/
  stateIndices : Fin numStateIndices → Fin n
  /-- The conditioning action variables (Julia `action_indices`). -/
  actionIndices : Fin numActionIndices → Fin m
  /-- `stateIndices` is strictly increasing. -/
  stateIndices_strictMono : StrictMono stateIndices
  /-- `actionIndices` is strictly increasing. -/
  actionIndices_strictMono : StrictMono actionIndices
  /-- The interval set for each value of the conditioning variables (Julia: the columns of
  `ambiguity_sets`, laid out by `sub2ind`). -/
  sets : ((j : Fin numStateIndices) → Fin (sv.dims (stateIndices j))) →
    ((k : Fin numActionIndices) → Fin (av.dims (actionIndices k))) →
    IntervalAmbiguity (Fin (sv.dims i))

namespace Marginal

variable {n m : ℕ} {sv : StateVars n} {av : ActionVars m} {i : Fin n} (Mg : Marginal sv av i)

/-- The values of the conditioning state variables in the joint state `s`, `s[state_indices]`.

Julia counterpart: the selection of `source` by `p.state_indices` in
`sub2ind(p::Marginal, action, source)` (`src/probabilities/Marginal.jl`). -/
def sourceOf (s : sv.State) : (j : Fin Mg.numStateIndices) → Fin (sv.dims (Mg.stateIndices j)) :=
  fun j => s (Mg.stateIndices j)

/-- The values of the conditioning action variables in the joint action `a`,
`a[action_indices]`.

Julia counterpart: the selection of `action` by `p.action_indices` in
`sub2ind(p::Marginal, action, source)` (`src/probabilities/Marginal.jl`). -/
def actionOf (a : av.Action) : (k : Fin Mg.numActionIndices) → Fin (av.dims (Mg.actionIndices k)) :=
  fun k => a (Mg.actionIndices k)

/-- The interval set of marginal `i` for joint state `s` and joint action `a`.

Julia counterpart: `getindex(p::Marginal, action, source)`, i.e. `marginal[a, s]`
(`src/probabilities/Marginal.jl`), which reads `ambiguity_sets[sub2ind(p, action, source)]`. -/
def get (s : sv.State) (a : av.Action) : IntervalAmbiguity (Fin (sv.dims i)) :=
  Mg.sets (Mg.sourceOf s) (Mg.actionOf a)

/-- The marginal set depends only on the conditioning variables: states and actions that agree on
`stateIndices` and `actionIndices` select the same interval set.

Julia counterpart: `getindex(p::Marginal, action, source)` (`src/probabilities/Marginal.jl`),
which reads only the `state_indices` and `action_indices` components. -/
theorem get_congr {s s' : sv.State} {a a' : av.Action}
    (hs : ∀ j, s (Mg.stateIndices j) = s' (Mg.stateIndices j))
    (ha : ∀ k, a (Mg.actionIndices k) = a' (Mg.actionIndices k)) : Mg.get s a = Mg.get s' a' := by
  have h1 : Mg.sourceOf s = Mg.sourceOf s' := funext hs
  have h2 : Mg.actionOf a = Mg.actionOf a' := funext ha
  simp only [get, h1, h2]

end Marginal

/-- A factored interval MDP: state and action variables, available joint actions, and one marginal
per state variable.

Julia counterpart: `FactoredRobustMarkovDecisionProcess` with `IntervalAmbiguitySets` marginals
(`IsFIMDP`, `src/models/FactoredRobustMarkovDecisionProcess.jl`); fields `state_vars`,
`action_vars`, `available_actions` and `transition` (accessor `marginals`). -/
structure FactoredIMDP (n m : ℕ) where
  /-- `|Sᵢ|` for each state variable (Julia `state_vars`). -/
  stateVars : StateVars n
  /-- `|Aₖ|` for each action variable (Julia `action_vars`). -/
  actionVars : ActionVars m
  /-- The available joint actions (Julia `available_actions`). -/
  availableActions : AvailableActions stateVars.State actionVars.Action
  /-- One marginal per state variable (Julia `transition`, accessor `marginals(mdp)`). -/
  marginals : (i : Fin n) → Marginal stateVars actionVars i

namespace FactoredIMDP

variable {n m : ℕ} (F : FactoredIMDP n m)

/-- The marginal interval set `Γⁱ = P(lⁱ, uⁱ)` of state variable `i` for `(s, a)`, with
`(lⁱ, uⁱ) = marginals[i][a, s]`.

Julia counterpart: `marginals(mdp)[i][a, s]` (`getindex(::Marginal, action, source)`,
`src/probabilities/Marginal.jl`) seen as a set of distributions on `1:state_vars[i]`. -/
def marginalSet (s : F.stateVars.State) (a : F.actionVars.Action) (i : Fin n) :
    AmbiguitySet (Fin (F.stateVars.dims i)) :=
  ((F.marginals i).get s a).toSet

/-- The product of the marginal interval sets for `(s, a)`, taken **literally**:
`Γ_{s,a} = ⨂ᵢ Γⁱ = {γ : γ(t) = ∏ᵢ γⁱ(tᵢ), γⁱ ∈ Γⁱ}`. It is not convexified; it is not convex in
general (`productSet_not_convex`).

Julia counterpart: `Γ_{s,a}` in the `FactoredRobustMarkovDecisionProcess` docstring
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def productSet (s : F.stateVars.State) (a : F.actionVars.Action) :
    AmbiguitySet F.stateVars.State :=
  AmbiguitySet.pi (F.marginalSet s a)

/-- Well-formedness of the factored model at `(s, a)`:

1. every choice of marginal distributions `γⁱ ∈ Γⁱ` gives a distribution on the joint state in
   `productSet s a`, namely `t ↦ ∏ᵢ γⁱ(tᵢ)`;
2. `productSet s a` is nonempty and closed (`AmbiguitySet.WellFormed`; convexity is not claimed).

Julia counterpart: the joint transition of `FactoredRobustMarkovDecisionProcess` (`IsFIMDP`,
`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
theorem toRMDP_wellFormed (s : F.stateVars.State) (a : F.actionVars.Action) :
    (∀ γ : (i : Fin n) → ProbVec (Fin (F.stateVars.dims i)), (∀ i, γ i ∈ F.marginalSet s a i) →
      ∃ p ∈ F.productSet s a, ∀ t, p t = ∏ i, γ i (t i)) ∧
    (F.productSet s a).WellFormed :=
  ⟨fun γ hγ => ⟨ProbVec.pi γ, ⟨γ, fun i _ => hγ i, rfl⟩, fun _ => rfl⟩,
    AmbiguitySet.WellFormed.pi fun i => ((F.marginals i).get s a).toSet_wellFormed⟩

/-- The factored IMDP as a robust MDP on joint states and joint actions, whose ambiguity sets are
the literal product sets `productSet s a = ⨂ᵢ P(lⁱ, uⁱ)` (no convex hull).

Julia counterpart: `FactoredRobustMarkovDecisionProcess` (`IsFIMDP`) as consumed by `bellman!`
(`src/bellman.jl`). -/
def toRMDP : RMDP F.stateVars.State F.actionVars.Action where
  toAvailableActions := F.availableActions
  ambiguity := F.productSet
  ambiguity_wellFormed s a := (F.toRMDP_wellFormed s a).2

/-- The ambiguity sets of `F.toRMDP` are exactly the product sets.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem toRMDP_ambiguity (s : F.stateVars.State) (a : F.actionVars.Action) :
    F.toRMDP.ambiguity s a = F.productSet s a := rfl

end FactoredIMDP

end IntervalMDP
