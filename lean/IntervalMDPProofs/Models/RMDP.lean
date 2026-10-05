import IntervalMDPProofs.Models.AmbiguitySet

/-!
# Robust MDPs

`RMDP S A` is the common target of every model in this project: interval MDPs
(`IMDP.toRMDP`), factored interval MDPs (`FactoredIMDP.toRMDP`) and products with a DFA
(`ProductProcess.toRMDP`) all convert to it, and the Bellman/VI theorems of later phases are
proved once, for `RMDP`.

Julia counterpart: `FactoredRobustMarkovDecisionProcess`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`) seen as a flat robust MDP (`IsRMDP`,
`IsIMDP` in `src/models/models.jl`), together with the available actions of
`src/available_actions.jl`.
-/

namespace IntervalMDP

/-- The actions available in each state: a nonempty finite set `available s ⊆ A`.

Julia counterpart: `AbstractAvailableActions` (`src/available_actions.jl`):
`AllAvailableActions` (every action, see `AvailableActions.all`) and `ListAvailableActions`
(an explicit list per state). Julia does not check that the lists are nonempty; the Lean model
requires it, because the optimum over an empty action set is undefined. -/
structure AvailableActions (S A : Type*) where
  /-- The available actions of state `s` (Julia `available(model, jₛ)`). -/
  available : S → Finset A
  /-- Every state has at least one available action. -/
  available_nonempty : ∀ s, (available s).Nonempty

namespace AvailableActions

variable {S A : Type*}

/-- Every action is available in every state.

Julia counterpart: `AllAvailableActions` (`src/available_actions.jl`), the default. -/
def all [Fintype A] [Nonempty A] : AvailableActions S A where
  available _ := Finset.univ
  available_nonempty _ := Finset.univ_nonempty

/-- With `AvailableActions.all`, every action is available.

Julia counterpart: `isavailable(::AllAvailableActions, jₛ, jₐ) = true`
(`src/available_actions.jl`). -/
theorem mem_all [Fintype A] [Nonempty A] (s : S) (a : A) :
    a ∈ (all : AvailableActions S A).available s :=
  Finset.mem_univ a

end AvailableActions

/-- Time-varying available actions: a nonempty action set for every time step `k` and state `s`.

Julia counterpart: `TimeVaryingAvailableActions` (`src/available_actions.jl`), a vector of
single-time-step available actions. -/
structure TimeVaryingAvailableActions (S A : Type*) where
  /-- The available actions at time step `k` (Julia `aa.actions[k]`). -/
  actions : ℕ → AvailableActions S A

/-- A robust MDP: available actions plus, for every state `s` and action `a`, a well-formed
ambiguity set `Γ_{s,a}` of successor distributions.

Julia counterpart: `FactoredRobustMarkovDecisionProcess` with one marginal (`IsRMDP` / `IsIMDP`,
`src/models/FactoredRobustMarkovDecisionProcess.jl`, `src/models/models.jl`). -/
structure RMDP (S A : Type*) [Fintype S] extends AvailableActions S A where
  /-- The ambiguity set `Γ_{s,a}` (Julia: `marginal[a, s]`, the set selected by `sub2ind`). -/
  ambiguity : S → A → AmbiguitySet S
  /-- Every ambiguity set is well formed: nonempty and closed (not necessarily convex; the
  factored product sets are not). -/
  ambiguity_wellFormed : ∀ s a, (ambiguity s a).WellFormed

namespace RMDP

variable {S A : Type*} [Fintype S]

/-- The same RMDP with the available actions replaced by `aa`; used to model time step `k` of
`TimeVaryingAvailableActions` (Julia: `available(aa.actions[k], jₛ)`).

Julia counterpart: `select_model(mp, k)` (`src/robust_value_iteration.jl`), which rebuilds the
model with `select_available_actions(available_actions(mp), k)`. -/
def withAvailable (M : RMDP S A) (aa : AvailableActions S A) : RMDP S A :=
  { M with toAvailableActions := aa }

/-- The RMDP seen at time step `k` of time-varying available actions.

Julia counterpart: `select_model(mp, k)` with
`select_available_actions(aa::TimeVaryingAvailableActions, k)`
(`src/robust_value_iteration.jl`). The relation between Julia's step `k` and the Lean index is
not fixed here. -/
def atTime (M : RMDP S A) (tv : TimeVaryingAvailableActions S A) (k : ℕ) : RMDP S A :=
  M.withAvailable (tv.actions k)

/-- Replacing the available actions keeps the ambiguity sets.

Julia counterpart: `select_model(mp, k)` (`src/robust_value_iteration.jl`) passes
`marginals(mp)` through unchanged. -/
@[simp] theorem withAvailable_ambiguity (M : RMDP S A) (aa : AvailableActions S A) :
    (M.withAvailable aa).ambiguity = M.ambiguity := rfl

end RMDP

end IntervalMDP
