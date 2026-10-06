import IntervalMDPProofs.Models.RMDP

/-!
# Strategies

A stationary strategy picks one action per state; a time-varying strategy picks one action per
state and time step, for a finite number of steps. A strategy is *valid* for a model when every
chosen action is available.

Julia counterpart: `StationaryStrategy`, `TimeVaryingStrategy` (`src/strategy.jl`) and
`GivenStrategyCache` (`src/strategy_cache.jl`). Julia's `checkstrategy` checks that each action
index is in `1:action_vars[i]`, which the Lean types enforce; it does not check membership in
`available(model, s)`, which is what `Valid` adds.
-/

namespace IntervalMDP

/-- A stationary (memoryless, time-independent) deterministic strategy `π : S → A`.

Julia counterpart: `StationaryStrategy` (`src/strategy.jl`), field `strategy` (an array of action
tuples over the source states). -/
structure StationaryStrategy (S A : Type*) where
  /-- The action chosen in each state (Julia `strategy[jₛ]`). -/
  strategy : S → A

/-- A time-varying deterministic strategy for `timeLength` steps: `πₖ : S → A` for `k < timeLength`.

Julia counterpart: `TimeVaryingStrategy` (`src/strategy.jl`), field `strategy::Vector{A}` with
`time_length = length(strategy)`; step `k` (0-based here) is Julia's `strategy[k + 1]`. -/
structure TimeVaryingStrategy (S A : Type*) where
  /-- The number of time steps (Julia `time_length(strategy)`). -/
  timeLength : ℕ
  /-- The action chosen at time step `k` in each state. -/
  strategy : Fin timeLength → S → A

namespace StationaryStrategy

variable {S A : Type*}

/-- `π` is valid for the available actions `aa` when `π(s) ∈ available s` for every state.

Julia counterpart: `checkstrategy(strategy::StationaryStrategy, system)` (`src/strategy.jl`),
which checks only the action range; availability is a Lean-side strengthening. -/
def Valid (π : StationaryStrategy S A) (aa : AvailableActions S A) : Prop :=
  ∀ s, π.strategy s ∈ aa.available s

/-- The time-varying strategy that plays `π` for `K` steps.

Julia counterpart: `getindex(strategy::StationaryStrategy, k)` (`src/strategy.jl`), which
returns the same strategy at every step. -/
def toTimeVarying (π : StationaryStrategy S A) (K : ℕ) : TimeVaryingStrategy S A where
  timeLength := K
  strategy _ := π.strategy

end StationaryStrategy

namespace TimeVaryingStrategy

variable {S A : Type*}

/-- `π` is valid for the available actions `aa` when `πₖ(s) ∈ available s` for every step and
state.

Julia counterpart: `checkstrategy(strategy::TimeVaryingStrategy, system)` (`src/strategy.jl`),
which checks only the action range; availability is a Lean-side strengthening. -/
def Valid (π : TimeVaryingStrategy S A) (aa : AvailableActions S A) : Prop :=
  ∀ k s, π.strategy k s ∈ aa.available s

/-- `π` is valid for time-varying available actions `tv` when `πₖ(s) ∈ available_k s`.

Julia counterpart: a `TimeVaryingStrategy` (`src/strategy.jl`) used with
`TimeVaryingAvailableActions` (`src/available_actions.jl`); Julia does not check availability. -/
def ValidAt (π : TimeVaryingStrategy S A) (tv : TimeVaryingAvailableActions S A) : Prop :=
  ∀ (k : Fin π.timeLength) s, π.strategy k s ∈ (tv.actions k.val).available s

end TimeVaryingStrategy

/-- A stationary strategy that is valid for `aa` gives a valid time-varying strategy of any
length.

Julia counterpart: none (Lean-side proof device). -/
theorem StationaryStrategy.Valid.toTimeVarying {S A : Type*} {π : StationaryStrategy S A}
    {aa : AvailableActions S A} (h : π.Valid aa) (K : ℕ) : (π.toTimeVarying K).Valid aa :=
  fun _ s => h s

/-- Every model admits a valid stationary strategy (choose any available action in each state;
`available s` is nonempty).

Julia counterpart: none (Lean-side proof device). -/
theorem AvailableActions.exists_valid {S A : Type*} (aa : AvailableActions S A) :
    ∃ π : StationaryStrategy S A, π.Valid aa :=
  ⟨⟨fun s => (aa.available_nonempty s).choose⟩, fun s => (aa.available_nonempty s).choose_spec⟩

end IntervalMDP
