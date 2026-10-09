import IntervalMDPProofs.Index.Marginal
import IntervalMDPProofs.Models.Strategy

/-!
# Strategy lookup

With a given strategy (`NonOptimizingStrategyCache`), `state_bellman!` (`src/bellman.jl`) reads the
action of state `jₛ` from the strategy array and evaluates only that action:

    jₐ = CartesianIndex(strategy_cache[jₛ])
    ambiguity_set = marginal[jₐ, jₛ]

The strategy array (`StationaryStrategy.strategy`, one step of `TimeVaryingStrategy.strategy`,
`src/strategy.jl`; wrapped by `ActiveGivenStrategyCache`, `src/strategy_cache.jl`) is an
`AbstractArray{NTuple{M, Int32}}` of shape `source_shape(system)`; `strategy_cache[jₛ]` with a
`CartesianIndex` `jₛ` reads the entry at the column-major linear index of `jₛ`.

`strategyAction` models this lookup on the integer level: the array is its data vector
(`StrategyArray`, 1-based linear position ↦ `Int32` action tuple) and the position is
`LinearIndices(source_shape)[jₛ]` in `N`-bit integers (`linearInt`). The index theorems carry the
overflow bound `∏ source_shape < 2 ^ (N - 1)` as a hypothesis.

## Validation does not guarantee availability (Finding F2)

`VerificationProblem` (`src/problem.jl`) validates a given strategy with `checkstrategy`
(`src/strategy.jl`), which checks the shape and `1 ≤ s[i] ≤ action_vars[i]` for every entry, but
not `isavailable(model, jₛ, jₐ)`. With `ListAvailableActions` (`src/available_actions.jl`) a
strategy can therefore pass validation and still look up an action that is not in
`available(model, jₛ)` (`checkStrategy_admits_unavailable`). The claim "for strategies that pass
validation the looked-up action is available" (`IntervalMDP.Index.strategyAction_available` in the
spec) is false for the Julia code; see Finding F2 in the inventory. The strongest statements that
hold are:

* `strategyAction_available_of_all`: for `AllAvailableActions` (the default, and the only option of
  `IntervalMarkovDecisionProcess`), every strategy that passes `checkstrategy` looks up an available
  action;
* `strategyAction_available_of_valid`: a strategy array that stores a Lean-valid strategy
  (`StationaryStrategy.Valid`, availability in every state) looks up an available action.
-/

namespace IntervalMDP.Index

variable {n m : ℕ}

/-- The data vector of a Julia strategy array: entry `k` (1-based, column-major) is the `Int32`
action tuple `(a[1], …, a[M])` stored at linear position `k`. Positions outside
`1..∏ source_shape` are never read under the overflow bound.

Julia counterpart: the `strategy` field of `StationaryStrategy` / one step of
`TimeVaryingStrategy` (`src/strategy.jl`), an `AbstractArray{NTuple{M, Int32}}`, as wrapped by
`ActiveGivenStrategyCache` (`src/strategy_cache.jl`). -/
abbrev StrategyArray (m : ℕ) : Type := ℤ → Fin m → ℤ

/-- The Julia action tuple `Tuple(jₐ)` of a Lean joint action (`toJulia` in every coordinate).

Julia counterpart: `Tuple(jₐ)` of a `CartesianIndex` action, as stored in a strategy array by
`_extract_strategy!` (`opt_index = Tuple(jₐ)`, `src/strategy_cache.jl`). -/
def actionTuple {av : ActionVars m} (a : av.Action) : Fin m → ℤ :=
  juliaTuple (dims := av.dims) a

/-- Strategy lookup: the action tuple `strategy_cache[jₛ]` of source state `jₛ`, read at the
column-major linear index `LinearIndices(source_shape)[jₛ]` computed in `N`-bit integers.
`CartesianIndex(·)` of the result is the action `jₐ` evaluated by `state_bellman!`.

Julia counterpart: `jₐ = CartesianIndex(strategy_cache[jₛ])` in `state_bellman!` with a
`NonOptimizingStrategyCache` (`src/bellman.jl`), via `getindex(cache::ActiveGivenStrategyCache, j)
= cache.strategy[j]` (`src/strategy_cache.jl`). -/
def strategyAction (N : ℕ) (sv : StateVars n) (strategy : StrategyArray m) (jₛ : sv.State) :
    Fin m → ℤ :=
  strategy (linearInt N sv.dims jₛ)

/-- The available actions of state `jₛ` as Julia action tuples, `Tuple.(available(model, jₛ))`.

Julia counterpart: `available(model, jₛ)` (`src/available_actions.jl`,
`src/models/FactoredRobustMarkovDecisionProcess.jl`) for the available actions `aa`. -/
def juliaAvailable {sv : StateVars n} {av : ActionVars m} (aa : AvailableActions sv.State av.Action)
    (jₛ : sv.State) : Set (Fin m → ℤ) :=
  actionTuple '' (aa.available jₛ : Set av.Action)

/-- Julia's validation of a strategy array: every stored tuple satisfies
`1 ≤ s[i] ≤ action_vars[i]` (the shape check `size(strategy) == source_shape(system)` is the
array length `∏ source_shape`). Availability is **not** checked.

Julia counterpart: `checkstrategy(strategy::AbstractArray, system::FactoredRMDP)`
(`src/strategy.jl`), called by the `VerificationProblem` constructor (`src/problem.jl`). -/
def checkStrategy (sv : StateVars n) (av : ActionVars m) (strategy : StrategyArray m) : Prop :=
  ∀ k ∈ juliaRange (∏ i, sv.dims i), ∀ i, 1 ≤ strategy k i ∧ strategy k i ≤ av.dims i

/-- The strategy array stores the Lean strategy `π`: the entry at the linear index of every source
state `jₛ` is the Julia tuple of `π(jₛ)`.

Julia counterpart: a `StationaryStrategy(strategy)` (`src/strategy.jl`) whose array holds
`Tuple(π(jₛ))` at `strategy[jₛ]`. -/
def Stores {sv : StateVars n} {av : ActionVars m} (strategy : StrategyArray m)
    (π : StationaryStrategy sv.State av.Action) : Prop :=
  ∀ jₛ : sv.State, strategy (linear sv.dims jₛ) = actionTuple (π.strategy jₛ)

/-- `actionTuple` is injective: distinct actions have distinct Julia tuples.

Julia counterpart: none (Lean-side proof device). -/
theorem actionTuple_injective {av : ActionVars m} :
    Function.Injective (actionTuple (av := av)) := by
  intro a b h
  funext i
  have hi := congrFun h i
  simp only [actionTuple, juliaTuple, toJulia, Nat.cast_inj] at hi
  exact Fin.ext (by omega)

/-- Under the overflow bound, the lookup reads the entry at the exact linear index of `jₛ`.

Julia counterpart: `strategy_cache[jₛ]` (`src/strategy_cache.jl`) with `Int` index arithmetic. -/
theorem strategyAction_eq_linear {N : ℕ} {sv : StateVars n} (hBound : ∏ i, sv.dims i < 2 ^ (N - 1))
    (strategy : StrategyArray m) (jₛ : sv.State) :
    strategyAction N sv strategy jₛ = strategy (linear sv.dims jₛ) := by
  rw [strategyAction, (linear_bijective (N := N) hBound).2 jₛ]

/-- **Strategy lookup, `AllAvailableActions`.** Under the overflow bound, every strategy that
passes Julia's validation (`checkstrategy`) looks up an available action when all actions are
available (`AvailableActions.all`, Julia `AllAvailableActions`, the default and the only option of
`IntervalMarkovDecisionProcess`).

Julia counterpart: `CartesianIndex(strategy_cache[jₛ])` in `state_bellman!` (`src/bellman.jl`)
after `checkstrategy` (`src/strategy.jl`), with `AllAvailableActions`
(`src/available_actions.jl`). -/
theorem strategyAction_available_of_all {N : ℕ} {sv : StateVars n} {av : ActionVars m}
    (hBound : ∏ i, sv.dims i < 2 ^ (N - 1)) {strategy : StrategyArray m}
    (hcheck : checkStrategy sv av strategy) (jₛ : sv.State) :
    strategyAction N sv strategy jₛ ∈
      juliaAvailable (AvailableActions.all : AvailableActions sv.State av.Action) jₛ := by
  rw [strategyAction_eq_linear hBound]
  have hk := (linear_bijective (N := N) (dims := sv.dims) hBound).1.mapsTo (Set.mem_univ jₛ)
  have hc := hcheck _ hk
  refine ⟨fun i => ⟨(strategy (linear sv.dims jₛ) i - 1).toNat, ?_⟩, Finset.mem_coe.mpr
    (AvailableActions.mem_all _ _), ?_⟩
  · have := hc i
    omega
  · funext i
    have := hc i
    simp only [actionTuple, juliaTuple, toJulia]
    omega

/-- **Strategy lookup, valid strategies.** Under the overflow bound, a strategy array that stores a
strategy `π` that is valid for the available actions (`π(jₛ) ∈ available(jₛ)` in every state)
looks up an available action in every state. Holds for any available actions, including
`ListAvailableActions`; Julia does not check the validity hypothesis (Finding F2).

Julia counterpart: `CartesianIndex(strategy_cache[jₛ])` in `state_bellman!` (`src/bellman.jl`)
for a `StationaryStrategy` (`src/strategy.jl`) whose actions are available. -/
theorem strategyAction_available_of_valid {N : ℕ} {sv : StateVars n} {av : ActionVars m}
    (hBound : ∏ i, sv.dims i < 2 ^ (N - 1)) {aa : AvailableActions sv.State av.Action}
    {π : StationaryStrategy sv.State av.Action} (hπ : π.Valid aa) {strategy : StrategyArray m}
    (hstore : Stores strategy π) (jₛ : sv.State) :
    strategyAction N sv strategy jₛ ∈ juliaAvailable aa jₛ := by
  rw [strategyAction_eq_linear hBound, hstore jₛ]
  exact ⟨π.strategy jₛ, Finset.mem_coe.mpr (hπ jₛ), rfl⟩

/-- **Finding F2.** Julia's validation does not imply availability: whenever some state `jₛ` has
an action `a` that is not available (possible with `ListAvailableActions`), the strategy array
that stores `Tuple(a)` everywhere passes `checkstrategy`, and its lookup at `jₛ` is not an
available action. So "every strategy that passes validation looks up an available action" is false
for every model with an unavailable action.

Julia counterpart: `checkstrategy(strategy::AbstractArray, system::FactoredRMDP)`
(`src/strategy.jl`), which checks only `1 ≤ s[i] ≤ action_vars[i]`, followed by
`CartesianIndex(strategy_cache[jₛ])` in `state_bellman!` (`src/bellman.jl`). -/
theorem checkStrategy_admits_unavailable (N : ℕ) {sv : StateVars n} {av : ActionVars m}
    (aa : AvailableActions sv.State av.Action) (jₛ : sv.State) {a : av.Action}
    (ha : a ∉ aa.available jₛ) :
    ∃ strategy : StrategyArray m, checkStrategy sv av strategy ∧
      strategyAction N sv strategy jₛ ∉ juliaAvailable aa jₛ := by
  refine ⟨fun _ => actionTuple a, fun _ _ i => ?_, ?_⟩
  · have := (a i).is_lt
    simp only [actionTuple, juliaTuple, toJulia]
    omega
  · rintro ⟨b, hb, hba⟩
    exact ha (actionTuple_injective hba ▸ Finset.mem_coe.mp hb)

end IntervalMDP.Index
