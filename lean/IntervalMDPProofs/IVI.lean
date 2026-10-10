import IntervalMDPProofs.VI.Strategy

/-!
# Interval value iteration: bounds, bracket and gap stopping (Phase 4a, 4b)

`_interval_value_iteration!` (`src/interval_value_iteration.jl`) runs

    initialize_ivi!(V_lower, V_upper, spec)   # V_lower = 𝟙_reach, V_upper = 1 - 𝟙_avoid
    nextiteration!(V_lower); nextiteration!(V_upper)
    ivi_step!(…, V_lower, V_upper, 0, mp, spec); k = 1
    while !term_criteria(V_lower, V_upper, k, mp)
        nextiteration!(V_lower); nextiteration!(V_upper)
        ivi_step!(…, V_lower, V_upper, k, mp, spec); k += 1
    end

and `ivi_step!` makes two `bellman!` calls with the same nature (`upper_bound = isoptimistic(spec)`)
and the same `maximize = ismaximize(spec)`:

1. on the **primary** bound (`V_lower` for `Pessimistic`, `V_upper` for `Optimistic`) with the
   optimizing strategy cache, which synthesizes the strategy `σ_k` (`extract_strategy!`);
2. on the **secondary** bound with `applied_strategy_cache(primary_cache)`, i.e. policy evaluation
   of `σ_k` (the `NonOptimizingStrategyCache` path);

then `step_postprocess_value_function!` on both bounds. IVI is only defined for reach-avoid
properties (`checkivisupported`, `src/specification.jl`).

This file transcribes that loop as `iviIter` (index `k` = the Julia counter `k`) and proves, for a
general `RMDP` (so for interval and factored IMDPs through `toRMDP`):

* `lower_le_upper`: `V_lower_k ≤ V_upper_k` for every `k`, all four satisfaction × strategy modes,
  all reach-avoid properties;
* `primary_sound`: the primary bound (the value Julia returns as `value_function`) is sound,
  `Sound sat V_primary_k V*`, all four modes, non-exact-time reach-avoid properties
  (`prop.toReachProperty.isExactTime = false`), via `Approx.iter_sound`;
* `lower_le_reachLfp`: `V_lower_k ≤ V*` when `sat = pessimistic` or `strat = maximize`,
  non-exact-time reach-avoid properties (`prop.toReachProperty.isExactTime = false`);
* `reachLfp_le_upper`: `V* ≤ V_upper_k` when `sat = optimistic` or `strat = minimize`,
  all reach-avoid properties (finite, infinite and exact time);
* `bracket_aligned`: `V_lower_k ≤ V* ≤ V_upper_k` for the two modes
  `(Pessimistic, Minimize)` and `(Optimistic, Maximize)`, non-exact-time reach-avoid properties
  (`prop.toReachProperty.isExactTime = false`).

Here `V* = reachLfp`, the least fixed point of the reach-avoid step (the infinite-horizon value,
Phase 3a). **The bracket fails for the other two modes** under Julia's strategy coupling: for
`(Pessimistic, Maximize)` the strategy synthesized on the lower bound can be worse than optimal for
the upper bound, so `V_upper_1 < V*`; dually for `(Optimistic, Minimize)` `V* < V_lower_1`
(inventory Finding F4; witnesses `Examples.ivi_upper_lt_reachLfp` and
`Examples.ivi_reachLfp_lt_lower`). The theorem `IVI.bracket` for all four modes is therefore not
stated.

**Stopping (4b).** For infinite-time reach-avoid the loop condition is `IVIInitialGapCriteria`:
the loop tests, after each call `k = 1, 2, …` (never on the initialised bounds), whether
`max_initial_gap = max(0, max_{s ∈ initial_states(mp)} (V_upper_k(s) - V_lower_k(s))) <
convergence_eps` (absolute gap), and returns the primary bound of the first such call. This is
modelled by `maxInitialGap` (a transcription of the loop of `max_initial_gap`),
`iviInitialGapCriteria`, `stopIndex` (the exit value of `k`) and `iviValueFunction` (the returned
`value_function`). Both theorems below assume that the loop exits, i.e. they take
`h : Terminates M sat strat prop o kind tol initial` as a hypothesis; termination is **not
proved** (the gap need not close, e.g. when an end component lies in the don't-care region; see
the design note in `src/interval_value_iteration.jl`). Proved:

* `gap_stop_sound_aligned`: assuming the loop exits (`Terminates`, not proved), the returned value
  is within `tol` of `V*` on the initial states, `abs (iviValueFunction h s - V* s) < tol`, for the
  two modes `(Pessimistic, Minimize)` and `(Optimistic, Maximize)`, non-exact-time reach-avoid
  properties (`prop.toReachProperty.isExactTime = false`), both strategy caches (via
  `bracket_aligned`);
* `gap_stop_primary_sound`: assuming the loop exits (`Terminates`, not proved), the returned value
  is sound, `Sound sat (iviValueFunction h) V*`, all four modes, non-exact-time reach-avoid
  properties (`prop.toReachProperty.isExactTime = false`), both strategy caches (via
  `primary_sound`).

The four-mode `IVI.gap_stop_sound` is false for Julia's coupling (Finding F4): in
`(Pessimistic, Maximize)` and `(Optimistic, Minimize)` the gap can be `0` at `k = 1` while the
returned value is `½` away from `V*` (`Examples.not_gap_stop_sound_pessimistic_maximize`,
`Examples.not_gap_stop_sound_optimistic_minimize`). It is therefore not stated.

**Scope.** Abstract (values in `ℝ`, exact operators), time-invariant model (`select_model(mp, k)
= mp`), `V*` = dynamic-programming least fixed point (not the path measure). The termination test
`IVIFixedIterationsCriteria` (finite horizon, `k = time_horizon`) is not modelled separately:
`iviIter k` at that `k` is its result. Floating point, threads, CUDA and the
Lean ↔ Julia correspondence are limitations L1–L6 of the inventory.
-/

namespace IntervalMDP.IVI

open Bellman Approx
open VI hiding step

/-! ### Reach-avoid properties -/

/-- The reach-avoid properties, the only properties IVI accepts. Finite- and infinite-time
reach-avoid share a constructor (the horizon and `convergence_eps` only enter the termination
criterion).

Julia counterpart: the concrete subtypes of `AbstractReachAvoid` (`src/specification.jl`):
`FiniteTimeReachAvoid`, `InfiniteTimeReachAvoid` and `ExactTimeReachAvoid`; `checkivisupported`
rejects every other property. -/
inductive ReachAvoidProperty (S : Type*) where
  /-- `FiniteTimeReachAvoid` / `InfiniteTimeReachAvoid(reach, avoid, …)`; `checkdisjoint`
  enforces `disjoint`. -/
  | reachAvoid (reach avoid : Finset S) (disjoint : Disjoint reach avoid)
  /-- `ExactTimeReachAvoid(reach, avoid, time_horizon)`. -/
  | exactTimeReachAvoid (reach avoid : Finset S) (disjoint : Disjoint reach avoid)

namespace ReachAvoidProperty

variable {S : Type*}

/-- The target set `reach(prop)`.

Julia counterpart: `reach(prop)` (`src/specification.jl`). -/
def reach : ReachAvoidProperty S → Finset S
  | reachAvoid reach .. | exactTimeReachAvoid reach .. => reach

/-- The avoid set `avoid(prop)`.

Julia counterpart: `avoid(prop)` (`src/specification.jl`). -/
def avoid : ReachAvoidProperty S → Finset S
  | reachAvoid _ avoid _ | exactTimeReachAvoid _ avoid _ => avoid

/-- `reach` and `avoid` are disjoint.

Julia counterpart: `checkdisjoint` in the reach-avoid constructors (`src/specification.jl`). -/
theorem disjoint_reach_avoid (prop : ReachAvoidProperty S) : Disjoint prop.reach prop.avoid := by
  cases prop <;> assumption

/-- The same property as a `VI.ReachProperty`, which fixes its step postprocessing.

Julia counterpart: dispatch of `step_postprocess_value_function!` on `AbstractReachAvoid` and
`ExactTimeReachAvoid` (`src/specification.jl`). -/
def toReachProperty : ReachAvoidProperty S → ReachProperty S
  | reachAvoid r a h => .reachAvoid r a h
  | exactTimeReachAvoid r a h => .exactTimeReachAvoid r a h

/-- `reach` is preserved by `toReachProperty`.

Julia counterpart: `reach(prop)` (`src/specification.jl`). -/
@[simp] theorem reach_toReachProperty (prop : ReachAvoidProperty S) :
    prop.toReachProperty.reach = prop.reach := by
  cases prop <;> rfl

variable [DecidableEq S]

/-- The step postprocessing sets the avoid states to `0`.

Julia counterpart: `value_function.current[avoid(prop)] .= 0.0` in
`step_postprocess_value_function!` (`src/specification.jl`). -/
theorem stepPostprocess_avoid (prop : ReachAvoidProperty S) {V : S → ℝ} {s : S}
    (hs : s ∈ prop.avoid) : stepPostprocessValueFunction prop.toReachProperty V s = 0 := by
  cases prop <;> simp_all [toReachProperty, stepPostprocessValueFunction, avoid]

end ReachAvoidProperty

/-! ### The IVI state and one step -/

/-- The state of IVI between two calls of `ivi_step!`: the current lower and upper value functions
and the content of the strategy cache.

Julia counterpart: `V_lower.current`, `V_upper.current` (`ValueFunction`,
`src/robust_value_iteration.jl`) and `strategy_cache` (`cur_strategy` of a
`TimeVaryingStrategyCache`, `strategy` of a `StationaryStrategyCache`, `src/strategy_cache.jl`) in
`_interval_value_iteration!` (`src/interval_value_iteration.jl`). -/
structure Bounds (S A : Type*) where
  /-- Julia `V_lower.current`. -/
  lower : S → ℝ
  /-- Julia `V_upper.current`. -/
  upper : S → ℝ
  /-- The action per state stored in the strategy cache. -/
  strategy : S → A

/-- The strategy cache IVI uses: `construct_ivi_strategy_cache` returns the problem's regular cache
for a `ControlSynthesisProblem` (time-varying for a finite horizon, stationary otherwise) and a
`StationaryStrategyCache` for a `VerificationProblem`.

Julia counterpart: `TimeVaryingStrategyCache` / `StationaryStrategyCache` built by
`construct_ivi_strategy_cache` (`src/strategy_cache.jl`). -/
inductive StrategyCacheKind where
  /-- `TimeVaryingStrategyCache`: every call starts from `first(available_actions)`. -/
  | timeVarying
  /-- `StationaryStrategyCache`: every call starts from the action stored by the previous call. -/
  | stationary

variable {S A : Type*} [Fintype S] [DecidableEq S] [DecidableEq A]

/-- The seed of `extract_strategy!` for state `s`, given the cache content `prev`:
`first(available_actions)` for a time-varying cache, the stored action for a stationary cache (its
guard always passes, since stored actions are available, `iviIter_strategy_mem`).

Julia counterpart: `neutral` in `extract_strategy!(::TimeVaryingStrategyCache |
::StationaryStrategyCache, …)` (`src/strategy_cache.jl`). -/
def cacheSeed {M : RMDP S A} (o : ActionOrder M) : StrategyCacheKind → (S → A) → S → A
  | .timeVarying, _, s => o.first s
  | .stationary, prev, s => prev s

omit [Fintype S] [DecidableEq S] in
/-- The primary bound, on which the strategy is synthesized: `V_lower` for `Pessimistic`,
`V_upper` for `Optimistic`. Julia returns it as `value_function` of the solution.

Julia counterpart: `primary_current` in `ivi_step!` and `_ivi_verification_solution`
(`src/interval_value_iteration.jl`). -/
def primary : SatisfactionMode → Bounds S A → S → ℝ
  | .pessimistic, B => B.lower
  | .optimistic, B => B.upper

omit [Fintype S] [DecidableEq S] in
/-- The secondary bound, to which the synthesized strategy is applied: `V_upper` for
`Pessimistic`, `V_lower` for `Optimistic`.

Julia counterpart: `secondary_current` in `ivi_step!` (`src/interval_value_iteration.jl`). -/
def secondary : SatisfactionMode → Bounds S A → S → ℝ
  | .pessimistic, B => B.upper
  | .optimistic, B => B.lower

omit [Fintype S] [DecidableEq S] [DecidableEq A] in
/-- Reassemble the bounds from the new primary and secondary value functions and the new strategy.

Julia counterpart: the assignment of `primary_current` / `secondary_current` to
`V_lower.current` / `V_upper.current` in `ivi_step!` (`src/interval_value_iteration.jl`). -/
def ofPrimary : SatisfactionMode → (S → ℝ) → (S → ℝ) → (S → A) → Bounds S A
  | .pessimistic, P, Q, σ => ⟨P, Q, σ⟩
  | .optimistic, P, Q, σ => ⟨Q, P, σ⟩

/-- The strategy synthesized by the first `bellman!` call of `ivi_step!`: in each state, the
action selected by `extract_strategy!` (`argoptAction`) for the values of the **primary** bound,
from the cache's seed, over Julia's iteration order.

Julia counterpart: `extract_strategy!` with `primary_cache` in the `bellman!` call on the primary
bound in `ivi_step!` (`src/interval_value_iteration.jl`, `src/strategy_cache.jl`). -/
noncomputable def iviStrategy (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (kind : StrategyCacheKind) (B : Bounds S A) : StationaryStrategy S A where
  strategy s :=
    argoptAction strat (stateActionBellman M sat (primary sat B) s) (cacheSeed o kind B.strategy s)
      (o.acts s)

/-- One call of `ivi_step!`, a transcription of its body in order:

1. `bellman!(…, primary_cache, primary_current, primary_previous, …)`: the strategy `σ` is
   synthesized on the primary bound, whose new value is `opt_val = values[σ(s)]`, i.e.
   `Tπ σ (primary)` (equal to `T (primary)`, `primary_step`);
2. `bellman!(…, applied_strategy_cache(primary_cache), secondary_current, …)`: policy evaluation
   `Tπ σ (secondary)` of the same strategy, same nature `sat`;
3. `step_postprocess_value_function!` on `V_lower` and `V_upper` (`V[reach] .= 1; V[avoid] .= 0`
   for `Finite/InfiniteTimeReachAvoid`, `V[avoid] .= 0` for `ExactTimeReachAvoid`).

Julia counterpart: `ivi_step!(workspace, strategy_cache, V_lower, V_upper, k, mp, spec)`
(`src/interval_value_iteration.jl`), with `upper_bound = isoptimistic(spec)` (`sat`) and
`maximize = ismaximize(spec)` (`strat`); the model is time-invariant. -/
noncomputable def step (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind)
    (B : Bounds S A) : Bounds S A :=
  let σ := iviStrategy M sat strat o kind B
  let primaryNext := stepPostprocessValueFunction prop.toReachProperty
    (Tπ M sat σ (primary sat B))
  let secondaryNext := stepPostprocessValueFunction prop.toReachProperty
    (Tπ M sat σ (secondary sat B))
  ofPrimary sat primaryNext secondaryNext σ.strategy

/-- The initial upper bound: `0` on `avoid(prop)`, `1` elsewhere.

Julia counterpart: `fill!(V_upper.current, one(R)); V_upper.current[avoid(prop)] .= zero(R)` in
`initialize_ivi!` (`src/specification.jl`). -/
def initializeUpper (prop : ReachAvoidProperty S) : S → ℝ :=
  fun s => if s ∈ prop.avoid then 0 else 1

/-- The initial IVI state: `V_lower = 𝟙_reach` (`initializeValueFunction`), `V_upper = 1 - 𝟙_avoid`
(`initializeUpper`), and the strategy cache at `first(available_actions)` (a fresh stationary cache
holds zero tuples, whose guard falls back to `first(available_actions)`; a time-varying cache does
not read it).

Julia counterpart: `initialize_ivi!(V_lower, V_upper, prop::AbstractReachAvoid)`
(`src/specification.jl`) and `construct_ivi_strategy_cache` (`src/strategy_cache.jl`). -/
def initializeIvi {M : RMDP S A} (o : ActionOrder M) (prop : ReachAvoidProperty S) :
    Bounds S A where
  lower := initializeValueFunction prop.toReachProperty
  upper := initializeUpper prop
  strategy := o.first

/-- The IVI iterates, a transcription of the loop of `_interval_value_iteration!`: `iviIter 0` is
the initialised state, and `iviIter (k + 1)` is `step` applied to `iviIter k` (Julia:
`nextiteration!` on both bounds, then `ivi_step!(…, k, …)`, then `k += 1`). The Julia counter `k`
after the loop equals the Lean index; the returned bounds are `iviIter k` for the first `k ≥ 1` at
which `term_criteria` holds (`IVIFixedIterationsCriteria`: `k = time_horizon`;
`IVIInitialGapCriteria`: initial-state gap `< convergence_eps`).

Julia counterpart: `_interval_value_iteration!` (`src/interval_value_iteration.jl`). -/
noncomputable def iviIter (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind) :
    ℕ → Bounds S A
  | 0 => initializeIvi o prop
  | k + 1 => step M sat strat prop o kind (iviIter M sat strat prop o kind k)

/-! ### Unfolding lemmas -/

/-- The lower bound after a step is the postprocessed policy evaluation of the synthesized
strategy, in both satisfaction modes (primary for `Pessimistic`, secondary for `Optimistic`).

Julia counterpart: `ivi_step!` (`src/interval_value_iteration.jl`). -/
theorem step_lower (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind) (B : Bounds S A) :
    (step M sat strat prop o kind B).lower = stepPostprocessValueFunction prop.toReachProperty
      (Tπ M sat (iviStrategy M sat strat o kind B) B.lower) := by
  cases sat <;> rfl

/-- The upper bound after a step is the postprocessed policy evaluation of the synthesized
strategy, in both satisfaction modes.

Julia counterpart: `ivi_step!` (`src/interval_value_iteration.jl`). -/
theorem step_upper (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind) (B : Bounds S A) :
    (step M sat strat prop o kind B).upper = stepPostprocessValueFunction prop.toReachProperty
      (Tπ M sat (iviStrategy M sat strat o kind B) B.upper) := by
  cases sat <;> rfl

/-- The strategy cache after a step holds the synthesized strategy.

Julia counterpart: `ivi_step!` (`src/interval_value_iteration.jl`). -/
theorem step_strategy (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind) (B : Bounds S A) :
    (step M sat strat prop o kind B).strategy = (iviStrategy M sat strat o kind B).strategy := by
  cases sat <;> rfl

/-- The primary bound after a step is the postprocessed policy evaluation of the synthesized
strategy on the primary bound.

Julia counterpart: `ivi_step!` (`src/interval_value_iteration.jl`). -/
theorem primary_step_eq (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind) (B : Bounds S A) :
    primary sat (step M sat strat prop o kind B) = stepPostprocessValueFunction
      prop.toReachProperty (Tπ M sat (iviStrategy M sat strat o kind B) (primary sat B)) := by
  cases sat <;> rfl

/-! ### The synthesized strategy -/

omit [DecidableEq S] in
/-- The seed is available whenever the stored actions are.

Julia counterpart: `neutral` in `extract_strategy!` (`src/strategy_cache.jl`). -/
theorem cacheSeed_mem {M : RMDP S A} (o : ActionOrder M) (kind : StrategyCacheKind)
    {prev : S → A} (hprev : ∀ s, prev s ∈ M.available s) (s : S) :
    cacheSeed o kind prev s ∈ M.available s := by
  cases kind
  · exact o.first_mem s
  · exact hprev s

omit [DecidableEq S] in
/-- The synthesized strategy is valid, and its policy evaluation on the primary bound is the
Bellman update `T` of the primary bound (`argopt_attains`).

Julia counterpart: `extract_strategy!` (`src/strategy_cache.jl`) in the primary `bellman!` call of
`ivi_step!` (`src/interval_value_iteration.jl`). -/
theorem iviStrategy_spec (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (kind : StrategyCacheKind) {B : Bounds S A}
    (hB : ∀ s, B.strategy s ∈ M.available s) :
    (iviStrategy M sat strat o kind B).Valid M.toAvailableActions ∧
      Tπ M sat (iviStrategy M sat strat o kind B) (primary sat B) = T M sat strat (primary sat B) :=
  ⟨fun s => (argopt_attains M sat strat (primary sat B) s (o.toFinset_acts s)
      (cacheSeed_mem o kind hB s)).1,
    funext fun s => (argopt_attains M sat strat (primary sat B) s (o.toFinset_acts s)
      (cacheSeed_mem o kind hB s)).2⟩

/-- Every action stored in the strategy cache is available, at every call.

Julia counterpart: `strategy_cache` across the calls of `ivi_step!`
(`src/interval_value_iteration.jl`, `src/strategy_cache.jl`). -/
theorem iviIter_strategy_mem (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind) (k : ℕ) (s : S) :
    (iviIter M sat strat prop o kind k).strategy s ∈ M.available s := by
  induction k generalizing s with
  | zero => exact o.first_mem s
  | succ k ih =>
    show (step M sat strat prop o kind (iviIter M sat strat prop o kind k)).strategy s ∈ _
    rw [step_strategy]
    exact (iviStrategy_spec M sat strat o kind ih).1 s

omit [DecidableEq S] [DecidableEq A] in
/-- Policy evaluation is monotone in the value function.

Julia counterpart: `state_bellman!` with a `NonOptimizingStrategyCache` (`src/bellman.jl`). -/
theorem Tπ_mono (M : RMDP S A) (sat : SatisfactionMode) (π : StationaryStrategy S A) :
    Monotone (Tπ M sat π) :=
  fun _ _ h s => stateActionBellman_mono M sat h s (π.strategy s)

/-- The primary bound follows robust value iteration: one IVI step updates it by `VI.step` (the
exact Bellman update `T` followed by the postprocessing), provided the stored actions are
available.

Julia counterpart: the primary `bellman!` call of `ivi_step!` (`src/interval_value_iteration.jl`)
versus `step!` (`src/robust_value_iteration.jl`). -/
theorem primary_step (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind) {B : Bounds S A}
    (hB : ∀ s, B.strategy s ∈ M.available s) :
    primary sat (step M sat strat prop o kind B) =
      VI.step M sat strat prop.toReachProperty (primary sat B) := by
  rw [primary_step_eq, (iviStrategy_spec M sat strat o kind hB).2]
  rfl

/-- The primary iterates are the robust value-iteration iterates `VI.step^[k]` from the initial
primary bound (`𝟙_reach` for `Pessimistic`, `1 - 𝟙_avoid` for `Optimistic`).

Julia counterpart: `_interval_value_iteration!` (`src/interval_value_iteration.jl`). -/
theorem primary_iviIter (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind) (k : ℕ) :
    primary sat (iviIter M sat strat prop o kind k) =
      (VI.step M sat strat prop.toReachProperty)^[k] (primary sat (initializeIvi o prop)) := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [Function.iterate_succ_apply', ← ih]
    exact primary_step M sat strat prop o kind (iviIter_strategy_mem M sat strat prop o kind k)

/-! ### `lower_le_upper` -/

/-- The initial bounds are ordered: `𝟙_reach ≤ 1 - 𝟙_avoid` (`reach ∩ avoid = ∅`).

Julia counterpart: `initialize_ivi!` (`src/specification.jl`). -/
theorem initializeIvi_lower_le_upper {M : RMDP S A} (o : ActionOrder M)
    (prop : ReachAvoidProperty S) :
    (initializeIvi o prop).lower ≤ (initializeIvi o prop).upper := by
  intro s
  show initializeValueFunction prop.toReachProperty s ≤ initializeUpper prop s
  by_cases hr : s ∈ prop.reach
  · have ha : s ∉ prop.avoid := Finset.disjoint_left.mp prop.disjoint_reach_avoid hr
    simp [initializeValueFunction, initializeUpper, hr, ha]
  · by_cases ha : s ∈ prop.avoid <;> simp [initializeValueFunction, initializeUpper, hr, ha]

/-- **`V_lower_k ≤ V_upper_k`** for every call `k`, all four satisfaction × strategy modes, every
reach-avoid property (finite, infinite and exact time), both strategy caches and a general `RMDP`.
Both bounds are updated with the policy evaluation of the same synthesized strategy and the same
nature, which is monotone, and the postprocessing is monotone.

Julia counterpart: `_interval_value_iteration!` / `ivi_step!` / `initialize_ivi!`
(`src/interval_value_iteration.jl`, `src/specification.jl`); the returned `residual`
`V_upper - V_lower` is nonnegative. -/
theorem lower_le_upper (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind) (k : ℕ) :
    (iviIter M sat strat prop o kind k).lower ≤ (iviIter M sat strat prop o kind k).upper := by
  induction k with
  | zero => exact initializeIvi_lower_le_upper o prop
  | succ k ih =>
    show (step M sat strat prop o kind (iviIter M sat strat prop o kind k)).lower ≤
      (step M sat strat prop o kind (iviIter M sat strat prop o kind k)).upper
    rw [step_lower, step_upper]
    exact stepPostprocessValueFunction_mono _ (Tπ_mono M sat _ ih)

/-! ### Soundness of the bounds -/

omit [DecidableEq A] in
/-- The initial lower bound is below the exact value: `𝟙_reach ≤ V*` (non-exact-time properties).

Julia counterpart: `initialize_ivi!` (`src/specification.jl`). -/
theorem initializeValueFunction_le_reachLfp (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) {prop : ReachAvoidProperty S}
    (hprop : prop.toReachProperty.isExactTime = false) :
    initializeValueFunction prop.toReachProperty ≤ reachLfp M sat strat prop.toReachProperty := by
  have h := initializeValueFunction_le_step M sat strat hprop
    (W := reachLfp M sat strat prop.toReachProperty)
    (fun s => (reachLfp_mem_unit M sat strat _ s).1)
  rwa [step_reachLfp] at h

omit [DecidableEq A] in
/-- The initial upper bound is above the exact value: `V* ≤ 1 - 𝟙_avoid` (`V*` is in `[0, 1]` and
the postprocessing sets the avoid states to `0`).

Julia counterpart: `initialize_ivi!` (`src/specification.jl`). -/
theorem reachLfp_le_initializeUpper (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (prop : ReachAvoidProperty S) :
    reachLfp M sat strat prop.toReachProperty ≤ initializeUpper prop := by
  intro s
  by_cases ha : s ∈ prop.avoid
  · have h : reachLfp M sat strat prop.toReachProperty s = 0 := by
      rw [← step_reachLfp M sat strat prop.toReachProperty]
      exact prop.stepPostprocess_avoid ha
    simp [initializeUpper, ha, h]
  · simpa [initializeUpper, ha] using (reachLfp_mem_unit M sat strat _ s).2

omit [DecidableEq A] in
/-- The robust value-iteration iterates from a sound start stay sound for `V*`, via
`Approx.iter_sound` (exact operator `VI.step` on both sides, started from its fixed point `V*`).

Julia counterpart: `step!` (`src/robust_value_iteration.jl`) iterated by
`_interval_value_iteration!` (`src/interval_value_iteration.jl`). -/
theorem iterate_sound (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) (m : SatisfactionMode) {W₀ : S → ℝ}
    (h₀ : Sound m W₀ (reachLfp M sat strat prop)) (k : ℕ) :
    Sound m ((VI.step M sat strat prop)^[k] W₀) (reachLfp M sat strat prop) := by
  have h := (iter_sound (T' := VI.step M sat strat prop) (T := VI.step M sat strat prop)
    (fun V => Sound.refl m (VI.step M sat strat prop V)) (Or.inl (step_mono M sat strat prop))
    h₀).1 k
  rwa [Function.iterate_fixed (step_reachLfp M sat strat prop)] at h

/-- **The primary bound is sound** (A6, the returned `value_function`): `Sound sat V_primary_k V*`,
i.e. `V_lower_k ≤ V*` for `Pessimistic` and `V* ≤ V_upper_k` for `Optimistic`, for every call `k`,
all four modes, both strategy caches, non-exact-time reach-avoid properties. Proved via
`Approx.iter_sound` (`iterate_sound`): the primary iterates are robust value iteration
(`primary_iviIter`).

Julia counterpart: `value_function` of the solution returned by `solve(…,
::IntervalValueIteration)` (`_ivi_verification_solution`, `src/interval_value_iteration.jl`). -/
theorem primary_sound (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    {prop : ReachAvoidProperty S} (hprop : prop.toReachProperty.isExactTime = false)
    (o : ActionOrder M) (kind : StrategyCacheKind) (k : ℕ) :
    Sound sat (primary sat (iviIter M sat strat prop o kind k))
      (reachLfp M sat strat prop.toReachProperty) := by
  rw [primary_iviIter]
  refine iterate_sound M sat strat _ sat ?_ k
  cases sat
  · exact initializeValueFunction_le_reachLfp M _ strat hprop
  · exact reachLfp_le_initializeUpper M _ strat prop

omit [DecidableEq S] in
/-- Policy evaluation of the synthesized strategy on the lower bound is at most the Bellman update
when `sat = pessimistic` (the lower bound is primary, so they are equal) or `strat = maximize` (a
fixed valid strategy is never better than the maximum, `Tπ_stepSound`).

Julia counterpart: the two `bellman!` calls of `ivi_step!` (`src/interval_value_iteration.jl`). -/
theorem Tπ_iviStrategy_lower_le (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (kind : StrategyCacheKind) {B : Bounds S A}
    (hB : ∀ s, B.strategy s ∈ M.available s)
    (hmode : sat = .pessimistic ∨ strat = .maximize) :
    Tπ M sat (iviStrategy M sat strat o kind B) B.lower ≤ T M sat strat B.lower := by
  obtain ⟨hvalid, hprim⟩ := iviStrategy_spec M sat strat o kind hB
  rcases hmode with rfl | rfl
  · exact hprim.le
  · exact Tπ_stepSound M sat .maximize hvalid B.lower

omit [DecidableEq S] in
/-- The Bellman update of the upper bound is at most the policy evaluation of the synthesized
strategy when `sat = optimistic` (the upper bound is primary, so they are equal) or
`strat = minimize` (a fixed valid strategy is never better than the minimum, `Tπ_stepSound`).

Julia counterpart: the two `bellman!` calls of `ivi_step!` (`src/interval_value_iteration.jl`). -/
theorem T_le_Tπ_iviStrategy_upper (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (kind : StrategyCacheKind) {B : Bounds S A}
    (hB : ∀ s, B.strategy s ∈ M.available s)
    (hmode : sat = .optimistic ∨ strat = .minimize) :
    T M sat strat B.upper ≤ Tπ M sat (iviStrategy M sat strat o kind B) B.upper := by
  obtain ⟨hvalid, hprim⟩ := iviStrategy_spec M sat strat o kind hB
  rcases hmode with rfl | rfl
  · exact hprim.ge
  · exact Tπ_stepSound M sat .minimize hvalid B.upper

/-- When `sat = pessimistic` or `strat = maximize`, the lower bound stays below the robust
value-iteration iterates from `𝟙_reach`.

Julia counterpart: `ivi_step!` (`src/interval_value_iteration.jl`) versus `step!`
(`src/robust_value_iteration.jl`). -/
theorem lower_le_iterate (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind)
    (hmode : sat = .pessimistic ∨ strat = .maximize) (k : ℕ) :
    (iviIter M sat strat prop o kind k).lower ≤
      (VI.step M sat strat prop.toReachProperty)^[k] (initializeIvi o prop).lower := by
  induction k with
  | zero => exact le_rfl
  | succ k ih =>
    rw [Function.iterate_succ_apply']
    show (step M sat strat prop o kind (iviIter M sat strat prop o kind k)).lower ≤ _
    rw [step_lower]
    exact stepPostprocessValueFunction_mono _
      ((Tπ_iviStrategy_lower_le M sat strat o kind
        (iviIter_strategy_mem M sat strat prop o kind k) hmode).trans (T_mono M sat strat ih))

/-- When `sat = optimistic` or `strat = minimize`, the upper bound stays above the robust
value-iteration iterates from `1 - 𝟙_avoid`.

Julia counterpart: `ivi_step!` (`src/interval_value_iteration.jl`) versus `step!`
(`src/robust_value_iteration.jl`). -/
theorem iterate_le_upper (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind)
    (hmode : sat = .optimistic ∨ strat = .minimize) (k : ℕ) :
    (VI.step M sat strat prop.toReachProperty)^[k] (initializeIvi o prop).upper ≤
      (iviIter M sat strat prop o kind k).upper := by
  induction k with
  | zero => exact le_rfl
  | succ k ih =>
    rw [Function.iterate_succ_apply']
    show _ ≤ (step M sat strat prop o kind (iviIter M sat strat prop o kind k)).upper
    rw [step_upper]
    exact stepPostprocessValueFunction_mono _ ((T_mono M sat strat ih).trans
      (T_le_Tπ_iviStrategy_upper M sat strat o kind
        (iviIter_strategy_mem M sat strat prop o kind k) hmode))

/-- **Lower half of the bracket**: `V_lower_k ≤ V*` for every call `k` when `sat = pessimistic` or
`strat = maximize` (three of the four modes; it fails for `(Optimistic, Minimize)`, Finding F4,
`Examples.ivi_reachLfp_lt_lower`). Non-exact-time reach-avoid, both strategy caches. The exact
part goes through `Approx.iter_sound` (`iterate_sound`).

Julia counterpart: `V_lower` of `_interval_value_iteration!` (`src/interval_value_iteration.jl`). -/
theorem lower_le_reachLfp (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    {prop : ReachAvoidProperty S} (hprop : prop.toReachProperty.isExactTime = false)
    (o : ActionOrder M) (kind : StrategyCacheKind)
    (hmode : sat = .pessimistic ∨ strat = .maximize) (k : ℕ) :
    (iviIter M sat strat prop o kind k).lower ≤ reachLfp M sat strat prop.toReachProperty :=
  (lower_le_iterate M sat strat prop o kind hmode k).trans
    (iterate_sound M sat strat _ .pessimistic
      (initializeValueFunction_le_reachLfp M sat strat hprop) k)

/-- **Upper half of the bracket**: `V* ≤ V_upper_k` for every call `k` when `sat = optimistic` or
`strat = minimize` (three of the four modes; it fails for `(Pessimistic, Maximize)`, Finding F4,
`Examples.ivi_upper_lt_reachLfp`). All reach-avoid properties, both strategy caches. The exact part
goes through `Approx.iter_sound` (`iterate_sound`).

Julia counterpart: `V_upper` of `_interval_value_iteration!` (`src/interval_value_iteration.jl`). -/
theorem reachLfp_le_upper (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind)
    (hmode : sat = .optimistic ∨ strat = .minimize) (k : ℕ) :
    reachLfp M sat strat prop.toReachProperty ≤ (iviIter M sat strat prop o kind k).upper :=
  le_trans (α := S → ℝ)
    (iterate_sound M sat strat _ .optimistic (reachLfp_le_initializeUpper M sat strat prop) k)
    (iterate_le_upper M sat strat prop o kind hmode k)

/-- **Bracket, restricted to the aligned modes** (A6, strongest proved form): for
`(Pessimistic, Minimize)` and `(Optimistic, Maximize)`, `V_lower_k ≤ V* ≤ V_upper_k` for every call
`k`, both strategy caches, non-exact-time reach-avoid properties. In these modes the strategy is
synthesized on the bound that the strategy-mode direction keeps sound. The unrestricted bracket
(`IVI.bracket`, all four modes) is false for Julia's coupling: Finding F4.

Julia counterpart: `_interval_value_iteration!` / `ivi_step!`
(`src/interval_value_iteration.jl`). -/
theorem bracket_aligned (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    {prop : ReachAvoidProperty S} (hprop : prop.toReachProperty.isExactTime = false)
    (o : ActionOrder M) (kind : StrategyCacheKind)
    (hmode : (sat = .pessimistic ∧ strat = .minimize) ∨ (sat = .optimistic ∧ strat = .maximize))
    (k : ℕ) :
    (iviIter M sat strat prop o kind k).lower ≤ reachLfp M sat strat prop.toReachProperty ∧
      reachLfp M sat strat prop.toReachProperty ≤ (iviIter M sat strat prop o kind k).upper := by
  rcases hmode with ⟨hs, hm⟩ | ⟨hs, hm⟩
  · exact ⟨lower_le_reachLfp M sat strat hprop o kind (Or.inl hs) k,
      reachLfp_le_upper M sat strat prop o kind (Or.inr hm) k⟩
  · exact ⟨lower_le_reachLfp M sat strat hprop o kind (Or.inr hm) k,
      reachLfp_le_upper M sat strat prop o kind (Or.inl hs) k⟩

/-! ### Stopping on the initial-state gap (Phase 4b) -/

omit [Fintype S] [DecidableEq S] [DecidableEq A] in
/-- The gap `V_upper(s) - V_lower(s)` between the two bounds at state `s`.

Julia counterpart: `d = V_upper[ci] - V_lower[ci]` in `max_initial_gap`, and the returned `gap`
(`gap .= V_upper.current .- V_lower.current`) of `_interval_value_iteration!`
(`src/interval_value_iteration.jl`). -/
def gap (B : Bounds S A) (s : S) : ℝ := B.upper s - B.lower s

omit [Fintype S] [DecidableEq S] [DecidableEq A] in
/-- One iteration of the loop of `max_initial_gap`: `if d > diff; diff = d; end` with
`d = V_upper[s] - V_lower[s]`.

Julia counterpart: the loop body of `max_initial_gap` (`src/interval_value_iteration.jl`). -/
noncomputable def maxGapStep (B : Bounds S A) (diff : ℝ) (s : S) : ℝ :=
  if gap B s > diff then gap B s else diff

omit [Fintype S] [DecidableEq S] [DecidableEq A] in
/-- The largest gap over the initial states, floored at `0`: a transcription of the loop of
`max_initial_gap` (`diff = 0`, then `maxGapStep` for each `s in initial`, in order). The initial
states are the list `initial_states(mp)`; for `AllStates()` Julia loops over every state, which is
`initial = (Finset.univ : Finset S).toList` here.

Julia counterpart: `max_initial_gap(V_lower, V_upper, initial_states(mp))`
(`src/interval_value_iteration.jl`). -/
noncomputable def maxInitialGap (B : Bounds S A) (initial : List S) : ℝ :=
  initial.foldl (maxGapStep B) 0

omit [Fintype S] [DecidableEq S] [DecidableEq A] in
/-- The termination test of IVI for infinite-time reach-avoid: the largest initial-state gap is
strictly below the tolerance, `max_{s ∈ initial} (V_upper(s) - V_lower(s)) < tol` (absolute, not
relative; floored at `0`). Julia's `tol` is `convergence_eps(prop)`.

Julia counterpart: `IVIInitialGapCriteria` and its call operator
`(f::IVIInitialGapCriteria)(V_lower, V_upper, k, mp)` (`src/interval_value_iteration.jl`), chosen
by `ivi_termination_criteria(prop, Val(false))` for `InfiniteTimeReachAvoid`. -/
def iviInitialGapCriteria (tol : ℝ) (initial : List S) (B : Bounds S A) : Prop :=
  maxInitialGap B initial < tol

omit [Fintype S] [DecidableEq S] [DecidableEq A] in
/-- The criterion is decidable (a comparison of reals), so the loop exit `stopIndex` is the least
call that satisfies it.

Julia counterpart: the `Bool` returned by `(f::IVIInitialGapCriteria)(V_lower, V_upper, k, mp)`
(`src/interval_value_iteration.jl`). -/
noncomputable instance iviInitialGapCriteria.decidable (tol : ℝ) (initial : List S)
    (B : Bounds S A) : Decidable (iviInitialGapCriteria tol initial B) :=
  inferInstanceAs (Decidable (maxInitialGap B initial < tol))

/-- The loop of `_interval_value_iteration!` exits under `IVIInitialGapCriteria`: some call
`k + 1 ≥ 1` satisfies the criterion (Julia tests the criterion only after `ivi_step!`, never on the
initialised bounds `iviIter 0`).

Julia counterpart: termination of `while !term_criteria(V_lower, V_upper, k, mp)` in
`_interval_value_iteration!` (`src/interval_value_iteration.jl`). -/
def Terminates (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachAvoidProperty S) (o : ActionOrder M) (kind : StrategyCacheKind) (tol : ℝ)
    (initial : List S) : Prop :=
  ∃ k, iviInitialGapCriteria tol initial (iviIter M sat strat prop o kind (k + 1))

/-- The value of the Julia counter `k` when the loop exits: the least `k ≥ 1` at which the
criterion holds. Transcription of

    ivi_step!(…, 0, …); k = 1
    while !term_criteria(V_lower, V_upper, k, mp)
        nextiteration!(…); ivi_step!(…, k, …); k += 1
    end

where after the `k`-th call of `ivi_step!` the bounds are `iviIter k`: the loop tests `k = 1, 2, …`
in order and stops at the first success, so the exit value is `Nat.find h + 1`.

Julia counterpart: `k` returned by `_interval_value_iteration!` (`num_iterations` of the solution,
`src/interval_value_iteration.jl`). -/
noncomputable def stopIndex {M : RMDP S A} {sat : SatisfactionMode} {strat : StrategyMode}
    {prop : ReachAvoidProperty S} {o : ActionOrder M} {kind : StrategyCacheKind} {tol : ℝ}
    {initial : List S} (h : Terminates M sat strat prop o kind tol initial) : ℕ :=
  Nat.find h + 1

/-- The value function IVI returns: the primary bound (`V_lower` for `Pessimistic`, `V_upper` for
`Optimistic`) after the last call, `iviIter (stopIndex h)`. The final
`postprocess_value_function!` is a no-op for reach-avoid properties (`AbstractReachability`), so it
is not modelled.

Julia counterpart: `value_function(solve(problem, ::IntervalValueIteration))` via
`_ivi_verification_solution` / `_ivi_control_synthesis_solution`
(`src/interval_value_iteration.jl`). -/
noncomputable def iviValueFunction {M : RMDP S A} {sat : SatisfactionMode} {strat : StrategyMode}
    {prop : ReachAvoidProperty S} {o : ActionOrder M} {kind : StrategyCacheKind} {tol : ℝ}
    {initial : List S} (h : Terminates M sat strat prop o kind tol initial) : S → ℝ :=
  primary sat (iviIter M sat strat prop o kind (stopIndex h))

omit [Fintype S] [DecidableEq S] [DecidableEq A] in
/-- `V` is within `tol` of `W` on the initial states: `abs (V s - W s) < tol` for every
`s ∈ initial`.

Julia counterpart: the guarantee "the gap between the upper and lower bound over the set of initial
states drops below `convergence_eps`" in the `solve(…, ::IntervalValueIteration)` docstring
(`src/interval_value_iteration.jl`). -/
def WithinOn (initial : List S) (tol : ℝ) (V W : S → ℝ) : Prop :=
  ∀ s ∈ initial, abs (V s - W s) < tol

omit [Fintype S] [DecidableEq S] [DecidableEq A] in
/-- The loop of `max_initial_gap` never decreases `diff`, and ends above the gap of every state it
visits.

Julia counterpart: `max_initial_gap` (`src/interval_value_iteration.jl`). -/
theorem le_foldl_maxGapStep (B : Bounds S A) (initial : List S) (d : ℝ) :
    d ≤ initial.foldl (maxGapStep B) d ∧
      ∀ s ∈ initial, gap B s ≤ initial.foldl (maxGapStep B) d := by
  induction initial generalizing d with
  | nil => simp
  | cons t l ih =>
    have hstep : d ≤ maxGapStep B d t ∧ gap B t ≤ maxGapStep B d t := by
      unfold maxGapStep
      split_ifs with h
      · exact ⟨h.le, le_rfl⟩
      · exact ⟨le_rfl, not_lt.mp h⟩
    obtain ⟨ih₁, ih₂⟩ := ih (maxGapStep B d t)
    refine ⟨hstep.1.trans ih₁, ?_⟩
    intro s hs
    rcases List.mem_cons.mp hs with rfl | hs
    · exact hstep.2.trans ih₁
    · exact ih₂ s hs

omit [Fintype S] [DecidableEq S] [DecidableEq A] in
/-- The gap at every initial state is at most `maxInitialGap`.

Julia counterpart: `max_initial_gap` (`src/interval_value_iteration.jl`). -/
theorem gap_le_maxInitialGap (B : Bounds S A) {initial : List S} {s : S} (hs : s ∈ initial) :
    gap B s ≤ maxInitialGap B initial :=
  (le_foldl_maxGapStep B initial 0).2 s hs

omit [Fintype S] [DecidableEq S] [DecidableEq A] in
/-- If the bounds bracket `V` and the initial-state gap is below `tol`, then the primary bound is
within `tol` of `V` on the initial states (both lie in `[V_lower(s), V_upper(s)]`, an interval of
length `< tol`).

Julia counterpart: `IVIInitialGapCriteria` (`src/interval_value_iteration.jl`). -/
theorem withinOn_of_bracket (sat : SatisfactionMode) {B : Bounds S A} {V : S → ℝ} {tol : ℝ}
    {initial : List S} (hlo : B.lower ≤ V) (hup : V ≤ B.upper)
    (hgap : iviInitialGapCriteria tol initial B) : WithinOn initial tol (primary sat B) V := by
  intro s hs
  have hg : B.upper s - B.lower s < tol :=
    lt_of_le_of_lt (gap_le_maxInitialGap B hs) hgap
  have h1 := hlo s
  have h2 := hup s
  rw [abs_lt]
  cases sat
  · exact ⟨by simp only [primary]; linarith, by simp only [primary]; linarith⟩
  · exact ⟨by simp only [primary]; linarith, by simp only [primary]; linarith⟩

/-- The loop stops after at least one call: `1 ≤ stopIndex h`.

Julia counterpart: `ivi_step!(…, 0, …); k = 1` before the `while` loop of
`_interval_value_iteration!` (`src/interval_value_iteration.jl`). -/
theorem one_le_stopIndex {M : RMDP S A} {sat : SatisfactionMode} {strat : StrategyMode}
    {prop : ReachAvoidProperty S} {o : ActionOrder M} {kind : StrategyCacheKind} {tol : ℝ}
    {initial : List S} (h : Terminates M sat strat prop o kind tol initial) :
    1 ≤ stopIndex h := by
  unfold stopIndex
  omega

/-- The criterion holds when the loop stops.

Julia counterpart: the exit test of `while !term_criteria(…)` in `_interval_value_iteration!`
(`src/interval_value_iteration.jl`). -/
theorem iviInitialGapCriteria_stopIndex {M : RMDP S A} {sat : SatisfactionMode}
    {strat : StrategyMode} {prop : ReachAvoidProperty S} {o : ActionOrder M}
    {kind : StrategyCacheKind} {tol : ℝ} {initial : List S}
    (h : Terminates M sat strat prop o kind tol initial) :
    iviInitialGapCriteria tol initial (iviIter M sat strat prop o kind (stopIndex h)) := by
  exact Nat.find_spec h

/-- The criterion fails at every earlier call `1 ≤ j < stopIndex h`, so the loop does not exit
before `stopIndex h`.

Julia counterpart: the `while !term_criteria(…)` loop of `_interval_value_iteration!`
(`src/interval_value_iteration.jl`). -/
theorem not_iviInitialGapCriteria_of_lt_stopIndex {M : RMDP S A} {sat : SatisfactionMode}
    {strat : StrategyMode} {prop : ReachAvoidProperty S} {o : ActionOrder M}
    {kind : StrategyCacheKind} {tol : ℝ} {initial : List S}
    (h : Terminates M sat strat prop o kind tol initial) {j : ℕ} (hj₁ : 1 ≤ j)
    (hj : j < stopIndex h) :
    ¬ iviInitialGapCriteria tol initial (iviIter M sat strat prop o kind j) := by
  obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
  have hi : i < Nat.find h := by
    have : i + 1 < Nat.find h + 1 := hj
    omega
  exact Nat.find_min h hi

/-- **Gap stopping in the aligned modes** (A6): for `(Pessimistic, Minimize)` and
`(Optimistic, Maximize)`, whenever the initial-state gap criterion holds at a call `k`, the primary
bound is within `tol` of `V*` on the initial states. Non-exact-time reach-avoid properties
(`prop.toReachProperty.isExactTime = false`), both strategy caches, every `k` (in particular
`k = stopIndex h`). Proof: `bracket_aligned` (via `Approx.iter_sound`) and `withinOn_of_bracket`.

Julia counterpart: `IVIInitialGapCriteria` and `_interval_value_iteration!`
(`src/interval_value_iteration.jl`). -/
theorem withinOn_of_iviInitialGapCriteria_aligned (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) {prop : ReachAvoidProperty S}
    (hprop : prop.toReachProperty.isExactTime = false) (o : ActionOrder M)
    (kind : StrategyCacheKind)
    (hmode : (sat = .pessimistic ∧ strat = .minimize) ∨ (sat = .optimistic ∧ strat = .maximize))
    {tol : ℝ} {initial : List S} {k : ℕ}
    (hgap : iviInitialGapCriteria tol initial (iviIter M sat strat prop o kind k)) :
    WithinOn initial tol (primary sat (iviIter M sat strat prop o kind k))
      (reachLfp M sat strat prop.toReachProperty) :=
  have hb := bracket_aligned M sat strat hprop o kind hmode k
  withinOn_of_bracket sat hb.1 hb.2 hgap

/-- **Gap stopping is sound in the aligned modes** (A6, strongest proved form of `gap_stop_sound`):
for `(Pessimistic, Minimize)` and `(Optimistic, Maximize)`, if IVI stops on
`IVIInitialGapCriteria(tol)`, the returned value function is within `tol` of `V*` on the initial
states, `abs (iviValueFunction h s - V* s) < tol`. Non-exact-time reach-avoid properties
(`prop.toReachProperty.isExactTime = false`), both strategy caches, a general `RMDP`. The
four-mode statement `IVI.gap_stop_sound` is false for Julia's strategy coupling (Finding F4,
witnesses `Examples.not_gap_stop_sound_pessimistic_maximize` and
`Examples.not_gap_stop_sound_optimistic_minimize`) and is not stated.

Julia counterpart: `solve(problem, ::IntervalValueIteration)` with `IVIInitialGapCriteria`
(`src/interval_value_iteration.jl`). -/
theorem gap_stop_sound_aligned (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    {prop : ReachAvoidProperty S} (hprop : prop.toReachProperty.isExactTime = false)
    (o : ActionOrder M) (kind : StrategyCacheKind)
    (hmode : (sat = .pessimistic ∧ strat = .minimize) ∨ (sat = .optimistic ∧ strat = .maximize))
    {tol : ℝ} {initial : List S} (h : Terminates M sat strat prop o kind tol initial) :
    WithinOn initial tol (iviValueFunction h) (reachLfp M sat strat prop.toReachProperty) :=
  withinOn_of_iviInitialGapCriteria_aligned M sat strat hprop o kind hmode
    (iviInitialGapCriteria_stopIndex h)

/-- **The returned value is sound in all four modes** (A6, the part of `gap_stop_sound` that holds
for every mode): if IVI stops on `IVIInitialGapCriteria(tol)`, the returned value function is a
sound one-sided bound, `Sound sat (iviValueFunction h) V*` (`≤ V*` for `Pessimistic`, `≥ V*` for
`Optimistic`), on all states. Non-exact-time reach-avoid properties
(`prop.toReachProperty.isExactTime = false`), both strategy caches. It gives no two-sided `tol`
guarantee in `(Pessimistic, Maximize)` and `(Optimistic, Minimize)` (Finding F4). Via
`primary_sound` (`Approx.iter_sound`).

Julia counterpart: `solve(problem, ::IntervalValueIteration)` with `IVIInitialGapCriteria`
(`src/interval_value_iteration.jl`). -/
theorem gap_stop_primary_sound (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    {prop : ReachAvoidProperty S} (hprop : prop.toReachProperty.isExactTime = false)
    (o : ActionOrder M) (kind : StrategyCacheKind) {tol : ℝ} {initial : List S}
    (h : Terminates M sat strat prop o kind tol initial) :
    Sound sat (iviValueFunction h) (reachLfp M sat strat prop.toReachProperty) :=
  primary_sound M sat strat hprop o kind (stopIndex h)

end IntervalMDP.IVI
