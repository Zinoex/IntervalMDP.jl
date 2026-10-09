import IntervalMDPProofs.VI.Reach

/-!
# Robust value iteration: safety (the −1/+1 shift)

IntervalMDP.jl computes safety with a shifted value function (`src/specification.jl`,
lines 803–813, `AbstractSafety`):

    initialize!(value_function, prop::AbstractSafety)          # current[avoid] .= -1.0
    step_postprocess_value_function!(value_function, prop)      # current[avoid] .= -1.0
    postprocess_value_function!(value_function, ::AbstractSafety)  # current .+= 1.0

inside the same `_value_iteration!` loop as reachability (`src/robust_value_iteration.jl`,
lines 170–206; `step!` at 236–249 is one `bellman!` call with `upper_bound = isoptimistic(spec)`,
`maximize = ismaximize(spec)`, then `step_postprocess_value_function!`). The array starts at `0`
(`ValueFunction`), so before the final `+ 1` the iterate is `0` on safe states and `-1` on `avoid`;
the satisfaction and strategy modes are passed to `bellman!` unchanged (no dualisation).

This file transcribes that loop as `safetyIter` (the shifted iterate, before the final `+ 1`) and
`SafetyProperty.postprocessValueFunction` (the final `+ 1`), and proves

* `safety_shift_eq`: `safetyIter k + 1` equals the reachability iteration `reachIter`
  (`VI/Reach.lean`) of the exact-time reach-avoid property with `reach = avoidᶜ` and the same
  `avoid`, i.e. the unshifted safety recursion `V₀ = 𝟙_{avoidᶜ}`, `V_{k+1} = T V_k` off `avoid`,
  `0` on `avoid` — for all four satisfaction × strategy modes and a general `RMDP`. The proof is
  translation equivariance of the Bellman operator, `Bellman.T_add_const`.

Not modelled (as in `VI/Reach.lean`): time-varying models and strategies, floating point, threads
and CUDA. No convergence or stopping (A4) claim is made for safety in this sub-phase.
-/

namespace IntervalMDP.VI

open Bellman

/-- A safety property: the set `avoid` of states that must never be visited. The Julia time
horizon and convergence threshold only enter the termination criterion, not the iteration map, so
finite- and infinite-time safety share this structure (`Property.toSafetyProperty`).

Julia counterpart: the concrete subtypes of `AbstractSafety` (`src/specification.jl`):
`FiniteTimeSafety(avoid, time_horizon)` and `InfiniteTimeSafety(avoid, convergence_eps)`. -/
structure SafetyProperty (S : Type*) where
  /-- The avoid set, Julia `avoid(prop)` (`src/specification.jl`). -/
  avoid : Finset S

/-- The safety property behind a `Property`, or `none` for the other properties.

Julia counterpart: dispatch of `initialize!` / `step_postprocess_value_function!` /
`postprocess_value_function!` on `AbstractSafety` (`src/specification.jl`); time horizon and
`convergence_eps` are dropped (they only enter `termination_criteria`,
`src/robust_value_iteration.jl`). -/
def _root_.IntervalMDP.Property.toSafetyProperty {S Q : Type*} :
    Property S Q → Option (SafetyProperty S)
  | .finiteTimeSafety avoid .. | .infiniteTimeSafety avoid .. => some ⟨avoid⟩
  | _ => none

namespace SafetyProperty

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- The shifted initial value function: `-1` on `avoid`, `0` elsewhere.

Julia counterpart: `ValueFunction(problem)` (all zeros, `src/robust_value_iteration.jl`) followed
by `initialize!(value_function, prop::AbstractSafety)`, `current[avoid(prop)] .= -1.0`
(`src/specification.jl`). -/
def initializeValueFunction (prop : SafetyProperty S) : S → ℝ :=
  fun s => if s ∈ prop.avoid then -1 else 0

/-- The shifted postprocessing after each Bellman step: `V[avoid] .= -1`.

Julia counterpart: `step_postprocess_value_function!(value_function, prop::AbstractSafety)`,
`current[avoid(prop)] .= -1.0` (`src/specification.jl`). -/
def stepPostprocessValueFunction (prop : SafetyProperty S) (V : S → ℝ) : S → ℝ :=
  fun s => if s ∈ prop.avoid then -1 else V s

/-- The final postprocessing after the loop, undoing the shift: `V .+= 1`.

Julia counterpart: `postprocess_value_function!(value_function, ::AbstractSafety)`,
`current .+= 1.0` (`src/specification.jl`), called once after the `while` loop of
`_value_iteration!` (`src/robust_value_iteration.jl`). -/
def postprocessValueFunction (_prop : SafetyProperty S) (V : S → ℝ) : S → ℝ :=
  V + Function.const S 1

/-- The unshifted reachability form of a safety property: exact-time reach-avoid with
`reach = avoidᶜ` and the same `avoid` (`ℙ[ω[k] ∉ O ∧ ∀ k' < k, ω[k'] ∉ O]`, which is the
`k`-step safety probability). Its iteration (`reachIter`, `VI/Reach.lean`) starts at
`𝟙_{avoidᶜ}`, applies `T` and resets `avoid` to `0`, without resetting `reach`.

Julia counterpart: `ExactTimeReachAvoid(reach, avoid, time_horizon)` (`src/specification.jl`) with
`reach` the complement of `avoid(prop)` of an `AbstractSafety` property. Julia does not build this
property; it is the reference the shifted safety iteration is compared with. -/
def toReachProperty (prop : SafetyProperty S) : ReachProperty S :=
  .exactTimeReachAvoid prop.avoidᶜ prop.avoid disjoint_compl_left

end SafetyProperty

variable {S A : Type*} [Fintype S] [DecidableEq S]

/-- One shifted safety value-iteration step: the robust Bellman update `T` followed by
`V[avoid] .= -1`.

Julia counterpart: `step!(workspace, strategy_cache, value_function, k, mp, spec)`
(`src/robust_value_iteration.jl`) with an `AbstractSafety` specification: `bellman!(…;
upper_bound = isoptimistic(spec), maximize = ismaximize(spec))` then
`step_postprocess_value_function!` (`src/specification.jl`). The model is time-invariant here. -/
noncomputable def safetyStep (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : SafetyProperty S) (V : S → ℝ) : S → ℝ :=
  prop.stepPostprocessValueFunction (T M sat strat V)

/-- The shifted safety value-iteration iterates, a transcription of the loop of
`_value_iteration!` for an `AbstractSafety` specification: `safetyIter 0` is the initialised
(shifted) value function, and `safetyIter (k + 1)` is `safetyStep` applied to `safetyIter k`
(Julia: `nextiteration!`, `step!(…, k, …)`, `k += 1`). The Julia counter `k` after the loop equals
the Lean index; the value Julia returns is
`prop.postprocessValueFunction (safetyIter M sat strat prop k)` (the final `current .+= 1.0`).

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) with
`FiniteTimeSafety` / `InfiniteTimeSafety` (`src/specification.jl`). -/
noncomputable def safetyIter (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : SafetyProperty S) : ℕ → S → ℝ
  | 0 => prop.initializeValueFunction
  | k + 1 => safetyStep M sat strat prop (safetyIter M sat strat prop k)

/-- **The −1/+1 shift.** Undoing the shift (`+ 1`, Julia's `postprocess_value_function!`) on the
`k`-th shifted safety iterate gives the `k`-th iterate of the unshifted safety recursion, i.e. of
`reachIter` for the exact-time reach-avoid property `reach = avoidᶜ`, `avoid`
(`SafetyProperty.toReachProperty`). Holds for all four satisfaction × strategy modes (Julia passes
them to `bellman!` unchanged) and a general `RMDP`; proved with `Bellman.T_add_const`.

Julia counterpart: `initialize!`, `step_postprocess_value_function!` and
`postprocess_value_function!` for `AbstractSafety` (`src/specification.jl`) inside
`_value_iteration!` (`src/robust_value_iteration.jl`). -/
theorem safety_shift_eq (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : SafetyProperty S) (k : ℕ) :
    prop.postprocessValueFunction (safetyIter M sat strat prop k) =
      reachIter M sat strat prop.toReachProperty k := by
  induction k with
  | zero =>
    funext s
    by_cases hs : s ∈ prop.avoid <;>
      simp [safetyIter, reachIter, SafetyProperty.postprocessValueFunction,
        SafetyProperty.initializeValueFunction, initializeValueFunction,
        SafetyProperty.toReachProperty, ReachProperty.reach, hs]
  | succ k ih =>
    funext s
    have hT := congrFun (T_add_const M sat strat (safetyIter M sat strat prop k) 1) s
    simp only [SafetyProperty.postprocessValueFunction] at ih hT
    rw [ih] at hT
    change _ = step M sat strat prop.toReachProperty (reachIter M sat strat _ k) s
    generalize reachIter M sat strat prop.toReachProperty k = R at hT ⊢
    by_cases hs : s ∈ prop.avoid
    · simp [safetyIter, safetyStep, step, SafetyProperty.postprocessValueFunction,
        SafetyProperty.stepPostprocessValueFunction, stepPostprocessValueFunction,
        SafetyProperty.toReachProperty, hs]
    · simp [safetyIter, safetyStep, step, SafetyProperty.postprocessValueFunction,
        SafetyProperty.stepPostprocessValueFunction, stepPostprocessValueFunction,
        SafetyProperty.toReachProperty, hs, hT]

/-- The shifted safety iterate is the unshifted one minus `1`: Julia's
`value_function.current` before `postprocess_value_function!` equals
`reachIter M sat strat prop.toReachProperty k - 1` (all four modes).

Julia counterpart: `value_function.current` in `_value_iteration!`
(`src/robust_value_iteration.jl`) for an `AbstractSafety` property (`src/specification.jl`). -/
theorem safetyIter_eq_sub_one (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : SafetyProperty S) (k : ℕ) :
    safetyIter M sat strat prop k =
      reachIter M sat strat prop.toReachProperty k - Function.const S 1 := by
  rw [← safety_shift_eq]
  funext s
  simp [SafetyProperty.postprocessValueFunction]

/-- The value returned by Julia for safety (after the final `+ 1`) lies in `[0, 1]` (all four
modes), from `safety_shift_eq` and `reachIter_mem_unit`.

Julia counterpart: the `value_function.current` returned by `_value_iteration!`
(`src/robust_value_iteration.jl`) for `FiniteTimeSafety` / `InfiniteTimeSafety`
(`src/specification.jl`). -/
theorem safetyIter_postprocess_mem_unit (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (prop : SafetyProperty S) (k : ℕ) (s : S) :
    prop.postprocessValueFunction (safetyIter M sat strat prop k) s ∈ Set.Icc (0 : ℝ) 1 := by
  rw [safety_shift_eq]
  exact reachIter_mem_unit M sat strat _ k s

end IntervalMDP.VI
