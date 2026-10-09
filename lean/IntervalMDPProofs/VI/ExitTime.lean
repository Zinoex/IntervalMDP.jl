import IntervalMDPProofs.VI.Reach
import Mathlib.Data.ENNReal.Real

/-!
# Robust value iteration: expected exit time

For `ExpectedExitTime(avoid_states, convergence_eps)` (`src/specification.jl`, lines 1092–1134)
IntervalMDP.jl runs the `_value_iteration!` loop (`src/robust_value_iteration.jl`, lines 170–206;
`step!` at 236–249) with

    initialize!(value_function, prop::ExpectedExitTime)        # current .= 1.0
                                                               # current[avoid(prop)] .= 0.0
    step_postprocess_value_function!(value_function, prop)     # current .+= 1.0
                                                               # current[avoid(prop)] .= 0.0
    postprocess_value_function!(value_function, ::AbstractHittingTime) = value_function  # no-op

so iterate `k` is the robust expected number of steps (capped at `k + 1`) spent outside `avoid`
before entering it: `V₀ = 𝟙_{avoidᶜ}` and `V_{k+1} = 0` on `avoid`, `T V_k + 1` elsewhere.

This file transcribes that loop as `exitIter` and proves, for all four satisfaction × strategy
modes and a general `RMDP`:

* `exitIter_succ`: the one-step recursion, as Julia computes it;
* `exitIter_mono`: the iterates are non-decreasing in `k`;
* `exitIter_sound` (A4): every iterate lies below every nonnegative real super-solution `W`
  (`exitStep W ≤ W`) of the exit-time Bellman equation, via `Approx.iter_sound`.

**The true value.** Expected exit times may be infinite (a strategy/adversary pair that never
leaves the safe set), so the target is the limit of the iterates in `ℝ≥0∞`, `exitValue`
(`⨆ k, V_k`, the Kleene limit of the dynamic-programming recursion from `0`, since
`V₀ = exitStep 0`). It lies above every iterate (`exitIter_le_exitValue`) and below every
nonnegative real super-solution (`exitValue_le_of_step_le`). **No convergence is claimed**: neither
that `exitValue` is finite nor that the iterates converge in `ℝ`, nor any error bound for Julia's
`convergence_eps` stopping criterion.

**Direction in optimistic mode.** As in `VI/Reach.lean`, `exitIter_sound` states
`Sound .pessimistic V_k W` (that is, `V_k ≤ W`) in every mode. For `sat = pessimistic` this is
`Sound sat`, the conservative direction. For `sat = optimistic` it is still a lower bound of the
optimistic value, which is **not** the conservative direction `Sound .optimistic`; no such claim is
made.

Not modelled: time-varying models and strategies, floating point (including overflow of diverging
values), threads and CUDA.
-/

namespace IntervalMDP.VI

open Bellman Approx

/-- An expected-exit-time property: the set `avoidStates` of unsafe states whose first hitting
time is measured.

Julia counterpart: `ExpectedExitTime(avoid_states, convergence_eps)` (`src/specification.jl`);
the field is `avoid_states`, read through `avoid(prop)`. `convergence_eps` only enters the
termination criterion, not the iteration map, and is dropped. -/
structure ExpectedExitTime (S : Type*) where
  /-- The unsafe set, Julia field `avoid_states` / accessor `avoid(prop)`
  (`src/specification.jl`). -/
  avoidStates : Finset S

/-- The expected-exit-time property behind a `Property`, or `none` for the other properties.

Julia counterpart: dispatch of `initialize!` / `step_postprocess_value_function!` on
`ExpectedExitTime` (`src/specification.jl`). -/
def _root_.IntervalMDP.Property.toExpectedExitTime {S Q : Type*} :
    Property S Q → Option (ExpectedExitTime S)
  | .expectedExitTime avoidStates .. => some ⟨avoidStates⟩
  | _ => none

namespace ExpectedExitTime

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- The initial value function: `1` everywhere (`current .= 1.0`), then `0` on `avoid`
(`current[avoid(prop)] .= 0.0`).

Julia counterpart: `initialize!(value_function, prop::ExpectedExitTime)`
(`src/specification.jl`). -/
def initializeValueFunction (prop : ExpectedExitTime S) : S → ℝ :=
  fun s => if s ∈ prop.avoidStates then 0 else 1

/-- The postprocessing after each Bellman step, in Julia's statement order: first add one step
(`current .+= 1.0`), then reset the unsafe states (`current[avoid(prop)] .= 0.0`).

Julia counterpart: `step_postprocess_value_function!(value_function, prop::ExpectedExitTime)`
(`src/specification.jl`). -/
def stepPostprocessValueFunction (prop : ExpectedExitTime S) (V : S → ℝ) : S → ℝ :=
  fun s => if s ∈ prop.avoidStates then 0 else (V + Function.const S 1 : S → ℝ) s

end ExpectedExitTime

variable {S A : Type*} [Fintype S] [DecidableEq S]

/-- One expected-exit-time value-iteration step: the robust Bellman update `T` followed by
`current .+= 1.0; current[avoid] .= 0.0`.

Julia counterpart: `step!(workspace, strategy_cache, value_function, k, mp, spec)`
(`src/robust_value_iteration.jl`) with an `ExpectedExitTime` specification: `bellman!(…;
upper_bound = isoptimistic(spec), maximize = ismaximize(spec))` then
`step_postprocess_value_function!` (`src/specification.jl`). The model is time-invariant here. -/
noncomputable def exitStep (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) (V : S → ℝ) : S → ℝ :=
  prop.stepPostprocessValueFunction (T M sat strat V)

/-- The expected-exit-time value-iteration iterates, a transcription of the loop of
`_value_iteration!`: `exitIter 0` is the initialised value function, and `exitIter (k + 1)` is
`exitStep` applied to `exitIter k` (Julia: `nextiteration!`, `step!(…, k, …)`, `k += 1`). The
Julia counter `k` after the loop equals the Lean index; `postprocess_value_function!` is the
identity for `AbstractHittingTime`, so the returned `value_function.current` is `exitIter k` for
the `k` at which `term_criteria` first holds (`CovergenceCriteria`: residual `< convergence_eps`).

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) with an
`ExpectedExitTime` specification (`src/specification.jl`). -/
noncomputable def exitIter (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) : ℕ → S → ℝ
  | 0 => prop.initializeValueFunction
  | k + 1 => exitStep M sat strat prop (exitIter M sat strat prop k)

/-- The exact robust expected exit time in `ℝ≥0∞`: the supremum (= limit, `exitIter_mono`) of
the iterates, possibly `∞`. Since `exitIter 0 = exitStep 0`
(`initializeValueFunction_eq_exitStep_zero`), this is the Kleene limit `⨆ k, exitStep^[k + 1] 0`
of the dynamic-programming recursion; it lies below every nonnegative real super-solution
(`exitValue_le_of_step_le`). No finiteness is claimed.

Julia counterpart: the value that `_value_iteration!` (`src/robust_value_iteration.jl`)
approximates for `ExpectedExitTime` (`src/specification.jl`), `𝔼^{π,η}_exit(O)` in its docstring. -/
noncomputable def exitValue (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) : S → ENNReal :=
  fun s => ⨆ k, ENNReal.ofReal (exitIter M sat strat prop k s)

/-- `exitIter` is the `k`-fold iterate of `exitStep` from the initial value function.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`). -/
theorem exitIter_eq_iterate (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) (k : ℕ) :
    exitIter M sat strat prop k = (exitStep M sat strat prop)^[k] prop.initializeValueFunction := by
  induction k with
  | zero => rfl
  | succ k ih => rw [Function.iterate_succ_apply', ← ih]; rfl

/-- The Julia initialisation is one step from the zero function: `V₀ = exitStep 0` (all four
modes), because `T 0 = 0` (`T_zero`).

Julia counterpart: `initialize!(value_function, prop::ExpectedExitTime)` and
`step_postprocess_value_function!` (`src/specification.jl`). -/
theorem initializeValueFunction_eq_exitStep_zero (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (prop : ExpectedExitTime S) :
    prop.initializeValueFunction = exitStep M sat strat prop 0 := by
  funext s
  by_cases hs : s ∈ prop.avoidStates <;>
    simp [ExpectedExitTime.initializeValueFunction, exitStep,
      ExpectedExitTime.stepPostprocessValueFunction, T_zero, hs]

omit [Fintype S] in
/-- The exit-time postprocessing is monotone.

Julia counterpart: `step_postprocess_value_function!(value_function, prop::ExpectedExitTime)`
(`src/specification.jl`). -/
theorem ExpectedExitTime.stepPostprocessValueFunction_mono (prop : ExpectedExitTime S)
    {V W : S → ℝ} (h : V ≤ W) :
    prop.stepPostprocessValueFunction V ≤ prop.stepPostprocessValueFunction W := by
  intro s
  by_cases hs : s ∈ prop.avoidStates <;>
    simp [ExpectedExitTime.stepPostprocessValueFunction, hs, h s]

/-- `exitStep` is monotone (all four modes).

Julia counterpart: `step!` (`src/robust_value_iteration.jl`). -/
theorem exitStep_mono (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) : Monotone (exitStep M sat strat prop) :=
  fun _ _ h => prop.stepPostprocessValueFunction_mono (T_mono M sat strat h)

/-- **One-step recursion.** As Julia computes it: after the Bellman update, add one step and reset
the unsafe states, `V_{k+1} s = 0` for `s ∈ avoid` and `V_{k+1} s = T V_k s + 1` otherwise (all
four satisfaction × strategy modes, general `RMDP`).

Julia counterpart: `step!` (`src/robust_value_iteration.jl`) followed by
`step_postprocess_value_function!(value_function, prop::ExpectedExitTime)`,
`current .+= 1.0; current[avoid(prop)] .= 0.0` (`src/specification.jl`). -/
theorem exitIter_succ (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) (k : ℕ) (s : S) :
    exitIter M sat strat prop (k + 1) s =
      if s ∈ prop.avoidStates then 0 else T M sat strat (exitIter M sat strat prop k) s + 1 := by
  simp [exitIter, exitStep, ExpectedExitTime.stepPostprocessValueFunction]

omit [Fintype S] in
/-- The initial value function is nonnegative.

Julia counterpart: `initialize!(value_function, prop::ExpectedExitTime)`
(`src/specification.jl`). -/
theorem ExpectedExitTime.initializeValueFunction_nonneg (prop : ExpectedExitTime S) :
    0 ≤ prop.initializeValueFunction := by
  intro s
  by_cases hs : s ∈ prop.avoidStates <;>
    simp [ExpectedExitTime.initializeValueFunction, hs]

/-- Every iterate is nonnegative (all four modes).

Julia counterpart: `value_function.current` in `_value_iteration!`
(`src/robust_value_iteration.jl`) for `ExpectedExitTime` (`src/specification.jl`). -/
theorem exitIter_nonneg (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) (k : ℕ) : 0 ≤ exitIter M sat strat prop k := by
  induction k with
  | zero => exact prop.initializeValueFunction_nonneg
  | succ k ih =>
    have h := exitStep_mono M sat strat prop ih
    rw [← initializeValueFunction_eq_exitStep_zero] at h
    exact le_trans prop.initializeValueFunction_nonneg h

/-- **Monotone iterates.** The expected-exit-time iterates are non-decreasing in `k`, for all four
satisfaction × strategy modes and a general `RMDP` (they may diverge; no limit in `ℝ` is claimed).

Julia counterpart: successive `value_function.current` in `_value_iteration!`
(`src/robust_value_iteration.jl`) for `ExpectedExitTime` (`src/specification.jl`). -/
theorem exitIter_mono (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) : Monotone (exitIter M sat strat prop) := by
  have h₀ :
      prop.initializeValueFunction ≤ exitStep M sat strat prop prop.initializeValueFunction := by
    have h := exitStep_mono M sat strat prop (exitIter_nonneg M sat strat prop 0)
    rwa [← initializeValueFunction_eq_exitStep_zero] at h
  have h := (exitStep_mono M sat strat prop).monotone_iterate_of_le_map h₀
  intro m n hmn
  rw [exitIter_eq_iterate, exitIter_eq_iterate]
  exact h hmn

/-- **A4: stopping at any finite `k` gives a lower bound.** Every expected-exit-time iterate lies
below every nonnegative real super-solution `W` of the exit-time Bellman equation
(`exitStep W ≤ W`), in all four satisfaction × strategy modes — in particular at the `k` where
Julia's termination criterion stops, and below the exact value `exitValue` whenever that is finite
(`exitValue_le_of_step_le`, `exitIter_le_exitValue`). Proved through `Approx.iter_sound`. No
convergence is claimed.

Direction: the conclusion is `Sound .pessimistic`, i.e. `V_k ≤ W`. For `sat = pessimistic` this is
soundness in the sense of `Sound sat`. For `sat = optimistic` the iterates are still lower bounds,
which is **not** the conservative direction (`Sound .optimistic` needs `W ≤ V_k`); not claimed.

Julia counterpart: the value returned by `_value_iteration!` (`src/robust_value_iteration.jl`)
for `ExpectedExitTime` (`src/specification.jl`) when `term_criteria` stops it. -/
theorem exitIter_sound (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) (k : ℕ) {W : S → ℝ} (hW : 0 ≤ W)
    (hstep : exitStep M sat strat prop W ≤ W) :
    Sound .pessimistic (exitIter M sat strat prop k) W := by
  have hmono := exitStep_mono M sat strat prop
  have h₀ : Sound .pessimistic prop.initializeValueFunction W := by
    rw [initializeValueFunction_eq_exitStep_zero M sat strat]
    exact le_trans (hmono hW) hstep
  have h := (iter_sound (fun V => Sound.refl .pessimistic (exitStep M sat strat prop V))
    (Or.inl hmono) h₀).1 k
  rw [← exitIter_eq_iterate] at h
  exact le_trans (α := S → ℝ) h (hmono.antitone_iterate_of_map_le hstep (Nat.zero_le k))

/-- Every iterate lies below the exact expected exit time `exitValue` (in `ℝ≥0∞`, all four
modes).

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) for `ExpectedExitTime`
(`src/specification.jl`). -/
theorem exitIter_le_exitValue (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) (k : ℕ) (s : S) :
    ENNReal.ofReal (exitIter M sat strat prop k s) ≤ exitValue M sat strat prop s :=
  le_iSup (f := (ENNReal.ofReal <| exitIter M sat strat prop · s)) k

/-- The exact expected exit time `exitValue` lies below every nonnegative real super-solution of
the exit-time Bellman equation (all four modes), from `exitIter_sound`; so whenever such a `W`
exists, `exitValue` is finite and `exitIter k ≤ exitValue ≤ W`.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) for `ExpectedExitTime`
(`src/specification.jl`). -/
theorem exitValue_le_of_step_le (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) {W : S → ℝ} (hW : 0 ≤ W)
    (hstep : exitStep M sat strat prop W ≤ W) (s : S) :
    exitValue M sat strat prop s ≤ ENNReal.ofReal (W s) :=
  iSup_le fun k => ENNReal.ofReal_le_ofReal (exitIter_sound M sat strat prop k hW hstep s)

end IntervalMDP.VI
