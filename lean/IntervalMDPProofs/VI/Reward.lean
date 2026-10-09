import IntervalMDPProofs.Bellman
import Mathlib.Topology.MetricSpace.Contracting

/-!
# Robust value iteration: discounted reward

For `FiniteTimeReward(reward, discount, time_horizon)` and
`InfiniteTimeReward(reward, discount, convergence_eps)` (`src/specification.jl`, lines 918–1067)
IntervalMDP.jl runs the `_value_iteration!` loop (`src/robust_value_iteration.jl`, lines 170–206;
`step!` at 236–249) with the `AbstractReward` methods (`src/specification.jl`, lines 927–935)

    initialize!(value_function, prop::AbstractReward)          # current .= reward(prop)
    step_postprocess_value_function!(value_function, prop)     # rmul!(current, discount(prop))
                                                               # current .+= reward(prop)
    postprocess_value_function!(value_function, ::AbstractReward) = value_function  # no-op

so, with `T` the robust Bellman operator computed by `bellman!` into `current` from `previous`,
`V₀ = r` and `V_{k+1} = ν · T V_k + r` (discount applied *after* the Bellman update, reward added
last). The termination criteria (`src/robust_value_iteration.jl`, lines 1–20) are
`FixedIterationsCriteria(time_horizon(prop))` (stop when `k ≥ K`, so `V_K = ∑_{j=0}^{K} νʲ …` has
`K + 1` reward terms, matching the `FiniteTimeReward` docstring) and
`CovergenceCriteria(convergence_eps(prop))`, which stops at the first `k ≥ 1` with
`maximum(abs, V_k - V_{k-1}) < convergence_eps` (`lastdiff!` stores `current - previous`).

Validation: `checkreward` requires `discount > 0` (both properties); `checkdiscountupperbound`
requires `discount < 1` for `InfiniteTimeReward` only. `FiniteTimeReward` accepts `ν ≥ 1`, where
the iteration is still well defined but is not a contraction; only `rewardIter_succ` applies there.

This file transcribes that loop as `rewardIter` and proves, for all four satisfaction × strategy
modes and a general `RMDP`:

* `rewardIter_succ`: the one-step recursion, as Julia computes it;
* `reward_contracting`: for `0 < ν < 1` the one-step map `rewardStep` is `ContractingWith ν` on
  `S → ℝ` with the sup distance (from `Bellman.T_nonexpansive`);
* `reward_error_bound` (A5): `dist V_k V* ≤ ν / (1 - ν) · dist V_k V_{k-1}`, where `V*` is the
  unique fixed point `rewardValue` (Mathlib `ContractingWith.fixedPoint`), so Julia's stop
  `‖V_k - V_{k-1}‖∞ < ε` guarantees `‖V_k - V*‖∞ < ν / (1 - ν) · ε` (`reward_stop_bound`).

Not modelled: time-varying models and strategies, floating point (including rounding in the
residual test), threads and CUDA. `V*` is the fixed point of the dynamic-programming recursion; its
relation to the path-measure expectation `𝔼^{π,η}[∑ νᵏ r(ω[k])]` is not proved.
-/

namespace IntervalMDP.VI

open Bellman

/-- A discounted-reward property: the state reward `reward : S → ℝ` and the discount factor
`discount = ν`, with the invariant `0 < ν` that `checkreward` enforces. The Julia time horizon and
convergence threshold only enter the termination criterion, so finite- and infinite-time rewards
share this structure (`Property.toRewardProperty`); `ν < 1` is required only by the theorems that
need it (`reward_contracting`, `reward_error_bound`).

Julia counterpart: the concrete subtypes of `AbstractReward` (`src/specification.jl`):
`FiniteTimeReward(reward, discount, time_horizon)` and
`InfiniteTimeReward(reward, discount, convergence_eps)`. -/
structure RewardProperty (S : Type*) where
  /-- The state reward, Julia `reward(prop)` (`src/specification.jl`). -/
  reward : S → ℝ
  /-- The discount factor `ν`, Julia `discount(prop)` (`src/specification.jl`). -/
  discount : ℝ
  /-- `checkreward`: "the discount factor must be greater than 0" (`src/specification.jl`). -/
  discount_pos : 0 < discount

/-- The reward property behind a `Property`, or `none` for the other properties.

Julia counterpart: dispatch of `initialize!` / `step_postprocess_value_function!` /
`postprocess_value_function!` on `AbstractReward` (`src/specification.jl`); time horizon and
`convergence_eps` are dropped (they only enter `termination_criteria`,
`src/robust_value_iteration.jl`). -/
def _root_.IntervalMDP.Property.toRewardProperty {S Q : Type*} :
    Property S Q → Option (RewardProperty S)
  | .finiteTimeReward reward discount discount_pos .. => some ⟨reward, discount, discount_pos⟩
  | .infiniteTimeReward reward discount discount_pos .. => some ⟨reward, discount, discount_pos⟩
  | _ => none

/-- Every infinite-time property with a reward has `ν < 1`: the hypothesis of
`reward_contracting` and `reward_error_bound` holds for every `InfiniteTimeReward` that Julia
accepts.

Julia counterpart: `checkdiscountupperbound(prop::InfiniteTimeReward)`, called from
`checkproperty` (`src/specification.jl`). -/
theorem _root_.IntervalMDP.Property.toRewardProperty_discount_lt_one {S Q : Type*}
    {p : Property S Q} {prop : RewardProperty S} (hinf : p.isFiniteTime = false)
    (h : p.toRewardProperty = some prop) : prop.discount < 1 := by
  cases p <;> simp only [Property.toRewardProperty, Property.isFiniteTime, reduceCtorEq,
    Option.some.injEq] at hinf h
  subst h
  assumption

namespace RewardProperty

variable {S : Type*}

/-- The discount factor as a nonnegative real, the Lipschitz constant of `rewardStep` in
`reward_contracting` (Mathlib's `ContractingWith` takes an `ℝ≥0`).

Julia counterpart: `discount(prop)` (`src/specification.jl`). -/
def discountNNReal (prop : RewardProperty S) : NNReal :=
  ⟨prop.discount, prop.discount_pos.le⟩

/-- The initial value function: `current .= reward(prop)`.

Julia counterpart: `initialize!(value_function, prop::AbstractReward)` (`src/specification.jl`). -/
def initializeValueFunction (prop : RewardProperty S) : S → ℝ :=
  prop.reward

/-- The postprocessing after each Bellman step, in Julia's statement order: first discount
(`rmul!(current, discount(prop))`), then add the reward (`current .+= reward(prop)`), i.e.
`ν • V + r`.

Julia counterpart: `step_postprocess_value_function!(value_function, prop::AbstractReward)`
(`src/specification.jl`). -/
def stepPostprocessValueFunction (prop : RewardProperty S) (V : S → ℝ) : S → ℝ :=
  prop.discount • V + prop.reward

/-- The final postprocessing after the loop: the identity.

Julia counterpart: `postprocess_value_function!(value_function, ::AbstractReward) =
value_function` (`src/specification.jl`), called once after the `while` loop of
`_value_iteration!` (`src/robust_value_iteration.jl`). -/
def postprocessValueFunction (_prop : RewardProperty S) (V : S → ℝ) : S → ℝ :=
  V

end RewardProperty

/-- Julia's convergence test on two successive iterates: the sup distance (Mathlib's `dist` on
`S → ℝ`, `maxₛ |V s - Vprev s|` for finite nonempty `S`) is below `convergenceEps`.

Julia counterpart: `(f::CovergenceCriteria)(V, k, u) = maximum(abs, u) < f.tol`
(`src/robust_value_iteration.jl`) with `u = lastdiff!(value_function) = current - previous` and
`tol = convergence_eps(prop)`. -/
def convergenceCriteria {S : Type*} [Fintype S] (convergenceEps : ℝ) (V Vprev : S → ℝ) : Prop :=
  dist V Vprev < convergenceEps

variable {S A : Type*} [Fintype S]

/-- One reward value-iteration step: the robust Bellman update `T` followed by
`rmul!(current, discount(prop)); current .+= reward(prop)`, i.e. `V ↦ ν • T V + r`.

Julia counterpart: `step!(workspace, strategy_cache, value_function, k, mp, spec)`
(`src/robust_value_iteration.jl`) with an `AbstractReward` specification: `bellman!(…;
upper_bound = isoptimistic(spec), maximize = ismaximize(spec))` then
`step_postprocess_value_function!` (`src/specification.jl`). The model is time-invariant here. -/
noncomputable def rewardStep (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) (V : S → ℝ) : S → ℝ :=
  prop.stepPostprocessValueFunction (T M sat strat V)

/-- The reward value-iteration iterates, a transcription of the loop of `_value_iteration!`:
`rewardIter 0` is the initialised value function `r`, and `rewardIter (k + 1)` is `rewardStep`
applied to `rewardIter k` (Julia: `nextiteration!`, `step!(…, k, …)`, `k += 1`). The Julia counter
`k` after the loop equals the Lean index; `postprocess_value_function!` is the identity for
`AbstractReward`, so the returned `value_function.current` is
`prop.postprocessValueFunction (rewardIter M sat strat prop k) = rewardIter M sat strat prop k`
for the `k` at which `term_criteria` first holds (`k = time_horizon(prop)` for
`FixedIterationsCriteria`; the first `k ≥ 1` with `convergenceCriteria convergence_eps
(rewardIter k) (rewardIter (k - 1))` for `CovergenceCriteria`).

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) with
`FiniteTimeReward` / `InfiniteTimeReward` (`src/specification.jl`). -/
noncomputable def rewardIter (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) : ℕ → S → ℝ
  | 0 => prop.initializeValueFunction
  | k + 1 => rewardStep M sat strat prop (rewardIter M sat strat prop k)

/-- `rewardIter` is the `k`-fold iterate of `rewardStep` from the initial value function.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`). -/
theorem rewardIter_eq_iterate (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) (k : ℕ) :
    rewardIter M sat strat prop k =
      (rewardStep M sat strat prop)^[k] prop.initializeValueFunction := by
  induction k with
  | zero => rfl
  | succ k ih => rw [Function.iterate_succ_apply', ← ih]; rfl

/-- **One-step recursion.** As Julia computes it: Bellman update, then discount, then add the
reward, `V_{k+1} s = ν · T V_k s + r s` (all four satisfaction × strategy modes, general `RMDP`,
any `ν > 0`, so also the finite-horizon case `ν ≥ 1`).

Julia counterpart: `step!` (`src/robust_value_iteration.jl`) followed by
`step_postprocess_value_function!(value_function, prop::AbstractReward)`,
`rmul!(current, discount(prop)); current .+= reward(prop)` (`src/specification.jl`). -/
theorem rewardIter_succ (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) (k : ℕ) (s : S) :
    rewardIter M sat strat prop (k + 1) s =
      prop.discount * T M sat strat (rewardIter M sat strat prop k) s + prop.reward s := by
  simp [rewardIter, rewardStep, RewardProperty.stepPostprocessValueFunction]

/-- The reward step scales sup distances by at most `ν`: `‖rewardStep V - rewardStep W‖∞ ≤
ν · ‖V - W‖∞` (all four modes, any `ν > 0`), from `Bellman.T_nonexpansive`.

Julia counterpart: `step!` (`src/robust_value_iteration.jl`) with `AbstractReward`
(`src/specification.jl`). -/
theorem rewardStep_dist_le (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) (V W : S → ℝ) :
    dist (rewardStep M sat strat prop V) (rewardStep M sat strat prop W) ≤
      prop.discount * dist V W := by
  simp only [rewardStep, RewardProperty.stepPostprocessValueFunction, dist_add_right,
    dist_smul₀, Real.norm_eq_abs, abs_of_pos prop.discount_pos]
  exact mul_le_mul_of_nonneg_left (T_nonexpansive M sat strat V W) prop.discount_pos.le

/-- **Contraction.** For `0 < ν < 1` the one-step reward map `V ↦ ν • T V + r` is a contraction
with constant `ν` on `S → ℝ` with the sup distance (Mathlib `ContractingWith`), for all four
satisfaction × strategy modes and a general `RMDP` (no convexity assumed). `0 < ν` is the
structure invariant; `ν < 1` is the row's hypothesis (it holds for every Julia
`InfiniteTimeReward`, `Property.toRewardProperty_discount_lt_one`).

Julia counterpart: `step!` (`src/robust_value_iteration.jl`) with `AbstractReward`
(`src/specification.jl`). -/
theorem reward_contracting (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) (hν : prop.discount < 1) :
    ContractingWith prop.discountNNReal (rewardStep M sat strat prop) :=
  ⟨hν, LipschitzWith.of_dist_le_mul (rewardStep_dist_le M sat strat prop)⟩

/-- The exact robust discounted value `V*` for `0 < ν < 1`: the unique fixed point
`V* = ν • T V* + r` of `rewardStep` (Banach, Mathlib `ContractingWith.fixedPoint`; uniqueness
`ContractingWith.fixedPoint_unique`, convergence `rewardIter_tendsto`).

Julia counterpart: the value that `_value_iteration!` (`src/robust_value_iteration.jl`)
approximates for `InfiniteTimeReward` (`src/specification.jl`). -/
noncomputable def rewardValue (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) (hν : prop.discount < 1) : S → ℝ :=
  ContractingWith.fixedPoint (rewardStep M sat strat prop) (reward_contracting M sat strat prop hν)

/-- `rewardValue` is a fixed point of the reward step: `V* = ν • T V* + r` (all four modes).

Julia counterpart: `step!` (`src/robust_value_iteration.jl`) with `InfiniteTimeReward`
(`src/specification.jl`). -/
theorem rewardValue_isFixedPt (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) (hν : prop.discount < 1) :
    Function.IsFixedPt (rewardStep M sat strat prop) (rewardValue M sat strat prop hν) :=
  ContractingWith.fixedPoint_isFixedPt _

/-- The reward iterates converge to `rewardValue` in the sup distance (all four modes,
`0 < ν < 1`).

Julia counterpart: successive `value_function.current` in `_value_iteration!`
(`src/robust_value_iteration.jl`) for `InfiniteTimeReward` (`src/specification.jl`). -/
theorem rewardIter_tendsto (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) (hν : prop.discount < 1) :
    Filter.Tendsto (rewardIter M sat strat prop) Filter.atTop
      (nhds (rewardValue M sat strat prop hν)) := by
  have h := ContractingWith.tendsto_iterate_fixedPoint (reward_contracting M sat strat prop hν)
    prop.initializeValueFunction
  rw [show rewardIter M sat strat prop = _ from funext (rewardIter_eq_iterate M sat strat prop)]
  exact h

/-- **A5: explicit error bound at any stopping index.** For `0 < ν < 1`, all four satisfaction ×
strategy modes and a general `RMDP`: `‖V_{k+1} - V*‖∞ ≤ ν / (1 - ν) · ‖V_{k+1} - V_k‖∞`, where
`V* = rewardValue` is the unique fixed point. Julia's `CovergenceCriteria` compares iterate `k + 1`
with iterate `k` (Julia's `k ≥ 1` is the Lean `k + 1`; the index is shifted so no truncated
`k - 1` appears). So the closed sup-ball of radius `ν / (1 - ν) · ‖V_{k+1} - V_k‖∞` around the
returned `V_{k+1}` contains `V*` (`reward_stop_bound` for Julia's `< ε` test).

Julia counterpart: the value returned by `_value_iteration!` (`src/robust_value_iteration.jl`) for
`InfiniteTimeReward` (`src/specification.jl`) when `CovergenceCriteria` stops it. -/
theorem reward_error_bound (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) (hν : prop.discount < 1) (k : ℕ) :
    dist (rewardIter M sat strat prop (k + 1)) (rewardValue M sat strat prop hν) ≤
      prop.discount / (1 - prop.discount) *
        dist (rewardIter M sat strat prop (k + 1)) (rewardIter M sat strat prop k) := by
  set V := rewardIter M sat strat prop k
  set Vs := rewardValue M sat strat prop hν
  have hfix : rewardStep M sat strat prop Vs = Vs := rewardValue_isFixedPt M sat strat prop hν
  have hstep : rewardIter M sat strat prop (k + 1) = rewardStep M sat strat prop V := rfl
  rw [hstep]
  have h₁ : dist (rewardStep M sat strat prop V) Vs ≤ prop.discount * dist V Vs := by
    simpa [hfix] using rewardStep_dist_le M sat strat prop V Vs
  have h₂ : dist V Vs ≤ dist V (rewardStep M sat strat prop V) +
      dist (rewardStep M sat strat prop V) Vs := dist_triangle _ _ _
  have h1ν : 0 < 1 - prop.discount := by linarith
  rw [dist_comm (rewardStep M sat strat prop V) V, div_mul_eq_mul_div, le_div_iff₀ h1ν]
  nlinarith [prop.discount_pos]

/-- **A5 at Julia's stop.** If Julia's convergence test holds at iterate `k + 1`,
`‖V_{k+1} - V_k‖∞ < ε`, then `‖V_{k+1} - V*‖∞ < ν / (1 - ν) · ε` (all four modes, `0 < ν < 1`):
every state's returned value lies strictly within `ν / (1 - ν) · ε` of the exact value.

Julia counterpart: `CovergenceCriteria(convergence_eps(prop))` and `_value_iteration!`
(`src/robust_value_iteration.jl`) for `InfiniteTimeReward` (`src/specification.jl`). -/
theorem reward_stop_bound (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) (hν : prop.discount < 1) {k : ℕ} {ε : ℝ}
    (hstop : convergenceCriteria ε (rewardIter M sat strat prop (k + 1))
      (rewardIter M sat strat prop k)) :
    dist (rewardIter M sat strat prop (k + 1)) (rewardValue M sat strat prop hν) <
      prop.discount / (1 - prop.discount) * ε := by
  have hc : 0 < prop.discount / (1 - prop.discount) :=
    div_pos prop.discount_pos (by linarith)
  exact lt_of_le_of_lt (reward_error_bound M sat strat prop hν k)
    (mul_lt_mul_of_pos_left hstop hc)

/-- **A5, pointwise.** The interval `[V_{k+1} s - c, V_{k+1} s + c]` with
`c = ν / (1 - ν) · ‖V_{k+1} - V_k‖∞` contains the exact value `V* s`, for every state `s` (all four
modes, `0 < ν < 1`).

Julia counterpart: the value returned by `_value_iteration!` (`src/robust_value_iteration.jl`) for
`InfiniteTimeReward` (`src/specification.jl`). -/
theorem reward_error_interval (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) (hν : prop.discount < 1) (k : ℕ) (s : S) :
    |rewardIter M sat strat prop (k + 1) s - rewardValue M sat strat prop hν s| ≤
      prop.discount / (1 - prop.discount) *
        dist (rewardIter M sat strat prop (k + 1)) (rewardIter M sat strat prop k) := by
  rw [← Real.dist_eq]
  exact le_trans (dist_le_pi_dist _ _ s) (reward_error_bound M sat strat prop hν k)

end IntervalMDP.VI
