import IntervalMDPProofs.Bellman

/-!
# Robust value iteration: reachability and reach-avoid

`_value_iteration!` (`src/robust_value_iteration.jl`, lines 170–206) runs

    initialize!(value_function, spec)            # V₀ = 𝟙_reach (zeros elsewhere)
    nextiteration!(value_function)
    step!(…, value_function, 0, mp, spec); k = 1
    while !term_criteria(value_function.current, k, lastdiff!(value_function))
        nextiteration!(value_function)
        step!(…, value_function, k, mp, spec); k += 1
    end

where `step!` (lines 236–249) is one `bellman!` call followed by
`step_postprocess_value_function!(value_function, spec)`. For the reachability properties
(`src/specification.jl`) the initialisation and postprocessing are:

| Julia type | `step_postprocess_value_function!` | Lean |
|---|---|---|
| `Finite/InfiniteTimeReachability` | `V[reach] .= 1` (lines 356–358) | `.reachability` |
| `Finite/InfiniteTimeReachAvoid` | `V[reach] .= 1; V[avoid] .= 0` (532–535) | `.reachAvoid` |
| `ExactTimeReachability` | nothing (497–499) | `.exactTimeReachability` |
| `ExactTimeReachAvoid` | `V[avoid] .= 0` (760–762) | `.exactTimeReachAvoid` |

All six share `initialize!(value_function, prop::AbstractReachability)` (lines 352–354:
`V[reach] .= 1` on the all-zero array of `ValueFunction`), and `postprocess_value_function!` is
`nothing` (line 360).

This file transcribes that loop as `reachIter` (iterate `k` = the Julia counter `k` after the
`k`-th `step!`) and proves, for all four satisfaction × strategy modes and a general `RMDP`:

* `reachIter_mem_unit`: every iterate lies in `[0, 1]` (all six property types);
* `reachIter_mono`: the iterates are non-decreasing in `k`;
* `reachIter_tendsto_lfp`: they converge to `reachLfp`, the least fixed point of the one-step map
  `step` on `S → [0, 1]` (Mathlib `OrderHom.lfp`);
* `reachIter_sound` (A4): `V_k ≤ reachLfp` for every `k`, via `Approx.iter_sound`.

The last three are stated for the non-exact-time properties (`isExactTime = false`). Exact-time
properties are excluded on purpose: their value `ℙ[ω[K] ∈ G]` is not monotone in the horizon `K`,
it is not a fixed point of `step`, and Julia runs them for exactly `K` steps
(`FixedIterationsCriteria`), so no stopping approximation is involved.

**Direction in optimistic mode.** `reachIter_sound` states `Sound .pessimistic V_k V*` (that is,
`V_k ≤ V*`) in every mode. For `sat = pessimistic` this is `Sound sat`, the conservative direction.
For `sat = optimistic` it is still `V_k ≤ V*` — a lower bound of the optimistic value — which is
**not** the conservative direction `Sound .optimistic` (`V* ≤ V_k`); no such claim is made.
Without A4/A6, no error bound is claimed for the reachability ε-criterion.

Not modelled: time-varying models (`select_model(mp, k)` with time-varying available actions or
labelling), time-varying strategies (`NonOptimizingStrategyCache` indexed by `k`), floating point,
threads and CUDA. A given stationary strategy is covered by applying the theorems to
`M.withAvailable (strategyAvailable π)` (`Bellman.Tπ_eq_T_strategyAvailable`).
-/

namespace IntervalMDP.VI

open Filter Topology Bellman Approx

/-- The reachability properties of IntervalMDP.jl, one constructor per
`step_postprocess_value_function!` method they dispatch to. The Julia time horizon and
convergence threshold only enter the termination criterion, not the iteration map, so finite- and
infinite-time variants share a constructor (`Property.toReachProperty`).

Julia counterpart: the concrete subtypes of `AbstractReachability` (`src/specification.jl`):
`FiniteTimeReachability`, `InfiniteTimeReachability`, `ExactTimeReachability` and, via
`AbstractReachAvoid`, `FiniteTimeReachAvoid`, `InfiniteTimeReachAvoid`, `ExactTimeReachAvoid`. -/
inductive ReachProperty (S : Type*) where
  /-- `FiniteTimeReachability` / `InfiniteTimeReachability(reach, …)`. -/
  | reachability (reach : Finset S)
  /-- `FiniteTimeReachAvoid` / `InfiniteTimeReachAvoid(reach, avoid, …)`; `checkdisjoint`
  enforces `disjoint`. -/
  | reachAvoid (reach avoid : Finset S) (disjoint : Disjoint reach avoid)
  /-- `ExactTimeReachability(reach, time_horizon)`. -/
  | exactTimeReachability (reach : Finset S)
  /-- `ExactTimeReachAvoid(reach, avoid, time_horizon)`. -/
  | exactTimeReachAvoid (reach avoid : Finset S) (disjoint : Disjoint reach avoid)

namespace ReachProperty

variable {S : Type*}

/-- The target set `reach(prop)`.

Julia counterpart: `reach(prop)` (`src/specification.jl`). -/
def reach : ReachProperty S → Finset S
  | reachability reach | reachAvoid reach .. | exactTimeReachability reach
  | exactTimeReachAvoid reach .. => reach

/-- Whether the property is an exact-time property (no reset of the target states).

Julia counterpart: `ExactTimeReachability` / `ExactTimeReachAvoid` (`src/specification.jl`), the
two types whose `step_postprocess_value_function!` does not reset `reach`. -/
def isExactTime : ReachProperty S → Bool
  | exactTimeReachability .. | exactTimeReachAvoid .. => true
  | _ => false

end ReachProperty

/-- The reachability property behind a `Property`, or `none` for the other properties.

Julia counterpart: dispatch of `initialize!` / `step_postprocess_value_function!` on
`AbstractReachability` (`src/specification.jl`); time horizon and `convergence_eps` are dropped
(they only enter `termination_criteria`, `src/robust_value_iteration.jl`). -/
def _root_.IntervalMDP.Property.toReachProperty {S Q : Type*} :
    Property S Q → Option (ReachProperty S)
  | .finiteTimeReachability reach .. | .infiniteTimeReachability reach .. =>
    some (.reachability reach)
  | .finiteTimeReachAvoid reach avoid h .. | .infiniteTimeReachAvoid reach avoid h .. =>
    some (.reachAvoid reach avoid h)
  | .exactTimeReachability reach .. => some (.exactTimeReachability reach)
  | .exactTimeReachAvoid reach avoid h .. => some (.exactTimeReachAvoid reach avoid h)
  | _ => none

variable {S A : Type*} [Fintype S] [DecidableEq S]

/-! ### The one-step map -/

/-- The initial value function `V₀ = 𝟙_reach`: `1` on `reach(prop)`, `0` elsewhere.

Julia counterpart: `ValueFunction(problem)` (all zeros, `src/robust_value_iteration.jl`) followed
by `initialize!(value_function, prop::AbstractReachability)`, `current[reach(prop)] .= 1.0`
(`src/specification.jl`), shared by all six reachability types. -/
def initializeValueFunction (prop : ReachProperty S) : S → ℝ :=
  fun s => if s ∈ prop.reach then 1 else 0

/-- The postprocessing after each Bellman step (statement order as in Julia: `reach` first, then
`avoid`, so `avoid` wins; they are disjoint anyway):

* reachability: `V[reach] .= 1`;
* reach-avoid: `V[reach] .= 1; V[avoid] .= 0`;
* exact-time reachability: unchanged;
* exact-time reach-avoid: `V[avoid] .= 0`.

Julia counterpart: `step_postprocess_value_function!` for `AbstractReachability`,
`AbstractReachAvoid`, `ExactTimeReachability` and `ExactTimeReachAvoid` (`src/specification.jl`). -/
def stepPostprocessValueFunction : ReachProperty S → (S → ℝ) → S → ℝ
  | .reachability reach, V, s => if s ∈ reach then 1 else V s
  | .reachAvoid reach avoid _, V, s => if s ∈ avoid then 0 else if s ∈ reach then 1 else V s
  | .exactTimeReachability _, V, s => V s
  | .exactTimeReachAvoid _ avoid _, V, s => if s ∈ avoid then 0 else V s

/-- One value-iteration step: the robust Bellman update `T` followed by the postprocessing.

Julia counterpart: `step!(workspace, strategy_cache, value_function, k, mp, spec)`
(`src/robust_value_iteration.jl`): `bellman!(…, value_function.current, value_function.previous,
select_model(mp, k); upper_bound = isoptimistic(spec), maximize = ismaximize(spec))` then
`step_postprocess_value_function!(value_function, spec)` (`src/specification.jl`). The model is
time-invariant here (`select_model(mp, k) = mp`). -/
noncomputable def step (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) (V : S → ℝ) : S → ℝ :=
  stepPostprocessValueFunction prop (T M sat strat V)

/-- The value-iteration iterates for a reachability property, a transcription of the loop of
`_value_iteration!`: `reachIter 0` is the initialised value function, and `reachIter (k + 1)` is
`step` applied to `reachIter k` (Julia: `nextiteration!` copies `current` to `previous`, then
`step!(…, k, …)` writes `current`, then `k += 1`). The Julia counter `k` after the loop equals the
Lean index (`k ≥ 1`; the first `step!` before the loop gives `k = 1`); the returned
`value_function.current` is `reachIter k` for the `k` at which `term_criteria` first holds
(`FixedIterationsCriteria`: `k = time_horizon`; `CovergenceCriteria`: residual `< ε`).

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) with a reachability
`Specification` (`src/specification.jl`). -/
noncomputable def reachIter (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) : ℕ → S → ℝ
  | 0 => initializeValueFunction prop
  | k + 1 => step M sat strat prop (reachIter M sat strat prop k)

/-- `reachIter` is the `k`-fold iterate of `step` from the initial value function.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`). -/
theorem reachIter_eq_iterate (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) (k : ℕ) :
    reachIter M sat strat prop k = (step M sat strat prop)^[k] (initializeValueFunction prop) := by
  induction k with
  | zero => rfl
  | succ k ih => rw [Function.iterate_succ_apply', ← ih]; rfl

/-! ### Properties of the Bellman operator on `[0, 1]` -/

omit [DecidableEq S] in
/-- The Bellman operator maps the zero function to zero (all four modes).

Julia counterpart: `bellman!` (`src/bellman.jl`). -/
theorem T_zero (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode) :
    T M sat strat 0 = 0 := by
  funext s
  have h : stateActionBellman M sat 0 s = 0 := by
    funext a
    obtain ⟨x, -, hx⟩ := innerOpt_mem sat (M.ambiguity_wellFormed s a) 0
    simp only [stateActionBellman, Pi.zero_apply]
    rw [← hx]
    simp [OMax.dot]
  cases strat <;> simp [T, extractValue, h]

omit [DecidableEq S] in
/-- The Bellman operator fixes constant functions: `T (const c) = const c` (all four modes).

Julia counterpart: `bellman!` (`src/bellman.jl`). -/
theorem T_const (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode) (c : ℝ) :
    T M sat strat (Function.const S c) = Function.const S c := by
  have h := T_add_const M sat strat 0 c
  rwa [zero_add, T_zero, zero_add] at h

omit [DecidableEq S] in
/-- The Bellman operator maps `[0, 1]`-valued functions to `[0, 1]`-valued functions.

Julia counterpart: `bellman!` (`src/bellman.jl`). -/
theorem T_mem_unit (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode) {V : S → ℝ}
    (hV : ∀ s, V s ∈ Set.Icc (0 : ℝ) 1) (s : S) : T M sat strat V s ∈ Set.Icc (0 : ℝ) 1 := by
  have h₀ := T_mono M sat strat (V := Function.const S 0) (W := V) (fun s => (hV s).1) s
  have h₁ := T_mono M sat strat (V := V) (W := Function.const S 1) (fun s => (hV s).2) s
  rw [T_const] at h₀ h₁
  exact ⟨h₀, h₁⟩

/-! ### Properties of the one-step map -/

omit [Fintype S] in
/-- The postprocessing is monotone.

Julia counterpart: `step_postprocess_value_function!` (`src/specification.jl`). -/
theorem stepPostprocessValueFunction_mono (prop : ReachProperty S) {V W : S → ℝ} (h : V ≤ W) :
    stepPostprocessValueFunction prop V ≤ stepPostprocessValueFunction prop W := by
  intro s
  cases prop <;> simp only [stepPostprocessValueFunction] <;> (try split_ifs) <;>
    first | exact le_rfl | exact h s

omit [Fintype S] in
/-- The postprocessing keeps values in `[0, 1]`.

Julia counterpart: `step_postprocess_value_function!` (`src/specification.jl`). -/
theorem stepPostprocessValueFunction_mem_unit (prop : ReachProperty S) {V : S → ℝ}
    (hV : ∀ s, V s ∈ Set.Icc (0 : ℝ) 1) (s : S) :
    stepPostprocessValueFunction prop V s ∈ Set.Icc (0 : ℝ) 1 := by
  cases prop <;> simp only [stepPostprocessValueFunction] <;> (try split_ifs) <;>
    first | exact hV s | exact ⟨le_rfl, zero_le_one⟩ | exact ⟨zero_le_one, le_rfl⟩

omit [Fintype S] in
/-- The postprocessing is continuous (each entry is a constant or a coordinate).

Julia counterpart: `step_postprocess_value_function!` (`src/specification.jl`). -/
theorem continuous_stepPostprocessValueFunction (prop : ReachProperty S) :
    Continuous (stepPostprocessValueFunction prop) := by
  refine continuous_pi fun s => ?_
  cases prop with
  | reachability reach =>
    by_cases hs : s ∈ reach <;> simp [stepPostprocessValueFunction, hs, continuous_apply,
      continuous_const]
  | reachAvoid reach avoid _ =>
    by_cases ha : s ∈ avoid <;> by_cases hs : s ∈ reach <;>
      simp [stepPostprocessValueFunction, ha, hs, continuous_apply,
      continuous_const]
  | exactTimeReachability reach => simp [stepPostprocessValueFunction, continuous_apply]
  | exactTimeReachAvoid reach avoid _ =>
    by_cases ha : s ∈ avoid <;> simp [stepPostprocessValueFunction, ha, continuous_apply,
      continuous_const]

/-- `step` is monotone (all four modes, all property types).

Julia counterpart: `step!` (`src/robust_value_iteration.jl`). -/
theorem step_mono (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) : Monotone (step M sat strat prop) :=
  fun _ _ h => stepPostprocessValueFunction_mono prop (T_mono M sat strat h)

/-- `step` maps `[0, 1]`-valued functions to `[0, 1]`-valued functions.

Julia counterpart: `step!` (`src/robust_value_iteration.jl`). -/
theorem step_mem_unit (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) {V : S → ℝ} (hV : ∀ s, V s ∈ Set.Icc (0 : ℝ) 1) (s : S) :
    step M sat strat prop V s ∈ Set.Icc (0 : ℝ) 1 :=
  stepPostprocessValueFunction_mem_unit prop (T_mem_unit M sat strat hV) s

/-- `step` is continuous (`T` is `1`-Lipschitz, `T_lipschitz`).

Julia counterpart: `step!` (`src/robust_value_iteration.jl`). -/
theorem continuous_step (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) : Continuous (step M sat strat prop) :=
  (continuous_stepPostprocessValueFunction prop).comp (T_lipschitz M sat strat).continuous

/-- For a non-exact-time property, the initial value function lies below `step W` for every
nonnegative `W`: on `reach` both are `1` (`reach` and `avoid` are disjoint), elsewhere the initial
value is `0`.

Julia counterpart: `initialize!` and `step!` (`src/specification.jl`,
`src/robust_value_iteration.jl`). -/
theorem initializeValueFunction_le_step (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) {prop : ReachProperty S} (hprop : prop.isExactTime = false)
    {W : S → ℝ} (hW : 0 ≤ W) : initializeValueFunction prop ≤ step M sat strat prop W := by
  intro s
  have hT : 0 ≤ T M sat strat W s := by
    have h := T_mono M sat strat hW s
    rwa [show (0 : S → ℝ) = Function.const S 0 from rfl, T_const] at h
  cases prop with
  | reachability reach =>
    by_cases hs : s ∈ reach <;>
      simp [initializeValueFunction, ReachProperty.reach, step, stepPostprocessValueFunction, hs,
        hT]
  | reachAvoid reach avoid hd =>
    by_cases ha : s ∈ avoid
    · have hs : s ∉ reach := fun hs => Finset.disjoint_left.mp hd hs ha
      simp [initializeValueFunction, ReachProperty.reach, step, stepPostprocessValueFunction, hs,
        ha]
    · by_cases hs : s ∈ reach <;>
        simp [initializeValueFunction, ReachProperty.reach, step, stepPostprocessValueFunction, hs,
          ha, hT]
  | exactTimeReachability _ => simp [ReachProperty.isExactTime] at hprop
  | exactTimeReachAvoid _ _ _ => simp [ReachProperty.isExactTime] at hprop

/-! ### The least fixed point on `S → [0, 1]` -/

/-- `0 ≤ 1` in `ℝ`, needed for the complete lattice `Set.Icc 0 1` (Mathlib
`Set.Icc.completeLattice`).

Julia counterpart: none (Lean-side proof device). -/
instance fact_zero_le_one_real : Fact ((0 : ℝ) ≤ 1) := ⟨zero_le_one⟩

/-- `step` as a monotone map on the complete lattice `S → [0, 1]` of `[0, 1]`-valued value
functions.

Julia counterpart: `step!` (`src/robust_value_iteration.jl`) restricted to probability-valued
value functions. -/
noncomputable def stepHom (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) : (S → Set.Icc (0 : ℝ) 1) →o (S → Set.Icc (0 : ℝ) 1) where
  toFun W s := ⟨step M sat strat prop (Subtype.val ∘ W) s,
    step_mem_unit M sat strat prop (fun s => (W s).2) s⟩
  monotone' _ _ h s := step_mono M sat strat prop (fun s => h s) s

/-- The least fixed point of the one-step map `step` on `S → [0, 1]` (Mathlib `OrderHom.lfp`),
read as a real-valued function. For a non-exact-time property this is the exact robust
reachability (reach-avoid) value `V*` of the dynamic-programming semantics
(`reachIter_tendsto_lfp`); it is also below every nonnegative real fixed point of `step`
(`reachLfp_le_of_fixedPt`).

Julia counterpart: the value that `_value_iteration!` (`src/robust_value_iteration.jl`)
approximates for `InfiniteTimeReachability` / `InfiniteTimeReachAvoid` (`src/specification.jl`). -/
noncomputable def reachLfp (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) : S → ℝ :=
  fun s => (OrderHom.lfp (stepHom M sat strat prop) s : ℝ)

/-- `reachLfp` is a fixed point of `step`.

Julia counterpart: `step!` (`src/robust_value_iteration.jl`). -/
theorem step_reachLfp (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) :
    step M sat strat prop (reachLfp M sat strat prop) = reachLfp M sat strat prop := by
  funext s
  exact congrArg (fun W => (W s : ℝ)) (OrderHom.map_lfp (stepHom M sat strat prop))

/-- `reachLfp` takes values in `[0, 1]`.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`). -/
theorem reachLfp_mem_unit (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) (s : S) : reachLfp M sat strat prop s ∈ Set.Icc (0 : ℝ) 1 :=
  (OrderHom.lfp (stepHom M sat strat prop) s).2

/-- `reachLfp` lies below every `[0, 1]`-valued pre-fixed point `W` (`step W ≤ W`) of `step`.

Julia counterpart: `step!` (`src/robust_value_iteration.jl`). -/
theorem reachLfp_le_of_step_le (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) {W : S → ℝ} (hW : ∀ s, W s ∈ Set.Icc (0 : ℝ) 1)
    (h : step M sat strat prop W ≤ W) : reachLfp M sat strat prop ≤ W := by
  let Wu : S → Set.Icc (0 : ℝ) 1 := fun s => ⟨W s, hW s⟩
  have h' : stepHom M sat strat prop Wu ≤ Wu := fun s => h s
  exact fun s => OrderHom.lfp_le _ h' s

/-! ### The four theorems -/

/-- **Iterates stay in `[0, 1]`.** Every value-iteration iterate for a reachability or reach-avoid
property (all six Julia types, including exact time) lies in `[0, 1]` pointwise, for all four
satisfaction × strategy modes and a general `RMDP`.

Julia counterpart: `value_function.current` after each `step!` in `_value_iteration!`
(`src/robust_value_iteration.jl`). -/
theorem reachIter_mem_unit (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) (k : ℕ) (s : S) :
    reachIter M sat strat prop k s ∈ Set.Icc (0 : ℝ) 1 := by
  induction k generalizing s with
  | zero =>
    simp only [reachIter, initializeValueFunction]
    split_ifs
    · exact ⟨zero_le_one, le_rfl⟩
    · exact ⟨le_rfl, zero_le_one⟩
  | succ k ih => exact step_mem_unit M sat strat prop ih s

/-- **Monotone iterates.** For a non-exact-time reachability or reach-avoid property, the
iterates are non-decreasing in `k` (from the indicator start `𝟙_reach`), all four satisfaction ×
strategy modes, general `RMDP`. Exact-time properties are excluded: their `K`-step value is not
monotone in `K`.

Julia counterpart: successive `value_function.current` in `_value_iteration!`
(`src/robust_value_iteration.jl`) for `FiniteTimeReachability`, `InfiniteTimeReachability`,
`FiniteTimeReachAvoid`, `InfiniteTimeReachAvoid` (`src/specification.jl`). -/
theorem reachIter_mono (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    {prop : ReachProperty S} (hprop : prop.isExactTime = false) :
    Monotone (reachIter M sat strat prop) := by
  refine monotone_nat_of_le_succ fun k => ?_
  induction k with
  | zero =>
    exact initializeValueFunction_le_step M sat strat hprop
      (fun s => (reachIter_mem_unit M sat strat prop 0 s).1)
  | succ k ih => exact step_mono M sat strat prop ih

/-- **A4: stopping at any finite `k` gives a lower bound.** For a non-exact-time reachability or
reach-avoid property, every iterate lies below the least fixed point: `V_k ≤ reachLfp`, in all four
satisfaction × strategy modes — in particular at the `k` where Julia's termination criterion stops.
Proved through `Approx.iter_sound` (the exact operator is `step` itself, started from `reachLfp`).

Direction: the conclusion is `Sound .pessimistic`, i.e. `V_k ≤ V*`. For `sat = pessimistic` this
is soundness in the sense of `Sound sat`. For `sat = optimistic` the iterates are still lower
bounds of the optimistic value, which is **not** the conservative direction (`Sound .optimistic`
needs `V* ≤ V_k`); that is not claimed. Without A4/A6, no error bound is claimed for the
reachability ε-criterion.

Julia counterpart: the value returned by `_value_iteration!` (`src/robust_value_iteration.jl`)
when `term_criteria` stops it (`FixedIterationsCriteria` or `CovergenceCriteria`). -/
theorem reachIter_sound (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    {prop : ReachProperty S} (hprop : prop.isExactTime = false) (k : ℕ) :
    Sound .pessimistic (reachIter M sat strat prop k) (reachLfp M sat strat prop) := by
  have h₀ : Sound .pessimistic (initializeValueFunction prop) (reachLfp M sat strat prop) := by
    have h := initializeValueFunction_le_step M sat strat hprop
      (fun s => (reachLfp_mem_unit M sat strat prop s).1)
    rwa [step_reachLfp] at h
  have h := (iter_sound (fun V => Sound.refl .pessimistic (step M sat strat prop V))
    (Or.inl (step_mono M sat strat prop)) h₀).1 k
  rwa [Function.iterate_fixed (step_reachLfp M sat strat prop), ← reachIter_eq_iterate] at h

/-- **Convergence to the least fixed point.** For a non-exact-time reachability or reach-avoid
property, the iterates converge pointwise to `reachLfp`, the least fixed point of `step` on
`S → [0, 1]`, in all four satisfaction × strategy modes and for a general `RMDP` (the continuity
needed comes from `T_lipschitz`).

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) for
`InfiniteTimeReachability` / `InfiniteTimeReachAvoid` (`src/specification.jl`). -/
theorem reachIter_tendsto_lfp (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    {prop : ReachProperty S} (hprop : prop.isExactTime = false) :
    Tendsto (reachIter M sat strat prop) atTop (𝓝 (reachLfp M sat strat prop)) := by
  have hmono := reachIter_mono M sat strat hprop
  have hbdd : ∀ s, BddAbove (Set.range fun k => reachIter M sat strat prop k s) :=
    fun s => ⟨1, by rintro _ ⟨k, rfl⟩; exact (reachIter_mem_unit M sat strat prop k s).2⟩
  let L : S → ℝ := fun s => ⨆ k, reachIter M sat strat prop k s
  have hL : Tendsto (reachIter M sat strat prop) atTop (𝓝 L) :=
    tendsto_pi_nhds.2 fun s => tendsto_atTop_ciSup (fun _ _ hab => hmono hab s) (hbdd s)
  have hfix : step M sat strat prop L = L := by
    have h₁ : Tendsto (fun k => reachIter M sat strat prop (k + 1)) atTop (𝓝 L) :=
      hL.comp (tendsto_add_atTop_nat 1)
    have h₂ := ((continuous_step M sat strat prop).tendsto L).comp hL
    exact tendsto_nhds_unique h₂ h₁
  have hLunit : ∀ s, L s ∈ Set.Icc (0 : ℝ) 1 := fun s =>
    ⟨le_trans (reachIter_mem_unit M sat strat prop 0 s).1 (le_ciSup (hbdd s) 0),
      ciSup_le fun k => (reachIter_mem_unit M sat strat prop k s).2⟩
  have hEq : L = reachLfp M sat strat prop := by
    apply le_antisymm
    · exact fun s => ciSup_le fun k => (reachIter_sound M sat strat hprop k) s
    · exact reachLfp_le_of_step_le M sat strat prop hLunit hfix.le
  rwa [← hEq]

/-- `reachLfp` is the least nonnegative real fixed point of `step` (not only the least one in
`S → [0, 1]`), for a non-exact-time property. Proved through `Approx.iter_sound` and
`reachIter_tendsto_lfp`.

Julia counterpart: `step!` (`src/robust_value_iteration.jl`). -/
theorem reachLfp_le_of_fixedPt (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    {prop : ReachProperty S} (hprop : prop.isExactTime = false) {W : S → ℝ} (hW : 0 ≤ W)
    (hfix : step M sat strat prop W = W) : reachLfp M sat strat prop ≤ W := by
  have h₀ : Sound .pessimistic (initializeValueFunction prop) W := by
    have h := initializeValueFunction_le_step M sat strat hprop hW
    rwa [hfix] at h
  have h := (iter_sound (fun V => Sound.refl .pessimistic (step M sat strat prop V))
    (Or.inl (step_mono M sat strat prop)) h₀).1
  refine le_of_tendsto_of_tendsto' (reachIter_tendsto_lfp M sat strat hprop) tendsto_const_nhds
    fun k => ?_
  have hk := h k
  rwa [Function.iterate_fixed hfix, ← reachIter_eq_iterate] at hk

end IntervalMDP.VI
