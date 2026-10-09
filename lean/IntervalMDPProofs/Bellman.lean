import IntervalMDPProofs.Models.RMDP
import IntervalMDPProofs.Models.Specification
import IntervalMDPProofs.OMax

/-!
# The robust Bellman operator on general RMDPs

The robust Bellman operator of IntervalMDP.jl (`bellman!` in `src/bellman.jl`) updates every
state `s` by

    T V s = opt_strat_{a ∈ available s} opt_sat_{p ∈ Γ_{s,a}} ⟨p, V⟩.

In Julia, `_bellman_helper!` loops over the states and calls `state_bellman!` with an
`OptimizingStrategyCache`. For each available action `jₐ`, `state_bellman!` stores
`state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)` (the inner optimum over
the ambiguity set) in `workspace.actions[jₐ]`. It then calls `extract_strategy!`, which returns the
maximum (`maximize = true`) or minimum (`maximize = false`) of these values over the available
actions (`src/strategy_cache.jl`). The two modes reach `bellman!` from `step!`
(`src/robust_value_iteration.jl`) as `upper_bound = isoptimistic(spec)` and
`maximize = ismaximize(spec)`.

The Lean definitions follow this structure:

* `innerOpt sat Γ V`: `sup` (optimistic, Julia `upper_bound = true`) or `inf` (pessimistic,
  `upper_bound = false`) of `⟨p, V⟩` over `p ∈ Γ`;
* `stateActionBellman M sat V s a = innerOpt sat (M.ambiguity s a) V`;
* `extractValue strat acts values`: `max` / `min` of `values` over the nonempty finite set `acts`;
* `T M sat strat V s`: `extractValue` over `M.available s` of `stateActionBellman`.

`T` is defined once, for the general `RMDP`. Interval MDPs, factored IMDPs and products reach it
through their `toRMDP`.

## Results

For **all four** satisfaction × strategy modes and for a **general** `RMDP`:

* `T_mono`: `V ≤ W → T V ≤ T W` (pointwise);
* `T_add_const`: `T (V + c) = T V + c` for a constant `c`;
* `T_nonexpansive`: `dist (T V) (T W) ≤ dist V W` for the sup-norm distance on `S → ℝ`.

They follow from one lemma, `T_le_add`: `V ≤ W + c` pointwise implies `T V ≤ T W + c`.

**No convexity is assumed.** The only hypotheses are the `RMDP` invariants: every ambiguity set
is nonempty and closed (`AmbiguitySet.WellFormed`, hence compact, so each inner optimum is
attained, `innerOpt_mem`), and every state has a nonempty finite set of available actions. These
hold for the non-convex product sets of factored IMDPs (`FactoredIMDP.productSet_not_convex`).

**Scope.** Abstract: values in `ℝ`, exact `sup`/`inf`/`max`/`min`. Floating-point rounding,
threaded and CUDA execution, and the Lean ↔ Julia correspondence are not proved (see the
inventory, limitations L1–L6).
-/

namespace IntervalMDP.Bellman

open OMax

variable {S A : Type*} [Fintype S]

/-! ### The inner optimum over an ambiguity set -/

/-- The set `{⟨p, V⟩ : p ∈ Γ}` of expected values of `V` under the distributions of `Γ`.

Julia counterpart: the values `dot(V, p)` over the feasible distributions `p` of one ambiguity
set; `state_action_bellman` (`src/bellman.jl`) returns their `sup` or `inf`. -/
def expectations (Γ : AmbiguitySet S) (V : S → ℝ) : Set ℝ := dot V '' Γ.vecs

/-- The inner optimum of `⟨p, V⟩` over `p ∈ Γ`: `sup` for `optimistic`, `inf` for `pessimistic`.

Julia counterpart: `state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)`
(`src/bellman.jl`) with `upper_bound = isoptimistic(spec)` (`src/robust_value_iteration.jl`,
`src/specification.jl`): `upper_bound = true` computes the upper bound (`sup`), `false` the lower
bound (`inf`). Julia computes the optimum with a workspace-specific algorithm (O-maximization,
vertex enumeration, ...); this definition is the exact value those algorithms target. -/
noncomputable def innerOpt (sat : SatisfactionMode) (Γ : AmbiguitySet S) (V : S → ℝ) : ℝ :=
  match sat with
  | .optimistic => sSup (expectations Γ V)
  | .pessimistic => sInf (expectations Γ V)

/-- `⟨x, V⟩` is continuous in the distribution vector `x`.

Julia counterpart: none (Lean-side proof device). -/
theorem continuous_dot (V : S → ℝ) : Continuous (dot V) := by
  unfold dot
  fun_prop

/-- For a well-formed `Γ`, the set of expected values is compact (the continuous image of the
compact set `Γ.vecs`).

Julia counterpart: none (Lean-side proof device). -/
theorem expectations_isCompact {Γ : AmbiguitySet S} (hΓ : Γ.WellFormed) (V : S → ℝ) :
    IsCompact (expectations Γ V) :=
  hΓ.isCompact.image (continuous_dot V)

/-- For a well-formed `Γ`, the set of expected values is nonempty.

Julia counterpart: none (Lean-side proof device). -/
theorem expectations_nonempty {Γ : AmbiguitySet S} (hΓ : Γ.WellFormed) (V : S → ℝ) :
    (expectations Γ V).Nonempty := by
  obtain ⟨p, hp⟩ := hΓ.nonempty
  exact ⟨dot V p, p, AmbiguitySet.coe_mem_vecs.mpr hp, rfl⟩

/-- The inner optimum over a well-formed (nonempty, closed, not necessarily convex) set is
attained: it is `⟨p, V⟩` for some `p ∈ Γ`. This is why `WellFormed` requires closedness.

Julia counterpart: `state_action_bellman` (`src/bellman.jl`) returns the value of a feasible
distribution (for intervals, the greedy O-maximization distribution, `OMax.omax_mem`). -/
theorem innerOpt_mem (sat : SatisfactionMode) {Γ : AmbiguitySet S} (hΓ : Γ.WellFormed)
    (V : S → ℝ) : innerOpt sat Γ V ∈ expectations Γ V := by
  cases sat
  · exact (expectations_isCompact hΓ V).sInf_mem (expectations_nonempty hΓ V)
  · exact (expectations_isCompact hΓ V).sSup_mem (expectations_nonempty hΓ V)

/-- If `V ≤ W + c` pointwise, then `⟨x, V⟩ ≤ ⟨x, W⟩ + c` for every distribution vector `x`
(entries `≥ 0`, summing to `1`).

Julia counterpart: none (Lean-side proof device). -/
theorem dot_le_dot_add {V W : S → ℝ} {c : ℝ} (h : ∀ s, V s ≤ W s + c) {x : S → ℝ}
    (hx : x ∈ stdSimplex ℝ S) : dot V x ≤ dot W x + c := by
  calc dot V x ≤ ∑ s, (W s + c) * x s :=
        Finset.sum_le_sum fun s _ => mul_le_mul_of_nonneg_right (h s) (hx.1 s)
    _ = dot W x + c := by
        simp only [add_mul, Finset.sum_add_distrib, ← Finset.mul_sum, hx.2, mul_one, dot]

/-- Core inequality for the inner optimum, in both satisfaction modes: if `V ≤ W + c`
pointwise, then `innerOpt sat Γ V ≤ innerOpt sat Γ W + c`. Only nonemptiness and boundedness
of the expected values are used, no convexity.

Julia counterpart: `state_action_bellman` (`src/bellman.jl`), both values of `upper_bound`. -/
theorem innerOpt_le_add (sat : SatisfactionMode) {Γ : AmbiguitySet S} (hΓ : Γ.WellFormed)
    {V W : S → ℝ} {c : ℝ} (h : ∀ s, V s ≤ W s + c) :
    innerOpt sat Γ V ≤ innerOpt sat Γ W + c := by
  have hne := expectations_nonempty hΓ V
  cases sat
  · -- pessimistic: `inf V ≤ ⟨x, V⟩ ≤ ⟨x, W⟩ + c` for every `x`, so `inf V - c ≤ inf W`
    have hbdd := (expectations_isCompact hΓ V).bddBelow
    have hle : sInf (expectations Γ V) - c ≤ sInf (expectations Γ W) := by
      refine le_csInf (expectations_nonempty hΓ W) ?_
      rintro _ ⟨x, hx, rfl⟩
      have h₁ : sInf (expectations Γ V) ≤ dot V x := csInf_le hbdd ⟨x, hx, rfl⟩
      linarith [dot_le_dot_add h (Γ.vecs_subset_stdSimplex hx)]
    simp only [innerOpt]
    linarith
  · -- optimistic: `⟨x, V⟩ ≤ ⟨x, W⟩ + c ≤ sup W + c` for every `x`
    have hbdd := (expectations_isCompact hΓ W).bddAbove
    show sSup (expectations Γ V) ≤ sSup (expectations Γ W) + c
    refine csSup_le hne ?_
    rintro _ ⟨x, hx, rfl⟩
    have h₁ : dot W x ≤ sSup (expectations Γ W) := le_csSup hbdd ⟨x, hx, rfl⟩
    linarith [dot_le_dot_add h (Γ.vecs_subset_stdSimplex hx)]

/-! ### The optimum over the available actions -/

/-- The optimum of `values` over a nonempty finite set of actions `acts`: `max` for `maximize`,
`min` for `minimize`.

Julia counterpart: the value returned by `extract_strategy!(strategy_cache, values,
available_actions, jₛ, maximize)` (`src/strategy_cache.jl`) for an `OptimizingStrategyCache` or
`NoStrategyCache` (`maximize ? maximum(values) : minimum(values)` over the available actions). -/
def extractValue (strat : StrategyMode) (acts : Finset A) (hacts : acts.Nonempty)
    (values : A → ℝ) : ℝ :=
  match strat with
  | .maximize => acts.sup' hacts values
  | .minimize => acts.inf' hacts values

/-- Core inequality for the action optimum, in both strategy modes: if `f a ≤ g a + c` for every
`a ∈ acts`, then `extractValue strat acts f ≤ extractValue strat acts g + c`.

Julia counterpart: `extract_strategy!` (`src/strategy_cache.jl`), both values of `maximize`. -/
theorem extractValue_le_add (strat : StrategyMode) {acts : Finset A} (hacts : acts.Nonempty)
    {f g : A → ℝ} {c : ℝ} (h : ∀ a ∈ acts, f a ≤ g a + c) :
    extractValue strat acts hacts f ≤ extractValue strat acts hacts g + c := by
  cases strat
  · -- maximize: `f a ≤ g a + c ≤ max g + c` for every available `a`
    show acts.sup' hacts f ≤ acts.sup' hacts g + c
    refine Finset.sup'_le hacts f fun a ha => ?_
    linarith [h a ha, Finset.le_sup' g ha]
  · -- minimize: `min f - c ≤ f a - c ≤ g a` for every available `a`
    have hle : acts.inf' hacts f - c ≤ acts.inf' hacts g :=
      Finset.le_inf' hacts _ fun a ha => by linarith [h a ha, Finset.inf'_le f ha]
    simp only [extractValue]
    linarith

/-! ### The Bellman operator -/

/-- The value of action `a` in state `s`: the inner optimum of `⟨p, V⟩` over `Γ_{s,a}`.

Julia counterpart: `workspace.actions[jₐ] = state_action_bellman(workspace, V, ambiguity_set,
budget, upper_bound)` with `ambiguity_set = marginal[jₐ, jₛ]`, in `state_bellman!`
(`src/bellman.jl`). -/
noncomputable def stateActionBellman (M : RMDP S A) (sat : SatisfactionMode) (V : S → ℝ) (s : S)
    (a : A) : ℝ :=
  innerOpt sat (M.ambiguity s a) V

/-- The robust Bellman operator
`T V s = opt_strat_{a ∈ available s} opt_sat_{p ∈ Γ_{s,a}} ⟨p, V⟩`: the inner optimum is `sup`
(optimistic) or `inf` (pessimistic), the outer one `max` (maximize) or `min` (minimize).

Julia counterpart: `bellman!(workspace, strategy_cache, Vres, V, model; upper_bound, maximize)`
(`src/bellman.jl`) with an `OptimizingStrategyCache`: `_bellman_helper!` loops over the states
`jₛ` and calls `state_bellman!`, which fills `workspace.actions[jₐ]` for `jₐ ∈ available(model,
jₛ)` (`stateActionBellman`) and sets `Vres[jₛ]` to `extract_strategy!(…, maximize)`
(`extractValue`). `step!` (`src/robust_value_iteration.jl`) passes
`upper_bound = isoptimistic(spec)` (`sat`) and `maximize = ismaximize(spec)` (`strat`). -/
noncomputable def T (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (V : S → ℝ) : S → ℝ :=
  fun s =>
    extractValue strat (M.available s) (M.available_nonempty s) (stateActionBellman M sat V s)

/-- Core inequality of the Bellman operator, all four modes, general `RMDP` (no convexity): if
`V ≤ W + c` pointwise, then `T V ≤ T W + c` pointwise.

Julia counterpart: `bellman!` / `state_bellman!` with an `OptimizingStrategyCache`
(`src/bellman.jl`). -/
theorem T_le_add (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode) {V W : S → ℝ}
    {c : ℝ} (h : ∀ s, V s ≤ W s + c) (s : S) : T M sat strat V s ≤ T M sat strat W s + c :=
  extractValue_le_add strat (M.available_nonempty s) fun a _ =>
    innerOpt_le_add sat (M.ambiguity_wellFormed s a) h

/-- **Monotonicity.** `V ≤ W` pointwise implies `T V ≤ T W` pointwise, for all four
satisfaction × strategy modes and a general `RMDP` (nonempty closed ambiguity sets; no convexity
assumed, so this holds for factored IMDPs).

Julia counterpart: `bellman!` / `state_bellman!` with an `OptimizingStrategyCache`
(`src/bellman.jl`). -/
theorem T_mono (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode) {V W : S → ℝ}
    (h : V ≤ W) : T M sat strat V ≤ T M sat strat W := by
  intro s
  simpa using T_le_add M sat strat (c := 0) (fun s => by simpa using h s) s

/-- `T` is a monotone map (`T_mono` in Mathlib's `Monotone` form, as used by
`Approx.iter_sound`). All four modes; no convexity assumed.

Julia counterpart: `bellman!` (`src/bellman.jl`). -/
theorem T_monotone (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode) :
    Monotone (T M sat strat) :=
  fun _ _ h => T_mono M sat strat h

/-- **Translation equivariance.** Adding a constant `c` to the value function adds `c` to its
Bellman update: `T (V + c) = T V + c`, for all four satisfaction × strategy modes and a general
`RMDP` (no convexity assumed). Uses `∑ₛ p(s) = 1`.

Julia counterpart: `bellman!` / `state_bellman!` with an `OptimizingStrategyCache`
(`src/bellman.jl`). -/
theorem T_add_const (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (V : S → ℝ) (c : ℝ) :
    T M sat strat (V + Function.const S c) = T M sat strat V + Function.const S c := by
  funext s
  apply le_antisymm
  · exact T_le_add M sat strat (by simp) s
  · have h := T_le_add M sat strat (V := V) (W := V + Function.const S c) (c := -c)
      (by simp) s
    simp only [Pi.add_apply, Function.const_apply]
    linarith

/-- **Non-expansiveness** in the sup norm: `‖T V - T W‖∞ ≤ ‖V - W‖∞`, written with Mathlib's
`dist` on `S → ℝ`, which for finite `S` is the sup distance `maxₛ |V s - W s|`. For all four
satisfaction × strategy modes and a general `RMDP` (no convexity assumed).

Julia counterpart: `bellman!` / `state_bellman!` with an `OptimizingStrategyCache`
(`src/bellman.jl`). -/
theorem T_nonexpansive (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (V W : S → ℝ) : dist (T M sat strat V) (T M sat strat W) ≤ dist V W := by
  have hVW : ∀ s, |V s - W s| ≤ dist V W := fun s => by
    simpa [Real.dist_eq] using dist_le_pi_dist V W s
  refine (dist_pi_le_iff dist_nonneg).mpr fun s => ?_
  have h₁ := T_le_add M sat strat (V := V) (W := W) (c := dist V W)
    (fun s => by linarith [(abs_le.mp (hVW s)).2]) s
  have h₂ := T_le_add M sat strat (V := W) (W := V) (c := dist V W)
    (fun s => by linarith [(abs_le.mp (hVW s)).1]) s
  rw [Real.dist_eq, abs_le]
  constructor <;> linarith

/-- `T` is `1`-Lipschitz for the sup distance (`T_nonexpansive` in Mathlib's `LipschitzWith`
form). All four modes; no convexity assumed.

Julia counterpart: `bellman!` (`src/bellman.jl`). -/
theorem T_lipschitz (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode) :
    LipschitzWith 1 (T M sat strat) :=
  LipschitzWith.of_dist_le_mul fun V W => by simpa using T_nonexpansive M sat strat V W

end IntervalMDP.Bellman
