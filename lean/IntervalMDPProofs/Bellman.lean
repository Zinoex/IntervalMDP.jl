import IntervalMDPProofs.Models.RMDP
import IntervalMDPProofs.Models.IMDP
import IntervalMDPProofs.Models.Specification
import IntervalMDPProofs.Models.Strategy
import IntervalMDPProofs.Approx.Lift
import IntervalMDPProofs.Index.Strategy
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

## Interval MDPs, strategy extraction and policy evaluation (2b)

* `T_interval_eq_omax`: for an IMDP in Julia's storage layout (`IntervalMDPLayout`, the columns of
  one `Marginal`), `T` of `IMDP.toRMDP` is the optimum over the available actions of the dense
  O-max values (`OMax.omax`, Phase 1c) of the sets in the columns `sub2ind(marginal, jₐ, jₛ)`
  (`marginalSub2ind`, Phase 1b; `IntervalMDPLayout.column_eq`), with sort direction
  `upper_bound = isoptimistic(sat)`. All four modes.
* `argopt_attains`: the action selected by `extract_strategy!` (`argoptAction`, a transcription of
  the `_extract_strategy!` loop with its strict `>`/`<` tie-breaking) is available and attains the
  outer optimum of `T`; `stationarySeed_available` shows that the seed of a stationary cache is
  always available on a model with fixed available actions. All four modes.
* `policy_eval_sound` (A8): policy evaluation `Tπ` (the `NonOptimizingStrategyCache` path) is the
  Bellman operator of the model restricted to the strategy's actions, and is sound for `T` in the
  strategy-mode direction (`Tπ V ≤ T V` for `maximize`, `≥` for `minimize`), also for every
  value-iteration iterate via `Approx.iter_sound`. Requires a valid strategy (`π(s) ∈ available s`),
  which Julia does not check (Finding F2, `Index.checkStrategy_admits_unavailable`).

Naming: `Bellman.stateActionBellman` is the exact inner optimum over a general ambiguity set;
`OMax.stateActionBellman` is the transcription of the dense O-max `state_action_bellman` loop.
`stateActionBellman_interval_eq_omax` relates them for interval sets.

**Scope.** Abstract: values in `ℝ`, exact `sup`/`inf`/`max`/`min`. Floating-point rounding,
threaded and CUDA execution, and the Lean ↔ Julia correspondence are not proved (see the
inventory, limitations L1–L6).
-/

namespace IntervalMDP.Bellman

open OMax Index

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
(`src/bellman.jl`). Not to be confused with `OMax.stateActionBellman`, the literal transcription of
the dense O-max method of `state_action_bellman`; for interval sets the two agree
(`stateActionBellman_interval_eq_omax`). -/
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

/-! ### Interval MDPs: the inner optimum is O-maximization -/

/-- Julia's `isoptimistic(spec)`, passed to `bellman!` as `upper_bound` and to `sortperm!` as
`rev`: `true` for `optimistic`, `false` for `pessimistic`.

Julia counterpart: `isoptimistic` (`src/specification.jl`), used as
`upper_bound = isoptimistic(spec)` in `step!` (`src/robust_value_iteration.jl`). -/
def isOptimistic : SatisfactionMode → Bool
  | .pessimistic => false
  | .optimistic => true

/-- For an interval ambiguity set, the expected values `expectations` (2a) are the O-max
`valueSet` (1c): `{⟨p, V⟩ : p ∈ P(l, u)}`.

Julia counterpart: none (Lean-side proof device). -/
theorem expectations_toSet (I : IntervalAmbiguity S) (V : S → ℝ) :
    expectations I.toSet V = valueSet I V := by
  ext x
  simp [expectations, valueSet, AmbiguitySet.vecs]

/-- For an interval ambiguity set, the inner optimum is dense O-maximization with the sort
direction `rev = upper_bound = isoptimistic(sat)`: `inf` (pessimistic) is `omax` with `V` sorted
ascending (`omax_eq_sInf`), `sup` (optimistic) is `omax` with `V` sorted descending
(`omax_eq_sSup`). Interval sets are convex, but only `omax_eq_sSup`/`omax_eq_sInf` are used.

Julia counterpart: `bellman_precomputation!` + `state_action_bellman(::DenseIntervalOMaxWorkspace,
…)` with `upper_bound = isoptimistic(spec)` (`src/bellman.jl`). -/
theorem innerOpt_interval_eq_omax {n : ℕ} (I : IntervalAmbiguity (Fin n)) (sat : SatisfactionMode)
    (V : Fin n → ℝ) : innerOpt sat I.toSet V = omax I ⟨V, isOptimistic sat⟩ := by
  cases sat
  · rw [omax_eq_sInf I _ rfl]
    simp only [innerOpt, expectations_toSet]
  · rw [omax_eq_sSup I _ rfl]
    simp only [innerOpt, expectations_toSet]

/-- An interval MDP in Julia's storage layout. `IntervalMarkovDecisionProcess(ambiguity_sets,
num_actions)` stores the interval sets as the columns of one `IntervalAmbiguitySets` and wraps them
in `Marginal(ambiguity_sets, (num_states,), (num_actions,))` (one state variable, one action
variable); the set of source state `jₛ` and action `jₐ` is the column `sub2ind(marginal, jₐ, jₛ)`.
States are `Fin ns` and actions `Fin na` (Julia `jₛ = s + 1`, `jₐ = a + 1`); the available
actions are general (`FactoredRMDP` with `ListAvailableActions`), `AllAvailableActions` being the
case built by `IntervalMarkovDecisionProcess`.

Julia counterpart: the `FactoredRobustMarkovDecisionProcess` returned by
`IntervalMarkovDecisionProcess(ambiguity_set::IntervalAmbiguitySets, num_actions)`
(`src/models/IntervalMarkovDecisionProcess.jl`, `src/probabilities/Marginal.jl`). -/
structure IntervalMDPLayout (ns na : ℕ) extends AvailableActions (Fin ns) (Fin na) where
  /-- Column `j` (1-based) of the `IntervalAmbiguitySets`, Julia `ambiguity_sets[j]`; only the
  columns `1..num_states * num_actions` are read (`IntervalMDPLayout.column_eq`). -/
  ambiguitySets : ℤ → IntervalAmbiguity (Fin ns)
  /-- There is at least one state (Julia `num_target(marginal) ≥ 1`). -/
  numStates_pos : 0 < ns
  /-- There is at least one action (Julia `num_actions ≥ 1`). -/
  numActions_pos : 0 < na

namespace IntervalMDPLayout

variable {ns na : ℕ} (C : IntervalMDPLayout ns na)

/-- The single state variable, of size `num_states`.

Julia counterpart: `state_vars = (num_target(marginal),)` in `IntervalMarkovDecisionProcess`
(`src/models/IntervalMarkovDecisionProcess.jl`). -/
def stateVars : StateVars 1 := ⟨fun _ => ns, fun _ => C.numStates_pos⟩

/-- The single action variable, of size `num_actions`.

Julia counterpart: `action_vars = (num_actions,)` of `Marginal(ambiguity_sets, source_dims,
action_vars)` (`src/probabilities/Marginal.jl`). -/
def actionVars : ActionVars 1 := ⟨fun _ => na, fun _ => C.numActions_pos⟩

/-- The marginal of an interval MDP: it conditions on state variable 1 and action variable 1
(`state_indices = (1,)`, `action_indices = (1,)`); its sets are the columns `ambiguity_sets[j]` at
`j = (jₛ - 1) * num_actions + jₐ`.

Julia counterpart: `Marginal(ambiguity_sets, (num_states,), (num_actions,))`
(`src/probabilities/Marginal.jl`), built by `IntervalMarkovDecisionProcess`
(`src/models/IntervalMarkovDecisionProcess.jl`). -/
def marginal : Marginal C.stateVars C.actionVars 0 where
  numStateIndices := 1
  numActionIndices := 1
  stateIndices _ := 0
  actionIndices _ := 0
  stateIndices_strictMono := Subsingleton.strictMono _
  actionIndices_strictMono := Subsingleton.strictMono _
  sets src act := C.ambiguitySets ((src 0 : ℕ) * na + (act 0 : ℕ) + 1)

/-- The joint state `(jₛ,)` of state `s` (one state variable).

Julia counterpart: the `CartesianIndex` source state `jₛ` of `state_bellman!` (`src/bellman.jl`). -/
def jointState (s : Fin ns) : C.stateVars.State := fun _ => s

/-- The joint action `(jₐ,)` of action `a` (one action variable).

Julia counterpart: the `CartesianIndex` action `jₐ` of `state_bellman!` (`src/bellman.jl`). -/
def jointAction (a : Fin na) : C.actionVars.Action := fun _ => a

/-- The column of the interval set of state `s` and action `a`: `sub2ind(marginal, jₐ, jₛ)`.

Julia counterpart: `sub2ind(marginal, jₐ, jₛ)` in `getindex(marginal, jₐ, jₛ)`
(`marginal[jₐ, jₛ]`) and `workspace.budget[sub2ind(marginal, jₐ, jₛ)]` of `state_bellman!`
(`src/bellman.jl`, `src/probabilities/Marginal.jl`). -/
def column (s : Fin ns) (a : Fin na) : ℤ :=
  marginalSub2ind C.marginal (Index.actionTuple (C.jointAction a))
    (juliaTuple (dims := C.stateVars.dims) (C.jointState s))

/-- The IMDP stored by the layout: `Γ_{s,a}` is the interval set in column
`sub2ind(marginal, jₐ, jₛ)`.

Julia counterpart: `marginal[jₐ, jₛ]` for the `IntervalMarkovDecisionProcess`
(`src/models/IntervalMarkovDecisionProcess.jl`, `src/probabilities/Marginal.jl`). -/
def toIMDP : IMDP (Fin ns) (Fin na) where
  toAvailableActions := C.toAvailableActions
  ambiguitySets s a := C.ambiguitySets (C.column s a)

/-- The column of `(s, a)` is `(jₛ - 1) * num_actions + jₐ` with `jₛ = s + 1`, `jₐ = a + 1`, in
`1..num_states * num_actions` (Phase 1b, `marginalSub2ind_eq_linear`, for this layout).

Julia counterpart: `sub2ind(marginal, jₐ, jₛ)` (`src/probabilities/Marginal.jl`). -/
theorem column_eq (s : Fin ns) (a : Fin na) : C.column s a = ((s : ℕ) * na + a + 1 : ℕ) := by
  simp [column, marginalSub2ind, marginalSourceDims, marginalActionVars, marginal, stateVars,
    actionVars, juliaTuple, Index.actionTuple, jointState, jointAction, toJulia, List.finRange_succ]

/-- Under the overflow bound `num_states * num_actions < 2 ^ (N - 1)`, the column computed in
`N`-bit integers is the exact column.

Julia counterpart: `sub2ind(marginal, jₐ, jₛ)` in `Int` arithmetic
(`src/probabilities/Marginal.jl`). -/
theorem columnInt_eq {N : ℕ} (hBound : ns * na < 2 ^ (N - 1)) (s : Fin ns) (a : Fin na) :
    marginalSub2indInt N C.marginal (Index.actionTuple (C.jointAction a))
      (juliaTuple (dims := C.stateVars.dims) (C.jointState s)) = C.column s a := by
  have hlt : (s : ℕ) * na + a + 1 < 2 ^ (N - 1) := by
    have h1 : (s : ℕ) * na + a + 1 ≤ ns * na := by
      have := s.is_lt
      have := a.is_lt
      nlinarith
    omega
  have h := machineInt_eq_self hlt
  rw [machineInt] at h
  rw [marginalSub2indInt, ← column, column_eq, h]

end IntervalMDPLayout

/-- The value of action `a` in state `s` of an interval MDP as Julia computes it: dense
O-maximization (`OMax.omax`, 1c) of the interval set in column `sub2ind(marginal, jₐ, jₛ)` (1b),
with `V` sorted in direction `rev = upper_bound = isoptimistic(sat)`.

Julia counterpart: `workspace.actions[jₐ] = state_action_bellman(workspace, V, marginal[jₐ, jₛ],
workspace.budget[sub2ind(marginal, jₐ, jₛ)], upper_bound)` in `state_bellman!` for
`DenseIntervalOMaxWorkspace` (`src/bellman.jl`). -/
noncomputable def intervalStateActionBellman {ns na : ℕ} (C : IntervalMDPLayout ns na)
    (sat : SatisfactionMode) (V : Fin ns → ℝ) (s : Fin ns) (a : Fin na) : ℝ :=
  omax (C.ambiguitySets (C.column s a)) ⟨V, isOptimistic sat⟩

/-- For an interval MDP, the inner optimum of `T` for state `s` and action `a` is the O-max value
of the interval set in column `sub2ind(marginal, jₐ, jₛ)`. Both satisfaction modes.

Julia counterpart: `state_action_bellman(::DenseIntervalOMaxWorkspace, …)` in `state_bellman!`
(`src/bellman.jl`). -/
theorem stateActionBellman_interval_eq_omax {ns na : ℕ} (C : IntervalMDPLayout ns na)
    (sat : SatisfactionMode) (V : Fin ns → ℝ) (s : Fin ns) (a : Fin na) :
    stateActionBellman C.toIMDP.toRMDP sat V s a = intervalStateActionBellman C sat V s a :=
  innerOpt_interval_eq_omax _ sat V

/-- **`T` for interval MDPs.** For the IMDP stored in Julia's layout (`IMDP.toRMDP` of
`IntervalMDPLayout.toIMDP`), the robust Bellman operator is the optimum over the available actions
(`max`/`min` by `strat`) of the O-max values (`omax`, 1c) of the interval sets in the columns
`sub2ind(marginal, jₐ, jₛ)` (1b), sorted in direction `upper_bound = isoptimistic(sat)`. All four
satisfaction × strategy modes.

Julia counterpart: `bellman!` → `state_bellman!` with an `OptimizingStrategyCache` and a
`DenseIntervalOMaxWorkspace` (`src/bellman.jl`), `extract_strategy!` (`src/strategy_cache.jl`). -/
theorem T_interval_eq_omax {ns na : ℕ} (C : IntervalMDPLayout ns na) (sat : SatisfactionMode)
    (strat : StrategyMode) (V : Fin ns → ℝ) (s : Fin ns) :
    T C.toIMDP.toRMDP sat strat V s =
      extractValue strat (C.available s) (C.available_nonempty s)
        (intervalStateActionBellman C sat V s) := by
  have h : stateActionBellman C.toIMDP.toRMDP sat V s = intervalStateActionBellman C sat V s :=
    funext (stateActionBellman_interval_eq_omax C sat V s)
  simp only [T, h]
  rfl

/-! ### Strategy extraction -/

/-- `optLE strat x y`: `y` is at least as good as `x` for the strategy mode, `x ≤ y` for
`maximize` and `y ≤ x` for `minimize`.

Julia counterpart: the negation of `gt(x, y)` with `gt = maximize ? (>) : (<)` in
`_extract_strategy!` (`src/strategy_cache.jl`). -/
def optLE : StrategyMode → ℝ → ℝ → Prop
  | .maximize, x, y => x ≤ y
  | .minimize, x, y => y ≤ x

/-- `optLE` is reflexive.

Julia counterpart: none (Lean-side proof device). -/
theorem optLE_refl (strat : StrategyMode) (x : ℝ) : optLE strat x x := by
  cases strat <;> exact le_refl x

/-- `optLE` is transitive.

Julia counterpart: none (Lean-side proof device). -/
theorem optLE_trans {strat : StrategyMode} {x y z : ℝ} (h₁ : optLE strat x y)
    (h₂ : optLE strat y z) : optLE strat x z := by
  cases strat
  · exact le_trans (α := ℝ) h₁ h₂
  · exact le_trans (α := ℝ) h₂ h₁

/-- One iteration of the `_extract_strategy!` loop: the candidate `jₐ` replaces the incumbent
`opt` only if it is **strictly** better, `values[jₐ] > values[opt]` (`maximize`) or
`values[jₐ] < values[opt]` (`minimize`); on ties the incumbent is kept.

Julia counterpart: the body `v = values[jₐ]; if gt(v, opt_val); opt_val = v; opt_index =
Tuple(jₐ); end` of `_extract_strategy!` (`src/strategy_cache.jl`), with `opt_val = values[opt]`. -/
noncomputable def argoptStep (strat : StrategyMode) (values : A → ℝ) (opt jₐ : A) : A :=
  match strat with
  | .maximize => if values opt < values jₐ then jₐ else opt
  | .minimize => if values jₐ < values opt then jₐ else opt

/-- The action selected by `extract_strategy!`: a left fold of `argoptStep` over the available
actions `acts`, in Julia's iteration order, starting from the incumbent `seed`. Ties keep the
earlier action (strict comparison), so the result is the first optimal action after `seed`.

The seed models Julia's `neutral`. `TimeVaryingStrategyCache` (and a fresh or reset
`StationaryStrategyCache`) start from `(typemin(R), first(available_actions))` (`typemax` for
`minimize`); in `ℝ` the first comparison then always succeeds, which is the same as starting from
`seed = first(available_actions)` with its own value. `StationaryStrategyCache` otherwise starts
from the previous action `s` with `values[s]` (`stationarySeed`).

Julia counterpart: `extract_strategy!(::TimeVaryingStrategyCache | ::StationaryStrategyCache,
values, available_actions, jₛ, maximize)` and the loop `for jₐ in available_actions` of
`_extract_strategy!` (`src/strategy_cache.jl`); the stored `cur_strategy[jₛ] = opt_index` is this
action and the returned `opt_val` is its value. -/
noncomputable def argoptAction (strat : StrategyMode) (values : A → ℝ) (seed : A) (acts : List A) :
    A :=
  acts.foldl (argoptStep strat values) seed

/-- One loop step returns the incumbent or the candidate, and the result is at least as good as
both.

Julia counterpart: the loop body of `_extract_strategy!` (`src/strategy_cache.jl`). -/
theorem argoptStep_spec (strat : StrategyMode) (values : A → ℝ) (opt jₐ : A) :
    (argoptStep strat values opt jₐ = opt ∨ argoptStep strat values opt jₐ = jₐ) ∧
      optLE strat (values opt) (values (argoptStep strat values opt jₐ)) ∧
      optLE strat (values jₐ) (values (argoptStep strat values opt jₐ)) := by
  cases strat
  · by_cases h : values opt < values jₐ
    · simp only [argoptStep, if_pos h, optLE]
      exact ⟨by simp, h.le, le_refl _⟩
    · simp only [argoptStep, if_neg h, optLE]
      exact ⟨by simp, le_refl _, not_lt.mp h⟩
  · by_cases h : values jₐ < values opt
    · simp only [argoptStep, if_pos h, optLE]
      exact ⟨by simp, h.le, le_refl _⟩
    · simp only [argoptStep, if_neg h, optLE]
      exact ⟨by simp, le_refl _, not_lt.mp h⟩

/-- The loop invariant of `_extract_strategy!`: the selected action is the seed or one of the
actions iterated over, and it is at least as good as the seed and every iterated action.

Julia counterpart: the loop `for jₐ in available_actions` of `_extract_strategy!`
(`src/strategy_cache.jl`). -/
theorem argoptAction_spec (strat : StrategyMode) (values : A → ℝ) (acts : List A) (seed : A) :
    (argoptAction strat values seed acts = seed ∨ argoptAction strat values seed acts ∈ acts) ∧
      ∀ a ∈ seed :: acts,
        optLE strat (values a) (values (argoptAction strat values seed acts)) := by
  induction acts generalizing seed with
  | nil => simp [argoptAction, optLE_refl]
  | cons b L ih =>
    obtain ⟨hmem, hle⟩ := ih (argoptStep strat values seed b)
    obtain ⟨hstep, hs, hb⟩ := argoptStep_spec strat values seed b
    have hrw : argoptAction strat values seed (b :: L) =
        argoptAction strat values (argoptStep strat values seed b) L := rfl
    have htop := hle _ List.mem_cons_self
    rw [hrw]
    refine ⟨?_, fun a ha => ?_⟩
    · rcases hmem with h | h
      · rcases hstep with h' | h'
        · exact Or.inl (h.trans h')
        · exact Or.inr (by rw [h, h']; exact List.mem_cons_self)
      · exact Or.inr (List.mem_cons_of_mem _ h)
    · rcases List.mem_cons.mp ha with rfl | ha
      · exact optLE_trans hs htop
      · rcases List.mem_cons.mp ha with rfl | ha
        · exact optLE_trans hb htop
        · exact hle a (List.mem_cons_of_mem _ ha)

/-- If `r ∈ acts` is at least as good as every action of `acts`, it attains the action optimum:
`extractValue strat acts values = values r`.

Julia counterpart: `extract_strategy!` (`src/strategy_cache.jl`), whose returned `opt_val` is the
`maximum`/`minimum` of `NoStrategyCache`. -/
theorem extractValue_eq_of_opt (strat : StrategyMode) {acts : Finset A} (hacts : acts.Nonempty)
    {values : A → ℝ} {r : A} (hr : r ∈ acts) (h : ∀ a ∈ acts, optLE strat (values a) (values r)) :
    extractValue strat acts hacts values = values r := by
  cases strat
  · exact le_antisymm (Finset.sup'_le hacts values h) (Finset.le_sup' values hr)
  · exact le_antisymm (Finset.inf'_le values hr) (Finset.le_inf' hacts values h)

/-- **Strategy extraction attains the optimum.** For any iteration order `acts` of the available
actions and any available seed (Julia's `first(available_actions)`, or the previous stationary
choice, `stationarySeed_available`), the action selected by `extract_strategy!` is available and
its value is the outer optimum of `T`: `stateActionBellman (argoptAction …) = T M sat strat V s`.
All four satisfaction × strategy modes; ties are broken as in Julia (strict `>`/`<`).

Julia counterpart: `extract_strategy!` / `_extract_strategy!` (`src/strategy_cache.jl`) called from
`state_bellman!` with an `OptimizingStrategyCache` (`src/bellman.jl`). -/
theorem argopt_attains [DecidableEq A] (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (V : S → ℝ) (s : S) {acts : List A}
    (hacts : acts.toFinset = M.available s) {seed : A} (hseed : seed ∈ M.available s) :
    argoptAction strat (stateActionBellman M sat V s) seed acts ∈ M.available s ∧
      stateActionBellman M sat V s (argoptAction strat (stateActionBellman M sat V s) seed acts) =
        T M sat strat V s := by
  obtain ⟨hmem, hle⟩ := argoptAction_spec strat (stateActionBellman M sat V s) acts seed
  have hin : argoptAction strat (stateActionBellman M sat V s) seed acts ∈ M.available s := by
    rcases hmem with h | h
    · rw [h]; exact hseed
    · rw [← hacts]; exact List.mem_toFinset.mpr h
  refine ⟨hin, (extractValue_eq_of_opt strat _ hin fun a ha => hle a ?_).symm⟩
  rw [← hacts] at ha
  exact List.mem_cons_of_mem _ (List.mem_toFinset.mp ha)

/-- The seed of `extract_strategy!` for a `StationaryStrategyCache` at state `s`, call `k` of
value iteration (value functions `V 0, V 1, …`). Call `0` starts from the zero tuple, so from
`first(available_actions)`; call `k + 1` starts from the action stored by call `k` when `keep k`
holds, otherwise from `first(available_actions)`. `keep k` is the outcome of Julia's guard
`!(all(iszero.(s)) || CartesianIndex(s) ∉ available_actions)`, left arbitrary here: before the
B-1 fix the guard compared the state index `jₛ` with the action list (Observation O11).

Julia counterpart: `neutral` in `extract_strategy!(::StationaryStrategyCache, …)`
(`src/strategy_cache.jl`) across successive `bellman!` calls (`src/robust_value_iteration.jl`). -/
noncomputable def stationarySeed (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (V : ℕ → S → ℝ) (s : S) (acts : List A) (first : A) (keep : ℕ → Bool) : ℕ → A
  | 0 => first
  | k + 1 =>
    if keep k then argoptAction strat (stateActionBellman M sat (V k) s)
      (stationarySeed M sat strat V s acts first keep k) acts
    else first

/-- On a model with fixed available actions, every seed of a `StationaryStrategyCache` is
available, whatever the outcome of the guard: it is `first(available_actions)` or an action selected
by an earlier call for the same state (`argopt_attains`). So the hypothesis `hseed` of
`argopt_attains` holds for every call of value iteration.

Julia counterpart: `extract_strategy!(::StationaryStrategyCache, …)` (`src/strategy_cache.jl`)
across successive `bellman!` calls (`src/robust_value_iteration.jl`). -/
theorem stationarySeed_available [DecidableEq A] (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (V : ℕ → S → ℝ) (s : S) {acts : List A}
    (hacts : acts.toFinset = M.available s) {first : A} (hfirst : first ∈ M.available s)
    (keep : ℕ → Bool) (k : ℕ) :
    stationarySeed M sat strat V s acts first keep k ∈ M.available s := by
  induction k with
  | zero => exact hfirst
  | succ k ih =>
    by_cases h : keep k = true
    · simp only [stationarySeed, h, if_true]
      exact (argopt_attains M sat strat (V k) s hacts ih).1
    · simp only [stationarySeed, h]
      exact hfirst

/-! ### Policy evaluation for a given strategy -/

/-- The available actions `{π(s)}` of a strategy: one action per state.

Julia counterpart: the single action `jₐ = CartesianIndex(strategy_cache[jₛ])` evaluated by
`state_bellman!` with a `NonOptimizingStrategyCache` (`src/bellman.jl`). -/
def strategyAvailable (π : StationaryStrategy S A) : AvailableActions S A where
  available s := {π.strategy s}
  available_nonempty _ := Finset.singleton_nonempty _

/-- Policy evaluation for a given strategy `π`: `Tπ V s = opt_{p ∈ Γ_{s,π(s)}} ⟨p, V⟩`, the inner
optimum (`sup` optimistic / `inf` pessimistic) of the strategy's action only. There is no outer
optimum: Julia's non-optimizing path ignores `maximize`. A time-varying strategy uses step `k`'s
decision rule at step `k` (`GivenStrategyCache[k]`).

Julia counterpart: `state_bellman!` with a `NonOptimizingStrategyCache` (`src/bellman.jl`):
`jₐ = CartesianIndex(strategy_cache[jₛ])` (`Index.strategyAction`), then
`Vres[jₛ] = state_action_bellman(workspace, V, marginal[jₐ, jₛ], budget, upper_bound)`. -/
noncomputable def Tπ (M : RMDP S A) (sat : SatisfactionMode) (π : StationaryStrategy S A)
    (V : S → ℝ) : S → ℝ :=
  fun s => stateActionBellman M sat V s (π.strategy s)

/-- The direction in which policy evaluation is sound for the optimal operator, as a
satisfaction mode for `Approx.Sound`: a fixed strategy is never better than the optimizing one, so
for `maximize` `Tπ V ≤ T V` (the direction of `Sound .pessimistic`) and for `minimize`
`Tπ V ≥ T V` (the direction of `Sound .optimistic`).

Julia counterpart: `strategy_mode(spec)` (`Maximize`/`Minimize`, `src/specification.jl`) of a
`VerificationProblem` with a given strategy (`src/problem.jl`). -/
def policyEvalMode : StrategyMode → SatisfactionMode
  | .maximize => .pessimistic
  | .minimize => .optimistic

/-- Policy evaluation is the robust Bellman operator of the model restricted to the strategy's
actions (`available s = {π(s)}`), in either strategy mode.

Julia counterpart: `state_bellman!` with a `NonOptimizingStrategyCache` (`src/bellman.jl`). -/
theorem Tπ_eq_T_strategyAvailable (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (π : StationaryStrategy S A) :
    Tπ M sat π = T (M.withAvailable (strategyAvailable π)) sat strat := by
  funext V s
  cases strat <;>
    simp [Tπ, T, extractValue, strategyAvailable, RMDP.withAvailable, stateActionBellman]

/-- One step of policy evaluation is sound for the optimal operator in the strategy-mode direction:
`Tπ V ≤ T V` for `maximize`, `Tπ V ≥ T V` for `minimize`, for a strategy that is valid for the
model (`π(s) ∈ available s`).

Julia counterpart: `state_bellman!` with a `NonOptimizingStrategyCache` versus an
`OptimizingStrategyCache` (`src/bellman.jl`). -/
theorem Tπ_stepSound (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    {π : StationaryStrategy S A} (hπ : π.Valid M.toAvailableActions) :
    Approx.StepSound (policyEvalMode strat) (Tπ M sat π) (T M sat strat) := by
  intro V
  cases strat
  · intro s
    exact Finset.le_sup' (stateActionBellman M sat V s) (hπ s)
  · intro s
    exact Finset.inf'_le (stateActionBellman M sat V s) (hπ s)

/-- **A8: given-strategy verification.** For a strategy `π` valid for the model:

1. *exact*: `Tπ` evaluates exactly the strategy's action, i.e. it is the Bellman operator of the
   model restricted to `available s = {π(s)}`;
2. *one step sound*: `Sound (policyEvalMode strat) (Tπ V) (T V)`, i.e. `Tπ V ≤ T V` for
   `maximize` and `Tπ V ≥ T V` for `minimize`;
3. *value iteration sound* (via `Approx.iter_sound` and `T_monotone`): the same holds for every
   iterate from a common start `V₀`.

All four satisfaction × strategy modes. Julia does not check `π.Valid` (Finding F2).

Julia counterpart: `state_bellman!` with a `NonOptimizingStrategyCache` (`src/bellman.jl`), iterated
by `_value_iteration!` (`src/robust_value_iteration.jl`) for a `VerificationProblem` with a given
strategy (`src/problem.jl`). -/
theorem policy_eval_sound (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    {π : StationaryStrategy S A} (hπ : π.Valid M.toAvailableActions) (V₀ : S → ℝ) :
    Tπ M sat π = T (M.withAvailable (strategyAvailable π)) sat strat ∧
      Approx.StepSound (policyEvalMode strat) (Tπ M sat π) (T M sat strat) ∧
      ∀ n, Approx.Sound (policyEvalMode strat) ((Tπ M sat π)^[n] V₀) ((T M sat strat)^[n] V₀) :=
  ⟨Tπ_eq_T_strategyAvailable M sat strat π, Tπ_stepSound M sat strat hπ,
    (Approx.iter_sound (Tπ_stepSound M sat strat hπ) (Or.inl (T_monotone M sat strat))
      (Approx.Sound.refl _ V₀)).1⟩

/-- Policy evaluation on an interval MDP is the O-max value (1c) of the interval set in column
`sub2ind(marginal, jₐ, jₛ)` (1b) of the strategy's action. Both satisfaction modes.

Julia counterpart: `state_bellman!` with a `NonOptimizingStrategyCache` and a
`DenseIntervalOMaxWorkspace` (`src/bellman.jl`). -/
theorem Tπ_interval_eq_omax {ns na : ℕ} (C : IntervalMDPLayout ns na) (sat : SatisfactionMode)
    (π : StationaryStrategy (Fin ns) (Fin na)) (V : Fin ns → ℝ) (s : Fin ns) :
    Tπ C.toIMDP.toRMDP sat π V s = intervalStateActionBellman C sat V s (π.strategy s) :=
  stateActionBellman_interval_eq_omax C sat V s (π.strategy s)

end IntervalMDP.Bellman
