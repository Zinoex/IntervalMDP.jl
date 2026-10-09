import IntervalMDPProofs.VI.Safety
import IntervalMDPProofs.VI.ExitTime
import IntervalMDPProofs.VI.Reward

/-!
# Synthesized strategies (A7)

For a `ControlSynthesisProblem`, `solve` (`src/robust_value_iteration.jl`) runs
`_value_iteration!` with an `OptimizingStrategyCache` and returns
`cachetostrategy(strategy_cache)` (`src/strategy_cache.jl`):

* finite horizon (`isfinitetime(prop)`): a `TimeVaryingStrategyCache`. Every `bellman!` call
  (`step!(…, k, …)`, `k = 0, 1, …`) runs `extract_strategy!` per state from the neutral
  `(typemin, first(available_actions))` and writes `cur_strategy[jₛ]`;
  `step_postprocess_strategy_cache!` pushes a copy, and `cachetostrategy` returns
  `TimeVaryingStrategy(reverse(strategy))`, so Julia's `strategy[i]` is the decision of call
  `K - i`;
* infinite horizon: a `StationaryStrategyCache`. Call `k + 1` seeds `extract_strategy!` with the
  action stored by call `k` and its current value (ties keep it); call `0` starts from the zero
  tuple, i.e. from `first(available_actions)`. The cache after the last call is returned as a
  `StationaryStrategy`.

Both caches choose the action with `argoptAction` (`Bellman.lean`, the `_extract_strategy!` loop
with strict `>`/`<`); they differ only in the seed (`synthesizedStrategy` with `timeVaryingSeed`
or `stationaryCacheSeed`). The model follows the **documented** behaviour of the stationary cache
(the previous action is always kept on ties). Before the B-1 fix (issue #119), Julia's guard
`jₛ ∉ available_actions` in `src/strategy_cache.jl` compared the state index with the actions and
reset the seed for every state whose index exceeds the number of actions (benchmark Finding B-1);
this is inventory Finding F3, witnessed in `Models/Examples.lean` (`b1_stationary_unsound`). The
fixed guard tests the cached action (`CartesianIndex(s) ∉ available_actions`).

The generic iteration `viIter` (`post ∘ T` from `V₀`) is the common shape of `reachIter`,
`safetyIter`, `exitIter` and `rewardIter` (`reachIter_eq_viIter`, …).

## Results

* `timeVarying_attains` (A7, finite horizon): evaluating the returned time-varying strategy
  (`policyEvalIter`, the `GivenStrategyCache` loop with `strategy[time_length - k]` at call `k`)
  gives exactly the computed `V_K`, for every step postprocessing (all property types) and all
  four satisfaction × strategy modes; instances `timeVarying_attains_reach` (all six reachability
  types), `timeVarying_attains_safety`, `timeVarying_attains_reward`.
* `stationary_sound` (A7, infinite horizon): for reachability / reach-avoid (`isExactTime = false`)
  the value of the returned stationary strategy (`strategyReachLfp`, the least fixed point of its
  policy-evaluation step) is at least the computed `V_{K+1}`, i.e.
  `Sound .pessimistic V_{K+1} V^σ`, in all four modes. The proof for `maximize` uses that the
  cache keeps its action on ties (`strategy_eq_of_T_eq`): along the iterates, a state switches only
  on a strict increase of its value; for `minimize` every valid strategy works
  (`viIter_le_superSolution_minimize`). Instances for the other infinite-horizon properties:
  `stationary_sound_exitTime` (every nonnegative real super-solution of the strategy's exit-time
  step) and `stationary_reward_error_bound` (`ν < 1`: the strategy's value lies in the A5
  interval around `V_{K+1}`).

**Direction.** As for A4 (`reachIter_sound`), the conclusion is `V_{K+1} ≤ V^σ` in every mode. For
`sat = optimistic` the conservative direction would be `V^σ ≤ V_{K+1}`, which is false already
for the exact value (the iterates approach from below); it is not claimed. Infinite-time safety is
not covered: its iterates decrease, so `V_K` is not a lower bound (no A4 for safety).

**Scope.** Abstract (values in `ℝ`, exact operators), time-invariant model (no
`TimeVaryingAvailableActions`), strategy value = dynamic-programming fixed point (not the path
measure). Floating point, threads, CUDA and the Lean ↔ Julia correspondence are limitations
L1–L5 of the inventory.
-/

namespace IntervalMDP.VI

open Bellman Approx
open OMax (dot)

variable {S A : Type*} [Fintype S] [DecidableEq A]

/-! ### Iteration order of the available actions -/

/-- Julia's iteration order of the available actions: `available_actions = available(model, jₛ)`
iterated by `for jₐ in available_actions` in `_extract_strategy!`, and its `first` element, which
seeds the time-varying cache.

Julia counterpart: `available(model, jₛ)` (`src/available_actions.jl`) as used by
`state_bellman!` (`src/bellman.jl`) and `extract_strategy!` (`src/strategy_cache.jl`). -/
structure ActionOrder (M : RMDP S A) where
  /-- The available actions of state `s` in Julia's iteration order. -/
  acts : S → List A
  /-- The list enumerates exactly the available actions. -/
  toFinset_acts : ∀ s, (acts s).toFinset = M.available s

namespace ActionOrder

variable {M : RMDP S A} (o : ActionOrder M)

/-- The iteration order of a state is nonempty (`available s` is nonempty).

Julia counterpart: none (Lean-side proof device). -/
theorem acts_ne_nil (s : S) : o.acts s ≠ [] := by
  intro h
  have h' := o.toFinset_acts s
  rw [h, List.toFinset_nil] at h'
  exact (M.available_nonempty s).ne_empty h'.symm

/-- `first(available_actions)`, the action of the neutral element of `extract_strategy!`.

Julia counterpart: `Tuple(first(available_actions))` in `extract_strategy!`
(`src/strategy_cache.jl`). -/
def first (s : S) : A :=
  (o.acts s).head (o.acts_ne_nil s)

/-- `first(available_actions)` is available.

Julia counterpart: `first(available_actions)` (`src/strategy_cache.jl`). -/
theorem first_mem (s : S) : o.first s ∈ M.available s := by
  rw [← o.toFinset_acts s]
  exact List.mem_toFinset.mpr (List.head_mem _)

end ActionOrder

/-! ### The generic value-iteration loop -/

omit [DecidableEq A] in
/-- The value-iteration iterates for a step postprocessing `post` from `V₀`: `viIter 0 = V₀` and
`viIter (k + 1) = post (T (viIter k))`. `reachIter`, `safetyIter`, `exitIter` and `rewardIter`
are instances (`reachIter_eq_viIter`, …).

Julia counterpart: the loop of `_value_iteration!` (`src/robust_value_iteration.jl`): `step!` is
`bellman!` followed by `step_postprocess_value_function!` (`src/specification.jl`). -/
noncomputable def viIter (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (post : (S → ℝ) → S → ℝ) (V₀ : S → ℝ) : ℕ → S → ℝ
  | 0 => V₀
  | k + 1 => post (T M sat strat (viIter M sat strat post V₀ k))

/-! ### The strategy caches -/

/-- The decision rule written by value-iteration call `k` (Julia's `step!(…, k, …)`, input value
function `V k`): in each state `s`, the action selected by `extract_strategy!` from the seed
`seed k s` over Julia's iteration order. With `timeVaryingSeed` this is `cur_strategy` of a
`TimeVaryingStrategyCache`, with `stationaryCacheSeed` the content of a `StationaryStrategyCache`
after call `k`.

Julia counterpart: `extract_strategy!(::TimeVaryingStrategyCache | ::StationaryStrategyCache, …)`
and `_extract_strategy!` (`src/strategy_cache.jl`), called per state by `state_bellman!`
(`src/bellman.jl`). -/
noncomputable def synthesizedStrategy (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (o : ActionOrder M) (V : ℕ → S → ℝ) (seed : ℕ → S → A) (k : ℕ) :
    StationaryStrategy S A where
  strategy s := argoptAction strat (stateActionBellman M sat (V k) s) (seed k s) (o.acts s)

/-- The seed of a `TimeVaryingStrategyCache`: `first(available_actions)` at every call (the
neutral `(typemin(R), first(available_actions))`, `typemax` for `minimize`).

Julia counterpart: `extract_strategy!(::TimeVaryingStrategyCache, …)` (`src/strategy_cache.jl`). -/
def timeVaryingSeed {M : RMDP S A} (o : ActionOrder M) : ℕ → S → A :=
  fun _ s => o.first s

/-- The seed of a `StationaryStrategyCache` at call `k`: `first(available_actions)` at call `0`
(zero tuple), and the action stored by call `k - 1` afterwards (`stationarySeed` with the guard
always passing — the documented behaviour, which the fixed Julia guard implements; before the B-1
fix (issue #119) Julia's guard reset the seed, Finding F3 / B-1).

Julia counterpart: `neutral` in `extract_strategy!(::StationaryStrategyCache, …)`
(`src/strategy_cache.jl`). -/
noncomputable def stationaryCacheSeed (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (o : ActionOrder M) (V : ℕ → S → ℝ) : ℕ → S → A :=
  fun k s => stationarySeed M sat strat V s (o.acts s) (o.first s) (fun _ => true) k

/-- The strategy returned for a finite horizon after `K` calls: Julia's `strategy[i]`
(Lean index `i - 1`) is the decision rule of call `K - i`, i.e. Lean index `j` holds call
`Fin.rev j = K - 1 - j`.

Julia counterpart: `cachetostrategy(::TimeVaryingStrategyCache) =
TimeVaryingStrategy(collect(reverse(strategy_cache.strategy)))` with
`step_postprocess_strategy_cache!` pushing `cur_strategy` after every call
(`src/strategy_cache.jl`). -/
noncomputable def timeVaryingCacheStrategy (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (o : ActionOrder M) (V : ℕ → S → ℝ) (K : ℕ) :
    TimeVaryingStrategy S A where
  timeLength := K
  strategy j := (synthesizedStrategy M sat strat o V (timeVaryingSeed o) (Fin.rev j : ℕ)).strategy

/-- The strategy returned for an infinite horizon after the calls `0, …, K`: the content of the
`StationaryStrategyCache` after call `K`, which produced the returned value `V (K + 1)`.

Julia counterpart: `cachetostrategy(::StationaryStrategyCache) =
StationaryStrategy(strategy_cache.strategy)` (`src/strategy_cache.jl`). -/
noncomputable def stationaryCacheStrategy (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (o : ActionOrder M) (V : ℕ → S → ℝ) (K : ℕ) : StationaryStrategy S A :=
  synthesizedStrategy M sat strat o V (stationaryCacheSeed M sat strat o V) K

/-! ### Policy evaluation of a time-varying strategy -/

omit [DecidableEq A] in
/-- The decision rule a `VerificationProblem` applies at call `k` of value iteration:
`strategy_cache[time_length(strategy_cache) - k]` (1-based), i.e. Lean index `Fin.rev k`.

Julia counterpart: `select_strategy_cache(strategy_cache::NonOptimizingStrategyCache, k)`
(`src/robust_value_iteration.jl`) and `getindex(::GivenStrategyCache, k)`
(`src/strategy_cache.jl`). -/
def selectStrategy (π : TimeVaryingStrategy S A) (k : Fin π.timeLength) : StationaryStrategy S A :=
  ⟨π.strategy (Fin.rev k)⟩

omit [DecidableEq A] in
/-- Value iteration of a `VerificationProblem` with a given time-varying strategy `π`: call `k`
applies policy evaluation `Tπ` with `selectStrategy π k`, then the postprocessing. Julia runs
exactly `time_length(π)` calls; beyond them the Lean iterate is left unchanged.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) with a
`GivenStrategyCache` (`src/strategy_cache.jl`) and `state_bellman!` on the
`NonOptimizingStrategyCache` path (`src/bellman.jl`). -/
noncomputable def policyEvalIter (M : RMDP S A) (sat : SatisfactionMode)
    (post : (S → ℝ) → S → ℝ) (V₀ : S → ℝ) (π : TimeVaryingStrategy S A) : ℕ → S → ℝ
  | 0 => V₀
  | k + 1 =>
    if h : k < π.timeLength then
      post (Tπ M sat (selectStrategy π ⟨k, h⟩) (policyEvalIter M sat post V₀ π k))
    else policyEvalIter M sat post V₀ π k

/-! ### Every synthesized decision rule attains `T` -/

/-- A decision rule synthesized from available seeds is valid and its policy evaluation at the
call's input `V k` is the Bellman update: `Tπ (synthesizedStrategy … k) (V k) = T (V k)`.

Julia counterpart: `extract_strategy!` (`src/strategy_cache.jl`) in `state_bellman!`
(`src/bellman.jl`); the stored action attains the returned `opt_val`. -/
theorem synthesizedStrategy_spec (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (V : ℕ → S → ℝ) {seed : ℕ → S → A} (k : ℕ)
    (hseed : ∀ s, seed k s ∈ M.available s) :
    (synthesizedStrategy M sat strat o V seed k).Valid M.toAvailableActions ∧
      Tπ M sat (synthesizedStrategy M sat strat o V seed k) (V k) = T M sat strat (V k) :=
  ⟨fun s => (argopt_attains M sat strat (V k) s (o.toFinset_acts s) (hseed s)).1,
    funext fun s => (argopt_attains M sat strat (V k) s (o.toFinset_acts s) (hseed s)).2⟩

/-- Every seed of the stationary cache is available.

Julia counterpart: `extract_strategy!(::StationaryStrategyCache, …)` (`src/strategy_cache.jl`). -/
theorem stationaryCacheSeed_mem (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (V : ℕ → S → ℝ) (k : ℕ) (s : S) :
    stationaryCacheSeed M sat strat o V k s ∈ M.available s :=
  stationarySeed_available M sat strat V s (o.toFinset_acts s) (o.first_mem s) _ k

/-- Call `k + 1` of the stationary cache is seeded with the action stored by call `k`.

Julia counterpart: `strategy_cache.strategy[jₛ]` read as `neutral` by
`extract_strategy!(::StationaryStrategyCache, …)` (`src/strategy_cache.jl`). -/
theorem stationaryCacheSeed_succ (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (V : ℕ → S → ℝ) (k : ℕ) (s : S) :
    stationaryCacheSeed M sat strat o V (k + 1) s =
      (stationaryCacheStrategy M sat strat o V k).strategy s := by
  simp [stationaryCacheSeed, stationarySeed, stationaryCacheStrategy, synthesizedStrategy]

/-- The returned time-varying strategy is valid: every action is available.

Julia counterpart: `cachetostrategy(::TimeVaryingStrategyCache)` (`src/strategy_cache.jl`). -/
theorem timeVaryingCacheStrategy_valid (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (o : ActionOrder M) (V : ℕ → S → ℝ) (K : ℕ) :
    (timeVaryingCacheStrategy M sat strat o V K).Valid M.toAvailableActions :=
  fun _ => (synthesizedStrategy_spec M sat strat o V _ fun s => o.first_mem s).1

/-- The returned stationary strategy is valid: every action is available.

Julia counterpart: `cachetostrategy(::StationaryStrategyCache)` (`src/strategy_cache.jl`). -/
theorem stationaryCacheStrategy_valid (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (o : ActionOrder M) (V : ℕ → S → ℝ) (K : ℕ) :
    (stationaryCacheStrategy M sat strat o V K).Valid M.toAvailableActions :=
  (synthesizedStrategy_spec M sat strat o V K (stationaryCacheSeed_mem M sat strat o V K)).1

/-! ### A7, finite horizon -/

/-- The decision rule applied at call `k < K` to the returned time-varying strategy is the one
synthesized by call `k`.

Julia counterpart: `select_strategy_cache` (`src/robust_value_iteration.jl`) on
`cachetostrategy(::TimeVaryingStrategyCache)` (`src/strategy_cache.jl`). -/
theorem selectStrategy_timeVaryingCacheStrategy (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (o : ActionOrder M) (V : ℕ → S → ℝ) {K k : ℕ}
    (h : k < (timeVaryingCacheStrategy M sat strat o V K).timeLength) :
    selectStrategy (timeVaryingCacheStrategy M sat strat o V K) ⟨k, h⟩ =
      synthesizedStrategy M sat strat o V (timeVaryingSeed o) k := by
  simp only [selectStrategy, timeVaryingCacheStrategy, Fin.rev_rev]

/-- Evaluating the returned time-varying strategy reproduces every computed iterate up to the
horizon: `policyEvalIter k = V_k` for `k ≤ K`.

Julia counterpart: `solve(VerificationProblem(mdp, spec, strategy))` with the strategy of
`solve(ControlSynthesisProblem(mdp, spec))` (`src/robust_value_iteration.jl`). -/
theorem policyEvalIter_timeVaryingCacheStrategy (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (o : ActionOrder M) (post : (S → ℝ) → S → ℝ) (V₀ : S → ℝ)
    {K k : ℕ} (hk : k ≤ K) :
    policyEvalIter M sat post V₀ (timeVaryingCacheStrategy M sat strat o
      (viIter M sat strat post V₀) K) k = viIter M sat strat post V₀ k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    have hk' : k <
        (timeVaryingCacheStrategy M sat strat o (viIter M sat strat post V₀) K).timeLength :=
      Nat.lt_of_succ_le hk
    rw [policyEvalIter, dif_pos hk', selectStrategy_timeVaryingCacheStrategy,
      ih (Nat.le_of_succ_le hk),
      (synthesizedStrategy_spec M sat strat o _ k fun s => o.first_mem s).2]
    rfl

/-- **A7, finite horizon.** Evaluating the time-varying strategy returned after `K` calls (policy
evaluation with Julia's `strategy[time_length - k]` at call `k`) gives exactly the computed `V_K`,
for every step postprocessing `post` and start `V₀` (all property types) and all four
satisfaction × strategy modes.

Julia counterpart: `cachetostrategy(::TimeVaryingStrategyCache)` (`src/strategy_cache.jl`)
evaluated by `_value_iteration!` with a `GivenStrategyCache` (`src/robust_value_iteration.jl`). -/
theorem timeVarying_attains (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (post : (S → ℝ) → S → ℝ) (V₀ : S → ℝ) (K : ℕ) :
    policyEvalIter M sat post V₀ (timeVaryingCacheStrategy M sat strat o
      (viIter M sat strat post V₀) K) K = viIter M sat strat post V₀ K :=
  policyEvalIter_timeVaryingCacheStrategy M sat strat o post V₀ le_rfl

/-! ### Instances of the generic loop -/

variable [DecidableEq S]

omit [DecidableEq A] in
/-- `reachIter` is `viIter` with the reachability postprocessing.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`). -/
theorem reachIter_eq_viIter (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) :
    reachIter M sat strat prop =
      viIter M sat strat (stepPostprocessValueFunction prop) (initializeValueFunction prop) := by
  funext k
  induction k with
  | zero => rfl
  | succ k ih => simp only [reachIter, viIter, step, ih]

omit [DecidableEq A] in
/-- `safetyIter` is `viIter` with the shifted safety postprocessing.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`). -/
theorem safetyIter_eq_viIter (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : SafetyProperty S) :
    safetyIter M sat strat prop =
      viIter M sat strat prop.stepPostprocessValueFunction prop.initializeValueFunction := by
  funext k
  induction k with
  | zero => rfl
  | succ k ih => simp only [safetyIter, viIter, safetyStep, ih]

omit [DecidableEq A] in
/-- `exitIter` is `viIter` with the exit-time postprocessing.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`). -/
theorem exitIter_eq_viIter (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ExpectedExitTime S) :
    exitIter M sat strat prop =
      viIter M sat strat prop.stepPostprocessValueFunction prop.initializeValueFunction := by
  funext k
  induction k with
  | zero => rfl
  | succ k ih => simp only [exitIter, viIter, exitStep, ih]

omit [DecidableEq A] [DecidableEq S] in
/-- `rewardIter` is `viIter` with the reward postprocessing.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`). -/
theorem rewardIter_eq_viIter (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : RewardProperty S) :
    rewardIter M sat strat prop =
      viIter M sat strat prop.stepPostprocessValueFunction prop.initializeValueFunction := by
  funext k
  induction k with
  | zero => rfl
  | succ k ih => simp only [rewardIter, viIter, rewardStep, ih]

/-- **A7, finite horizon, reachability.** `timeVarying_attains` for the six reachability types
(`FiniteTimeReachability`, `FiniteTimeReachAvoid`, exact time; also the infinite-time types when
run for a fixed `K`).

Julia counterpart: `cachetostrategy(::TimeVaryingStrategyCache)` (`src/strategy_cache.jl`) for an
`AbstractReachability` specification (`src/specification.jl`). -/
theorem timeVarying_attains_reach (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (prop : ReachProperty S) (K : ℕ) :
    policyEvalIter M sat (stepPostprocessValueFunction prop) (initializeValueFunction prop)
      (timeVaryingCacheStrategy M sat strat o (reachIter M sat strat prop) K) K =
      reachIter M sat strat prop K := by
  rw [reachIter_eq_viIter]
  exact timeVarying_attains M sat strat o _ _ K

/-- **A7, finite horizon, safety.** `timeVarying_attains` for `FiniteTimeSafety`, on the shifted
iterates (the final `+ 1` of `postprocess_value_function!` is applied to both sides alike).

Julia counterpart: `cachetostrategy(::TimeVaryingStrategyCache)` (`src/strategy_cache.jl`) for an
`AbstractSafety` specification (`src/specification.jl`). -/
theorem timeVarying_attains_safety (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (prop : SafetyProperty S) (K : ℕ) :
    policyEvalIter M sat prop.stepPostprocessValueFunction prop.initializeValueFunction
      (timeVaryingCacheStrategy M sat strat o (safetyIter M sat strat prop) K) K =
      safetyIter M sat strat prop K := by
  rw [safetyIter_eq_viIter]
  exact timeVarying_attains M sat strat o _ _ K

omit [DecidableEq S] in
/-- **A7, finite horizon, reward.** `timeVarying_attains` for `FiniteTimeReward`.

Julia counterpart: `cachetostrategy(::TimeVaryingStrategyCache)` (`src/strategy_cache.jl`) for an
`AbstractReward` specification (`src/specification.jl`). -/
theorem timeVarying_attains_reward (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (prop : RewardProperty S) (K : ℕ) :
    policyEvalIter M sat prop.stepPostprocessValueFunction prop.initializeValueFunction
      (timeVaryingCacheStrategy M sat strat o (rewardIter M sat strat prop) K) K =
      rewardIter M sat strat prop K := by
  rw [rewardIter_eq_viIter]
  exact timeVarying_attains M sat strat o _ _ K

/-! ### Step postprocessing of reset/shift shape -/

/-- The shape shared by the `step_postprocess_value_function!` methods of the reachability and
exit-time properties: states in `reset` are set to `resetValue`, every other state gets `shift`
added.

Julia counterpart: `step_postprocess_value_function!` for `AbstractReachability`,
`AbstractReachAvoid`, `ExactTimeReachability`, `ExactTimeReachAvoid` (`V[reach] .= 1`,
`V[avoid] .= 0`) and `ExpectedExitTime` (`current .+= 1.0; current[avoid] .= 0.0`)
(`src/specification.jl`). -/
structure StepPostprocess (S : Type*) where
  /-- The states set to a constant (Julia `reach`, `avoid`). -/
  reset : Finset S
  /-- The constant of a reset state (`1` on `reach`, `0` on `avoid`). -/
  resetValue : S → ℝ
  /-- The constant added to every other state (`1.0` for exit time, `0` for reachability). -/
  shift : S → ℝ

namespace StepPostprocess

/-- Applying the postprocessing: `resetValue t` on `reset`, `V t + shift t` elsewhere.

Julia counterpart: `step_postprocess_value_function!` (`src/specification.jl`). -/
def apply (P : StepPostprocess S) (V : S → ℝ) : S → ℝ :=
  fun t => if t ∈ P.reset then P.resetValue t else V t + P.shift t

omit [Fintype S] in
/-- The postprocessing is monotone.

Julia counterpart: `step_postprocess_value_function!` (`src/specification.jl`). -/
theorem apply_mono (P : StepPostprocess S) {V W : S → ℝ} (h : V ≤ W) : P.apply V ≤ P.apply W := by
  intro t
  simp only [apply]
  split_ifs
  · exact le_rfl
  · linarith [h t]

end StepPostprocess

/-- The reset/shift form of a reachability postprocessing: reset `reach` to `1` and `avoid` to
`0` (`avoid` first, as `stepPostprocessValueFunction`), shift `0`.

Julia counterpart: `step_postprocess_value_function!` for the reachability types
(`src/specification.jl`). -/
def ReachProperty.stepPostprocess : ReachProperty S → StepPostprocess S
  | .reachability reach => ⟨reach, fun _ => 1, fun _ => 0⟩
  | .reachAvoid reach avoid _ => ⟨avoid ∪ reach, fun t => if t ∈ avoid then 0 else 1, fun _ => 0⟩
  | .exactTimeReachability _ => ⟨∅, fun _ => 0, fun _ => 0⟩
  | .exactTimeReachAvoid _ avoid _ => ⟨avoid, fun _ => 0, fun _ => 0⟩

omit [Fintype S] in
/-- The reset/shift form agrees with `stepPostprocessValueFunction`.

Julia counterpart: `step_postprocess_value_function!` (`src/specification.jl`). -/
theorem ReachProperty.stepPostprocess_apply (prop : ReachProperty S) :
    prop.stepPostprocess.apply = stepPostprocessValueFunction prop := by
  funext V t
  cases prop <;>
    simp only [ReachProperty.stepPostprocess, StepPostprocess.apply,
      stepPostprocessValueFunction, Finset.mem_union, Finset.notMem_empty, add_zero] <;>
    split_ifs <;> simp_all

/-- The reset/shift form of the exit-time postprocessing: reset `avoid` to `0`, shift `1`.

Julia counterpart: `step_postprocess_value_function!(value_function, prop::ExpectedExitTime)`
(`src/specification.jl`). -/
def ExpectedExitTime.stepPostprocess (prop : ExpectedExitTime S) : StepPostprocess S :=
  ⟨prop.avoidStates, fun _ => 0, fun _ => 1⟩

omit [Fintype S] in
/-- The reset/shift form agrees with `ExpectedExitTime.stepPostprocessValueFunction`.

Julia counterpart: `step_postprocess_value_function!` (`src/specification.jl`). -/
theorem ExpectedExitTime.stepPostprocess_apply (prop : ExpectedExitTime S) :
    prop.stepPostprocess.apply = prop.stepPostprocessValueFunction := by
  funext V t
  simp only [ExpectedExitTime.stepPostprocess, StepPostprocess.apply,
    ExpectedExitTime.stepPostprocessValueFunction]
  split_ifs <;> simp_all

/-! ### Supporting lemmas for the stationary cache -/

omit [DecidableEq A] [DecidableEq S] in
/-- `_extract_strategy!` for `maximize` returns its seed or a strictly better action.

Julia counterpart: the strict `gt(v, opt_val)` of `_extract_strategy!` (`src/strategy_cache.jl`). -/
theorem argoptAction_eq_or_lt (values : A → ℝ) (seed : A) (acts : List A) :
    argoptAction .maximize values seed acts = seed ∨
      values seed < values (argoptAction .maximize values seed acts) := by
  induction acts generalizing seed with
  | nil => exact Or.inl rfl
  | cons b L ih =>
    have hrw : argoptAction .maximize values seed (b :: L) =
        argoptAction .maximize values (argoptStep .maximize values seed b) L := rfl
    rw [hrw]
    by_cases h : values seed < values b
    · have hb : argoptStep .maximize values seed b = b := by simp [argoptStep, h]
      rw [hb]
      rcases ih b with h' | h'
      · exact Or.inr (by rw [h']; exact h)
      · exact Or.inr (h.trans h')
    · have hs : argoptStep .maximize values seed b = seed := by simp [argoptStep, h]
      rw [hs]
      exact ih seed

omit [DecidableEq A] [DecidableEq S] in
/-- The value of one action is monotone in the value function (both satisfaction modes).

Julia counterpart: `state_action_bellman` (`src/bellman.jl`). -/
theorem stateActionBellman_mono (M : RMDP S A) (sat : SatisfactionMode) {V W : S → ℝ}
    (h : V ≤ W) (s : S) (a : A) :
    stateActionBellman M sat V s a ≤ stateActionBellman M sat W s a := by
  have h' := innerOpt_le_add sat (M.ambiguity_wellFormed s a) (V := V) (W := W) (c := 0)
    (fun u => by simpa using h u)
  simpa [Bellman.stateActionBellman] using h'

omit [DecidableEq S] in
/-- **Strict switching.** If call `k + 1` of the stationary cache (`maximize`) changes the action
of state `s`, then the Bellman value of `s` strictly increases from call `k` to call `k + 1`
(for non-decreasing iterates `V`).

Julia counterpart: `extract_strategy!(::StationaryStrategyCache, …)` and the strict `>` of
`_extract_strategy!` (`src/strategy_cache.jl`). -/
theorem T_lt_of_switch (M : RMDP S A) (sat : SatisfactionMode) (o : ActionOrder M)
    {V : ℕ → S → ℝ} (hV : Monotone V) (k : ℕ) (s : S)
    (h : (stationaryCacheStrategy M sat .maximize o V (k + 1)).strategy s ≠
      (stationaryCacheStrategy M sat .maximize o V k).strategy s) :
    T M sat .maximize (V k) s < T M sat .maximize (V (k + 1)) s := by
  have hk := (synthesizedStrategy_spec M sat .maximize o V k
    (stationaryCacheSeed_mem M sat .maximize o V k)).2
  have hk1 := (synthesizedStrategy_spec M sat .maximize o V (k + 1)
    (stationaryCacheSeed_mem M sat .maximize o V (k + 1))).2
  have hseed := stationaryCacheSeed_succ M sat .maximize o V k s
  rcases argoptAction_eq_or_lt (stateActionBellman M sat (V (k + 1)) s)
      (stationaryCacheSeed M sat .maximize o V (k + 1) s) (o.acts s) with h' | h'
  · exact absurd (h'.trans hseed) h
  · rw [← congrFun hk s, ← congrFun hk1 s]
    calc stateActionBellman M sat (V k) s
          ((stationaryCacheStrategy M sat .maximize o V k).strategy s)
        ≤ stateActionBellman M sat (V (k + 1)) s
            ((stationaryCacheStrategy M sat .maximize o V k).strategy s) :=
          stateActionBellman_mono M sat (hV (Nat.le_succ k)) s _
      _ < _ := by rw [← hseed]; exact h'

omit [DecidableEq S] in
/-- **Stagnation keeps the action.** If the Bellman value of `s` is the same at calls `j ≤ m`
(non-decreasing iterates, `maximize`), the stationary cache holds the same action for `s` after
both calls.

Julia counterpart: `extract_strategy!(::StationaryStrategyCache, …)` (`src/strategy_cache.jl`),
whose ties keep the previous action. -/
theorem strategy_eq_of_T_eq (M : RMDP S A) (sat : SatisfactionMode) (o : ActionOrder M)
    {V : ℕ → S → ℝ} (hV : Monotone V) {j m : ℕ} (hjm : j ≤ m) (s : S)
    (h : T M sat .maximize (V j) s = T M sat .maximize (V m) s) :
    (stationaryCacheStrategy M sat .maximize o V j).strategy s =
      (stationaryCacheStrategy M sat .maximize o V m).strategy s := by
  induction m, hjm using Nat.le_induction with
  | base => rfl
  | succ m hjm ih =>
    have h₁ : T M sat .maximize (V j) s ≤ T M sat .maximize (V m) s :=
      T_mono M sat .maximize (hV hjm) s
    have h₂ : T M sat .maximize (V m) s ≤ T M sat .maximize (V (m + 1)) s :=
      T_mono M sat .maximize (hV (Nat.le_succ m)) s
    have hm : T M sat .maximize (V j) s = T M sat .maximize (V m) s := le_antisymm h₁ (h ▸ h₂)
    rw [ih hm]
    by_contra hne
    have := T_lt_of_switch M sat o hV m s (Ne.symm hne)
    linarith

omit [DecidableEq A] [DecidableEq S] in
/-- For every pair `V`, `W` there is one distribution `p ∈ Γ` with
`innerOpt Γ V ≤ ⟨p, V⟩` and `⟨p, W⟩ ≤ innerOpt Γ W`: a minimizer for `W` (pessimistic) or a
maximizer for `V` (optimistic).

Julia counterpart: `state_action_bellman` (`src/bellman.jl`). -/
theorem exists_dot_bracket (sat : SatisfactionMode) {Γ : AmbiguitySet S} (hΓ : Γ.WellFormed)
    (V W : S → ℝ) : ∃ p ∈ Γ.vecs, innerOpt sat Γ V ≤ dot V p ∧ dot W p ≤ innerOpt sat Γ W := by
  cases sat
  · obtain ⟨p, hp, hpW⟩ := innerOpt_mem .pessimistic hΓ W
    refine ⟨p, hp, ?_, hpW.le⟩
    exact csInf_le (expectations_isCompact hΓ V).bddBelow ⟨p, hp, rfl⟩
  · obtain ⟨p, hp, hpV⟩ := innerOpt_mem .optimistic hΓ V
    refine ⟨p, hp, hpV.ge, ?_⟩
    exact le_csSup (expectations_isCompact hΓ W).bddAbove ⟨p, hp, rfl⟩

omit [DecidableEq A] [DecidableEq S] in
/-- If `g ≥ 0` and `⟨p, g⟩ ≤ 0` for a distribution `p`, then `g` vanishes on some state with
`p u > 0`.

Julia counterpart: none (Lean-side proof device). -/
theorem exists_eq_zero_of_dot_nonpos {p g : S → ℝ} (hp : p ∈ stdSimplex ℝ S) (hg : 0 ≤ g)
    (h : dot g p ≤ 0) : ∃ u, g u = 0 := by
  have hterm : ∀ u ∈ Finset.univ, 0 ≤ g u * p u := fun u _ => mul_nonneg (hg u) (hp.1 u)
  have hsum : ∑ u, g u * p u = 0 := le_antisymm h (Finset.sum_nonneg hterm)
  have hzero := (Finset.sum_eq_zero_iff_of_nonneg hterm).mp hsum
  by_contra hne
  have hp0 : ∀ u, p u = 0 := fun u => by
    rcases mul_eq_zero.mp (hzero u (Finset.mem_univ u)) with h' | h'
    · exact absurd ⟨u, h'⟩ hne
    · exact h'
  have h1 := hp.2
  simp [hp0] at h1

omit [DecidableEq A] in
/-- Policy evaluation with the stationary strategy, `minimize`: the iterates stay below every
super-solution `W` of the strategy's step (`P.apply (Tπ π W) ≤ W`) above `V₀`. Holds for every
valid strategy (`T ≤ Tπ` for `minimize`), whatever the tie-breaking of the cache.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) with `Minimize` and a
`StationaryStrategyCache` (`src/strategy_cache.jl`). -/
theorem viIter_le_superSolution_minimize (M : RMDP S A) (sat : SatisfactionMode)
    (P : StepPostprocess S) {V₀ W : S → ℝ} {π : StationaryStrategy S A}
    (hπ : π.Valid M.toAvailableActions) (hV₀ : V₀ ≤ W) (hW : P.apply (Tπ M sat π W) ≤ W)
    (k : ℕ) : viIter M sat .minimize P.apply V₀ k ≤ W := by
  induction k with
  | zero => exact hV₀
  | succ k ih =>
    have hT : T M sat .minimize W ≤ Tπ M sat π W := Tπ_stepSound M sat .minimize hπ W
    exact (P.apply_mono ((T_mono M sat .minimize ih).trans hT)).trans hW

/-- One backward step of the `maximize` argument: if `t` attains the maximal excess
`δ = max (V_{K+1} - W)` and `V_{j+1} t = V_{K+1} t` (`j ≤ K`), then some successor `u` under the
strategy's action also attains `δ` and has `V_j u = V_{K+1} u`.

Julia counterpart: `extract_strategy!(::StationaryStrategyCache, …)` (`src/strategy_cache.jl`)
within `_value_iteration!` (`src/robust_value_iteration.jl`). -/
theorem stationary_backward_step (M : RMDP S A) (sat : SatisfactionMode) (o : ActionOrder M)
    (P : StepPostprocess S) {V : ℕ → S → ℝ} {W : S → ℝ}
    (hV : ∀ i, V (i + 1) = P.apply (T M sat .maximize (V i))) (hmono : Monotone V) {K j : ℕ}
    (hjK : j ≤ K) {δ : ℝ} (hδ : 0 < δ) (hmax : ∀ u, V (K + 1) u - W u ≤ δ)
    (hW : P.apply (Tπ M sat (stationaryCacheStrategy M sat .maximize o V K) W) ≤ W) {t : S}
    (ht : V (K + 1) t - W t = δ) (htj : V (j + 1) t = V (K + 1) t) :
    ∃ u, V (K + 1) u - W u = δ ∧ V j u = V (K + 1) u := by
  set σ := stationaryCacheStrategy M sat .maximize o V
  have hWt := hW t
  have hreset : t ∉ P.reset := by
    intro hmem
    have hK : V (K + 1) t = P.resetValue t := by
      rw [hV]
      simp [StepPostprocess.apply, hmem]
    simp only [StepPostprocess.apply, hmem, if_true] at hWt
    linarith
  have hVsucc : ∀ i, V (i + 1) t = T M sat .maximize (V i) t + P.shift t := fun i => by
    rw [hV]
    simp only [StepPostprocess.apply, if_neg hreset]
  have hTeq : T M sat .maximize (V j) t = T M sat .maximize (V K) t := by
    have := htj
    rw [hVsucc j, hVsucc K] at this
    linarith
  have hσ : (σ j).strategy t = (σ K).strategy t := strategy_eq_of_T_eq M sat o hmono hjK t hTeq
  have hspec := congrFun (synthesizedStrategy_spec M sat .maximize o V j
    (stationaryCacheSeed_mem M sat .maximize o V j)).2 t
  have hTj : T M sat .maximize (V j) t =
      innerOpt sat (M.ambiguity t ((σ K).strategy t)) (V j) := by
    rw [← hspec]
    show innerOpt sat (M.ambiguity t ((σ j).strategy t)) (V j) = _
    rw [hσ]
  obtain ⟨p, hp, hpV, hpW⟩ :=
    exists_dot_bracket sat (M.ambiguity_wellFormed t ((σ K).strategy t)) (V j) W
  have hps : p ∈ stdSimplex ℝ S := AmbiguitySet.vecs_subset_stdSimplex _ hp
  have hWt' : innerOpt sat (M.ambiguity t ((σ K).strategy t)) W + P.shift t ≤ W t := by
    have h' : P.apply (Tπ M sat (σ K) W) t =
        innerOpt sat (M.ambiguity t ((σ K).strategy t)) W + P.shift t := by
      simp only [StepPostprocess.apply, if_neg hreset]
      rfl
    rw [← h']
    exact hWt
  -- the excess function `g = W - V_j + δ` is nonnegative and has `⟨p, g⟩ ≤ 0`
  set g : S → ℝ := W - V j + Function.const S δ
  have hg : 0 ≤ g := fun u => by
    have h1 := hmax u
    have h2 : V j u ≤ V (K + 1) u := hmono (by omega : j ≤ K + 1) u
    simp only [g, Pi.add_apply, Pi.sub_apply, Function.const_apply, Pi.zero_apply]
    linarith
  have hdot : dot g p = dot W p - dot (V j) p + δ := by
    simp only [g, dot, Pi.add_apply, Pi.sub_apply, Function.const_apply, add_mul, sub_mul,
      Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hps.2, mul_one]
  have hgp : dot g p ≤ 0 := by
    have hVt :
        V (K + 1) t = innerOpt sat (M.ambiguity t ((σ K).strategy t)) (V j) + P.shift t := by
      rw [← htj, hVsucc j, hTj]
    rw [hdot]
    linarith
  obtain ⟨u, hu⟩ := exists_eq_zero_of_dot_nonpos hps hg hgp
  refine ⟨u, ?_, ?_⟩
  · have h1 := hmax u
    have h2 : V j u ≤ V (K + 1) u := hmono (by omega : j ≤ K + 1) u
    simp only [g, Pi.add_apply, Pi.sub_apply, Function.const_apply] at hu
    linarith
  · have h1 := hmax u
    have h2 : V j u ≤ V (K + 1) u := hmono (by omega : j ≤ K + 1) u
    simp only [g, Pi.add_apply, Pi.sub_apply, Function.const_apply] at hu
    linarith

/-- **Stationary strategy, generic form.** For a reset/shift postprocessing `P` with
non-decreasing iterates, the value `V_{K+1}` returned after the calls `0, …, K` lies below every
super-solution `W` of the returned strategy's step (`P.apply (Tπ σ W) ≤ W`) with `V₀ ≤ W`, in all
four satisfaction × strategy modes.

Julia counterpart: `cachetostrategy(::StationaryStrategyCache)` (`src/strategy_cache.jl`) after
`_value_iteration!` (`src/robust_value_iteration.jl`). -/
theorem stationary_le_superSolution (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (P : StepPostprocess S) {V₀ W : S → ℝ}
    (hmono : Monotone (viIter M sat strat P.apply V₀)) (K : ℕ) (hV₀ : V₀ ≤ W)
    (hW : P.apply (Tπ M sat (stationaryCacheStrategy M sat strat o
      (viIter M sat strat P.apply V₀) K) W) ≤ W) :
    viIter M sat strat P.apply V₀ (K + 1) ≤ W := by
  cases strat
  case minimize =>
    exact viIter_le_superSolution_minimize M sat P
      (stationaryCacheStrategy_valid M sat .minimize o _ K) hV₀ hW (K + 1)
  case maximize =>
    set V := viIter M sat .maximize P.apply V₀ with hVdef
    by_contra hne
    obtain ⟨t, ht⟩ : ∃ t, W t < V (K + 1) t := by
      by_contra h'
      exact hne fun u => not_lt.mp fun hlt => h' ⟨u, hlt⟩
    obtain ⟨t₀, -, ht₀⟩ := Finset.exists_max_image Finset.univ
      (fun u => V (K + 1) u - W u) ⟨t, Finset.mem_univ t⟩
    set δ := V (K + 1) t₀ - W t₀
    have hδ : 0 < δ := by
      have := ht₀ t (Finset.mem_univ t)
      linarith
    have hmax : ∀ u, V (K + 1) u - W u ≤ δ := fun u => ht₀ u (Finset.mem_univ u)
    -- backwards induction: some state attains `δ` with `V_{K+1-i} = V_{K+1}`
    have key : ∀ i, i ≤ K + 1 → ∃ u, V (K + 1) u - W u = δ ∧ V (K + 1 - i) u = V (K + 1) u := by
      intro i
      induction i with
      | zero => exact fun _ => ⟨t₀, rfl, rfl⟩
      | succ i ih =>
        intro hi
        obtain ⟨u, hu, hu'⟩ := ih (by omega)
        have hidx : K + 1 - i = (K - i) + 1 := by omega
        rw [hidx] at hu'
        obtain ⟨v, hv, hv'⟩ := stationary_backward_step M sat o P (fun _ => rfl) hmono
          (by omega : K - i ≤ K) hδ hmax hW hu hu'
        refine ⟨v, hv, ?_⟩
        rw [show K + 1 - (i + 1) = K - i by omega]
        exact hv'
    obtain ⟨u, hu, hu'⟩ := key (K + 1) le_rfl
    rw [Nat.sub_self] at hu'
    have h0 : V 0 u ≤ W u := hV₀ u
    rw [hu'] at h0
    linarith

/-! ### A7, infinite horizon -/

/-- The value of a given stationary strategy `π` for a reachability property: the least fixed
point of its policy-evaluation step (`reachLfp` of the model restricted to `available s = {π s}`,
`Tπ_eq_T_strategyAvailable`). This is what `solve(VerificationProblem(mdp, spec, π))`
approximates for `InfiniteTimeReachability` / `InfiniteTimeReachAvoid` (`reachIter_tendsto_lfp`).

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) with a
`GivenStrategyCache` (`src/strategy_cache.jl`). -/
noncomputable def strategyReachLfp (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (prop : ReachProperty S) (π : StationaryStrategy S A) : S → ℝ :=
  reachLfp (M.withAvailable (strategyAvailable π)) sat strat prop

/-- **A7, infinite horizon (reachability / reach-avoid).** For `InfiniteTimeReachability` and
`InfiniteTimeReachAvoid` (`isExactTime = false`), the stationary strategy returned after the calls
`0, …, K` achieves at least the returned value: `Sound .pessimistic V_{K+1} V^σ`, i.e.
`V_{K+1} ≤ strategyReachLfp σ`, for all four satisfaction × strategy modes and every `K`
(in particular the `K` at which `CovergenceCriteria` stops). For `optimistic` this is a lower bound,
not the conservative direction (as for A4). Models the documented cache (ties keep the previous
action), which the fixed Julia guard implements; before the B-1 fix (issue #119) Julia's guard
differed (Finding F3, benchmark B-1; re-check pending).

Julia counterpart: `cachetostrategy(::StationaryStrategyCache)` (`src/strategy_cache.jl`) returned
by `solve(::ControlSynthesisProblem)` (`src/robust_value_iteration.jl`), evaluated by
`solve(VerificationProblem(mdp, spec, strategy))`. -/
theorem stationary_sound (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) {prop : ReachProperty S} (hprop : prop.isExactTime = false) (K : ℕ) :
    Sound .pessimistic (reachIter M sat strat prop (K + 1)) (strategyReachLfp M sat strat prop
      (stationaryCacheStrategy M sat strat o (reachIter M sat strat prop) K)) := by
  have hiter : reachIter M sat strat prop =
      viIter M sat strat prop.stepPostprocess.apply (initializeValueFunction prop) := by
    rw [ReachProperty.stepPostprocess_apply]
    exact reachIter_eq_viIter M sat strat prop
  have hmono : Monotone (viIter M sat strat prop.stepPostprocess.apply
      (initializeValueFunction prop)) := hiter ▸ reachIter_mono M sat strat hprop
  rw [sound_pessimistic, hiter]
  set σ := stationaryCacheStrategy M sat strat o
    (viIter M sat strat prop.stepPostprocess.apply (initializeValueFunction prop)) K
  set M' := M.withAvailable (strategyAvailable σ)
  have hfix := step_reachLfp M' sat strat prop
  have hT : T M' sat strat = Tπ M sat σ := (Tπ_eq_T_strategyAvailable M sat strat σ).symm
  have hnn : 0 ≤ reachLfp M' sat strat prop := fun s => (reachLfp_mem_unit M' sat strat prop s).1
  refine stationary_le_superSolution M sat strat o _ hmono K ?_ ?_
  · exact (initializeValueFunction_le_step M' sat strat hprop hnn).trans hfix.le
  · rw [congrFun (ReachProperty.stepPostprocess_apply prop) (Tπ M sat σ _), ← hT]
    exact hfix.le

/-- **A7, infinite horizon (expected exit time).** For `ExpectedExitTime`, the value `V_{K+1}`
returned with the stationary strategy `σ` lies below every nonnegative real super-solution `W` of
`σ`'s exit-time step (the form of A4's `exitIter_sound`; the exact value of `σ` is the least
such `W` when it is finite), all four modes.

Julia counterpart: `cachetostrategy(::StationaryStrategyCache)` (`src/strategy_cache.jl`) for an
`ExpectedExitTime` specification (`src/specification.jl`). -/
theorem stationary_sound_exitTime (M : RMDP S A) (sat : SatisfactionMode) (strat : StrategyMode)
    (o : ActionOrder M) (prop : ExpectedExitTime S) (K : ℕ) {W : S → ℝ} (hW : 0 ≤ W)
    (hstep : exitStep (M.withAvailable (strategyAvailable (stationaryCacheStrategy M sat strat o
      (exitIter M sat strat prop) K))) sat strat prop W ≤ W) :
    Sound .pessimistic (exitIter M sat strat prop (K + 1)) W := by
  have hiter : exitIter M sat strat prop =
      viIter M sat strat prop.stepPostprocess.apply prop.initializeValueFunction := by
    rw [ExpectedExitTime.stepPostprocess_apply]
    exact exitIter_eq_viIter M sat strat prop
  have hmono : Monotone (viIter M sat strat prop.stepPostprocess.apply
      prop.initializeValueFunction) := hiter ▸ exitIter_mono M sat strat prop
  rw [hiter] at hstep
  rw [sound_pessimistic, hiter]
  set σ := stationaryCacheStrategy M sat strat o
    (viIter M sat strat prop.stepPostprocess.apply prop.initializeValueFunction) K
  set M' := M.withAvailable (strategyAvailable σ)
  have hT : T M' sat strat = Tπ M sat σ := (Tπ_eq_T_strategyAvailable M sat strat σ).symm
  refine stationary_le_superSolution M sat strat o _ hmono K ?_ ?_
  · rw [initializeValueFunction_eq_exitStep_zero M' sat strat prop]
    exact (exitStep_mono M' sat strat prop hW).trans hstep
  · rw [congrFun (ExpectedExitTime.stepPostprocess_apply prop) (Tπ M sat σ W), ← hT]
    exact hstep

omit [DecidableEq S] in
/-- The value of a given stationary strategy `π` for a discounted reward with `ν < 1`: the unique
fixed point `rewardValue` of its policy-evaluation step (the model restricted to
`available s = {π s}`, `Tπ_eq_T_strategyAvailable`), what
`solve(VerificationProblem(mdp, spec, π))` approximates for `InfiniteTimeReward`.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) with a
`GivenStrategyCache` (`src/strategy_cache.jl`). -/
noncomputable def strategyRewardValue (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (prop : RewardProperty S) (hν : prop.discount < 1)
    (π : StationaryStrategy S A) : S → ℝ :=
  rewardValue (M.withAvailable (strategyAvailable π)) sat strat prop hν

omit [DecidableEq S] in
/-- **A7, infinite horizon (discounted reward).** For `InfiniteTimeReward` (`ν < 1`), the exact
value of the returned stationary strategy `σ` (the fixed point `rewardValue` of its
policy-evaluation step) lies in the A5 interval around the returned value:
`‖V_{K+1} − V^σ‖∞ ≤ ν/(1−ν)·‖V_{K+1} − V_K‖∞`, all four modes.

Julia counterpart: `cachetostrategy(::StationaryStrategyCache)` (`src/strategy_cache.jl`) for an
`InfiniteTimeReward` specification (`src/specification.jl`). -/
theorem stationary_reward_error_bound (M : RMDP S A) (sat : SatisfactionMode)
    (strat : StrategyMode) (o : ActionOrder M) (prop : RewardProperty S) (hν : prop.discount < 1)
    (K : ℕ) :
    dist (rewardIter M sat strat prop (K + 1)) (strategyRewardValue M sat strat prop hν
      (stationaryCacheStrategy M sat strat o (rewardIter M sat strat prop) K)) ≤
      prop.discount / (1 - prop.discount) *
        dist (rewardIter M sat strat prop (K + 1)) (rewardIter M sat strat prop K) := by
  set V := rewardIter M sat strat prop
  set σ := stationaryCacheStrategy M sat strat o V K
  set M' := M.withAvailable (strategyAvailable σ)
  set Vσ := strategyRewardValue M sat strat prop hν σ
  have hT : T M' sat strat = Tπ M sat σ := (Tπ_eq_T_strategyAvailable M sat strat σ).symm
  have hspec := (synthesizedStrategy_spec M sat strat o V K
    (stationaryCacheSeed_mem M sat strat o V K)).2
  have hstep : V (K + 1) = rewardStep M' sat strat prop (V K) := by
    show rewardStep M sat strat prop (V K) = _
    simp only [rewardStep]
    rw [hT]
    exact congrArg prop.stepPostprocessValueFunction hspec.symm
  have hfix : rewardStep M' sat strat prop Vσ = Vσ := rewardValue_isFixedPt M' sat strat prop hν
  have hc := rewardStep_dist_le M' sat strat prop (V K) Vσ
  rw [← hstep, hfix] at hc
  have htri := dist_triangle (V K) (V (K + 1)) Vσ
  have hν0 := prop.discount_pos
  have h1 : 0 < 1 - prop.discount := by linarith
  rw [div_mul_eq_mul_div, le_div_iff₀ h1, dist_comm (V K) (V (K + 1))] at *
  nlinarith [dist_nonneg (x := V (K + 1)) (y := Vσ), dist_nonneg (x := V (K + 1)) (y := V K)]

end IntervalMDP.VI
