import IntervalMDPProofs.Models.Factored
import IntervalMDPProofs.Models.Product
import IntervalMDPProofs.Models.Strategy
import IntervalMDPProofs.VI.Strategy

/-!
# Worked examples from the IntervalMDP.jl docstrings

* `docIMDP`: the 3-state, 2-action IMDP of the `RobustValueIteration` / `solve` docstring
  (`src/robust_value_iteration.jl`).
* `docFIMDP`: the fIMDP with state variables `(2, 3)` and action variables `(1, 2)` of the
  `FactoredRobustMarkovDecisionProcess` docstring
  (`src/models/FactoredRobustMarkovDecisionProcess.jl`).

Every structure invariant is discharged by `norm_num`. Julia's 1-based indices are 0-based here
(state `1` is `0 : Fin 3`, and so on). This is the only file with concrete instances.

The file also contains `binaryFIMDP`, the two-binary-variable fIMDP behind
`IntervalMDP.FactoredIMDP.productSet_not_convex`: the ambiguity set of a factored IMDP is not
convex in general (arXiv:2411.11803, arXiv:2508.00707).

Finally, `selfLoopRMDP` (two states, a self-loop tied with a goal action) is the witness
`b1_stationary_unsound` of inventory Finding F3: with Julia's guard in
`extract_strategy!(::StationaryStrategyCache, …)` (`src/strategy_cache.jl`, benchmark B-1) the
returned stationary strategy is not sound, while the documented cache is (`VI.stationary_sound`).
-/

noncomputable section

namespace IntervalMDP.Examples

open Finset

/-- A matrix pair of interval ambiguity sets, one per column (rows = targets, columns = sets).

Julia counterpart: `IntervalAmbiguitySets(; lower, upper)`
(`src/probabilities/IntervalAmbiguitySets.jl`); the invariants are `checkprobabilities`. Used here
to transcribe the docstring matrices verbatim. -/
structure IntervalAmbiguitySets (n k : ℕ) where
  /-- The lower-bound matrix (Julia `lower`). -/
  lower : Matrix (Fin n) (Fin k) ℝ
  /-- The upper-bound matrix (Julia `upper`). -/
  upper : Matrix (Fin n) (Fin k) ℝ
  /-- `0 ≤ lower`. -/
  lower_nonneg : ∀ i j, 0 ≤ lower i j
  /-- `lower ≤ upper`. -/
  lower_le_upper : ∀ i j, lower i j ≤ upper i j
  /-- `upper ≤ 1`. -/
  upper_le_one : ∀ i j, upper i j ≤ 1
  /-- Column sums of `lower` are at most one. -/
  sum_lower_le_one : ∀ j, ∑ i, lower i j ≤ 1
  /-- Column sums of `upper` are at least one. -/
  one_le_sum_upper : ∀ j, 1 ≤ ∑ i, upper i j

/-- Column `j` as an `IntervalAmbiguity` (Julia `p[j]`, an `IntervalAmbiguitySet`).

Julia counterpart: `getindex(p::IntervalAmbiguitySets, j::Integer)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
def IntervalAmbiguitySets.get {n k : ℕ} (P : IntervalAmbiguitySets n k) (j : Fin k) :
    IntervalAmbiguity (Fin n) where
  lower i := P.lower i j
  upper i := P.upper i j
  lower_nonneg i := P.lower_nonneg i j
  lower_le_upper i := P.lower_le_upper i j
  upper_le_one i := P.upper_le_one i j
  sum_lower_le_one := P.sum_lower_le_one j
  one_le_sum_upper := P.one_le_sum_upper j

/-! ### The IMDP of the `RobustValueIteration` docstring -/

/-- `prob1` of the `RobustValueIteration` docstring: the sets of state 1 (columns = actions).

Julia counterpart: `prob1 = IntervalAmbiguitySets(; lower, upper)` in the `RobustValueIteration`
docstring (`src/robust_value_iteration.jl`). -/
def prob1 : IntervalAmbiguitySets 3 2 where
  lower := !![0.0, 0.5; 0.1, 0.3; 0.2, 0.1]
  upper := !![0.5, 0.7; 0.6, 0.5; 0.7, 0.3]
  lower_nonneg := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  lower_le_upper := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  upper_le_one := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  sum_lower_le_one := by intro j; fin_cases j <;> norm_num [Fin.sum_univ_succ]
  one_le_sum_upper := by intro j; fin_cases j <;> norm_num [Fin.sum_univ_succ]

/-- `prob2` of the `RobustValueIteration` docstring: the sets of state 2.

Julia counterpart: `prob2 = IntervalAmbiguitySets(; lower, upper)` in the `RobustValueIteration`
docstring (`src/robust_value_iteration.jl`). -/
def prob2 : IntervalAmbiguitySets 3 2 where
  lower := !![0.1, 0.2; 0.2, 0.3; 0.3, 0.4]
  upper := !![0.6, 0.6; 0.5, 0.5; 0.4, 0.4]
  lower_nonneg := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  lower_le_upper := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  upper_le_one := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  sum_lower_le_one := by intro j; fin_cases j <;> norm_num [Fin.sum_univ_succ]
  one_le_sum_upper := by intro j; fin_cases j <;> norm_num [Fin.sum_univ_succ]

/-- `prob3` of the `RobustValueIteration` docstring: state 3 is absorbing.

Julia counterpart: `prob3 = IntervalAmbiguitySets(; lower, upper)` in the `RobustValueIteration`
docstring (`src/robust_value_iteration.jl`). -/
def prob3 : IntervalAmbiguitySets 3 2 where
  lower := !![0.0, 0.0; 0.0, 0.0; 1.0, 1.0]
  upper := !![0.0, 0.0; 0.0, 0.0; 1.0, 1.0]
  lower_nonneg := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  lower_le_upper := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  upper_le_one := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  sum_lower_le_one := by intro j; fin_cases j <;> norm_num [Fin.sum_univ_succ]
  one_le_sum_upper := by intro j; fin_cases j <;> norm_num [Fin.sum_univ_succ]

/-- The per-state sets `[prob1, prob2, prob3]` (Julia `transition_probs`).

Julia counterpart: `transition_probs = [prob1, prob2, prob3]` in the `RobustValueIteration`
docstring (`src/robust_value_iteration.jl`). -/
def docTransitionProbs : Fin 3 → IntervalAmbiguitySets 3 2 := ![prob1, prob2, prob3]

/-- The 3-state, 2-action IMDP of the `RobustValueIteration` docstring
(`src/robust_value_iteration.jl`):
`IntervalMarkovDecisionProcess([prob1, prob2, prob3], [1])`, with all actions available.

Julia counterpart: `mdp = IntervalMarkovDecisionProcess(transition_probs, initial_state)` in the
`RobustValueIteration` docstring (`src/robust_value_iteration.jl`); constructor in
`src/models/IntervalMarkovDecisionProcess.jl`. -/
def docIMDP : IMDP (Fin 3) (Fin 2) where
  toAvailableActions := AvailableActions.all
  ambiguitySets s a := (docTransitionProbs s).get a

/-- `docIMDP` as a robust MDP (all invariants already discharged above).

Julia counterpart: the `FactoredRobustMarkovDecisionProcess` that
`IntervalMarkovDecisionProcess` returns for the `RobustValueIteration` docstring example
(`src/robust_value_iteration.jl`, `src/models/IntervalMarkovDecisionProcess.jl`). -/
def docRMDP : RMDP (Fin 3) (Fin 2) := docIMDP.toRMDP

/-! ### The fIMDP of the `FactoredRobustMarkovDecisionProcess` docstring -/

/-- `state_vars = (2, 3)`.

Julia counterpart: `state_vars = (2, 3)` in the `FactoredRobustMarkovDecisionProcess` docstring
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def docStateVars : StateVars 2 where
  dims := ![2, 3]
  dims_pos := by intro i; fin_cases i <;> decide

/-- `action_vars = (1, 2)`.

Julia counterpart: `action_vars = (1, 2)` in the `FactoredRobustMarkovDecisionProcess` docstring
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def docActionVars : ActionVars 2 where
  dims := ![1, 2]
  dims_pos := by intro k; fin_cases k <;> decide

/-- The 6 interval sets of `marginal1` (target: state variable 1, 2 values). Column layout
`(a¹, s¹, s²)` with actions first, then states, column-major.

Julia counterpart: the `IntervalAmbiguitySets(; lower, upper)` argument of `marginal1` in the
`FactoredRobustMarkovDecisionProcess` docstring
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def marginal1Sets : IntervalAmbiguitySets 2 6 where
  lower := !![1/15, 7/30, 1/15, 13/30, 4/15, 1/6; 2/5, 7/30, 1/30, 11/30, 2/15, 1/10]
  upper := !![17/30, 7/10, 2/3, 4/5, 7/10, 2/3; 9/10, 13/15, 9/10, 5/6, 4/5, 14/15]
  lower_nonneg := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  lower_le_upper := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  upper_le_one := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  sum_lower_le_one := by intro j; fin_cases j <;> norm_num [Fin.sum_univ_succ]
  one_le_sum_upper := by intro j; fin_cases j <;> norm_num [Fin.sum_univ_succ]

/-- The 6 interval sets of `marginal2` (target: state variable 2, 3 values). Column layout
`(a², s²)`, actions first.

Julia counterpart: the `IntervalAmbiguitySets(; lower, upper)` argument of `marginal2` in the
`FactoredRobustMarkovDecisionProcess` docstring
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def marginal2Sets : IntervalAmbiguitySets 3 6 where
  lower := !![1/30, 1/3, 1/6, 1/15, 2/5, 2/15; 4/15, 1/4, 1/6, 1/30, 2/15, 1/30;
    2/15, 7/30, 1/10, 7/30, 7/15, 1/5]
  upper := !![2/3, 7/15, 4/5, 11/30, 19/30, 1/2; 23/30, 4/5, 23/30, 3/5, 7/10, 8/15;
    7/15, 4/5, 23/30, 7/10, 7/15, 23/30]
  lower_nonneg := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  lower_le_upper := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  upper_le_one := by intro i j; fin_cases i <;> fin_cases j <;> norm_num
  sum_lower_le_one := by intro j; fin_cases j <;> norm_num [Fin.sum_univ_succ]
  one_le_sum_upper := by intro j; fin_cases j <;> norm_num [Fin.sum_univ_succ]

/-- `marginal1 = Marginal(…, state_indices = (1, 2), action_indices = (1,), …)`: the next value of
state variable 1 depends on both state variables and on action variable 1. The column of
`(s¹, s², a¹)` is `a¹ + 1·(s¹ + 2·s²)` (0-based; `finProdFinEquiv (x, y) = y + n·x`).

Julia counterpart: `marginal1` in the `FactoredRobustMarkovDecisionProcess` docstring
(`src/models/FactoredRobustMarkovDecisionProcess.jl`); `Marginal` is defined in
`src/probabilities/Marginal.jl`. -/
def marginal1 : Marginal docStateVars docActionVars 0 where
  numStateIndices := 2
  numActionIndices := 1
  stateIndices := ![0, 1]
  actionIndices := ![0]
  stateIndices_strictMono := Fin.strictMono_iff_lt_succ.mpr (by decide)
  actionIndices_strictMono := Fin.strictMono_iff_lt_succ.mpr (by decide)
  sets src act := marginal1Sets.get
    (finProdFinEquiv (finProdFinEquiv ((src 1 : Fin 3), (src 0 : Fin 2)), (act 0 : Fin 1)))

/-- `marginal2 = Marginal(…, state_indices = (2,), action_indices = (2,), …)`: the next value of
state variable 2 depends on state variable 2 and action variable 2. The column of `(s², a²)` is
`a² + 2·s²` (0-based).

Julia counterpart: `marginal2` in the `FactoredRobustMarkovDecisionProcess` docstring
(`src/models/FactoredRobustMarkovDecisionProcess.jl`); `Marginal` is defined in
`src/probabilities/Marginal.jl`. -/
def marginal2 : Marginal docStateVars docActionVars 1 where
  numStateIndices := 1
  numActionIndices := 1
  stateIndices := ![1]
  actionIndices := ![1]
  stateIndices_strictMono := Fin.strictMono_iff_lt_succ.mpr (by decide)
  actionIndices_strictMono := Fin.strictMono_iff_lt_succ.mpr (by decide)
  sets src act := marginal2Sets.get (finProdFinEquiv ((src 0 : Fin 3), (act 0 : Fin 2)))

/-- The fIMDP of the `FactoredRobustMarkovDecisionProcess` docstring
(`src/models/FactoredRobustMarkovDecisionProcess.jl`):
`FactoredRobustMarkovDecisionProcess((2, 3), (1, 2), (marginal1, marginal2), [(1, 1)])`, with all
joint actions available (`AllAvailableActions`).

Julia counterpart: `mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, …)` with
the marginals `(marginal1, marginal2)` in the `FactoredRobustMarkovDecisionProcess` docstring
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def docFIMDP : FactoredIMDP 2 2 where
  stateVars := docStateVars
  actionVars := docActionVars
  availableActions := AvailableActions.all
  marginals := Fin.cons marginal1 (Fin.cons marginal2 finZeroElim)

/-- `docFIMDP` as a robust MDP on joint states and actions.

Julia counterpart: the same `mdp` (an `IsFIMDP` model) of the
`FactoredRobustMarkovDecisionProcess` docstring
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def docFRMDP : RMDP docStateVars.State docActionVars.Action := docFIMDP.toRMDP

/-! ### Factored ambiguity is not convex

The smallest factored IMDP whose product set is not convex: two binary state variables, a single
action, and unconstrained marginals (`lower = [0, 0]`, `upper = [1, 1]`). Julia setting (not run):

    full = IntervalAmbiguitySets(; lower = zeros(2, 2), upper = ones(2, 2))
    m1 = Marginal(full, (1,), (1,), (2,), (1,))
    m2 = Marginal(full, (2,), (1,), (2,), (1,))
    FactoredRobustMarkovDecisionProcess((2, 2), (1,), (m1, m2))

Every column is unconstrained, so the conditioning variable is irrelevant; the Lean model below
drops it (`binaryMarginal` conditions on nothing).

The joint distributions `[1 0; 0 0]` and `[0 0; 0 1]` are products of feasible marginals, their
average `[0.5 0; 0 0.5]` is not a product.
-/

/-- The unconstrained interval set on `{0, 1}`: `lower = [0, 0]`, `upper = [1, 1]`, so `P(l, u)`
is the whole simplex.

Julia counterpart: `IntervalAmbiguitySets(; lower = [0.0; 0.0;;], upper = [1.0; 1.0;;])`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
def fullBinary : IntervalAmbiguity (Fin 2) where
  lower _ := 0
  upper _ := 1
  lower_nonneg _ := le_refl 0
  lower_le_upper _ := zero_le_one
  upper_le_one _ := le_refl 1
  sum_lower_le_one := by simp
  one_le_sum_upper := by norm_num [Fin.sum_univ_two]

/-- Every distribution on `{0, 1}` lies in the unconstrained interval set.

Julia counterpart: none (Lean-side proof device). -/
theorem mem_fullBinary_toSet (p : ProbVec (Fin 2)) : p ∈ fullBinary.toSet :=
  fun s => ⟨p.nonneg s, p.le_one s⟩

/-- `state_vars = (2, 2)`: two binary state variables.

Julia counterpart: the `state_vars` argument `(2, 2)` of `FactoredRobustMarkovDecisionProcess`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def binaryStateVars : StateVars 2 where
  dims _ := 2
  dims_pos _ := two_pos

/-- `action_vars = (1,)`: a single action.

Julia counterpart: the `action_vars` argument `(1,)` of `FactoredRobustMarkovDecisionProcess`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def binaryActionVars : ActionVars 1 where
  dims _ := 1
  dims_pos _ := one_pos

/-- Marginal `i` of `binaryFIMDP`: no conditioning variables, and the unconstrained set
`fullBinary` (Julia: `Marginal(full, (i,), (1,), (2,), (1,))` with all columns unconstrained,
`src/probabilities/Marginal.jl`).

Julia counterpart: `Marginal` (`src/probabilities/Marginal.jl`). -/
def binaryMarginal (i : Fin 2) : Marginal binaryStateVars binaryActionVars i where
  numStateIndices := 0
  numActionIndices := 0
  stateIndices := finZeroElim
  actionIndices := finZeroElim
  stateIndices_strictMono a := a.elim0
  actionIndices_strictMono a := a.elim0
  sets _ _ := fullBinary

/-- The two-binary-variable fIMDP with unconstrained marginals (Julia:
`FactoredRobustMarkovDecisionProcess((2, 2), (1,), (m1, m2))`,
`src/models/FactoredRobustMarkovDecisionProcess.jl`).

Julia counterpart: `FactoredRobustMarkovDecisionProcess`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def binaryFIMDP : FactoredIMDP 2 1 where
  stateVars := binaryStateVars
  actionVars := binaryActionVars
  availableActions := AvailableActions.all
  marginals := binaryMarginal

/-- Two binary variables: if every marginal set contains both point masses `δ₀` and `δ₁`, the
product set `⨂ᵢ Γᵢ` is not convex. The point masses at `(0, 0)` and at `(1, 1)` are products, but
their midpoint is not: it would need `γ¹(0)γ²(1) = 0` while `γ¹(0)γ²(0) = γ¹(1)γ²(1) = 1/2`.

Julia counterpart: `Γ_{s,a} = ⨂ᵢ Γⁱ` of `FactoredRobustMarkovDecisionProcess`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
theorem pi_not_convex_of_dirac_mem {Γ : Fin 2 → AmbiguitySet (Fin 2)}
    (h0 : ∀ i, ProbVec.dirac 0 ∈ Γ i) (h1 : ∀ i, ProbVec.dirac 1 ∈ Γ i) :
    ¬ (AmbiguitySet.pi Γ).IsConvex := by
  intro h
  set x : (Fin 2 → Fin 2) → ℝ :=
    ⇑(ProbVec.pi (X := fun _ : Fin 2 => Fin 2) (fun _ => ProbVec.dirac 0)) with hx
  set y : (Fin 2 → Fin 2) → ℝ :=
    ⇑(ProbVec.pi (X := fun _ : Fin 2 => Fin 2) (fun _ => ProbVec.dirac 1)) with hy
  have hxm : x ∈ (AmbiguitySet.pi Γ).vecs := ⟨_, ⟨_, fun i _ => h0 i, rfl⟩, rfl⟩
  have hym : y ∈ (AmbiguitySet.pi Γ).vecs := ⟨_, ⟨_, fun i _ => h1 i, rfl⟩, rfl⟩
  obtain ⟨p, ⟨γ, -, rfl⟩, hp⟩ :=
    h hxm hym (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num)
  have h00 := congrFun hp ![0, 0]
  have h01 := congrFun hp ![0, 1]
  have h11 := congrFun hp ![1, 1]
  simp [hx, hy, ProbVec.pi_apply, Fin.prod_univ_two] at h00 h01 h11
  rcases h01 with h | h
  · rw [h] at h00; norm_num at h00
  · rw [h] at h11; norm_num at h11

/-- The product set of `binaryFIMDP` is not convex, at every state and action.

Julia counterpart: `Γ_{s,a} = ⨂ᵢ Γⁱ` of `FactoredRobustMarkovDecisionProcess`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
theorem binaryFIMDP_productSet_not_convex (s : binaryFIMDP.stateVars.State)
    (a : binaryFIMDP.actionVars.Action) : ¬ (binaryFIMDP.productSet s a).IsConvex :=
  pi_not_convex_of_dirac_mem (fun _ => mem_fullBinary_toSet _) (fun _ => mem_fullBinary_toSet _)

end IntervalMDP.Examples

namespace IntervalMDP.FactoredIMDP

/-- **The ambiguity set of a factored IMDP is not convex in general.** There is a factored IMDP
(`Examples.binaryFIMDP`: two binary state variables, unconstrained marginals) and a state–action
pair whose literal product set `productSet s a = ⨂ᵢ Γⁱ` is not convex: the point masses at `(0, 0)`
and `(1, 1)` are products, their midpoint is not.

This is known in the literature: arXiv:2411.11803 (the IntervalMDP.jl author's paper on factored
IMDPs) and arXiv:2508.00707 (Schnitzer et al.). Consequently `AmbiguitySet.WellFormed` does not
include convexity and `FactoredIMDP.toRMDP` uses `productSet` itself, not a convex hull.

Julia counterpart: `Γ_{s,a}` of `FactoredRobustMarkovDecisionProcess`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
theorem productSet_not_convex :
    ∃ (F : FactoredIMDP 2 1) (s : F.stateVars.State) (a : F.actionVars.Action),
      ¬ (F.productSet s a).IsConvex :=
  ⟨Examples.binaryFIMDP, fun _ => ⟨0, two_pos⟩, fun _ => ⟨0, one_pos⟩,
    Examples.binaryFIMDP_productSet_not_convex _ _⟩

end IntervalMDP.FactoredIMDP


/-! ### Finding F3: the stationary cache with Julia's guard (benchmark B-1)

Two states (`0` = goal, `1` = `s`), two actions, point-mass transitions: in `s`, action `0` is a
self-loop and action `1` goes to the goal; the goal is absorbing. Specification
`InfiniteTimeReachability([goal])`, `Maximize`, either satisfaction mode. Value iteration gives
`V₁ = V₂ = (1, 1)`. At call `1` both actions of `s` have value `1`. The documented stationary cache
keeps the goal action chosen at call `0` (`stationary_sound` holds); Julia's guard
`jₛ ∉ available_actions` resets the seed to `first(available_actions)` for every state whose index
exceeds the number of actions (benchmark B-1), so the tie goes to the self-loop, whose value is `0`.
Julia reproduction (3 states, `s` at index 3): reported `V = [1, 0, 1]`, policy evaluation of the
returned strategy `[0, 0, 0]` at `s` (inventory F3). -/

namespace IntervalMDP.Examples

open Bellman VI Approx

/-- The successor of `(s, a)`: `s = 1` with action `0` loops, every other pair goes to the goal
`0`.

Julia counterpart: the point-mass columns of `IntervalAmbiguitySets(; lower, upper)`
(`src/probabilities/IntervalAmbiguitySets.jl`) in the F3 reproduction. -/
def selfLoopTarget (s a : Fin 2) : Fin 2 :=
  if s = 1 ∧ a = 0 then 1 else 0

/-- The two-state self-loop model of Finding F3, with point-mass ambiguity sets.

Julia counterpart: `IntervalMarkovDecisionProcess`
(`src/models/IntervalMarkovDecisionProcess.jl`) of the F3 reproduction. -/
def selfLoopRMDP : RMDP (Fin 2) (Fin 2) where
  toAvailableActions := AvailableActions.all
  ambiguity s a := {ProbVec.dirac (selfLoopTarget s a)}
  ambiguity_wellFormed s a :=
    { nonempty := Set.singleton_nonempty _
      closed := by
        simp only [AmbiguitySet.vecs, Set.image_singleton]
        exact isClosed_singleton }

/-- Julia's action order `[1, 2]` (0-based `[0, 1]`) in every state.

Julia counterpart: `AllAvailableActions` (`src/available_actions.jl`). -/
def selfLoopOrder : ActionOrder selfLoopRMDP where
  acts _ := [0, 1]
  toFinset_acts _ := by
    ext a
    fin_cases a <;> simp [selfLoopRMDP, AvailableActions.all]

/-- `InfiniteTimeReachability([goal])`.

Julia counterpart: `InfiniteTimeReachability` (`src/specification.jl`). -/
def selfLoopProp : ReachProperty (Fin 2) :=
  .reachability {0}

/-- The stationary strategy Julia returns after the calls `0, 1` when the guard
`jₛ ∉ available_actions` resets the seed at every call (`keep = false` in `stationarySeed`).

Julia counterpart: `cachetostrategy(::StationaryStrategyCache)` (`src/strategy_cache.jl`) with the
guard of `extract_strategy!` failing (benchmark B-1). -/
def b1Strategy (sat : SatisfactionMode) : StationaryStrategy (Fin 2) (Fin 2) :=
  synthesizedStrategy selfLoopRMDP sat .maximize selfLoopOrder
    (reachIter selfLoopRMDP sat .maximize selfLoopProp)
    (fun k s => stationarySeed selfLoopRMDP sat .maximize
      (reachIter selfLoopRMDP sat .maximize selfLoopProp) s (selfLoopOrder.acts s)
      (selfLoopOrder.first s) (fun _ => false) k) 1

/-- Action values in the self-loop model: `stateActionBellman V s a = V (selfLoopTarget s a)`.

Julia counterpart: `state_action_bellman` (`src/bellman.jl`) on a point-mass column. -/
theorem selfLoop_stateActionBellman (sat : SatisfactionMode) (V : Fin 2 → ℝ) (s a : Fin 2) :
    stateActionBellman selfLoopRMDP sat V s a = V (selfLoopTarget s a) := by
  have hexp : expectations (selfLoopRMDP.ambiguity s a) V = {V (selfLoopTarget s a)} := by
    simp [expectations, selfLoopRMDP, AmbiguitySet.vecs, Set.image_singleton, OMax.dot,
      ProbVec.dirac]
  cases sat <;> simp [stateActionBellman, innerOpt, hexp]

/-- **Witness for Finding F3.** With Julia's guard resetting the seed (benchmark B-1), the
returned stationary strategy is not sound: `V₂(s) = 1` but the strategy's value at `s` is `0`,
in both satisfaction modes. (`stationary_sound` proves soundness for the documented cache.)

Julia counterpart: `extract_strategy!(::StationaryStrategyCache, …)` (`src/strategy_cache.jl`)
and `solve` (`src/robust_value_iteration.jl`); reproduction in inventory Finding F3. -/
theorem b1_stationary_unsound (sat : SatisfactionMode) :
    ¬ Sound .pessimistic (reachIter selfLoopRMDP sat .maximize selfLoopProp 2)
      (strategyReachLfp selfLoopRMDP sat .maximize selfLoopProp (b1Strategy sat)) := by
  set V := reachIter selfLoopRMDP sat .maximize selfLoopProp
  set W := initializeValueFunction selfLoopProp
  have hQ := selfLoop_stateActionBellman sat
  have hpost0 : ∀ X : Fin 2 → ℝ, stepPostprocessValueFunction selfLoopProp X 0 = 1 := fun X => by
    simp [stepPostprocessValueFunction, selfLoopProp]
  have hpost1 : ∀ X : Fin 2 → ℝ, stepPostprocessValueFunction selfLoopProp X 1 = X 1 := fun X => by
    simp [stepPostprocessValueFunction, selfLoopProp]
  have hW0 : W 0 = 1 := by simp [W, initializeValueFunction, selfLoopProp, ReachProperty.reach]
  have hW1 : W 1 = 0 := by simp [W, initializeValueFunction, selfLoopProp, ReachProperty.reach]
  have hgoal : ∀ k, V (k + 1) 0 = 1 := fun k => hpost0 (T selfLoopRMDP sat .maximize (V k))
  have hge : ∀ k, V k 0 ≤ V (k + 1) 1 := fun k => by
    show V k 0 ≤ stepPostprocessValueFunction selfLoopProp (T selfLoopRMDP sat .maximize (V k)) 1
    rw [hpost1]
    calc V k 0 = stateActionBellman selfLoopRMDP sat (V k) 1 1 := by
          rw [hQ]; simp [selfLoopTarget]
      _ ≤ T selfLoopRMDP sat .maximize (V k) 1 :=
          Finset.le_sup' (stateActionBellman selfLoopRMDP sat (V k) 1) (Finset.mem_univ (1 : Fin 2))
  have hV11 : V 1 1 = 1 := by
    have h := hge 0
    have h0 : V 0 0 = 1 := hW0
    rw [h0] at h
    exact le_antisymm (reachIter_mem_unit _ _ _ _ 1 1).2 h
  have hV21 : V 2 1 = 1 := by
    have h := hge 1
    rw [hgoal 0] at h
    exact le_antisymm (reachIter_mem_unit _ _ _ _ 2 1).2 h
  -- the B-1 strategy plays the self-loop in `s`
  have hσ : (b1Strategy sat).strategy 1 = 0 := by
    show argoptAction .maximize (stateActionBellman selfLoopRMDP sat (V 1) 1)
      (stationarySeed selfLoopRMDP sat .maximize V 1 [0, 1] (selfLoopOrder.first 1)
        (fun _ => false) 1) [0, 1] = 0
    have hseed : stationarySeed selfLoopRMDP sat .maximize V 1 [0, 1] (selfLoopOrder.first 1)
        (fun _ => false) 1 = 0 := rfl
    rw [hseed]
    simp [argoptAction, argoptStep, hQ, selfLoopTarget, hV11, hgoal 0]
  -- `W = 𝟙_{goal}` is a fixed point of the strategy's step, so its value at `s` is `0`
  set M' := selfLoopRMDP.withAvailable (strategyAvailable (b1Strategy sat))
  have hT : T M' sat .maximize = Tπ selfLoopRMDP sat (b1Strategy sat) :=
    (Tπ_eq_T_strategyAvailable selfLoopRMDP sat .maximize _).symm
  have hWunit : ∀ s, W s ∈ Set.Icc (0 : ℝ) 1 := by
    refine Fin.forall_fin_two.mpr ⟨?_, ?_⟩
    · rw [hW0]; exact ⟨zero_le_one, le_rfl⟩
    · rw [hW1]; exact ⟨le_rfl, zero_le_one⟩
  have hfix : step M' sat .maximize selfLoopProp W ≤ W := by
    refine Fin.forall_fin_two.mpr ⟨?_, ?_⟩
    · show stepPostprocessValueFunction selfLoopProp (T M' sat .maximize W) 0 ≤ W 0
      rw [hpost0, hW0]
    · show stepPostprocessValueFunction selfLoopProp (T M' sat .maximize W) 1 ≤ W 1
      rw [hpost1, hT]
      show stateActionBellman selfLoopRMDP sat W 1 ((b1Strategy sat).strategy 1) ≤ W 1
      rw [hσ, hQ]
      simp [selfLoopTarget]
  have hval := reachLfp_le_of_step_le M' sat .maximize selfLoopProp hWunit hfix 1
  intro hsound
  have h : V 2 1 ≤ reachLfp M' sat .maximize selfLoopProp 1 := hsound 1
  rw [hV21] at h
  rw [hW1] at hval
  linarith

end IntervalMDP.Examples

end
