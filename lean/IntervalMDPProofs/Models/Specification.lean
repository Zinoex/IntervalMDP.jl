import Mathlib

/-!
# Properties and specifications

`Property S Q` has one constructor per concrete Julia property struct in `src/specification.jl`.
`S` is the type of MDP states and `Q` the type of DFA states (used only by the DFA properties).
The validity checks of Julia's `checkproperty` are carried as constructor arguments:

* `1 ≤ timeHorizon` for finite-time properties (`checktimehorizon`);
* `0 < convergenceEps` for infinite-time properties (`checkconvergence`);
* `Disjoint reach avoid` for reach-avoid properties (`checkdisjoint`);
* `0 < discount` for rewards (`checkreward`) and `discount < 1` for infinite-time rewards
  (`checkdiscountupperbound`).

State-bounds checks (`checkstatebounds`) are enforced by the types.
-/

namespace IntervalMDP

/-- A model-checking property.

Julia counterpart: the concrete subtypes of `Property` in `src/specification.jl` (one constructor
each). Field names follow the Julia fields (`reach`, `avoid`, `time_horizon`, `convergence_eps`,
`reward`, `discount`, `avoid_states`) in camelCase. -/
inductive Property (S Q : Type*) where
  /-- `FiniteTimeDFAReachability(reach, time_horizon)`: reach DFA states `reach` within `K` steps. -/
  | finiteTimeDFAReachability (reach : Finset Q) (timeHorizon : ℕ)
      (timeHorizon_pos : 1 ≤ timeHorizon)
  /-- `InfiniteTimeDFAReachability(reach, convergence_eps)`: eventually reach DFA states `reach`. -/
  | infiniteTimeDFAReachability (reach : Finset Q) (convergenceEps : ℝ)
      (convergenceEps_pos : 0 < convergenceEps)
  /-- `FiniteTimeDFASafety(avoid, time_horizon)`: avoid DFA states `avoid` for `K` steps. -/
  | finiteTimeDFASafety (avoid : Finset Q) (timeHorizon : ℕ) (timeHorizon_pos : 1 ≤ timeHorizon)
  /-- `InfiniteTimeDFASafety(avoid, convergence_eps)`: always avoid DFA states `avoid`. -/
  | infiniteTimeDFASafety (avoid : Finset Q) (convergenceEps : ℝ)
      (convergenceEps_pos : 0 < convergenceEps)
  /-- `FiniteTimeReachability(reach, time_horizon)`: reach `reach` within `K` steps. -/
  | finiteTimeReachability (reach : Finset S) (timeHorizon : ℕ)
      (timeHorizon_pos : 1 ≤ timeHorizon)
  /-- `InfiniteTimeReachability(reach, convergence_eps)`: eventually reach `reach`. -/
  | infiniteTimeReachability (reach : Finset S) (convergenceEps : ℝ)
      (convergenceEps_pos : 0 < convergenceEps)
  /-- `ExactTimeReachability(reach, time_horizon)`: be in `reach` at exactly step `K`. -/
  | exactTimeReachability (reach : Finset S) (timeHorizon : ℕ)
      (timeHorizon_pos : 1 ≤ timeHorizon)
  /-- `FiniteTimeReachAvoid(reach, avoid, time_horizon)`: reach `reach` within `K` steps while
  avoiding `avoid`. -/
  | finiteTimeReachAvoid (reach avoid : Finset S) (disjoint : Disjoint reach avoid)
      (timeHorizon : ℕ) (timeHorizon_pos : 1 ≤ timeHorizon)
  /-- `InfiniteTimeReachAvoid(reach, avoid, convergence_eps)`: eventually reach `reach` while
  avoiding `avoid`. -/
  | infiniteTimeReachAvoid (reach avoid : Finset S) (disjoint : Disjoint reach avoid)
      (convergenceEps : ℝ) (convergenceEps_pos : 0 < convergenceEps)
  /-- `ExactTimeReachAvoid(reach, avoid, time_horizon)`: be in `reach` at exactly step `K`, avoiding
  `avoid` before. -/
  | exactTimeReachAvoid (reach avoid : Finset S) (disjoint : Disjoint reach avoid)
      (timeHorizon : ℕ) (timeHorizon_pos : 1 ≤ timeHorizon)
  /-- `FiniteTimeSafety(avoid, time_horizon)`: avoid `avoid` for `K` steps. -/
  | finiteTimeSafety (avoid : Finset S) (timeHorizon : ℕ) (timeHorizon_pos : 1 ≤ timeHorizon)
  /-- `InfiniteTimeSafety(avoid, convergence_eps)`: always avoid `avoid`. -/
  | infiniteTimeSafety (avoid : Finset S) (convergenceEps : ℝ)
      (convergenceEps_pos : 0 < convergenceEps)
  /-- `FiniteTimeReward(reward, discount, time_horizon)`: `𝔼[∑_{k=0}^{K} νᵏ r(ω[k])]`. -/
  | finiteTimeReward (reward : S → ℝ) (discount : ℝ) (discount_pos : 0 < discount)
      (timeHorizon : ℕ) (timeHorizon_pos : 1 ≤ timeHorizon)
  /-- `InfiniteTimeReward(reward, discount, convergence_eps)`: `𝔼[∑_{k=0}^{∞} νᵏ r(ω[k])]` with
  `0 < ν < 1`. -/
  | infiniteTimeReward (reward : S → ℝ) (discount : ℝ) (discount_pos : 0 < discount)
      (discount_lt_one : discount < 1) (convergenceEps : ℝ)
      (convergenceEps_pos : 0 < convergenceEps)
  /-- `ExpectedExitTime(avoid_states, convergence_eps)`: the expected number of steps before
  entering `avoidStates`. -/
  | expectedExitTime (avoidStates : Finset S) (convergenceEps : ℝ)
      (convergenceEps_pos : 0 < convergenceEps)

namespace Property

variable {S Q : Type*}

/-- Whether the property has a finite time horizon.

Julia counterpart: `isfinitetime(prop)` (`src/specification.jl`). -/
def isFiniteTime : Property S Q → Bool
  | finiteTimeDFAReachability .. | finiteTimeDFASafety .. | finiteTimeReachability ..
  | exactTimeReachability .. | finiteTimeReachAvoid .. | exactTimeReachAvoid ..
  | finiteTimeSafety .. | finiteTimeReward .. => true
  | _ => false

/-- Whether the property is defined on a product with a DFA.

Julia counterpart: `ProductProperty` vs `BasicProperty` (`src/specification.jl`). -/
def isProductProperty : Property S Q → Bool
  | finiteTimeDFAReachability .. | infiniteTimeDFAReachability ..
  | finiteTimeDFASafety .. | infiniteTimeDFASafety .. => true
  | _ => false

end Property

/-- Whether the uncertainty is resolved adversarially or cooperatively: `pessimistic` takes the
minimum over each ambiguity set, `optimistic` the maximum.

Julia counterpart: `@enum SatisfactionMode Pessimistic Optimistic` (`src/specification.jl`). -/
inductive SatisfactionMode where
  /-- Julia `Pessimistic`: the adversary minimises over the ambiguity set. -/
  | pessimistic
  /-- Julia `Optimistic`: the adversary maximises over the ambiguity set. -/
  | optimistic
  deriving Repr

/-- Whether the strategy maximises or minimises the property.

Julia counterpart: `@enum StrategyMode Maximize Minimize` (`src/specification.jl`). -/
inductive StrategyMode where
  /-- Julia `Maximize`. -/
  | maximize
  /-- Julia `Minimize`. -/
  | minimize
  deriving Repr

/-- A specification: a property together with a satisfaction mode and a strategy mode.

Julia counterpart: `Specification` (`src/specification.jl`), fields `prop`, `satisfaction` and
`strategy`. -/
structure Specification (S Q : Type*) where
  /-- The property (Julia `prop`). -/
  prop : Property S Q
  /-- Pessimistic or optimistic (Julia `satisfaction`). -/
  satisfaction : SatisfactionMode
  /-- Maximize or minimize (Julia `strategy`). -/
  strategy : StrategyMode

end IntervalMDP
