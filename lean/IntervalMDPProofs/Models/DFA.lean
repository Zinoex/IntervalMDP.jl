import IntervalMDPProofs.Models.Distribution

/-!
# DFAs and labellings

A deterministic finite automaton `D = (Q, Λ, δ, q₀, Q_ac)` reads the labels `Λ = 2^{AP}` of the
states visited by the MDP. A labelling maps each MDP state to a label, either deterministically
(`Labelling`) or with probabilities (`ProbLabelling`).

Julia counterpart: `DFA` (`src/models/DFA.jl`), `TransitionFunction`
(`src/probabilities/TransitionFunction.jl`), `DeterministicLabelling`
(`src/probabilities/DeterministicLabelling.jl`) and `ProbabilisticLabelling`
(`src/probabilities/ProbabilisticLabelling.jl`).
-/

namespace IntervalMDP

/-- A deterministic finite automaton over the label alphabet `Λ`.

Julia counterpart: `DFA` (`src/models/DFA.jl`) with fields `transition` (here `δ`) and
`initial_state` (here `q₀`). Julia keeps the accepting states in the DFA properties
(`DFAReachability.reach`, `src/specification.jl`); they are stored here as `accepting`, as in the
formal tuple `(Q, 2^{AP}, δ, q₀, Q_ac)` of the `DFA` docstring. -/
structure DFA (Q Λ : Type*) where
  /-- The transition function `δ : Q × Λ → Q`; total by its type (Julia `dfa[q, l]`, i.e.
  `TransitionFunction.transition[l, q]`, checked in range by `checktransition`). -/
  δ : Q → Λ → Q
  /-- The initial state `q₀` (Julia `initial_state`, checked in range by `checkdfa`). -/
  q₀ : Q
  /-- The accepting states `Q_ac`. -/
  accepting : Finset Q

/-- A deterministic labelling `L : S → Λ`.

Julia counterpart: `DeterministicLabelling` (`src/probabilities/DeterministicLabelling.jl`), field
`map`; `lf[s]` is `map s`. -/
structure Labelling (S Λ : Type*) where
  /-- The label of each state (Julia `map`). -/
  map : S → Λ

/-- A probabilistic labelling `L : S → 𝒟(Λ)`: each state gets a distribution over labels.

Julia counterpart: `ProbabilisticLabelling` (`src/probabilities/ProbabilisticLabelling.jl`), field
`map` (labels on the rows, states on the columns; `checklabellingprobs` requires every column to
sum to one). -/
structure ProbLabelling (S Λ : Type*) [Fintype Λ] where
  /-- The label distribution of each state (Julia column `map[:, s]`, i.e. `lf[s]`). -/
  map : S → ProbVec Λ

/-- A labelling of either kind.

Julia counterpart: `AbstractLabelling` (`src/probabilities/Labelling.jl`), whose concrete
subtypes are `DeterministicLabelling` and `ProbabilisticLabelling`. -/
inductive AbstractLabelling (S Λ : Type*) [Fintype Λ] where
  /-- A deterministic labelling (Julia `DeterministicLabelling`). -/
  | deterministic (L : Labelling S Λ)
  /-- A probabilistic labelling (Julia `ProbabilisticLabelling`). -/
  | probabilistic (L : ProbLabelling S Λ)

namespace AbstractLabelling

variable {S Λ : Type*} [Fintype Λ] [DecidableEq Λ]

/-- The label distribution of state `s`: the point mass `δ_{L(s)}` for a deterministic labelling,
and `L(s)` for a probabilistic one.

Julia counterpart: `lf[idx]` in `_bellman_helper!` (`src/bellman.jl`), i.e. `getindex` of
`DeterministicLabelling` (`src/probabilities/DeterministicLabelling.jl`) or of
`ProbabilisticLabelling` (`src/probabilities/ProbabilisticLabelling.jl`). -/
def dist : AbstractLabelling S Λ → S → ProbVec Λ
  | deterministic L, s => ProbVec.dirac (L.map s)
  | probabilistic L, s => L.map s

/-- For a deterministic labelling the label distribution is a point mass.

Julia counterpart: `getindex(dl::DeterministicLabelling, s...)`
(`src/probabilities/DeterministicLabelling.jl`). -/
@[simp] theorem dist_deterministic (L : Labelling S Λ) (s : S) :
    (deterministic L).dist s = ProbVec.dirac (L.map s) := rfl

/-- For a probabilistic labelling the label distribution is the stored column.

Julia counterpart: `getindex(pl::ProbabilisticLabelling, s)`, the column `map[:, s]`
(`src/probabilities/ProbabilisticLabelling.jl`). -/
@[simp] theorem dist_probabilistic (L : ProbLabelling S Λ) (s : S) :
    (probabilistic L).dist s = L.map s := rfl

end AbstractLabelling

end IntervalMDP
