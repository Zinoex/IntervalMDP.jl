import IntervalMDPProofs.Models.RMDP
import IntervalMDPProofs.Models.DFA

/-!
# Product of an RMDP with a DFA

The product process has states `(s, q) ∈ S × Q` and the actions of the MDP. From `(s, q)` under
action `a`, the MDP moves to `t` with some `p ∈ Γ_{s,a}`, and the DFA moves to `δ(q, l)` where
`l` is the label of the **successor** `t` (drawn from `L(t)` for a probabilistic labelling):

  `p_prod((t, q')) = p(t) · ∑_{l : δ(q, l) = q'} L(t)(l)`.

Julia counterpart: `ProductProcess` (`src/models/ProductProcess.jl`) and its Bellman operator
`_bellman_helper!` (`src/bellman.jl`), which reads `V[idx, dfa[state, lf[idx]]]` (deterministic)
or `∑ₗ prob · V[idx, dfa[state, l]]` (probabilistic) for each successor `idx`. Note: the
`ProductProcess` docstring writes `δ_{q, L(s)}` with the *source* label; the code uses the
successor's label, which is what is modeled here.
-/

namespace IntervalMDP

/-- The product of a robust MDP with a DFA via a labelling.

Julia counterpart: `ProductProcess` (`src/models/ProductProcess.jl`) with fields `mdp`, `dfa` and
`labelling_func` (here `labelling`). `mdp` is any `RMDP`, so IMDPs and factored IMDPs enter through
their `toRMDP`. -/
structure ProductProcess (S A Q Λ : Type*) [Fintype S] [Fintype Λ] where
  /-- The underlying robust MDP (Julia `mdp`). -/
  mdp : RMDP S A
  /-- The automaton (Julia `dfa`). -/
  dfa : DFA Q Λ
  /-- The labelling of MDP states (Julia `labelling_func`). -/
  labelling : AbstractLabelling S Λ

namespace ProductProcess

variable {S A Q Λ : Type*} [Fintype S] [Fintype Q] [DecidableEq Q] [Fintype Λ] [DecidableEq Λ]
variable (P : ProductProcess S A Q Λ)

/-- The distribution of the next DFA state when the DFA is in `q` and the MDP moves to `t`:
`q' ↦ ∑_{l : δ(q, l) = q'} L(t)(l)`.

Julia counterpart: `dfa[state, lf[idx]]` (deterministic) and the loop over
`enumerate(lf[idx])` (probabilistic) in `_bellman_helper!` (`src/bellman.jl`). -/
def nextDFA (q : Q) (t : S) : ProbVec Q := (P.labelling.dist t).map (P.dfa.δ q)

/-- Attach the DFA step to a successor distribution `p` from DFA state `q`:
`(lift q p)(t, q') = p(t) · nextDFA q t q'`.

Julia counterpart: the loop over successors `idx` in `_bellman_helper!` for a `ProductProcess`
(`src/bellman.jl`), which pairs each successor with the next DFA state. -/
def lift (q : Q) (p : ProbVec S) : ProbVec (S × Q) := p.prodKernel (P.nextDFA q)

/-- `lift q` as a linear map on vectors: `x ↦ ((t, q') ↦ x(t) · nextDFA q t q')`.

Julia counterpart: none (Lean-side proof device). -/
def liftLinear (q : Q) : (S → ℝ) →ₗ[ℝ] (S × Q → ℝ) where
  toFun x z := x z.1 * P.nextDFA q z.1 z.2
  map_add' x y := by funext z; simp [add_mul]
  map_smul' c x := by funext z; simp [mul_assoc]

/-- `lift q` acts on vectors as the linear map `liftLinear q`.

Julia counterpart: none (Lean-side proof device). -/
theorem coe_lift (q : Q) (p : ProbVec S) : (P.lift q p : S × Q → ℝ) = P.liftLinear q p := rfl

/-- The product ambiguity set of product state `z = (s, q)` and action `a`:
`{lift q p : p ∈ Γ_{s,a}}`.

Julia counterpart: `Γ^{prod}_{z,a}` (`ProductProcess` docstring, `src/models/ProductProcess.jl`),
with the successor-label convention of `src/bellman.jl`. -/
def ambiguity (z : S × Q) (a : A) : AmbiguitySet (S × Q) :=
  (P.mdp.ambiguity z.1 a).map (P.lift z.2)

/-- Every product ambiguity set is well formed (nonempty and closed), for deterministic and for
probabilistic labellings: it is the image of the well-formed set `Γ_{s,a}` under the linear map
`liftLinear q`. No convexity of `Γ_{s,a}` is needed, so this covers factored IMDPs too.

Julia counterpart: `ProductProcess` (`src/models/ProductProcess.jl`) with either a
`DeterministicLabelling` or a `ProbabilisticLabelling`. -/
theorem toRMDP_wellFormed (z : S × Q) (a : A) : (P.ambiguity z a).WellFormed :=
  (P.mdp.ambiguity_wellFormed z.1 a).map (P.coe_lift z.2)

/-- The product process as a robust MDP on `S × Q`; the available actions of `(s, q)` are those of
`s`.

Julia counterpart: `ProductProcess` as consumed by `bellman!` (`src/bellman.jl`);
`available_actions(proc) = available_actions(markov_process(proc))`. -/
def toRMDP : RMDP (S × Q) A where
  available z := P.mdp.available z.1
  available_nonempty z := P.mdp.available_nonempty z.1
  ambiguity := P.ambiguity
  ambiguity_wellFormed := P.toRMDP_wellFormed

/-- The ambiguity sets of `P.toRMDP`.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem toRMDP_ambiguity (z : S × Q) (a : A) :
    P.toRMDP.ambiguity z a = (P.mdp.ambiguity z.1 a).map (P.lift z.2) := rfl

/-- With a deterministic labelling `L`, the product moves to `(t, δ(q, L(t)))` with probability
`p(t)` and nowhere else (Julia: `V[idx, dfa[state, lf[idx]]]`).

Julia counterpart: `V[idx, dfa[state, lf[idx]]]` in `_bellman_helper!` (`src/bellman.jl`) for a
`DeterministicLabelling`. -/
theorem lift_deterministic {L : Labelling S Λ} (hL : P.labelling = .deterministic L)
    (q : Q) (p : ProbVec S) (t : S) (q' : Q) :
    P.lift q p (t, q') = if q' = P.dfa.δ q (L.map t) then p t else 0 := by
  simp [lift, ProbVec.prodKernel_apply, nextDFA, hL, ProbVec.map_dirac]

end ProductProcess

end IntervalMDP
