import IntervalMDPProofs.Bellman

/-!
# Factored IMDPs: successor index and vertex enumeration (Phase 5a)

For a factored IMDP (`FactoredIMDP`, `Models/Factored.lean`) the default Bellman algorithm is
vertex enumeration, `state_action_bellman(::FactoredVertexIteratorWorkspace, V, ambiguity_sets,
upper_bound)` in `src/bellman.jl`:

    iterators = vertex_generator.(ambiguity_sets, workspace.result_vectors)
    optval = upper_bound ? typemin(R) : typemax(R)
    optfunc = upper_bound ? max : min
    for marginal_vertices in Iterators.product(iterators...)
        v = sum(V[I] * prod(r -> marginal_vertices[r][I[r]], eachindex(ambiguity_sets)) for
                I in CartesianIndices(num_target.(ambiguity_sets)))
        optval = optfunc(optval, v)
    end

Each marginal iterator (`IntervalAmbiguitySetVertexIterator`,
`src/probabilities/IntervalAmbiguitySets.jl`) runs the greedy loop `lower + allocation` along a
permutation vector, starting from `1:d`, and then jumps to the lexicographically next permutation
that differs within the first `break_idx` entries (`nextPermutation`), stopping when there is none
or when `iszero(budget)`. Lean transcribes all of this literally: `vertexLoop`/`vertexOf` (greedy
loop), `nextInSuffix`/`findSwap`/`swapSort`/`nextPermutation` (skip), `vertexRun`/`vertices`
(iteration protocol), `productList`/`vertexEnumeration` (`Iterators.product`, first factor
fastest), `vertexSum` and `vertexValue` (the sum and the `max`/`min` fold over `EReal`, with
`typemin = ⊥`, `typemax = ⊤`).

## Main results

* `IntervalMDP.Index.factored_successor_eq` — hypothesis: the overflow bound
  `∏ r, state_vars[r] < 2 ^ (N - 1)` (`hBound`) and a value array storing `W`
  (`hV : StoresValues`). The index `I ∈ CartesianIndices(num_target.(ambiguity_sets))` is in
  bijection with the joint successor states (`successorState`), the `N`-bit read `V[I]` is
  `W (successorState I)`, and `prod(r -> γ[r][I[r]], …)` is the product distribution at that state.
* `IntervalMDP.Factored.vertices_complete` — no hypotheses beyond the model invariants; dense
  marginal storage only (sparse marginals not modelled, Observation O16): the loop
  over `Iterators.product(iterators...)` visits a tuple `q` iff every `q r` is an extreme point
  (vertex) of the marginal set `P(lʳ, uʳ)` of `(s, a)`. It rests on `mem_vertices_iff` (the
  iterator with its permutation skipping yields exactly the greedy points of all permutations;
  `vertexLoop_prefix`, `nextPermutation_spec`, `vertexOf_mem_vertexRun`) and
  `mem_vertices_iff_extremePoints` (these greedy points are exactly the extreme points).
* `IntervalMDP.Factored.vertexValue_eq_opt` — no hypotheses beyond the model invariants; dense
  marginal storage only (sparse marginals not modelled, Observation O16), since `vertexValue` is
  built on the dense `vertices` transcription; all satisfaction modes `sat` (`upper_bound = isoptimistic(sat)`): `vertexValue` is the exact inner
  optimum `Bellman.innerOpt sat (productSet s a) W` over the literal, **non-convex** product set
  (finite, `sup` for optimistic, `inf` for pessimistic). The strategy mode only acts afterwards
  across actions, so all four satisfaction × strategy modes are covered. This is the exact
  reference value `Bellman.stateActionBellman F.toRMDP sat W s a`
  (`vertexValue_eq_stateActionBellman`, same scope) for A2/A3 (Phases 5b, 5c).

The exactness proof follows Schnitzer, Abate, Parker, "Efficient Solution and Learning of Robust
Factored MDPs", arXiv:2508.00707, **Theorem 1** (proof in Appendix A; numbering checked in arXiv
v1 and v2 and in the AAAI-26 version): every marginal set is the convex hull of its vertices
(`vecs_eq_convexHull_vertices`, Krein–Milman), so by multilinearity every product distribution's
expectation is a convex combination of the expectations under products of marginal vertices
(`dot_piVec_eq_sum`, `exists_vertex_expansion`; the paper's two-factor statement extended to any
number of factors with `Finset.prod_univ_sum`). The product set is never convexified.

**Scope.** Abstract: values in `ℝ`/`EReal`, exact arithmetic. Dense marginal storage only
(`support(p) = 1:d`); for sparse marginals Julia permutes only the support (not modelled).
Limitations L1–L6 of the inventory apply: floating-point rounding (including the
`budget <= gap` and `iszero(budget)` tests and `budget -= gap`), overflow beyond the stated index
bound, threaded/CUDA execution, the Lean ↔ Julia correspondence (literal transcription, not
proved), DP ↔ path-measure semantics; L6 (LP solvers) does not arise here.
-/

namespace IntervalMDP

open Finset Index OMax

namespace Factored

variable {n m : ℕ}

/-! ### The product-of-marginals index -/

/-- The marginal interval sets of the state–action pair `(s, a)`, one per state variable `r`:
`ambiguity_sets[r] = marginals(model)[r][jₐ, jₛ]`.

Julia counterpart: `ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)` in
`state_bellman!(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`). -/
def ambiguitySets (F : FactoredIMDP n m) (s : F.stateVars.State) (a : F.actionVars.Action)
    (r : Fin n) : IntervalAmbiguity (Fin (F.stateVars.dims r)) :=
  (F.marginals r).get s a

/-- The number of target values `d` of an interval ambiguity set on `Fin d`.

Julia counterpart: `num_target(p::IntervalAmbiguitySet) = length(p.lower)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
def numTarget {d : ℕ} (_A : IntervalAmbiguity (Fin d)) : ℕ := d

/-- The array size `num_target.(ambiguity_sets)` whose `CartesianIndices` the vertex-enumeration
sum ranges over: `numTargets F s a r = |S_r|`.

Julia counterpart: `num_target.(ambiguity_sets)` in
`state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`); the model
constructor checks `num_target(marginal r) == state_vars[r]` (`check_transition`,
`src/models/FactoredRobustMarkovDecisionProcess.jl`), which the Lean types enforce. -/
def numTargets (F : FactoredIMDP n m) (s : F.stateVars.State) (a : F.actionVars.Action) :
    Fin n → ℕ :=
  fun r => numTarget (ambiguitySets F s a r)

/-- The joint successor state addressed by the product-of-marginals index
`I ∈ CartesianIndices(num_target.(ambiguity_sets))`: the state whose `r`-th variable has value
`I[r]`.

Julia counterpart: the `CartesianIndex` `I` in `V[I] * prod(r -> marginal_vertices[r][I[r]], …)`
of `state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`), read as a joint
state of `state_values(model)`. -/
def successorState (F : FactoredIMDP n m) (s : F.stateVars.State) (a : F.actionVars.Action)
    (I : CartesianIndex (numTargets F s a)) : F.stateVars.State :=
  fun r => I r

/-- The Julia value array `V` (of size `state_values(model)`, column-major) stores the Lean value
function `W`: the entry at the linear index of every joint state `t` is `W t`.

Julia counterpart: the value array `V` passed to `bellman!` / `state_action_bellman`
(`src/bellman.jl`), allocated with size `state_values(mp)` in `src/robust_value_iteration.jl`. -/
def StoresValues (sv : StateVars n) (Vdata : ℤ → ℝ) (W : sv.State → ℝ) : Prop :=
  ∀ t : sv.State, Vdata (linear sv.dims t) = W t

/-- Julia's read `V[I]`: the entry of the value array at `LinearIndices(size(V))[I]`, computed in
`N`-bit integers, where `size(V) = state_values(model)`.

Julia counterpart: `V[I]` with `I ∈ CartesianIndices(num_target.(ambiguity_sets))` in
`state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`). -/
def readValue (N : ℕ) (F : FactoredIMDP n m) (s : F.stateVars.State) (a : F.actionVars.Action)
    (Vdata : ℤ → ℝ) (I : CartesianIndex (numTargets F s a)) : ℝ :=
  Vdata (linearInt N F.stateVars.dims (successorState F s a I))

/-- `successorState` is a bijection from the product-of-marginals indices onto the joint states.

Julia counterpart: `CartesianIndices(num_target.(ambiguity_sets))` (`src/bellman.jl`) visits every
joint successor state exactly once. -/
theorem successorState_bijective (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) : Function.Bijective (successorState F s a) :=
  ⟨fun _ _ h => funext fun r => congrFun h r, fun t => ⟨fun r => t r, rfl⟩⟩

end Factored

namespace Index

variable {n m : ℕ}

open Factored in
/-- **Factored successor index.** In `state_action_bellman(::FactoredVertexIteratorWorkspace, …)`
the index `I ∈ CartesianIndices(num_target.(ambiguity_sets))` addresses the joint successor state
`successorState I` (with `I[r]` the value of variable `r`): the indices are in bijection with the
joint states, and under the overflow bound `∏ state_vars < 2 ^ (N - 1)` the `N`-bit read `V[I]` of a
value array storing `W` is `W (successorState I)`, while the factor
`prod(r -> γ[r][I[r]], …)` is the product distribution `ProbVec.pi γ` at that state.

Julia counterpart: `V[I] * prod(r -> marginal_vertices[r][I[r]], eachindex(ambiguity_sets))` for
`I in CartesianIndices(num_target.(ambiguity_sets))` (`src/bellman.jl`); the same index pattern
appears in the factored O-max loops of that file. -/
theorem factored_successor_eq {N : ℕ} (F : FactoredIMDP n m)
    (hBound : ∏ r, F.stateVars.dims r < 2 ^ (N - 1)) (s : F.stateVars.State)
    (a : F.actionVars.Action) {Vdata : ℤ → ℝ} {W : F.stateVars.State → ℝ}
    (hV : StoresValues F.stateVars Vdata W) :
    Function.Bijective (successorState F s a) ∧
      ∀ (I : CartesianIndex (numTargets F s a)) (γ : ∀ r, ProbVec (Fin (F.stateVars.dims r))),
        readValue N F s a Vdata I = W (successorState F s a I) ∧
          ProbVec.pi γ (successorState F s a I) = ∏ r, γ r (I r) := by
  refine ⟨successorState_bijective F s a, fun I γ => ⟨?_, rfl⟩⟩
  rw [readValue, (linear_bijective (N := N) hBound).2]
  exact hV _

end Index

namespace Factored

variable {n m : ℕ}

/-! ### The marginal vertex loop -/

/-- Julia's in-place update `v[i] += x` of the result vector, for the 1-based index `i`.

Julia counterpart: `v[i] += budget` / `v[i] += gap(it.set, i)` in
`Base.iterate(::IntervalAmbiguitySetVertexIterator, …)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
def addAt {d : ℕ} (v : Fin d → ℝ) (i : ℕ) (x : ℝ) : Fin d → ℝ :=
  fun t => if toJulia t = i then v t + x else v t

/-- Literal transcription of the vertex loop of `Base.iterate(::IntervalAmbiguitySetVertexIterator,
…)`:

    for (j, i) in enumerate(permutation)
        if budget <= gap(it.set, i)
            v[i] += budget; break_idx = j; break
        else
            v[i] += gap(it.set, i); budget -= gap(it.set, i)
        end
    end

The arguments are the remaining `permutation`, the `enumerate` counter `j`, `budget` and `v`; the
result is `(v, break_idx)`, with `break_idx = 0` if the loop does not break. For a dense set
`support(it.set)[i] = i`, so the support lookup is the identity.

Julia counterpart: the `for (j, i) in enumerate(permutation)` loop of both `Base.iterate` methods
of `IntervalAmbiguitySetVertexIterator` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
noncomputable def vertexLoop {d : ℕ} (gap : Fin d → ℝ) :
    List ℕ → ℕ → ℝ → (Fin d → ℝ) → (Fin d → ℝ) × ℕ
  | [], _, _, v => (v, 0)
  | i :: perm, j, budget, v =>
    if budget ≤ juliaGet gap i then (addAt v i budget, j)
    else vertexLoop gap perm (j + 1) (budget - juliaGet gap i) (addAt v i (juliaGet gap i))

/-- The vertex (and break index) the iterator computes for one permutation vector:
`copyto!(v, lower(it.set)); budget = one(R) - sum(v)`, then `vertexLoop`.

Julia counterpart: the body of `Base.iterate(it::IntervalAmbiguitySetVertexIterator[, state])`
after the permutation is fixed (`src/probabilities/IntervalAmbiguitySets.jl`). -/
noncomputable def vertexOf {d : ℕ} (A : IntervalAmbiguity (Fin d)) (perm : List ℕ) :
    (Fin d → ℝ) × ℕ :=
  vertexLoop A.gap perm 1 (1 - ∑ t, A.lower t) A.lower

/-- An index that does not occur in `perm` gets no allocation.

Julia counterpart: none (Lean-side proof device). -/
theorem allocation_of_not_mem {d : ℕ} (gap : Fin d → ℝ) {k : ℕ} :
    ∀ (perm : List ℕ) (b : ℝ), k ∉ perm → allocation gap perm b k = 0
  | [], _, _ => rfl
  | i :: perm, b, hk => by
    rw [List.mem_cons, not_or] at hk
    simp only [allocation]
    rw [if_neg (fun h => hk.1 h.symm)]
    exact allocation_of_not_mem gap perm _ hk.2

/-- The vertex loop adds the O-max allocation to `v`: entry `t` becomes
`v t + allocation gap perm budget (toJulia t)` (`OMax.allocation`, the amount `min(budget, gap[i])`
of the greedy loop), for a duplicate-free `perm` and a nonnegative gap.

Julia counterpart: the `for (j, i) in enumerate(permutation)` loop of
`Base.iterate(::IntervalAmbiguitySetVertexIterator, …)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem vertexLoop_fst {d : ℕ} {gap : Fin d → ℝ} (hg : ∀ t, 0 ≤ gap t) :
    ∀ (perm : List ℕ) (j : ℕ) (b : ℝ) (v : Fin d → ℝ), perm.Nodup → ∀ t,
      (vertexLoop gap perm j b v).1 t = v t + allocation gap perm b (toJulia t)
  | [], _, _, v, _, t => by simp [vertexLoop, allocation]
  | i :: perm, j, b, v, hnd, t => by
    rw [List.nodup_cons] at hnd
    by_cases hb : b ≤ juliaGet gap i
    · rw [vertexLoop, if_pos hb]
      simp only [addAt, allocation, min_eq_left hb]
      by_cases ht : toJulia t = i
      · rw [if_pos ht, if_pos ht.symm]
      · rw [if_neg ht, if_neg (Ne.symm ht), sub_self, allocation_zero hg, add_zero]
    · rw [vertexLoop, if_neg hb, vertexLoop_fst hg perm _ _ _ hnd.2 t]
      have hmin : min b (juliaGet gap i) = juliaGet gap i := min_eq_right (le_of_not_ge hb)
      simp only [addAt, allocation, hmin]
      by_cases ht : toJulia t = i
      · rw [if_pos ht, if_pos ht.symm, ht, allocation_of_not_mem gap perm _ hnd.1, add_zero]
      · rw [if_neg ht, if_neg (Ne.symm ht)]

/-- The break index is `0` (no break) or a position `j ≤ break_idx < j + length(permutation)`.

Julia counterpart: `break_idx` in `Base.iterate(::IntervalAmbiguitySetVertexIterator, …)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem vertexLoop_snd {d : ℕ} (gap : Fin d → ℝ) :
    ∀ (perm : List ℕ) (j : ℕ) (b : ℝ) (v : Fin d → ℝ),
      (vertexLoop gap perm j b v).2 = 0 ∨
        (j ≤ (vertexLoop gap perm j b v).2 ∧ (vertexLoop gap perm j b v).2 < j + perm.length)
  | [], _, _, _ => Or.inl rfl
  | i :: perm, j, b, v => by
    by_cases hb : b ≤ juliaGet gap i
    · rw [vertexLoop, if_pos hb]
      right
      simp
    · rw [vertexLoop, if_neg hb]
      rcases vertexLoop_snd gap perm (j + 1) (b - juliaGet gap i) (addAt v i (juliaGet gap i))
        with h | h
      · exact Or.inl h
      · right
        simp only [List.length_cons]
        omega

/-- If the loop breaks, it reads only the prefix of `permutation` up to the break index.

Julia counterpart: none (Lean-side proof device). -/
theorem vertexLoop_take {d : ℕ} (gap : Fin d → ℝ) :
    ∀ (perm : List ℕ) (j : ℕ) (b : ℝ) (v : Fin d → ℝ), (vertexLoop gap perm j b v).2 ≠ 0 →
      vertexLoop gap (perm.take ((vertexLoop gap perm j b v).2 + 1 - j)) j b v =
        vertexLoop gap perm j b v
  | [], _, _, _, h => absurd rfl h
  | i :: perm, j, b, v, h => by
    by_cases hb : b ≤ juliaGet gap i
    · simp only [vertexLoop, if_pos hb, Nat.add_sub_cancel_left, List.take_succ_cons,
        List.take_zero]
    · have hrec := vertexLoop_take gap perm (j + 1) (b - juliaGet gap i)
        (addAt v i (juliaGet gap i))
      simp only [vertexLoop, if_neg hb] at h hrec ⊢
      have hge := (vertexLoop_snd gap perm (j + 1) (b - juliaGet gap i)
        (addAt v i (juliaGet gap i))).resolve_left h
      have hc : (vertexLoop gap perm (j + 1) (b - juliaGet gap i)
          (addAt v i (juliaGet gap i))).2 + 1 - j =
          ((vertexLoop gap perm (j + 1) (b - juliaGet gap i)
            (addAt v i (juliaGet gap i))).2 + 1 - (j + 1)) + 1 := by omega
      rw [hc, List.take_succ_cons, vertexLoop, if_neg hb, hrec h]

/-- If the loop breaks within `L₁`, appending `L₂` does not change its result.

Julia counterpart: none (Lean-side proof device). -/
theorem vertexLoop_append {d : ℕ} (gap : Fin d → ℝ) (L₂ : List ℕ) :
    ∀ (L₁ : List ℕ) (j : ℕ) (b : ℝ) (v : Fin d → ℝ), (vertexLoop gap L₁ j b v).2 ≠ 0 →
      vertexLoop gap (L₁ ++ L₂) j b v = vertexLoop gap L₁ j b v
  | [], _, _, _, h => absurd rfl h
  | i :: L₁, j, b, v, h => by
    by_cases hb : b ≤ juliaGet gap i
    · simp only [List.cons_append, vertexLoop, if_pos hb]
    · simp only [vertexLoop, if_neg hb] at h
      simp only [List.cons_append, vertexLoop, if_neg hb]
      exact vertexLoop_append gap L₂ L₁ _ _ _ h

/-- **Skipped permutations give the same vertex.** If the loop on `perm` breaks at
`break_idx ≠ 0` and `π` agrees with `perm` on the first `break_idx` entries, the loop on `π` gives
the same vertex and break index. This is why the iterator may skip all permutations with the prefix
`permutation[1:break_idx]`.

Julia counterpart: "Skip permutations that would lead to the same vertex based on the prefix
1:last_break_idx" in `Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem vertexLoop_prefix {d : ℕ} (gap : Fin d → ℝ) {perm π : List ℕ} (b : ℝ) (v : Fin d → ℝ)
    (h : (vertexLoop gap perm 1 b v).2 ≠ 0)
    (htake : π.take (vertexLoop gap perm 1 b v).2 = perm.take (vertexLoop gap perm 1 b v).2) :
    vertexLoop gap π 1 b v = vertexLoop gap perm 1 b v := by
  have h1 := vertexLoop_take gap perm 1 b v h
  rw [Nat.add_sub_cancel] at h1
  have h2 : (vertexLoop gap (perm.take (vertexLoop gap perm 1 b v).2) 1 b v).2 ≠ 0 := by
    rw [h1]; exact h
  rw [← List.take_append_drop (vertexLoop gap perm 1 b v).2 π, htake,
    vertexLoop_append gap _ _ 1 b v h2, h1]

/-- If the loop does not break on a nonempty list, the budget exceeds the sum of the gaps it
visited.

Julia counterpart: none (Lean-side proof device). -/
theorem sum_lt_of_vertexLoop_snd_eq_zero {d : ℕ} (gap : Fin d → ℝ) :
    ∀ (L : List ℕ) (j : ℕ) (b : ℝ) (v : Fin d → ℝ), 0 < j → (vertexLoop gap L j b v).2 = 0 →
      L ≠ [] → (L.map (juliaGet gap)).sum < b
  | [], _, _, _, _, _, hL => absurd rfl hL
  | i :: L, j, b, v, hj, h, _ => by
    by_cases hb : b ≤ juliaGet gap i
    · rw [vertexLoop, if_pos hb] at h
      exact absurd h (Nat.pos_iff_ne_zero.mp hj)
    · rw [vertexLoop, if_neg hb] at h
      rw [List.map_cons, List.sum_cons]
      by_cases hL : L = []
      · subst hL
        simp only [List.map_nil, List.sum_nil, add_zero]
        exact lt_of_not_ge hb
      · have := sum_lt_of_vertexLoop_snd_eq_zero gap L (j + 1) _ _ (Nat.succ_pos j) h hL
        linarith

/-- Summing `g` over a permutation vector of `1:d` is summing over all Julia indices.

Julia counterpart: none (Lean-side proof device). -/
theorem sum_map_of_perm {d : ℕ} {perm : List ℕ} (h : perm.Perm (List.range' 1 d)) (g : ℕ → ℝ) :
    (perm.map g).sum = ∑ t : Fin d, g (toJulia t) := by
  rw [(h.map g).sum_eq, ← map_toJulia_finRange, List.map_map, Fin.sum_univ_def]
  rfl

/-- Summing `juliaGet f` over a permutation vector of `1:d` is `∑ₜ f t`.

Julia counterpart: none (Lean-side proof device). -/
theorem sum_map_juliaGet {d : ℕ} {perm : List ℕ} (h : perm.Perm (List.range' 1 d))
    (f : Fin d → ℝ) : (perm.map (juliaGet f)).sum = ∑ t, f t := by
  rw [sum_map_of_perm h]
  simp only [juliaGet_toJulia]

/-- For a permutation vector of `1:d`, the vertex loop always breaks (`break_idx ≥ 1`), because
`budget ≤ ∑ gap`.

Julia counterpart: `break_idx` in `Base.iterate(::IntervalAmbiguitySetVertexIterator, …)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem vertexOf_snd_ne_zero {d : ℕ} (A : IntervalAmbiguity (Fin d)) {perm : List ℕ}
    (h : perm.Perm (List.range' 1 d)) : (vertexOf A perm).2 ≠ 0 := by
  intro h0
  have hsum := A.budget_le_sum_gap
  simp only [IntervalAmbiguity.budget] at hsum
  by_cases hp : perm = []
  · subst hp
    have hd : d = 0 := by simpa using h.length_eq.symm
    subst hd
    norm_num at hsum
  · have hlt := sum_lt_of_vertexLoop_snd_eq_zero A.gap perm 1 _ A.lower Nat.one_pos h0 hp
    rw [sum_map_juliaGet h] at hlt
    linarith

/-- The vertex of a permutation vector of `1:d` is `lower + allocation`.

Julia counterpart: the vector `v` returned by `Base.iterate(::IntervalAmbiguitySetVertexIterator,
…)` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem vertexOf_fst {d : ℕ} (A : IntervalAmbiguity (Fin d)) {perm : List ℕ}
    (h : perm.Perm (List.range' 1 d)) (t : Fin d) :
    (vertexOf A perm).1 t = A.lower t + allocation A.gap perm A.budget (toJulia t) :=
  vertexLoop_fst A.gap_nonneg perm 1 _ A.lower (h.nodup_iff.2 List.nodup_range') t

/-- With zero budget (`∑ lower = 1`) every permutation gives the vertex `lower`.

Julia counterpart: the `iszero(budget)` exit of `Base.iterate(::IntervalAmbiguitySetVertexIterator,
state)` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem vertexOf_fst_of_budget_eq_zero {d : ℕ} (A : IntervalAmbiguity (Fin d)) {perm : List ℕ}
    (h : perm.Perm (List.range' 1 d)) (hb : A.budget = 0) : (vertexOf A perm).1 = A.lower := by
  funext t
  rw [vertexOf_fst A h, hb, allocation_zero A.gap_nonneg, add_zero]

/-! ### Skipping to the next permutation -/

/-- The 1-based read `permutation[k]` of a permutation vector (`0` out of range).

Julia counterpart: `permutation[k]` in `Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
def permGet (perm : List ℕ) (k : ℕ) : ℕ := perm.getD (k - 1) 0

/-- One iteration of the inner loop `for k in (j + 1):length(permutation)`: if
`permutation[k] > permutation[j]` and (`next_in_suffix` is `nothing` or
`permutation[k] < permutation[next_in_suffix]`), set `next_in_suffix = k`.

Julia counterpart: the body of the inner `for k` loop in
`Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
def suffixStep (perm : List ℕ) (j : ℕ) (next : Option ℕ) (k : ℕ) : Option ℕ :=
  if permGet perm j < permGet perm k then
    match next with
    | none => some k
    | some nk => if permGet perm k < permGet perm nk then some k else some nk
  else next

/-- Literal transcription of the inner loop: the position `k > j` of the smallest entry of
`permutation[j+1:end]` that is larger than `permutation[j]` (`nothing` if there is none).

Julia counterpart: `next_in_suffix` after `for k in (j + 1):length(permutation)` in
`Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
def nextInSuffix (perm : List ℕ) (j : ℕ) : Option ℕ :=
  (List.range' (j + 1) (perm.length - j)).foldl (suffixStep perm j) none

/-- Literal transcription of the outer loop `for j in last_break_idx:-1:1`: the first `j`
(counting down from `last_break_idx`) with a `next_in_suffix`, together with that position.

Julia counterpart: the `for j in last_break_idx:-1:1` loop in
`Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`); `nothing` is the `isnothing(break_j)` exit. -/
def findSwap (perm : List ℕ) : ℕ → Option (ℕ × ℕ)
  | 0 => none
  | j + 1 =>
    match nextInSuffix perm (j + 1) with
    | some k => some (j + 1, k)
    | none => findSwap perm j

/-- The permutation after `permutation[j], permutation[next_in_suffix] =
permutation[next_in_suffix], permutation[j]` and `sort!(@view(permutation[(break_j + 1):end]))`,
for `jk = (break_j, next_in_suffix)`.

Julia counterpart: the swap and `sort!` in `Base.iterate(::IntervalAmbiguitySetVertexIterator,
state)` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
def swapSort (perm : List ℕ) (jk : ℕ × ℕ) : List ℕ :=
  let swapped := (perm.set (jk.1 - 1) (permGet perm jk.2)).set (jk.2 - 1) (permGet perm jk.1)
  swapped.take jk.1 ++ (swapped.drop jk.1).mergeSort

/-- The next permutation vector the iterator visits after `perm`, whose vertex broke at
`last_break_idx` (`none`: the iterator stops). It is the lexicographically next permutation that
differs from `perm` within the first `last_break_idx` entries (`nextPermutation_spec`).

Julia counterpart: the first part of `Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
with `state = (permutation, last_break_idx)` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
def nextPermutation (perm : List ℕ) (lastBreak : ℕ) : Option (List ℕ) :=
  (findSwap perm lastBreak).map (swapSort perm)

/-- Invariants of one `suffixStep`.

Julia counterpart: none (Lean-side proof device). -/
theorem suffixStep_spec (perm : List ℕ) (j : ℕ) (init : Option ℕ) (k₁ : ℕ) :
    (suffixStep perm j init k₁ = none → init = none ∧ ¬ permGet perm j < permGet perm k₁) ∧
    (∀ x, suffixStep perm j init k₁ = some x →
      init = some x ∨ (x = k₁ ∧ permGet perm j < permGet perm k₁)) ∧
    (permGet perm j < permGet perm k₁ →
      ∃ x, suffixStep perm j init k₁ = some x ∧ permGet perm x ≤ permGet perm k₁) ∧
    (∀ k₀, init = some k₀ →
      ∃ x, suffixStep perm j init k₁ = some x ∧ permGet perm x ≤ permGet perm k₀) := by
  unfold suffixStep
  by_cases hc : permGet perm j < permGet perm k₁
  · rw [if_pos hc]
    cases init with
    | none => simp [hc]
    | some nk =>
      by_cases hlt : permGet perm k₁ < permGet perm nk
      · simp only [if_pos hlt, reduceCtorEq, false_imp_iff, true_and, Option.some.injEq]
        refine ⟨fun x hx => Or.inr ⟨hx.symm, hc⟩, fun _ => ⟨k₁, rfl, le_refl _⟩,
          fun k₀ hk₀ => ⟨k₁, rfl, ?_⟩⟩
        rw [← hk₀]; exact hlt.le
      · simp only [if_neg hlt, reduceCtorEq, false_imp_iff, true_and, Option.some.injEq]
        refine ⟨fun x hx => Or.inl hx, fun _ => ⟨nk, rfl, not_lt.mp hlt⟩,
          fun k₀ hk₀ => ⟨nk, rfl, ?_⟩⟩
        rw [hk₀]
  · rw [if_neg hc]
    refine ⟨fun h => ⟨h, hc⟩, fun x hx => Or.inl hx, fun h => absurd h hc,
      fun k₀ hk₀ => ⟨k₀, hk₀, le_refl _⟩⟩

/-- Loop invariant of the inner `for k` loop: the result is `nothing` only if no position is a
candidate (`permutation[k] > permutation[j]`), and otherwise a candidate with the smallest entry.

Julia counterpart: none (Lean-side proof device). -/
theorem foldl_suffixStep (perm : List ℕ) (j : ℕ) : ∀ (ks : List ℕ) (init : Option ℕ),
    (ks.foldl (suffixStep perm j) init = none →
      init = none ∧ ∀ k ∈ ks, ¬ permGet perm j < permGet perm k) ∧
    (∀ k, ks.foldl (suffixStep perm j) init = some k →
      (init = some k ∨ (k ∈ ks ∧ permGet perm j < permGet perm k)) ∧
      (∀ k' ∈ ks, permGet perm j < permGet perm k' → permGet perm k ≤ permGet perm k') ∧
      (∀ k₀, init = some k₀ → permGet perm k ≤ permGet perm k₀))
  | [], init => by
    refine ⟨fun h => ⟨h, by simp⟩, fun k hk => ⟨Or.inl hk, by simp, fun k₀ hk₀ => ?_⟩⟩
    rw [List.foldl_nil] at hk
    rw [hk] at hk₀
    cases hk₀
    exact le_refl _
  | k₁ :: ks, init => by
    rw [List.foldl_cons]
    obtain ⟨hn, hs⟩ := foldl_suffixStep perm j ks (suffixStep perm j init k₁)
    obtain ⟨S1, S2, S3, S4⟩ := suffixStep_spec perm j init k₁
    refine ⟨fun h => ?_, fun k hk => ?_⟩
    · obtain ⟨h1, h2⟩ := hn h
      obtain ⟨h3, h4⟩ := S1 h1
      exact ⟨h3, fun k hk => (List.mem_cons.1 hk).elim (fun e => e ▸ h4) (h2 k)⟩
    · obtain ⟨h1, h2, h3⟩ := hs k hk
      refine ⟨?_, fun k' hk' hc => ?_, fun k₀ hk₀ => ?_⟩
      · rcases h1 with h1 | h1
        · rcases S2 k h1 with h | ⟨rfl, hc⟩
          · exact Or.inl h
          · exact Or.inr ⟨List.mem_cons_self .., hc⟩
        · exact Or.inr ⟨List.mem_cons_of_mem _ h1.1, h1.2⟩
      · rcases List.mem_cons.1 hk' with rfl | hk'
        · obtain ⟨x, hx, hle⟩ := S3 hc
          exact (h3 x hx).trans hle
        · exact h2 k' hk' hc
      · obtain ⟨x, hx, hle⟩ := S4 k₀ hk₀
        exact (h3 x hx).trans hle

/-- If the inner loop finds `next_in_suffix = k`, then `j < k ≤ length(permutation)`,
`permutation[k] > permutation[j]`, and `permutation[k]` is the smallest such entry after `j`.

Julia counterpart: `next_in_suffix` in `Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem nextInSuffix_some {perm : List ℕ} {j k : ℕ} (h : nextInSuffix perm j = some k) :
    j < k ∧ k ≤ perm.length ∧ permGet perm j < permGet perm k ∧
      ∀ k', j < k' → k' ≤ perm.length → permGet perm j < permGet perm k' →
        permGet perm k ≤ permGet perm k' := by
  obtain ⟨h1, h2, -⟩ := (foldl_suffixStep perm j _ none).2 k h
  rcases h1 with h1 | ⟨hk, hc⟩
  · cases h1
  · rw [List.mem_range'_1] at hk
    refine ⟨by omega, by omega, hc, fun k' hjk hk' hc' => h2 k' ?_ hc'⟩
    rw [List.mem_range'_1]
    omega

/-- If the inner loop finds no `next_in_suffix`, no entry after position `j` is larger than
`permutation[j]`.

Julia counterpart: the `isnothing(next_in_suffix)` branch in
`Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem nextInSuffix_none {perm : List ℕ} {j : ℕ} (h : nextInSuffix perm j = none) :
    ∀ k', j < k' → k' ≤ perm.length → ¬ permGet perm j < permGet perm k' := by
  intro k' hjk hk'
  refine ((foldl_suffixStep perm j _ none).1 h).2 k' ?_
  rw [List.mem_range'_1]
  omega

/-- If the outer loop finds `(break_j, next_in_suffix) = (j, k)`, then `1 ≤ j ≤ last_break_idx`,
the inner loop at `j` found `k`, and it found nothing at every `j' ∈ (j, last_break_idx]`.

Julia counterpart: `break_j` in `Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem findSwap_some {perm : List ℕ} : ∀ {b j k : ℕ}, findSwap perm b = some (j, k) →
    1 ≤ j ∧ j ≤ b ∧ nextInSuffix perm j = some k ∧
      ∀ j', j < j' → j' ≤ b → nextInSuffix perm j' = none
  | 0, _, _, h => by simp [findSwap] at h
  | b + 1, j, k, h => by
    unfold findSwap at h
    cases hn : nextInSuffix perm (b + 1) with
    | some k' =>
      rw [hn] at h
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨by omega, le_refl _, hn, fun j' h1 h2 => absurd h2 (by omega)⟩
    | none =>
      rw [hn] at h
      obtain ⟨h1, h2, h3, h4⟩ := findSwap_some h
      refine ⟨h1, by omega, h3, fun j' hj' hj'' => ?_⟩
      rcases Nat.lt_or_ge j' (b + 1) with hlt | hge
      · exact h4 j' hj' (by omega)
      · rw [show j' = b + 1 by omega]; exact hn

/-- If the outer loop finds nothing, the inner loop finds nothing for any
`j ∈ 1:last_break_idx`.

Julia counterpart: the `isnothing(break_j)` exit of
`Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem findSwap_none {perm : List ℕ} : ∀ {b : ℕ}, findSwap perm b = none →
    ∀ j', 1 ≤ j' → j' ≤ b → nextInSuffix perm j' = none
  | 0, _, j', h1, h2 => absurd h2 (by omega)
  | b + 1, h, j', h1, h2 => by
    unfold findSwap at h
    cases hn : nextInSuffix perm (b + 1) with
    | some k' => rw [hn] at h; cases h
    | none =>
      rw [hn] at h
      rcases Nat.lt_or_ge j' (b + 1) with hlt | hge
      · exact findSwap_none h j' h1 (by omega)
      · rw [show j' = b + 1 by omega]; exact hn

/-- `permGet` is list indexing within range.

Julia counterpart: none (Lean-side proof device). -/
theorem permGet_eq {perm : List ℕ} {k : ℕ} (h : k - 1 < perm.length) :
    permGet perm k = perm[k - 1] := by
  simp [permGet, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]

/-- `swapSort perm (j, k)` for `1 ≤ j < k ≤ length(perm)` keeps the first `j - 1` entries, puts
`permutation[k]` at position `j`, is sorted after position `j`, and is a permutation of `perm`.

Julia counterpart: the swap and `sort!(@view(permutation[(break_j + 1):end]))` in
`Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem swapSort_eq {perm : List ℕ} {j k : ℕ} (hj : 1 ≤ j) (hjk : j < k) (hk : k ≤ perm.length) :
    ∃ S : List ℕ, swapSort perm (j, k) = perm.take (j - 1) ++ permGet perm k :: S ∧
      S.Pairwise (· ≤ ·) ∧ (swapSort perm (j, k)).Perm perm := by
  have hj1 : j - 1 < perm.length := by omega
  have hk1 : k - 1 < perm.length := by omega
  have hsp : ((perm.set (j - 1) (permGet perm k)).set (k - 1) (permGet perm j)).Perm perm := by
    rw [permGet_eq hk1, permGet_eq hj1]
    exact List.set_set_perm hj1 hk1
  have htake : ((perm.set (j - 1) (permGet perm k)).set (k - 1) (permGet perm j)).take j =
      perm.take (j - 1) ++ [permGet perm k] := by
    apply List.ext_getElem
    · simp only [List.length_take, List.length_set, List.length_append, List.length_singleton]
      omega
    · intro t h1 _
      simp only [List.length_take, List.length_set] at h1
      have hkt : k - 1 ≠ t := by omega
      rw [List.getElem_take, List.getElem_set, if_neg hkt, List.getElem_set]
      by_cases hjt : j - 1 = t
      · rw [if_pos hjt, List.getElem_append_right (by simp; omega)]
        simp
      · rw [if_neg hjt, List.getElem_append_left (by simp; omega), List.getElem_take]
  refine ⟨(((perm.set (j - 1) (permGet perm k)).set (k - 1) (permGet perm j)).drop j).mergeSort,
    ?_, ?_, ?_⟩
  · simp only [swapSort]
    rw [htake, List.append_assoc, List.singleton_append]
  · have h := List.pairwise_mergeSort (le := fun a b : ℕ => decide (a ≤ b))
      (fun a b c hab hbc => by simp only [decide_eq_true_eq] at *; omega)
      (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega)
      (((perm.set (j - 1) (permGet perm k)).set (k - 1) (permGet perm j)).drop j)
    exact h.imp (fun hab => of_decide_eq_true hab)
  · simp only [swapSort]
    refine List.Perm.trans ?_ hsp
    conv_rhs => rw [← List.take_append_drop j
      ((perm.set (j - 1) (permGet perm k)).set (k - 1) (permGet perm j))]
    exact List.Perm.append_left _ (List.mergeSort_perm _ _)

/-- A sorted list is lexicographically at most every permutation of it.

Julia counterpart: none (Lean-side proof device). -/
theorem lt_or_eq_of_sorted_perm : ∀ {l l' : List ℕ}, l.Pairwise (· ≤ ·) → l.Perm l' →
    l < l' ∨ l = l'
  | [], _, _, hp => Or.inr (List.Perm.nil_eq hp)
  | _ :: _, [], _, hp => absurd hp.length_eq (by simp)
  | a :: t, c :: t', hs, hp => by
    rw [List.pairwise_cons] at hs
    have hc : c ∈ a :: t := hp.symm.subset (List.mem_cons_self ..)
    have hac : a ≤ c := by
      rcases List.mem_cons.1 hc with h | h
      · exact h ▸ le_refl a
      · exact hs.1 c h
    rcases lt_or_eq_of_le hac with h | h
    · exact Or.inl (List.cons_lt_cons_iff.2 (Or.inl h))
    · subst h
      rcases lt_or_eq_of_sorted_perm hs.2 hp.cons_inv with h | h
      · exact Or.inl (List.cons_lt_cons_iff.2 (Or.inr ⟨rfl, h⟩))
      · exact Or.inr (h ▸ rfl)

/-- Every permutation the iterator moves to is a permutation of the previous one and
lexicographically larger.

Julia counterpart: `Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem nextPermutation_some {perm σ : List ℕ} {b : ℕ} (h : nextPermutation perm b = some σ) :
    σ.Perm perm ∧ perm < σ := by
  unfold nextPermutation at h
  cases hf : findSwap perm b with
  | none => rw [hf] at h; cases h
  | some jk =>
    obtain ⟨j, k⟩ := jk
    rw [hf, Option.map_some, Option.some.injEq] at h
    subst h
    obtain ⟨hj, -, hn, -⟩ := findSwap_some hf
    obtain ⟨hjk, hk, hc, -⟩ := nextInSuffix_some hn
    obtain ⟨S, hS, -, hp⟩ := swapSort_eq hj hjk hk
    refine ⟨hp, ?_⟩
    rw [hS]
    conv_lhs => rw [← List.take_append_drop (j - 1) perm,
      List.drop_eq_getElem_cons (by omega : j - 1 < perm.length)]
    apply List.append_left_lt
    rw [List.cons_lt_cons_iff, ← permGet_eq (by omega)]
    exact Or.inl hc

/-- **The skip reaches every vertex class.** Let `π` be a permutation of `perm` that is
lexicographically larger and differs from `perm` within the first `last_break_idx` entries. Then
the iterator does not stop at `perm`, and the permutation it moves to is lexicographically at most
`π`. (All permutations strictly between `perm` and the next one share `perm`'s prefix of length
`last_break_idx`, hence its vertex: `vertexLoop_prefix`.)

Julia counterpart: `Base.iterate(::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem nextPermutation_spec {perm π : List ℕ} (hπ : π.Perm perm) (hlt : perm < π) {b : ℕ}
    (htake : π.take b ≠ perm.take b) :
    ∃ σ, nextPermutation perm b = some σ ∧ (σ < π ∨ σ = π) := by
  have hlen : π.length = perm.length := hπ.length_eq
  rcases List.lt_iff_exists.1 hlt with ⟨-, h⟩ | ⟨i, h1, h2, hagree, hi⟩
  · omega
  have htake_i : ∀ c ≤ i, π.take c = perm.take c := by
    intro c hc
    apply List.ext_getElem
    · simp [hlen]
    · intro t ht1 _
      simp only [List.length_take] at ht1
      simp only [List.getElem_take]
      exact (hagree t (by omega)).symm
  have hib : i < b := by
    by_contra hb
    exact htake (htake_i b (by omega))
  have hpd : perm = perm.take i ++ perm[i] :: perm.drop (i + 1) := by
    rw [← List.drop_eq_getElem_cons h1, List.take_append_drop]
  have hπd : π = perm.take i ++ π[i] :: π.drop (i + 1) := by
    rw [← htake_i i le_rfl, ← List.drop_eq_getElem_cons h2, List.take_append_drop]
  have hmem : π[i] ∈ perm.drop (i + 1) := by
    have hp' : (π[i] :: π.drop (i + 1)).Perm (perm[i] :: perm.drop (i + 1)) := by
      have h3 := hπ
      rw [hπd] at h3
      conv at h3 => rhs; rw [hpd]
      exact (List.perm_append_left_iff _).1 h3
    rcases List.mem_cons.1 (hp'.subset (List.mem_cons_self ..)) with he | he
    · exact absurd he (ne_of_gt hi)
    · exact he
  obtain ⟨q, hq, hqe⟩ := List.getElem_of_mem hmem
  rw [List.getElem_drop] at hqe
  simp only [List.length_drop] at hq
  have hcand_le : i + 1 + q + 1 ≤ perm.length := by omega
  have hpg_i : permGet perm (i + 1) = perm[i] := by
    rw [permGet_eq (by omega)]; rfl
  have hpg_k₀ : permGet perm (i + 1 + q + 1) = π[i] := by
    rw [permGet_eq (by omega), ← hqe]; rfl
  have hcand : permGet perm (i + 1) < permGet perm (i + 1 + q + 1) := by
    rw [hpg_i, hpg_k₀]; exact hi
  have hsome : nextInSuffix perm (i + 1) ≠ none := fun hn =>
    nextInSuffix_none hn (i + 1 + q + 1) (by omega) hcand_le hcand
  cases hf : findSwap perm b with
  | none => exact absurd (findSwap_none hf (i + 1) (by omega) (by omega)) hsome
  | some jk =>
    obtain ⟨j, k⟩ := jk
    obtain ⟨hj, -, hn, hmax⟩ := findSwap_some hf
    have hij : i + 1 ≤ j := by
      by_contra hc
      exact hsome (hmax (i + 1) (by omega) (by omega))
    obtain ⟨hjk, hk, -, hmin⟩ := nextInSuffix_some hn
    obtain ⟨S, hS, hsorted, hSp⟩ := swapSort_eq hj hjk hk
    refine ⟨swapSort perm (j, k), by simp [nextPermutation, hf], ?_⟩
    rw [hS]
    rcases Nat.lt_or_ge (i + 1) j with hlt' | hge
    · left
      have hsplit : perm.take (j - 1) =
          perm.take i ++ perm[i] :: (perm.drop (i + 1)).take (j - 1 - (i + 1)) := by
        have e : j - 1 = i + ((j - 1 - (i + 1)) + 1) := by omega
        rw [e, List.take_add, List.drop_eq_getElem_cons h1, List.take_succ_cons]
        congr 3
        omega
      rw [hsplit, hπd, List.append_assoc, List.cons_append]
      exact List.append_left_lt (List.cons_lt_cons_iff.2 (Or.inl hi))
    · have hji : j = i + 1 := by omega
      subst hji
      rw [Nat.add_sub_cancel] at hS ⊢
      have hy : permGet perm k ≤ π[i] := by
        rw [← hpg_k₀]; exact hmin _ (by omega) hcand_le hcand
      rcases lt_or_eq_of_le hy with hy | hy
      · left
        rw [hπd]
        exact List.append_left_lt (List.cons_lt_cons_iff.2 (Or.inl hy))
      · have hSπ : S.Perm (π.drop (i + 1)) := by
          have h3 : (perm.take i ++ permGet perm k :: S).Perm
              (perm.take i ++ π[i] :: π.drop (i + 1)) := by
            rw [← hS, ← hπd]; exact hSp.trans hπ.symm
          rw [hy] at h3
          exact ((List.perm_append_left_iff _).1 h3).cons_inv
        rcases lt_or_eq_of_sorted_perm hsorted hSπ with h | h
        · left
          rw [hπd, hy]
          exact List.append_left_lt (List.cons_lt_cons_iff.2 (Or.inr ⟨rfl, h⟩))
        · right
          rw [hπd, hy, h]

/-! ### The marginal vertex iterator -/

/-- Literal transcription of the iteration protocol of `IntervalAmbiguitySetVertexIterator`:
the vertex of the current `permutation` (`iterate(it)` starts with `permutation = 1:d`), then
`iterate(it, (permutation, break_idx))` moves to `nextPermutation permutation break_idx` and stops
(`return nothing`) when there is none or when `iszero(budget)`. The natural-number `fuel` is a Lean
device that bounds the number of iterations; `vertices` uses `fuel = d!`, the number of
permutations, which suffices (`vertexOf_mem_vertexRun`).

Julia counterpart: `Base.iterate(it::IntervalAmbiguitySetVertexIterator)` and
`Base.iterate(it::IntervalAmbiguitySetVertexIterator, state)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
noncomputable def vertexRun {d : ℕ} (A : IntervalAmbiguity (Fin d)) :
    ℕ → List ℕ → List (Fin d → ℝ)
  | 0, _ => []
  | fuel + 1, perm =>
    (vertexOf A perm).1 ::
      match nextPermutation perm (vertexOf A perm).2 with
      | none => []
      | some σ => if 1 - ∑ t, A.lower t = 0 then [] else vertexRun A fuel σ

/-- The vertices the iterator yields for a dense interval ambiguity set on `Fin d`, in iteration
order: `vertices(p) = map(copy, vertex_generator(p))`.

Julia counterpart: `vertices(p::IntervalAmbiguitySet)` and `vertex_generator(p)`
(`src/probabilities/IntervalAmbiguitySets.jl`), as iterated in
`state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`). -/
noncomputable def vertices {d : ℕ} (A : IntervalAmbiguity (Fin d)) : List (Fin d → ℝ) :=
  vertexRun A d.factorial (List.range' 1 d)

/-- Every vector the iterator yields is the vertex of some permutation vector of `1:d`.

Julia counterpart: `Base.iterate(::IntervalAmbiguitySetVertexIterator[, state])`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem mem_vertexRun {d : ℕ} (A : IntervalAmbiguity (Fin d)) :
    ∀ (fuel : ℕ) (perm : List ℕ), perm.Perm (List.range' 1 d) → ∀ v ∈ vertexRun A fuel perm,
      ∃ π, π.Perm (List.range' 1 d) ∧ (vertexOf A π).1 = v
  | 0, _, _, v, hv => by simp [vertexRun] at hv
  | fuel + 1, perm, hp, v, hv => by
    rw [vertexRun, List.mem_cons] at hv
    rcases hv with rfl | hv
    · exact ⟨perm, hp, rfl⟩
    · cases hn : nextPermutation perm (vertexOf A perm).2 with
      | none => rw [hn] at hv; simp at hv
      | some σ =>
        rw [hn] at hv
        dsimp only at hv
        split_ifs at hv with hb
        · simp at hv
        · exact mem_vertexRun A fuel σ ((nextPermutation_some hn).1.trans hp) v hv

open Classical in
/-- The number of permutation vectors of `1:d` that are lexicographically at least `perm`; it
bounds the number of iterations left.

Julia counterpart: none (Lean-side proof device). -/
noncomputable def permsFrom (d : ℕ) (perm : List ℕ) : ℕ :=
  ((List.range' 1 d).permutations.toFinset.filter (fun π => perm < π ∨ perm = π)).card

/-- A permutation vector counts itself.

Julia counterpart: none (Lean-side proof device). -/
theorem permsFrom_pos {d : ℕ} {perm : List ℕ} (h : perm.Perm (List.range' 1 d)) :
    0 < permsFrom d perm := by
  classical
  unfold permsFrom
  refine Finset.card_pos.2 ⟨perm, ?_⟩
  simp only [Finset.mem_filter, List.mem_toFinset, List.mem_permutations]
  exact ⟨h, Or.inr trivial⟩

/-- There are at most `d!` permutation vectors.

Julia counterpart: none (Lean-side proof device). -/
theorem permsFrom_le (d : ℕ) (perm : List ℕ) : permsFrom d perm ≤ d.factorial := by
  classical
  unfold permsFrom
  refine (Finset.card_filter_le _ _).trans ((List.toFinset_card_le _).trans ?_)
  rw [List.length_permutations, List.length_range']

/-- Moving to a lexicographically larger permutation vector decreases `permsFrom`.

Julia counterpart: none (Lean-side proof device). -/
theorem permsFrom_lt {d : ℕ} {ρ σ : List ℕ} (hρ : ρ.Perm (List.range' 1 d)) (h : ρ < σ) :
    permsFrom d σ < permsFrom d ρ := by
  classical
  unfold permsFrom
  apply Finset.card_lt_card
  rw [Finset.ssubset_iff_of_subset]
  · refine ⟨ρ, ?_, ?_⟩
    · simp only [Finset.mem_filter, List.mem_toFinset, List.mem_permutations]
      exact ⟨hρ, Or.inr trivial⟩
    · simp only [Finset.mem_filter, not_and, not_or]
      intro _
      refine ⟨fun h' => List.lt_irrefl ρ (List.lt_trans h h'), fun h' => ?_⟩
      subst h'
      exact List.lt_irrefl _ h
  · intro x hx
    simp only [Finset.mem_filter] at hx ⊢
    refine ⟨hx.1, Or.inl ?_⟩
    rcases hx.2 with h' | h'
    · exact List.lt_trans h h'
    · exact h' ▸ h

/-- `1:d` is lexicographically at most every permutation vector of `1:d`.

Julia counterpart: `permutation = collect(1:length(support(it.set)))` in
`Base.iterate(it::IntervalAmbiguitySetVertexIterator)` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem range_le_of_perm {d : ℕ} {π : List ℕ} (h : π.Perm (List.range' 1 d)) :
    List.range' 1 d < π ∨ List.range' 1 d = π :=
  lt_or_eq_of_sorted_perm ((List.pairwise_lt_range' 1).imp le_of_lt) h.symm

/-- **Completeness of the iterator.** Started at `ρ` with enough fuel, the iteration yields the
vertex of every permutation vector `π ≥ ρ` (lexicographically).

Julia counterpart: `Base.iterate(::IntervalAmbiguitySetVertexIterator[, state])`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem vertexOf_mem_vertexRun {d : ℕ} (A : IntervalAmbiguity (Fin d)) :
    ∀ (fuel : ℕ) (ρ : List ℕ), ρ.Perm (List.range' 1 d) → permsFrom d ρ ≤ fuel →
      ∀ π, π.Perm (List.range' 1 d) → (ρ < π ∨ ρ = π) → (vertexOf A π).1 ∈ vertexRun A fuel ρ
  | 0, ρ, hρ, hfuel, _, _, _ => absurd hfuel (by have := permsFrom_pos hρ; omega)
  | fuel + 1, ρ, hρ, hfuel, π, hπ, hle => by
    rw [vertexRun]
    have hb := vertexOf_snd_ne_zero A hρ
    by_cases ht : π.take (vertexOf A ρ).2 = ρ.take (vertexOf A ρ).2
    · have he : vertexOf A π = vertexOf A ρ :=
        vertexLoop_prefix A.gap (1 - ∑ t, A.lower t) A.lower hb ht
      rw [he]
      exact List.mem_cons_self ..
    · have hlt : ρ < π := hle.resolve_right (fun h => ht (h ▸ rfl))
      obtain ⟨σ, hσ, hσπ⟩ := nextPermutation_spec (hπ.trans hρ.symm) hlt ht
      obtain ⟨hσp, hρσ⟩ := nextPermutation_some hσ
      rw [hσ]
      dsimp only
      by_cases hb0 : 1 - ∑ t, A.lower t = 0
      · rw [vertexOf_fst_of_budget_eq_zero A hπ hb0, ← vertexOf_fst_of_budget_eq_zero A hρ hb0]
        exact List.mem_cons_self ..
      · rw [if_neg hb0]
        apply List.mem_cons_of_mem
        exact vertexOf_mem_vertexRun A fuel σ (hσp.trans hρ)
          (by have := permsFrom_lt hρ hρσ; omega) π hπ hσπ

/-- **The marginal iterator enumerates exactly the greedy vertices.** A vector is yielded by the
iterator iff it is the vertex `lower + allocation` of some permutation vector of `1:d`, although
the iterator skips permutations.

Julia counterpart: `vertices(p::IntervalAmbiguitySet)` / `vertex_generator(p)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem mem_vertices_iff {d : ℕ} (A : IntervalAmbiguity (Fin d)) (v : Fin d → ℝ) :
    v ∈ vertices A ↔ ∃ π, π.Perm (List.range' 1 d) ∧ (vertexOf A π).1 = v := by
  constructor
  · exact mem_vertexRun A _ _ (List.Perm.refl _) v
  · rintro ⟨π, hπ, rfl⟩
    exact vertexOf_mem_vertexRun A _ _ (List.Perm.refl _) (permsFrom_le d _) π hπ
      (range_le_of_perm hπ)

/-! ### The marginal vertices are the extreme points of `P(l, u)` -/

/-- Of two distinct indices, at least one gets allocation `0` or its full gap: the greedy loop
leaves at most one coordinate strictly between its bounds.

Julia counterpart: none (Lean-side proof device). -/
theorem allocation_boundary {d : ℕ} (gap : Fin d → ℝ) (hg : ∀ t, 0 ≤ gap t) :
    ∀ (perm : List ℕ) (b : ℝ) (k k' : ℕ), k ≠ k' →
      (allocation gap perm b k = 0 ∨ allocation gap perm b k = juliaGet gap k) ∨
        (allocation gap perm b k' = 0 ∨ allocation gap perm b k' = juliaGet gap k')
  | [], _, _, _, _ => Or.inl (Or.inl rfl)
  | i :: perm, b, k, k', hkk => by
    simp only [allocation]
    by_cases hb : b ≤ juliaGet gap i
    · rw [min_eq_left hb, sub_self]
      by_cases hk : i = k
      · right; left; rw [if_neg (by omega), allocation_zero hg]
      · left; left; rw [if_neg hk, allocation_zero hg]
    · rw [min_eq_right (le_of_not_ge hb)]
      by_cases hk : i = k
      · left; right; rw [if_pos hk, hk]
      · by_cases hk' : i = k'
        · right; right; rw [if_pos hk', hk']
        · rw [if_neg hk, if_neg hk']
          exact allocation_boundary gap hg perm _ k k' hkk

/-- The vertex of a permutation vector of `1:d` lies in `P(l, u)`.

Julia counterpart: the vector `v` returned by `Base.iterate(::IntervalAmbiguitySetVertexIterator,
…)` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem vertexOf_mem_vecs {d : ℕ} (A : IntervalAmbiguity (Fin d)) {perm : List ℕ}
    (h : perm.Perm (List.range' 1 d)) : (vertexOf A perm).1 ∈ A.toSet.vecs := by
  rw [A.vecs_toSet]
  have hnd : perm.Nodup := h.nodup_iff.2 List.nodup_range'
  have h0 := fun t : Fin d => allocation_nonneg A.gap_nonneg perm A.budget (toJulia t)
    A.budget_eq.2
  have h1 := fun t : Fin d => allocation_le_gap A.gap_nonneg perm A.budget (toJulia t)
  have hsum : ∑ t : Fin d, allocation A.gap perm A.budget (toJulia t) = A.budget := by
    have hs := sum_allocation A.gap_nonneg perm A.budget A.budget_eq.2 hnd
    rw [sum_map_of_perm h, sum_map_juliaGet h] at hs
    rw [hs, min_eq_left A.budget_le_sum_gap]
  refine ⟨⟨fun t => ?_, ?_⟩, fun t => ?_, fun t => ?_⟩
  · rw [vertexOf_fst A h]; linarith [A.lower_nonneg t, h0 t]
  · rw [Finset.sum_congr rfl (fun t _ => vertexOf_fst A h t), Finset.sum_add_distrib, hsum,
      IntervalAmbiguity.budget]
    ring
  · rw [vertexOf_fst A h]; linarith [h0 t]
  · have := h1 t
    rw [juliaGet_toJulia] at this
    rw [vertexOf_fst A h]; linarith [A.lower_add_gap t]

/-- A point of `P(l, u)` with at most one coordinate strictly between its bounds is an extreme
point of `P(l, u)`.

Julia counterpart: none (Lean-side proof device). -/
theorem mem_extremePoints_of_boundary {d : ℕ} (A : IntervalAmbiguity (Fin d)) {v : Fin d → ℝ}
    (hv : v ∈ A.toSet.vecs)
    (hbd : ∀ t t', t ≠ t' → (v t = A.lower t ∨ v t = A.upper t) ∨
      (v t' = A.lower t' ∨ v t' = A.upper t')) :
    v ∈ Set.extremePoints ℝ A.toSet.vecs := by
  rw [mem_extremePoints]
  refine ⟨hv, ?_⟩
  rw [A.vecs_toSet] at hv ⊢
  have key : ∀ x y : Fin d → ℝ, x ∈ stdSimplex ℝ (Fin d) ∩ Set.Icc A.lower A.upper →
      y ∈ stdSimplex ℝ (Fin d) ∩ Set.Icc A.lower A.upper → ∀ a b : ℝ, 0 < a → 0 < b →
      a + b = 1 → a • x + b • y = v → x = v := by
    intro x y hx hy a b ha hb hab hxy
    have hc : ∀ t, v t = a * x t + b * y t := fun t => by
      rw [← hxy]; simp [smul_eq_mul]
    have hbnd : ∀ t, (v t = A.lower t ∨ v t = A.upper t) → x t = v t := by
      intro t ht
      have hxl := hx.2.1 t
      have hxu := hx.2.2 t
      have hyl := hy.2.1 t
      have hyu := hy.2.2 t
      have hct := hc t
      rcases ht with ht | ht
      · have e1 : 0 ≤ a * (x t - A.lower t) := mul_nonneg ha.le (by linarith)
        have e2 : 0 ≤ b * (y t - A.lower t) := mul_nonneg hb.le (by linarith)
        have e0 : a * (x t - A.lower t) + b * (y t - A.lower t) = 0 := by
          linear_combination -hct + ht - A.lower t * hab
        have e3 : a * (x t - A.lower t) = 0 := by linarith
        rcases mul_eq_zero.1 e3 with e | e
        · exact absurd e ha.ne'
        · linarith
      · have e1 : 0 ≤ a * (A.upper t - x t) := mul_nonneg ha.le (by linarith)
        have e2 : 0 ≤ b * (A.upper t - y t) := mul_nonneg hb.le (by linarith)
        have e0 : a * (A.upper t - x t) + b * (A.upper t - y t) = 0 := by
          linear_combination hct - ht + A.upper t * hab
        have e3 : a * (A.upper t - x t) = 0 := by linarith
        rcases mul_eq_zero.1 e3 with e | e
        · exact absurd e ha.ne'
        · linarith
    funext t
    by_cases ht : v t = A.lower t ∨ v t = A.upper t
    · exact hbnd t ht
    · have hother : ∀ t' ∈ Finset.univ.erase t, x t' = v t' := fun t' ht' =>
        hbnd t' ((hbd t t' (Ne.symm (Finset.ne_of_mem_erase ht'))).resolve_left ht)
      have hxs := hx.1.2
      have hvs := hv.1.2
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ t)] at hxs hvs
      rw [Finset.sum_congr rfl hother] at hxs
      linarith
  intro x₁ hx₁ x₂ hx₂ hseg
  obtain ⟨a, b, ha, hb, hab, hx⟩ := hseg
  refine ⟨key x₁ x₂ hx₁ hx₂ a b ha hb hab hx, key x₂ x₁ hx₂ hx₁ b a hb ha (by linarith) ?_⟩
  rw [add_comm]; exact hx

/-- The vertex of a permutation vector of `1:d` is an extreme point of `P(l, u)`.

Julia counterpart: the vector `v` returned by `Base.iterate(::IntervalAmbiguitySetVertexIterator,
…)` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem vertexOf_mem_extremePoints {d : ℕ} (A : IntervalAmbiguity (Fin d)) {perm : List ℕ}
    (h : perm.Perm (List.range' 1 d)) :
    (vertexOf A perm).1 ∈ Set.extremePoints ℝ A.toSet.vecs := by
  refine mem_extremePoints_of_boundary A (vertexOf_mem_vecs A h) fun t t' htt => ?_
  have hne : toJulia t ≠ toJulia t' := fun e => htt (toJulia_injective e)
  have hg := fun s : Fin d => (juliaGet_toJulia A.gap s)
  rcases allocation_boundary A.gap A.gap_nonneg perm A.budget _ _ hne with h' | h'
  · left
    rw [vertexOf_fst A h]
    rcases h' with h' | h'
    · left; rw [h', add_zero]
    · right; rw [h', hg, A.lower_add_gap]
  · right
    rw [vertexOf_fst A h]
    rcases h' with h' | h'
    · left; rw [h', add_zero]
    · right; rw [h', hg, A.lower_add_gap]

/-- At an extreme point of `P(l, u)` at most one coordinate lies strictly between its bounds
(otherwise moving mass between two such coordinates stays in `P(l, u)` in both directions).

Julia counterpart: none (Lean-side proof device). -/
theorem boundary_of_mem_extremePoints {d : ℕ} (A : IntervalAmbiguity (Fin d)) {v : Fin d → ℝ}
    (hv : v ∈ Set.extremePoints ℝ A.toSet.vecs) :
    ∀ t t', t ≠ t' → (v t = A.lower t ∨ v t = A.upper t) ∨
      (v t' = A.lower t' ∨ v t' = A.upper t') := by
  intro t t' htt
  by_contra hcon
  simp only [not_or] at hcon
  obtain ⟨⟨hl, hu⟩, ⟨hl', hu'⟩⟩ := hcon
  rw [mem_extremePoints, A.vecs_toSet] at hv
  obtain ⟨⟨⟨hnn, hs⟩, hlo, hup⟩, hext⟩ := hv
  have hlt : A.lower t < v t := lt_of_le_of_ne (hlo t) (Ne.symm hl)
  have hut : v t < A.upper t := lt_of_le_of_ne (hup t) hu
  have hlt' : A.lower t' < v t' := lt_of_le_of_ne (hlo t') (Ne.symm hl')
  have hut' : v t' < A.upper t' := lt_of_le_of_ne (hup t') hu'
  set ε := min (min (v t - A.lower t) (A.upper t - v t))
    (min (v t' - A.lower t') (A.upper t' - v t')) with hε
  have hεpos : 0 < ε := by
    simp only [hε, lt_min_iff]; refine ⟨⟨?_, ?_⟩, ?_, ?_⟩ <;> linarith
  have hε1 : ε ≤ v t - A.lower t := (min_le_left _ _).trans (min_le_left _ _)
  have hε2 : ε ≤ A.upper t - v t := (min_le_left _ _).trans (min_le_right _ _)
  have hε3 : ε ≤ v t' - A.lower t' := (min_le_right _ _).trans (min_le_left _ _)
  have hε4 : ε ≤ A.upper t' - v t' := (min_le_right _ _).trans (min_le_right _ _)
  have hmem : ∀ c : ℝ, (c = ε ∨ c = -ε) →
      (fun s => v s + (if s = t then c else 0) - (if s = t' then c else 0)) ∈
        stdSimplex ℝ (Fin d) ∩ Set.Icc A.lower A.upper := by
    intro c hc
    have hbound : ∀ s, A.lower s ≤ v s + (if s = t then c else 0) - (if s = t' then c else 0) ∧
        v s + (if s = t then c else 0) - (if s = t' then c else 0) ≤ A.upper s := by
      intro s
      by_cases h1 : s = t
      · subst h1
        rw [if_pos rfl, if_neg htt]
        rcases hc with rfl | rfl <;> constructor <;> linarith
      · by_cases h2 : s = t'
        · subst h2
          rw [if_neg h1, if_pos rfl]
          rcases hc with rfl | rfl <;> constructor <;> linarith
        · rw [if_neg h1, if_neg h2]
          exact ⟨by linarith [hlo s], by linarith [hup s]⟩
    refine ⟨⟨fun s => le_trans (A.lower_nonneg s) (hbound s).1, ?_⟩,
      fun s => (hbound s).1, fun s => (hbound s).2⟩
    rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.sum_ite_eq',
      Finset.sum_ite_eq', if_pos (Finset.mem_univ _), if_pos (Finset.mem_univ _), hs]
    ring
  have hseg : v ∈ openSegment ℝ
      (fun s => v s + (if s = t then ε else 0) - (if s = t' then ε else 0))
      (fun s => v s + (if s = t then -ε else 0) - (if s = t' then -ε else 0)) := by
    refine ⟨1 / 2, 1 / 2, by norm_num, by norm_num, by norm_num, ?_⟩
    funext s
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    split_ifs <;> ring
  have := (hext _ (hmem ε (Or.inl rfl)) _ (hmem (-ε) (Or.inr rfl)) hseg).1
  have ht := congrFun this t
  simp only at ht
  rw [if_pos trivial, if_neg htt, sub_zero] at ht
  linarith

/-- Greedy allocation along a list whose first part `F` gets the full gap: the allocations of
`F ++ R` are `w` on `F` if `R` alone allocates `w` from its total.

Julia counterpart: none (Lean-side proof device). -/
theorem allocation_full_prefix {d : ℕ} (gap : Fin d → ℝ) (hg : ∀ t, 0 ≤ gap t) (w : ℕ → ℝ)
    (R : List ℕ) (hR0 : ∀ k ∈ R, 0 ≤ w k)
    (hR : ∀ k ∈ R, allocation gap R (R.map w).sum k = w k) :
    ∀ (F : List ℕ) (b : ℝ), (∀ k ∈ F, w k = juliaGet gap k) →
      b = (F.map w).sum + (R.map w).sum → ∀ k ∈ F ++ R, allocation gap (F ++ R) b k = w k
  | [], b, _, hb, k, hk => by
    simp only [List.map_nil, List.sum_nil, zero_add] at hb
    subst hb
    exact hR k hk
  | f :: F, b, hF, hb, k, hk => by
    have hFf : ∀ k ∈ F, w k = juliaGet gap k := fun k hk => hF k (List.mem_cons_of_mem _ hk)
    have hsF : 0 ≤ (F.map w).sum := List.sum_nonneg (by
      simp only [List.mem_map]
      rintro _ ⟨k, hk, rfl⟩
      rw [hFf k hk]; exact juliaGet_nonneg hg k)
    have hsR : 0 ≤ (R.map w).sum := List.sum_nonneg (by
      simp only [List.mem_map]
      rintro _ ⟨k, hk, rfl⟩
      exact hR0 k hk)
    have hwf := hF f (List.mem_cons_self ..)
    rw [List.map_cons, List.sum_cons] at hb
    have hmin : min b (juliaGet gap f) = juliaGet gap f := min_eq_right (by linarith)
    rw [List.cons_append]
    simp only [allocation, hmin]
    by_cases hfk : f = k
    · rw [if_pos hfk, ← hfk, hwf]
    · rw [if_neg hfk]
      rcases List.mem_cons.1 (List.cons_append ▸ hk) with h | h
      · exact absurd h.symm hfk
      · exact allocation_full_prefix gap hg w R hR0 hR F _ hFf (by linarith) k h

/-- Every Julia index of `1:d` is `toJulia` of a Lean index.

Julia counterpart: none (Lean-side proof device). -/
theorem exists_toJulia_of_mem_range {d k : ℕ} (hk : k ∈ List.range' 1 d) :
    ∃ t : Fin d, toJulia t = k := by
  rw [List.mem_range'_1] at hk
  exact ⟨⟨k - 1, by omega⟩, by simp only [toJulia]; omega⟩

/-- Every point of `P(l, u)` with at most one coordinate strictly between its bounds is the vertex
of some permutation vector of `1:d`: list the coordinates at their upper bound first, then the
interior one, then those at their lower bound.

Julia counterpart: the vertices produced by `Base.iterate(::IntervalAmbiguitySetVertexIterator,
…)` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem exists_perm_vertexOf_eq {d : ℕ} (A : IntervalAmbiguity (Fin d)) {v : Fin d → ℝ}
    (hv : v ∈ A.toSet.vecs)
    (hbd : ∀ t t', t ≠ t' → (v t = A.lower t ∨ v t = A.upper t) ∨
      (v t' = A.lower t' ∨ v t' = A.upper t')) :
    ∃ π, π.Perm (List.range' 1 d) ∧ (vertexOf A π).1 = v := by
  classical
  rw [A.vecs_toSet] at hv
  obtain ⟨⟨_, hs⟩, hlo, hup⟩ := hv
  set w : ℕ → ℝ := juliaGet (v - A.lower) with hw
  set J := List.range' 1 d
  set full : ℕ → Bool := fun k => decide (juliaGet v k = juliaGet A.upper k)
  set inner : ℕ → Bool := fun k => decide (juliaGet A.lower k < juliaGet v k)
  set F := J.filter full
  set R := J.filter (fun k => !full k)
  set M := R.filter inner
  set Z := R.filter (fun k => !inner k)
  have hRp : (M ++ Z).Perm R := List.filter_append_perm inner R
  have hπ : (F ++ (M ++ Z)).Perm J :=
    ((List.Perm.append_left F hRp)).trans (List.filter_append_perm full J)
  have hJ : ∀ k ∈ J, ∃ t : Fin d, toJulia t = k := fun k hk => exists_toJulia_of_mem_range hk
  have hwt : ∀ t : Fin d, w (toJulia t) = v t - A.lower t := fun t => by
    simp only [hw, juliaGet_toJulia, Pi.sub_apply]
  have hRJ : ∀ k ∈ R, k ∈ J := fun k hk => List.mem_of_mem_filter hk
  have hMR : ∀ k ∈ M, k ∈ R := fun k hk => List.mem_of_mem_filter hk
  have hZR : ∀ k ∈ Z, k ∈ R := fun k hk => List.mem_of_mem_filter hk
  have hw0 : ∀ k ∈ J, 0 ≤ w k := fun k hk => by
    obtain ⟨t, rfl⟩ := hJ k hk; rw [hwt]; linarith [hlo t]
  have hwF : ∀ k ∈ F, w k = juliaGet A.gap k := fun k hk => by
    obtain ⟨t, rfl⟩ := hJ k (List.mem_of_mem_filter hk)
    have h := (List.mem_filter.1 hk).2
    simp only [full, decide_eq_true_eq, juliaGet_toJulia] at h
    rw [hwt, juliaGet_toJulia, h, ← A.lower_add_gap]; ring
  have hwZ : ∀ k ∈ Z, w k = 0 := fun k hk => by
    obtain ⟨t, rfl⟩ := hJ k (hRJ k (hZR k hk))
    have h := (List.mem_filter.1 hk).2
    simp only [inner, Bool.not_eq_true', decide_eq_false_iff_not, not_lt,
      juliaGet_toJulia] at h
    rw [hwt]; linarith [hlo t]
  have hint : ∀ k ∈ M, ∃ t : Fin d, toJulia t = k ∧ ¬ (v t = A.lower t ∨ v t = A.upper t) :=
    fun k hk => by
      obtain ⟨t, rfl⟩ := hJ k (hRJ k (hMR k hk))
      refine ⟨t, rfl, ?_⟩
      have h1 := (List.mem_filter.1 hk).2
      have h2 := (List.mem_filter.1 (hMR _ hk)).2
      simp only [inner, full, decide_eq_true_eq, Bool.not_eq_true', decide_eq_false_iff_not,
        juliaGet_toJulia] at h1 h2
      rintro (h | h)
      · linarith
      · exact h2 h
  have hMnd : M.Nodup :=
    ((List.nodup_range').filter _).filter _
  have hRalloc : ∀ k ∈ M ++ Z, allocation A.gap (M ++ Z) ((M ++ Z).map w).sum k = w k := by
    match hM : M, hMnd, hint with
    | [], _, _ =>
      intro k hk
      have hz : (Z.map w).sum = 0 := List.sum_eq_zero (by
        simp only [List.mem_map]
        rintro _ ⟨k, hk, rfl⟩
        exact hwZ k hk)
      simp only [List.nil_append, hz]
      rw [allocation_zero A.gap_nonneg, hwZ k hk]
    | [m], _, hint' =>
      intro k hk
      have hz : (Z.map w).sum = 0 := List.sum_eq_zero (by
        simp only [List.mem_map]
        rintro _ ⟨k, hk, rfl⟩
        exact hwZ k hk)
      obtain ⟨t, rfl, -⟩ := hint' m (List.mem_singleton_self _)
      have hmle : w (toJulia t) ≤ juliaGet A.gap (toJulia t) := by
        rw [hwt, juliaGet_toJulia]; linarith [hup t, A.lower_add_gap t]
      simp only [List.singleton_append, List.map_cons, List.sum_cons, hz, add_zero, allocation,
        min_eq_left hmle, sub_self]
      by_cases hkt : toJulia t = k
      · rw [if_pos hkt, hkt]
      · rw [if_neg hkt, allocation_zero A.gap_nonneg]
        rcases List.mem_cons.1 hk with h | h
        · exact absurd h.symm hkt
        · exact (hwZ k h).symm
    | m₁ :: m₂ :: _, hnd, hint' =>
      obtain ⟨t₁, rfl, h₁⟩ := hint' m₁ (List.mem_cons_self ..)
      obtain ⟨t₂, rfl, h₂⟩ := hint' m₂ (List.mem_cons_of_mem _ (List.mem_cons_self ..))
      have hne : t₁ ≠ t₂ := by
        rintro rfl
        exact (List.nodup_cons.1 hnd).1 (List.mem_cons_self ..)
      exact absurd (hbd t₁ t₂ hne) (by tauto)
  have hRw0 : ∀ k ∈ M ++ Z, 0 ≤ w k := fun k hk => hw0 k (hRJ k (hRp.subset hk))
  have hbud : A.budget = (F.map w).sum + ((M ++ Z).map w).sum := by
    rw [← List.sum_append, ← List.map_append, sum_map_of_perm hπ]
    simp only [hwt, Finset.sum_sub_distrib, hs, IntervalAmbiguity.budget]
  refine ⟨F ++ (M ++ Z), hπ, funext fun t => ?_⟩
  have hmemt : toJulia t ∈ F ++ (M ++ Z) := by
    rw [hπ.mem_iff]
    show toJulia t ∈ List.range' 1 d
    rw [← map_toJulia_finRange]
    exact List.mem_map_of_mem (List.mem_finRange t)
  rw [vertexOf_fst A hπ, allocation_full_prefix A.gap A.gap_nonneg w (M ++ Z) hRw0 hRalloc F
    A.budget hwF hbud (toJulia t) hmemt, hwt]
  ring

/-- **Marginal vertex enumeration is complete and exact.** The iterator for a dense interval
ambiguity set yields exactly the extreme points (vertices) of `P(l, u)`.

Julia counterpart: `vertices(p::IntervalAmbiguitySet)` / `vertex_generator(p)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem mem_vertices_iff_extremePoints {d : ℕ} (A : IntervalAmbiguity (Fin d)) (v : Fin d → ℝ) :
    v ∈ vertices A ↔ v ∈ Set.extremePoints ℝ A.toSet.vecs := by
  rw [mem_vertices_iff]
  constructor
  · rintro ⟨π, hπ, rfl⟩
    exact vertexOf_mem_extremePoints A hπ
  · intro hv
    exact exists_perm_vertexOf_eq A hv.1
      (boundary_of_mem_extremePoints A hv)

/-! ### The product enumeration -/

/-- Literal transcription of `Iterators.product(L₁, …, L_k)`: all tuples `(x₁, …, x_k)` with
`xᵣ ∈ Lᵣ`, the first component varying fastest.

Julia counterpart: `Iterators.product(iterators...)` (Base) in
`state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`). -/
def productList : {k : ℕ} → {X : Fin k → Type} → ((r : Fin k) → List (X r)) →
    List ((r : Fin k) → X r)
  | 0, _, _ => [fun r => r.elim0]
  | _ + 1, _, L =>
    (productList (fun r => L r.succ)).flatMap (fun rest => (L 0).map (fun x => Fin.cons x rest))

/-- `Iterators.product` visits exactly the tuples whose components lie in the factor lists.

Julia counterpart: `Iterators.product(iterators...)` (Base) in
`state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`). -/
theorem mem_productList : ∀ {k : ℕ} {X : Fin k → Type} (L : (r : Fin k) → List (X r))
    (q : (r : Fin k) → X r), q ∈ productList L ↔ ∀ r, q r ∈ L r
  | 0, _, _, q => by
    simp only [productList, List.mem_singleton]
    exact ⟨fun _ r => r.elim0, fun _ => funext fun r => r.elim0⟩
  | _ + 1, _, L, q => by
    simp only [productList, List.mem_flatMap, List.mem_map]
    constructor
    · rintro ⟨rest, hrest, x, hx, rfl⟩ r
      refine Fin.cases ?_ (fun r' => ?_) r
      · simpa using hx
      · simpa using (mem_productList _ rest).1 hrest r'
    · intro h
      exact ⟨Fin.tail q, (mem_productList _ _).2 (fun r => h r.succ), q 0, h 0,
        Fin.cons_self_tail q⟩

/-- The tuples of marginal vertices that vertex enumeration visits for `(s, a)`:
`Iterators.product(vertex_generator.(ambiguity_sets, workspace.result_vectors)...)`.

Julia counterpart: `for marginal_vertices in Iterators.product(iterators...)` with
`iterators = vertex_generator.(ambiguity_sets, workspace.result_vectors)` in
`state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`). -/
noncomputable def vertexEnumeration (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) : List ((r : Fin n) → Fin (F.stateVars.dims r) → ℝ) :=
  productList (fun r => vertices (ambiguitySets F s a r))

/-- **Vertex enumeration is complete.** The loop
`for marginal_vertices in Iterators.product(iterators...)` of
`state_action_bellman(::FactoredVertexIteratorWorkspace, …)` visits a tuple `q` iff every
component `q r` is a vertex (extreme point) of the marginal interval set `P(lʳ, uʳ)` of `(s, a)`.
So it enumerates exactly the products of marginal vertices. No hypotheses beyond the model
invariants (the iterator's permutation skipping is modelled). Dense marginal storage only (sparse
marginals not modelled, O16).

Julia counterpart: `state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`)
with `IntervalAmbiguitySetVertexIterator` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem vertices_complete (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) (q : (r : Fin n) → Fin (F.stateVars.dims r) → ℝ) :
    q ∈ vertexEnumeration F s a ↔ ∀ r, q r ∈ Set.extremePoints ℝ (F.marginalSet s a r).vecs := by
  rw [vertexEnumeration, mem_productList]
  exact forall_congr' fun r => mem_vertices_iff_extremePoints _ _

/-! ### The vertex-enumeration value -/

/-- The value `v` of one tuple of marginal vertices:
`sum(V[I] * prod(r -> marginal_vertices[r][I[r]], eachindex(ambiguity_sets)) for I in
CartesianIndices(num_target.(ambiguity_sets)))`, with `V[I] = W (successorState I)`
(`Index.factored_successor_eq`).

Julia counterpart: the `sum(…)` in `state_action_bellman(::FactoredVertexIteratorWorkspace, …)`
(`src/bellman.jl`). -/
noncomputable def vertexSum (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) (W : F.stateVars.State → ℝ)
    (mv : (r : Fin n) → Fin (F.stateVars.dims r) → ℝ) : ℝ :=
  ∑ I : CartesianIndex (numTargets F s a), W (successorState F s a I) * ∏ r, mv r (I r)

/-- One loop iteration `optval = optfunc(optval, v)` with `optfunc = upper_bound ? max : min`.

Julia counterpart: the body of `for marginal_vertices in Iterators.product(iterators...)` in
`state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`). -/
noncomputable def optStep (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) (W : F.stateVars.State → ℝ) (upperBound : Bool) (optval : EReal)
    (mv : (r : Fin n) → Fin (F.stateVars.dims r) → ℝ) : EReal :=
  if upperBound then max optval (vertexSum F s a W mv) else min optval (vertexSum F s a W mv)

/-- Literal transcription of `state_action_bellman(workspace::FactoredVertexIteratorWorkspace, V,
ambiguity_sets, upper_bound)`:

    optval = upper_bound ? typemin(R) : typemax(R)
    optfunc = upper_bound ? max : min
    for marginal_vertices in Iterators.product(iterators...)
        v = sum(V[I] * prod(r -> marginal_vertices[r][I[r]], …) for I in CartesianIndices(…))
        optval = optfunc(optval, v)
    end

over `EReal` (`typemin = ⊥`, `typemax = ⊤`), with the loop order of `Iterators.product`
(`vertexEnumeration`) kept. `W` is the value function stored in `V`.

Julia counterpart: `state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`). -/
noncomputable def vertexValue (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) (W : F.stateVars.State → ℝ) (upperBound : Bool) : EReal :=
  (vertexEnumeration F s a).foldl (optStep F s a W upperBound) (if upperBound then ⊥ else ⊤)

/-- `optval = max(optval, f(x))`.

Julia counterpart: `optfunc(optval, v)` with `optfunc = max` (`src/bellman.jl`). -/
noncomputable def maxStep {α : Type*} (f : α → ℝ) (o : EReal) (x : α) : EReal := max o (f x)

/-- `optval = min(optval, f(x))`.

Julia counterpart: `optfunc(optval, v)` with `optfunc = min` (`src/bellman.jl`). -/
noncomputable def minStep {α : Type*} (f : α → ℝ) (o : EReal) (x : α) : EReal := min o (f x)

/-- A `max` fold returns its seed or one of the values, and bounds the seed and all values.

Julia counterpart: none (Lean-side proof device). -/
theorem foldl_maxStep {α : Type*} (f : α → ℝ) : ∀ (L : List α) (acc : EReal),
    (L.foldl (maxStep f) acc = acc ∨ ∃ x ∈ L, L.foldl (maxStep f) acc = f x) ∧
      acc ≤ L.foldl (maxStep f) acc ∧ ∀ x ∈ L, (f x : EReal) ≤ L.foldl (maxStep f) acc
  | [], acc => ⟨Or.inl rfl, le_rfl, by simp⟩
  | x :: L, acc => by
    obtain ⟨h1, h2, h3⟩ := foldl_maxStep f L (maxStep f acc x)
    rw [List.foldl_cons]
    refine ⟨?_, le_trans (le_max_left _ _) h2, fun y hy => ?_⟩
    · rcases h1 with h | ⟨y, hy, h⟩
      · rw [h, maxStep]
        rcases max_choice acc (f x : EReal) with e | e
        · exact Or.inl e
        · exact Or.inr ⟨x, List.mem_cons_self .., e⟩
      · exact Or.inr ⟨y, List.mem_cons_of_mem _ hy, h⟩
    · rcases List.mem_cons.1 hy with rfl | hy
      · exact le_trans (le_max_right _ _) h2
      · exact h3 y hy

/-- A `min` fold returns its seed or one of the values, and is below the seed and all values.

Julia counterpart: none (Lean-side proof device). -/
theorem foldl_minStep {α : Type*} (f : α → ℝ) : ∀ (L : List α) (acc : EReal),
    (L.foldl (minStep f) acc = acc ∨ ∃ x ∈ L, L.foldl (minStep f) acc = f x) ∧
      L.foldl (minStep f) acc ≤ acc ∧ ∀ x ∈ L, L.foldl (minStep f) acc ≤ (f x : EReal)
  | [], acc => ⟨Or.inl rfl, le_rfl, by simp⟩
  | x :: L, acc => by
    obtain ⟨h1, h2, h3⟩ := foldl_minStep f L (minStep f acc x)
    rw [List.foldl_cons]
    refine ⟨?_, le_trans h2 (min_le_left _ _), fun y hy => ?_⟩
    · rcases h1 with h | ⟨y, hy, h⟩
      · rw [h, minStep]
        rcases min_choice acc (f x : EReal) with e | e
        · exact Or.inl e
        · exact Or.inr ⟨x, List.mem_cons_self .., e⟩
      · exact Or.inr ⟨y, List.mem_cons_of_mem _ hy, h⟩
    · rcases List.mem_cons.1 hy with rfl | hy
      · exact le_trans h2 (min_le_right _ _)
      · exact h3 y hy

/-- The sum of `vertexSum` is the expectation `⟨⊗ᵣ mv r, W⟩` of `W` under the product of the
marginal vectors (`AmbiguitySet.piVec`).

Julia counterpart: the `sum(…)` in `state_action_bellman(::FactoredVertexIteratorWorkspace, …)`
(`src/bellman.jl`). -/
theorem vertexSum_eq_dot (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) (W : F.stateVars.State → ℝ)
    (mv : (r : Fin n) → Fin (F.stateVars.dims r) → ℝ) :
    vertexSum F s a W mv = dot W (AmbiguitySet.piVec mv) :=
  Fintype.sum_bijective (successorState F s a) (successorState_bijective F s a) _ _
    (fun _ => rfl)

/-- A dense interval set `P(l, u)` is the convex hull of its finitely many iterator vertices
(Krein–Milman, `closure_convexHull_extremePoints`, plus `mem_vertices_iff_extremePoints`).

Julia counterpart: none (Lean-side proof device). -/
theorem vecs_eq_convexHull_vertices {d : ℕ} [DecidableEq (Fin d → ℝ)]
    (A : IntervalAmbiguity (Fin d)) :
    A.toSet.vecs = convexHull ℝ ((vertices A).toFinset : Set (Fin d → ℝ)) := by
  have hext : Set.extremePoints ℝ A.toSet.vecs = ((vertices A).toFinset : Set (Fin d → ℝ)) := by
    ext v
    simp only [Finset.mem_coe, List.mem_toFinset, mem_vertices_iff_extremePoints]
  have hK := closure_convexHull_extremePoints A.toSet_wellFormed.isCompact A.toSet_convex
  rw [hext] at hK
  rw [← hK]
  exact (Set.Finite.isCompact_convexHull ℝ (Finset.finite_toSet _)).isClosed.closure_eq

/-- **Multilinear expansion** (the argument of arXiv:2508.00707, Theorem 1, proof in Appendix A):
if every marginal vector is a convex combination `y r = ∑_{v ∈ V r} w r v • v` of marginal
vertices, the expectation under the product `⊗ᵣ y r` is the convex combination, with weights
`∏ᵣ w r (q r)`, of the expectations under the vertex products `⊗ᵣ q r`.

Julia counterpart: none (Lean-side proof device). -/
theorem dot_piVec_eq_sum {d : Fin n → ℕ} (V : ∀ r, Finset (Fin (d r) → ℝ))
    (w : ∀ r, (Fin (d r) → ℝ) → ℝ) (y : ∀ r, Fin (d r) → ℝ)
    (hy : ∀ r, y r = ∑ v ∈ V r, w r v • v) (W : ((r : Fin n) → Fin (d r)) → ℝ) :
    dot W (AmbiguitySet.piVec y) =
      ∑ q ∈ Fintype.piFinset V, (∏ r, w r (q r)) * dot W (AmbiguitySet.piVec q) := by
  have hprod : ∀ t : (r : Fin n) → Fin (d r), AmbiguitySet.piVec y t =
      ∑ q ∈ Fintype.piFinset V, (∏ r, w r (q r)) * AmbiguitySet.piVec q t := by
    intro t
    simp only [AmbiguitySet.piVec, hy, Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    rw [Finset.prod_univ_sum]
    exact Finset.sum_congr rfl fun q _ => Finset.prod_mul_distrib
  simp only [dot, hprod, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun q _ => Finset.sum_congr rfl fun t _ => by ring

/-- The weights `∏ᵣ w r (q r)` of the multilinear expansion are a convex combination.

Julia counterpart: none (Lean-side proof device). -/
theorem sum_prod_weights {d : Fin n → ℕ} (V : ∀ r, Finset (Fin (d r) → ℝ))
    (w : ∀ r, (Fin (d r) → ℝ) → ℝ) (hw : ∀ r, ∑ v ∈ V r, w r v = 1) :
    ∑ q ∈ Fintype.piFinset V, ∏ r, w r (q r) = 1 := by
  rw [← Finset.prod_univ_sum]
  simp [hw]

open Classical in
/-- The marginal vertices of variable `r` at `(s, a)` as a finite set (the values the marginal
iterator yields).

Julia counterpart: the vectors yielded by `vertex_generator(ambiguity_sets[r], …)`
(`src/probabilities/IntervalAmbiguitySets.jl`) in
`state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`). -/
noncomputable def vertexFinset (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) (r : Fin n) : Finset (Fin (F.stateVars.dims r) → ℝ) :=
  (vertices (ambiguitySets F s a r)).toFinset

/-- Every expectation over the product set is a convex combination of the expectations under the
products of iterator vertices.

Julia counterpart: none (Lean-side proof device). -/
theorem exists_vertex_expansion (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) (W : F.stateVars.State → ℝ) {x : ℝ}
    (hx : x ∈ Bellman.expectations (F.productSet s a) W) :
    ∃ c : ((r : Fin n) → Fin (F.stateVars.dims r) → ℝ) → ℝ,
      (∀ q ∈ Fintype.piFinset (vertexFinset F s a), 0 ≤ c q) ∧
      ∑ q ∈ Fintype.piFinset (vertexFinset F s a), c q = 1 ∧
      x = ∑ q ∈ Fintype.piFinset (vertexFinset F s a), c q * dot W (AmbiguitySet.piVec q) := by
  classical
  obtain ⟨_, hy, rfl⟩ := hx
  rw [FactoredIMDP.productSet, AmbiguitySet.vecs_pi] at hy
  obtain ⟨y, hy, rfl⟩ := hy
  have hyr : ∀ r, y r ∈ convexHull ℝ (vertexFinset F s a r : Set (Fin (F.stateVars.dims r) → ℝ)) :=
    fun r => by
      unfold vertexFinset
      convert hy r (Set.mem_univ r)
      exact (vecs_eq_convexHull_vertices (ambiguitySets F s a r)).symm
  choose w hw0 hw1 hwy using fun r => Finset.mem_convexHull.1 (hyr r)
  refine ⟨fun q => ∏ r, w r (q r), fun q hq => Finset.prod_nonneg fun r _ =>
    hw0 r (q r) (Fintype.mem_piFinset.1 hq r), sum_prod_weights _ w hw1, ?_⟩
  refine dot_piVec_eq_sum _ w y (fun r => ?_) W
  rw [← hwy r, Finset.centerMass_eq_of_sum_1 _ _ (hw1 r)]
  rfl

/-- The enumeration is nonempty: it visits the tuple of the first vertices of the iterators.

Julia counterpart: `Iterators.product(iterators...)` in
`state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`). -/
theorem vertexEnumeration_ne_nil (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) : vertexEnumeration F s a ≠ [] := by
  intro h
  have hq : (fun r => (vertexOf (ambiguitySets F s a r)
      (List.range' 1 (F.stateVars.dims r))).1) ∈ vertexEnumeration F s a := by
    rw [vertexEnumeration, mem_productList]
    exact fun r => (mem_vertices_iff _ _).2 ⟨_, List.Perm.refl _, rfl⟩
  rw [h] at hq
  exact List.not_mem_nil hq

/-- The enumeration visits exactly the tuples of `Fintype.piFinset (vertexFinset F s a)`.

Julia counterpart: none (Lean-side proof device). -/
theorem mem_vertexEnumeration_iff (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) (q : (r : Fin n) → Fin (F.stateVars.dims r) → ℝ) :
    q ∈ vertexEnumeration F s a ↔ q ∈ Fintype.piFinset (vertexFinset F s a) := by
  rw [vertexEnumeration, mem_productList, Fintype.mem_piFinset]
  simp only [vertexFinset, List.mem_toFinset]

/-- Every enumerated value is the expectation of `W` under a distribution of `productSet`.

Julia counterpart: `v` in `state_action_bellman(::FactoredVertexIteratorWorkspace, …)`
(`src/bellman.jl`). -/
theorem vertexSum_mem_expectations (F : FactoredIMDP n m) (s : F.stateVars.State)
    (a : F.actionVars.Action) (W : F.stateVars.State → ℝ)
    {q : (r : Fin n) → Fin (F.stateVars.dims r) → ℝ} (hq : q ∈ vertexEnumeration F s a) :
    vertexSum F s a W q ∈ Bellman.expectations (F.productSet s a) W := by
  rw [vertexSum_eq_dot]
  refine ⟨AmbiguitySet.piVec q, ?_, rfl⟩
  rw [FactoredIMDP.productSet, AmbiguitySet.vecs_pi]
  exact ⟨q, fun r _ => ((vertices_complete F s a q).1 hq r).1, rfl⟩

/-- **Vertex enumeration is exact** (arXiv:2508.00707, Theorem 1). For every satisfaction mode
`sat` (`upper_bound = isoptimistic(sat)`), the value returned by
`state_action_bellman(::FactoredVertexIteratorWorkspace, V, ambiguity_sets, upper_bound)`
(`vertexValue`, a fold over the products of marginal vertices) is the exact inner optimum
`opt_{γ ∈ productSet s a} ⟨γ, W⟩` (`sup` for `optimistic`, `inf` for `pessimistic`) over the
literal, non-convex product set; it is finite. The strategy mode (maximize / minimize) acts only
after this, across actions (`extract_strategy!`), so all four satisfaction × strategy modes are
covered. No hypotheses beyond the model invariants. Dense marginal storage only (sparse marginals
not modelled, O16): `vertexValue` is built on the dense `vertices` transcription. This value is
`Bellman.stateActionBellman F.toRMDP sat W s a`, the exact reference for A2/A3.

Julia counterpart: `state_action_bellman(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`),
called from `state_bellman!` with `upper_bound = isoptimistic(spec)`. -/
theorem vertexValue_eq_opt (F : FactoredIMDP n m) (sat : SatisfactionMode)
    (s : F.stateVars.State) (a : F.actionVars.Action) (W : F.stateVars.State → ℝ) :
    vertexValue F s a W (Bellman.isOptimistic sat) =
      (Bellman.innerOpt sat (F.productSet s a) W : EReal) := by
  classical
  obtain ⟨q₀, hq₀⟩ := List.exists_mem_of_ne_nil _ (vertexEnumeration_ne_nil F s a)
  -- every expectation is a convex combination of enumerated values
  have hcomb : ∀ x ∈ Bellman.expectations (F.productSet s a) W, ∃ c :
      ((r : Fin n) → Fin (F.stateVars.dims r) → ℝ) → ℝ,
      (∀ q ∈ Fintype.piFinset (vertexFinset F s a), 0 ≤ c q) ∧
      ∑ q ∈ Fintype.piFinset (vertexFinset F s a), c q = 1 ∧
      x = ∑ q ∈ Fintype.piFinset (vertexFinset F s a), c q * vertexSum F s a W q := by
    intro x hx
    obtain ⟨c, hc0, hc1, hcx⟩ := exists_vertex_expansion F s a W hx
    refine ⟨c, hc0, hc1, ?_⟩
    simp only [vertexSum_eq_dot]
    exact hcx
  cases sat with
  | optimistic =>
    have hstep : optStep F s a W true = maxStep (vertexSum F s a W) := by
      funext o q; simp [optStep, maxStep]
    simp only [vertexValue, Bellman.isOptimistic, hstep, if_true]
    obtain ⟨h1, -, h3⟩ := foldl_maxStep (vertexSum F s a W) (vertexEnumeration F s a) ⊥
    rcases h1 with h | ⟨q, hq, h⟩
    · exfalso
      have := h3 q₀ hq₀
      rw [h, le_bot_iff] at this
      exact EReal.coe_ne_bot _ this
    · rw [h]
      have hmax : ∀ q' ∈ Fintype.piFinset (vertexFinset F s a),
          vertexSum F s a W q' ≤ vertexSum F s a W q := fun q' hq' =>
        EReal.coe_le_coe_iff.1 (h ▸ h3 q' ((mem_vertexEnumeration_iff F s a q').2 hq'))
      have hG : IsGreatest (Bellman.expectations (F.productSet s a) W) (vertexSum F s a W q) := by
        refine ⟨vertexSum_mem_expectations F s a W hq, fun x hx => ?_⟩
        obtain ⟨c, hc0, hc1, rfl⟩ := hcomb x hx
        calc ∑ q' ∈ Fintype.piFinset (vertexFinset F s a), c q' * vertexSum F s a W q'
            ≤ ∑ q' ∈ Fintype.piFinset (vertexFinset F s a), c q' * vertexSum F s a W q :=
              Finset.sum_le_sum fun q' hq' => mul_le_mul_of_nonneg_left (hmax q' hq') (hc0 q' hq')
          _ = vertexSum F s a W q := by rw [← Finset.sum_mul, hc1, one_mul]
      have he : Bellman.innerOpt .optimistic (F.productSet s a) W = vertexSum F s a W q :=
        hG.csSup_eq
      rw [he]
  | pessimistic =>
    have hstep : optStep F s a W false = minStep (vertexSum F s a W) := by
      funext o q; simp [optStep, minStep]
    simp only [vertexValue, Bellman.isOptimistic, hstep, Bool.false_eq_true, if_false]
    obtain ⟨h1, -, h3⟩ := foldl_minStep (vertexSum F s a W) (vertexEnumeration F s a) ⊤
    rcases h1 with h | ⟨q, hq, h⟩
    · exfalso
      have := h3 q₀ hq₀
      rw [h, top_le_iff] at this
      exact EReal.coe_ne_top _ this
    · rw [h]
      have hmin : ∀ q' ∈ Fintype.piFinset (vertexFinset F s a),
          vertexSum F s a W q ≤ vertexSum F s a W q' := fun q' hq' =>
        EReal.coe_le_coe_iff.1 (h ▸ h3 q' ((mem_vertexEnumeration_iff F s a q').2 hq'))
      have hL : IsLeast (Bellman.expectations (F.productSet s a) W) (vertexSum F s a W q) := by
        refine ⟨vertexSum_mem_expectations F s a W hq, fun x hx => ?_⟩
        obtain ⟨c, hc0, hc1, rfl⟩ := hcomb x hx
        calc vertexSum F s a W q
            = ∑ q' ∈ Fintype.piFinset (vertexFinset F s a), c q' * vertexSum F s a W q := by
              rw [← Finset.sum_mul, hc1, one_mul]
          _ ≤ ∑ q' ∈ Fintype.piFinset (vertexFinset F s a), c q' * vertexSum F s a W q' :=
              Finset.sum_le_sum fun q' hq' => mul_le_mul_of_nonneg_left (hmin q' hq') (hc0 q' hq')
      have he : Bellman.innerOpt .pessimistic (F.productSet s a) W = vertexSum F s a W q :=
        hL.csInf_eq
      rw [he]

/-- The vertex-enumeration value is the exact one-step value of action `a` in state `s` of the
factored model, `Bellman.stateActionBellman F.toRMDP sat W s a` (all four modes; no hypotheses
beyond the model invariants). Dense marginal storage only (sparse marginals not modelled, O16), as
for `vertexValue_eq_opt`.

Julia counterpart: `workspace.actions[jₐ] = state_action_bellman(workspace, V, ambiguity_sets,
upper_bound)` in `state_bellman!(::FactoredVertexIteratorWorkspace, …)` (`src/bellman.jl`). -/
theorem vertexValue_eq_stateActionBellman (F : FactoredIMDP n m) (sat : SatisfactionMode)
    (s : F.stateVars.State) (a : F.actionVars.Action) (W : F.stateVars.State → ℝ) :
    vertexValue F s a W (Bellman.isOptimistic sat) =
      (Bellman.stateActionBellman F.toRMDP sat W s a : EReal) :=
  vertexValue_eq_opt F sat s a W

end Factored

end IntervalMDP
