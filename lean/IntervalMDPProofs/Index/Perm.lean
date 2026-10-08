import IntervalMDPProofs.Index.Julia
import IntervalMDPProofs.Models.IntervalAmbiguity

/-!
# The sort permutation of O-maximization

The dense O-max Bellman update (`src/bellman.jl`) first sorts the value vector once,

    sortperm!(permutation(workspace), V; rev = upper_bound, scratch = scratch(workspace))

(ascending for `upper_bound = false`, descending for `upper_bound = true`), and then, per
ambiguity set, runs the greedy loop of `gap_value(V, gap, budget, perm)`:

    res = zero(T)
    for i in perm
        p = min(budget, gap[i])
        res += p * V[i]
        budget -= p
        if budget <= zero(T)
            break
        end
    end
    return res

`SortedPerm` models the `sortperm!` call, `gapValue` transcribes the loop. We prove that `perm` is a
permutation of `1..n` that orders `V` correctly (`sortedPerm_bijective`), that its entries fit the
`Vector{Int32}` of the workspace (`sortedPerm_fits_int32`), and that the greedy loop visits every
index exactly once, where stopping early only skips indices whose allocation would be `0`
(`greedy_visits_once`, `gapValue_eq_sum_allocation`).
-/

namespace IntervalMDP.Index

/-- The inputs of the call `sortperm!(perm, V; rev = upper_bound)`: the value vector `V` (length
`n`) and the sort direction `upperBound` (`rev = true` sorts descending). The output `perm` is
`SortedPerm.perm`.

Julia counterpart: `bellman_precomputation!(::DenseIntervalOMaxWorkspace, V, upper_bound)`, which
calls `sortperm!(permutation(workspace), V; rev = upper_bound)` (`src/bellman.jl`). -/
structure SortedPerm (n : ℕ) where
  /-- The value vector `V` that is sorted. -/
  V : Fin n → ℝ
  /-- Julia `upper_bound`, passed as `rev`: `true` sorts `V` in descending order. -/
  upperBound : Bool

namespace SortedPerm

variable {n : ℕ}

/-- The order `sortperm!` sorts by: `a` may precede `b` iff `a ≤ b` (`rev = false`, Julia's
`Forward` ordering) or `b ≤ a` (`rev = true`, Julia's `Reverse` ordering).

Julia counterpart: the ordering `ord(isless, identity, rev)` of `sortperm!` used in
`bellman_precomputation!` (`src/bellman.jl`). -/
def ordered (rev : Bool) (a b : ℝ) : Prop := if rev then b ≤ a else a ≤ b

/-- The Boolean comparison of two Lean indices by their values, in the direction `upperBound`.

Julia counterpart: the `lt`/`by` comparison `sortperm!` applies to `V[i]`, `V[j]` in
`bellman_precomputation!` (`src/bellman.jl`). -/
noncomputable def le (s : SortedPerm n) (i j : Fin n) : Bool :=
  if s.upperBound then decide (s.V j ≤ s.V i) else decide (s.V i ≤ s.V j)

/-- The sorted order as Lean indices: `0, …, n - 1` sorted by `le` with a stable merge sort. Julia
documents `sortperm` as stable, so ties keep increasing index order in both.

Julia counterpart: the result of `sortperm!(perm, V; rev = upper_bound)` in
`bellman_precomputation!` (`src/bellman.jl`), as 0-based indices. -/
noncomputable def order (s : SortedPerm n) : List (Fin n) := (List.finRange n).mergeSort s.le

/-- The permutation vector `perm` with Julia (1-based) entries: `perm = map toJulia order`.

Julia counterpart: `permutation(workspace)` after `sortperm!(permutation(workspace), V;
rev = upper_bound)` (`src/bellman.jl`, `src/workspace.jl`). -/
noncomputable def perm (s : SortedPerm n) : List ℕ := s.order.map toJulia

end SortedPerm

/-- Literal transcription of the loop of `gap_value(V, gap, budget, perm)`: for each `i` in `perm`
(in order), `p = min(budget, gap[i])`, `res += p * V[i]`, `budget -= p`, and `break` once
`budget <= 0`; the result is `res`. Julia starts with `res = zero(T)`, i.e. `gapValue V gap perm
budget 0`. The structure (loop order, the `- p`/`+ p * V[i]` updates, the early exit) is kept.

Julia counterpart: `gap_value(V, gap, budget, perm)` (`src/bellman.jl`). -/
noncomputable def gapValue {n : ℕ} (V gap : Fin n → ℝ) : List ℕ → ℝ → ℝ → ℝ
  | [], _, res => res
  | i :: perm, budget, res =>
    let p := min budget (juliaGet gap i)
    let res := res + p * juliaGet V i
    let budget := budget - p
    if budget ≤ 0 then res else gapValue V gap perm budget res

/-- The indices the loop of `gap_value` visits before it exits (including the one at which it
breaks), in order.

Julia counterpart: the iterations of `for i in perm` in `gap_value` (`src/bellman.jl`). -/
noncomputable def visited {n : ℕ} (gap : Fin n → ℝ) : List ℕ → ℝ → List ℕ
  | [], _ => []
  | i :: perm, budget =>
    let budget := budget - min budget (juliaGet gap i)
    i :: (if budget ≤ 0 then [] else visited gap perm budget)

/-- The allocation `p` that the loop of `gap_value`, run without the early exit, gives to the
Julia index `k`: `p = min(budget, gap[k])` with the budget left when `k` is reached (`0` if `k`
does not occur in `perm`).

Julia counterpart: the value `p = min(budget, gap[i])` in `gap_value` (`src/bellman.jl`). -/
def allocation {n : ℕ} (gap : Fin n → ℝ) : List ℕ → ℝ → ℕ → ℝ
  | [], _, _ => 0
  | i :: perm, budget, k =>
    let p := min budget (juliaGet gap i)
    if i = k then p else allocation gap perm (budget - p) k

/-- `juliaGet` of a nonnegative vector is nonnegative.

Julia counterpart: none (Lean-side proof device). -/
theorem juliaGet_nonneg {n : ℕ} {gap : Fin n → ℝ} (h : ∀ i, 0 ≤ gap i) (k : ℕ) :
    0 ≤ juliaGet gap k := by
  unfold juliaGet; split_ifs
  · exact h _
  · exact le_refl 0

/-- With no budget left, the loop allocates nothing.

Julia counterpart: none (Lean-side proof device). -/
theorem allocation_zero {n : ℕ} {gap : Fin n → ℝ} (h : ∀ i, 0 ≤ gap i) :
    ∀ (perm : List ℕ) (k : ℕ), allocation gap perm 0 k = 0
  | [], _ => rfl
  | i :: perm, k => by
    have hg := juliaGet_nonneg h i
    simp only [allocation, min_eq_left hg, sub_zero]
    split_ifs
    · rfl
    · exact allocation_zero h perm k

/-- After one step the budget is nonnegative: `budget - min(budget, g) ≥ 0`.

Julia counterpart: none (Lean-side proof device). -/
theorem budget_sub_min_nonneg (b g : ℝ) : 0 ≤ b - min b g :=
  sub_nonneg.2 (min_le_left b g)

/-- Indices that the loop skips after its early exit would get allocation `0`.

Julia counterpart: none (Lean-side proof device). -/
theorem allocation_eq_zero_of_not_visited {n : ℕ} {gap : Fin n → ℝ} (h : ∀ i, 0 ≤ gap i) :
    ∀ (perm : List ℕ) (b : ℝ) (k : ℕ), k ∉ visited gap perm b → allocation gap perm b k = 0
  | [], _, _, _ => rfl
  | i :: perm, b, k, hk => by
    have hb' := budget_sub_min_nonneg b (juliaGet gap i)
    simp only [visited, List.mem_cons, not_or] at hk
    simp only [allocation, if_neg (Ne.symm hk.1)]
    by_cases h0 : b - min b (juliaGet gap i) ≤ 0
    · rw [le_antisymm h0 hb']; exact allocation_zero h perm k
    · rw [if_neg h0] at hk
      exact allocation_eq_zero_of_not_visited h perm _ k hk.2

/-- The early-exit loop computes the same value as the full loop over `perm`: `res` plus
`∑_{k ∈ perm} allocation k * V[k]`, for a duplicate-free `perm` and nonnegative gaps.

Julia counterpart: none (Lean-side proof device). -/
theorem gapValue_eq_res_add_sum {n : ℕ} (V : Fin n → ℝ) {gap : Fin n → ℝ} (h : ∀ i, 0 ≤ gap i) :
    ∀ (perm : List ℕ) (b res : ℝ), perm.Nodup →
      gapValue V gap perm b res = res + (perm.map fun k => allocation gap perm b k * juliaGet V k).sum
  | [], _, _, _ => by simp [gapValue]
  | i :: perm, b, res, hnd => by
    rw [List.nodup_cons] at hnd
    have hb' := budget_sub_min_nonneg b (juliaGet gap i)
    have htail : (perm.map fun k => allocation gap (i :: perm) b k * juliaGet V k) =
        perm.map fun k => allocation gap perm (b - min b (juliaGet gap i)) k * juliaGet V k := by
      refine List.map_congr_left (fun k hk => ?_)
      have hik : i ≠ k := fun hik => hnd.1 (hik ▸ hk)
      simp only [allocation, if_neg hik]
    rw [List.map_cons, List.sum_cons, htail]
    simp only [gapValue, allocation]
    split_ifs with h0
    · have hz : b - min b (juliaGet gap i) = 0 := le_antisymm h0 hb'
      simp [hz, allocation_zero h]
    · rw [gapValue_eq_res_add_sum V h perm _ _ hnd.2]
      ring

/-- `sortperm!` returns a permutation of `1..n` that orders `V`: `perm ~ 1:n`, and
`V[perm[1]], V[perm[2]], …` is ascending (`rev = false`) or descending (`rev = true`).

Julia counterpart: `sortperm!(perm, V; rev = upper_bound)` in `bellman_precomputation!`
(`src/bellman.jl`). -/
theorem sortedPerm_bijective {n : ℕ} (s : SortedPerm n) :
    s.perm.Perm (List.range' 1 n) ∧
      (s.perm.map (juliaGet s.V)).Pairwise (SortedPerm.ordered s.upperBound) := by
  constructor
  · rw [← map_toJulia_finRange]
    exact (List.mergeSort_perm _ _).map toJulia
  · have hcomp : juliaGet s.V ∘ toJulia = s.V := funext (juliaGet_toJulia s.V)
    rw [SortedPerm.perm, List.map_map, hcomp, List.pairwise_map]
    have hsort := List.pairwise_mergeSort (le := s.le) ?trans ?total (List.finRange n)
    · refine hsort.imp (fun {a b} hab => ?_)
      unfold SortedPerm.le at hab
      unfold SortedPerm.ordered
      split_ifs at hab ⊢ <;> simpa using hab
    case trans =>
      intro a b c hab hbc
      unfold SortedPerm.le at *
      split_ifs at * <;> simp only [decide_eq_true_eq] at * <;> linarith
    case total =>
      intro a b
      unfold SortedPerm.le
      split_ifs <;> simp only [Bool.or_eq_true, decide_eq_true_eq] <;> exact le_total _ _

/-- The entries of `perm` fit the workspace's `Vector{Int32}`: if `n < 2 ^ 31`, every entry is held
exactly by an `Int32`.

Julia counterpart: `permutation::Vector{Int32}` of `DenseIntervalOMaxWorkspace`
(`src/workspace.jl`), filled by `sortperm!` (`src/bellman.jl`). -/
theorem sortedPerm_fits_int32 {n : ℕ} (s : SortedPerm n) (hBound : n < 2 ^ 31) :
    ∀ k ∈ s.perm, machineInt 32 k = k := by
  intro k hk
  apply machineInt_eq_self
  have hmem := (sortedPerm_bijective s).1.subset hk
  rw [List.mem_range'_1] at hmem
  simp only [Nat.add_one_sub_one]
  omega

/-- The greedy O-max loop of `gap_value` over `perm` visits every index exactly once
(`perm` contains each Julia index `toJulia i` once), and stopping early only skips indices whose
allocation would be `0`.

Julia counterpart: the loop `for i in perm … break` of `gap_value(V, gap, budget, perm)`
(`src/bellman.jl`), with `gap`/`budget` of an `IntervalAmbiguitySet`. -/
theorem greedy_visits_once {n : ℕ} (s : SortedPerm n) (A : IntervalAmbiguity (Fin n)) :
    (∀ i : Fin n, s.perm.count (toJulia i) = 1) ∧
      ∀ i : Fin n, toJulia i ∉ visited A.gap s.perm A.budget →
        allocation A.gap s.perm A.budget (toJulia i) = 0 := by
  constructor
  · intro i
    rw [(sortedPerm_bijective s).1.count_eq, ← map_toJulia_finRange]
    exact List.count_eq_one_of_mem ((List.nodup_finRange n).map toJulia_injective)
      (List.mem_map_of_mem (List.mem_finRange i))
  · intro i hi
    exact allocation_eq_zero_of_not_visited A.gap_nonneg s.perm A.budget (toJulia i) hi

/-- Stopping early does not change the result: `gap_value` equals `∑ᵢ allocation i * V[i]`, the
value of the full loop in which every index is visited exactly once.

Julia counterpart: `gap_value(V, gap, budget, perm)` (`src/bellman.jl`), with `gap`/`budget` of an
`IntervalAmbiguitySet`. -/
theorem gapValue_eq_sum_allocation {n : ℕ} (s : SortedPerm n) (A : IntervalAmbiguity (Fin n)) :
    gapValue s.V A.gap s.perm A.budget 0 =
      ∑ i : Fin n, allocation A.gap s.perm A.budget (toJulia i) * s.V i := by
  have hp := (sortedPerm_bijective s).1
  have hnd : s.perm.Nodup := hp.nodup_iff.2 List.nodup_range'
  have hsum : ∀ g : ℕ → ℝ, (s.perm.map g).sum = ∑ i : Fin n, g (toJulia i) := by
    intro g
    unfold SortedPerm.perm SortedPerm.order
    rw [List.map_map, ((List.mergeSort_perm (List.finRange n) s.le).map _).sum_eq,
      Fin.sum_univ_def]
    rfl
  rw [gapValue_eq_res_add_sum s.V A.gap_nonneg s.perm A.budget 0 hnd, zero_add,
    hsum]
  simp only [juliaGet_toJulia]

end IntervalMDP.Index
