import IntervalMDPProofs.Index.Perm
import IntervalMDPProofs.Index.Sparse

/-!
# Dense and sparse O-maximization

## Dense path

The dense interval Bellman update (`src/bellman.jl`) evaluates one ambiguity set as

    state_action_bellman(workspace::DenseIntervalOMaxWorkspace, V, ambiguity_set, budget,
                         upper_bound) =
        dot(V, lower(ambiguity_set)) +
        gap_value(V, gap(ambiguity_set), budget, permutation(workspace))

where `permutation(workspace)` was filled once per Bellman step by
`bellman_precomputation!`, i.e. `sortperm!(perm, V; rev = upper_bound)`, and `gap_value` is the
greedy loop transcribed as `IntervalMDP.Index.gapValue` (`Index/Perm.lean`).

`omax A s` transcribes this value for the ambiguity set `A` and the sort `s`. We prove that it is
exact:

* `omax_mem`: it is `⟨p, V⟩` for the greedy distribution `p = lower + allocation`, and
  `p ∈ P(l, u)`;
* `omax_eq_sSup`: for `upper_bound = true` (descending `perm`) it is `sup {⟨p, V⟩ : p ∈ P(l, u)}`;
* `omax_eq_sInf`: for `upper_bound = false` (ascending `perm`) it is `inf {⟨p, V⟩ : p ∈ P(l, u)}`;
* `omax_tie_invariant`: any two permutations that sort `V` in the same direction (stable or not,
  i.e. ordering ties in `V` differently) give the same value.

**Mode coverage.** O-maximization is mode-agnostic: the satisfaction mode (pessimistic /
optimistic) and the strategy mode (maximize / minimize) reach `state_action_bellman` only
through `upper_bound = isoptimistic(spec)` (`src/robust_value_iteration.jl`), which is `rev` of
the sort; the strategy mode acts afterwards, across actions. Both values of `upper_bound` are
covered (`omax_eq_sSup`, `omax_eq_sInf`), so all four satisfaction × strategy modes are covered.

**Scope.** Abstract (exact real arithmetic). The Julia loop exits early once `budget <= 0`; Lean
models this exactly through `gapValue` and removes the early exit with
`gapValue_eq_res_add_sum` / `greedy_visits_once` (`Index/Perm.lean`): the skipped indices get
`p_i = min(budget_i, gap_i) = 0`.

Proof idea: the greedy allocation has a threshold `c` (`IsThreshold`): along a descending `perm`
it fills `gap_i` for `V_i > c` and allocates `0` for `V_i < c`. Then for every feasible `q`,
`(q_i - l_i) (V_i - c) ≤ a_i (V_i - c)` termwise, and summing (both allocations total `budget`)
gives `⟨q, V⟩ ≤ ⟨l + a, V⟩`. The ascending case applies this to `-V`.

## Sparse path

For a sparse gap column, `state_action_bellman(::SparseIntervalOMaxWorkspace, V, ambiguity_set,
budget, upper_bound)` (`src/bellman.jl`, lines 536–553) does not sort `V`. It writes the pairs
`zip(V[support], nonzeros(gap))` into `workspace.values_gaps`, sorts them with
`sort!(Vp_workspace; rev = upper_bound, by = first)` and returns
`dot(V, lower(ambiguity_set)) + gap_value(Vp_workspace, budget)` (`gap_value(Vp, budget)`, lines
555–571). `omaxSparse` transcribes this for a `SparseIntervalAmbiguity` (an `IntervalAmbiguity`
whose gap is a CSC column) and any sort result `SortedValuesGaps`. `omaxSparse_eq_omax` proves that
it equals the dense `omax`, for both `upper_bound` values (all four modes), and for every sort of
the pairs by their first component (stable or not, ties ordered arbitrarily).

Proof idea: by `sparse_zip_correct` the sorted pairs are `(V[i], gap[i])` for the support rows `i`
in some order `L` sorted by `V`. `L` is a sublist of a full sorted permutation vector of `1:n`
(`exists_permutation_sublist`, via the stability of `mergeSort`), the rows it skips have `gap = 0`
and allocate nothing (`gapValue_sublist`), so the sparse loop equals the dense loop along a valid
dense permutation; `omax_tie_invariant` then gives `omax`.
-/

namespace IntervalMDP.OMax

open Finset IntervalMDP.Index

variable {n : ℕ}

/-- The inner product `⟨p, V⟩ = ∑ₛ V(s) p(s)`.

Julia counterpart: `dot(V, lower(ambiguity_set))` (LinearAlgebra's `dot`) in
`state_action_bellman(::DenseIntervalOMaxWorkspace, …)` (`src/bellman.jl`). -/
def dot {S : Type*} [Fintype S] (V p : S → ℝ) : ℝ := ∑ s, V s * p s

/-- The set `{⟨p, V⟩ : p ∈ P(l, u)}` of values that the distributions of the ambiguity set give to
`V`; the exact one-step optimum is its `sSup` (maximizing) or `sInf` (minimizing).

Julia counterpart: the optimization problem that `state_action_bellman` solves for an
`IntervalAmbiguitySet` (`src/bellman.jl`, `src/probabilities/IntervalAmbiguitySets.jl`). -/
def valueSet {S : Type*} [Fintype S] (A : IntervalAmbiguity S) (V : S → ℝ) : Set ℝ :=
  {x | ∃ p ∈ A.toSet, dot V p = x}

/-- A permutation vector `perm` that `sortperm!(perm, V; rev = upper_bound)` may return with any
sorting algorithm: `perm` is a permutation of `1..n` and `V[perm[1]], V[perm[2]], …` is
descending (`upper_bound = true`) or ascending (`upper_bound = false`). Ties in `V` may be ordered
arbitrarily. The stable merge sort of `SortedPerm.perm` is one instance (`stablePermutation`).

Julia counterpart: `permutation(workspace)` of `DenseIntervalOMaxWorkspace` (`src/workspace.jl`)
after `bellman_precomputation!` (`src/bellman.jl`). -/
structure Permutation (s : SortedPerm n) where
  /-- The Julia (1-based) permutation vector. -/
  perm : List ℕ
  /-- `perm` is a permutation of `1:n`. -/
  perm_perm : perm.Perm (List.range' 1 n)
  /-- `V[perm[1]], V[perm[2]], …` is sorted in direction `rev = upper_bound`. -/
  sorted : (perm.map (juliaGet s.V)).Pairwise (SortedPerm.ordered s.upperBound)

/-- The permutation computed by the stable `sortperm!` (`SortedPerm.perm`), as a `Permutation`.

Julia counterpart: `sortperm!(permutation(workspace), V; rev = upper_bound)` in
`bellman_precomputation!(::DenseIntervalOMaxWorkspace, V, upper_bound)` (`src/bellman.jl`). -/
noncomputable def stablePermutation (s : SortedPerm n) : Permutation s where
  perm := s.perm
  perm_perm := (sortedPerm_bijective s).1
  sorted := (sortedPerm_bijective s).2

/-- Transcription of `state_action_bellman(::DenseIntervalOMaxWorkspace, V, ambiguity_set, budget,
upper_bound)`: `dot(V, lower(ambiguity_set)) + gap_value(V, gap(ambiguity_set), budget, perm)`,
with `budget = 1 - ∑ lower` (precomputed in the workspace) and `perm = permutation(workspace)`.

Julia counterpart: `state_action_bellman(::DenseIntervalOMaxWorkspace, …)` (`src/bellman.jl`). -/
noncomputable def stateActionBellman (A : IntervalAmbiguity (Fin n)) (V : Fin n → ℝ)
    (perm : List ℕ) : ℝ :=
  dot V A.lower + gapValue V A.gap perm A.budget 0

/-- Dense O-maximization: `⟨lower, V⟩ + gap_value(V, gap, budget, perm)` with the permutation
`perm = sortperm(V; rev = upper_bound)` of `bellman_precomputation!`. For `upper_bound = true`
this is the maximum of `⟨p, V⟩` over `P(l, u)` (`omax_eq_sSup`), for `upper_bound = false` the
minimum (`omax_eq_sInf`).

Julia counterpart: `bellman_precomputation!(::DenseIntervalOMaxWorkspace, V, upper_bound)`
followed by `state_action_bellman(::DenseIntervalOMaxWorkspace, …)` (`src/bellman.jl`). -/
noncomputable def omax (A : IntervalAmbiguity (Fin n)) (s : SortedPerm n) : ℝ :=
  stateActionBellman A s.V s.perm

/-- With a nonnegative budget, every allocation `p = min(budget, gap[i])` is nonnegative.

Julia counterpart: none (Lean-side proof device). -/
theorem allocation_nonneg {gap : Fin n → ℝ} (h : ∀ i, 0 ≤ gap i) :
    ∀ (perm : List ℕ) (b : ℝ) (k : ℕ), 0 ≤ b → 0 ≤ allocation gap perm b k
  | [], _, _, _ => le_refl 0
  | i :: perm, b, k, hb => by
    simp only [allocation]
    split_ifs
    · exact le_min hb (juliaGet_nonneg h i)
    · exact allocation_nonneg h perm _ k (budget_sub_min_nonneg b _)

/-- Every allocation is at most the gap: `p = min(budget, gap[k]) ≤ gap[k]`.

Julia counterpart: none (Lean-side proof device). -/
theorem allocation_le_gap {gap : Fin n → ℝ} (h : ∀ i, 0 ≤ gap i) :
    ∀ (perm : List ℕ) (b : ℝ) (k : ℕ), allocation gap perm b k ≤ juliaGet gap k
  | [], _, k => juliaGet_nonneg h k
  | i :: perm, b, k => by
    simp only [allocation]
    split_ifs with hik
    · subst hik; exact min_le_right _ _
    · exact allocation_le_gap h perm _ k

/-- On the tail of `i :: perm`, the allocation is that of `perm` with the remaining budget.

Julia counterpart: none (Lean-side proof device). -/
theorem allocation_cons_of_ne {gap : Fin n → ℝ} {i k : ℕ} (perm : List ℕ) (b : ℝ) (hik : i ≠ k) :
    allocation gap (i :: perm) b k = allocation gap perm (b - min b (juliaGet gap i)) k := by
  simp only [allocation, if_neg hik]

/-- The full greedy loop allocates `min(budget, ∑ gap)` in total, for a duplicate-free `perm` and a
nonnegative budget.

Julia counterpart: none (Lean-side proof device). -/
theorem sum_allocation {gap : Fin n → ℝ} (h : ∀ i, 0 ≤ gap i) :
    ∀ (perm : List ℕ) (b : ℝ), 0 ≤ b → perm.Nodup →
      (perm.map (allocation gap perm b)).sum = min b (perm.map (juliaGet gap)).sum
  | [], b, hb, _ => by simp [allocation, hb]
  | i :: perm, b, hb, hnd => by
    rw [List.nodup_cons] at hnd
    have htail : perm.map (allocation gap (i :: perm) b) =
        perm.map (allocation gap perm (b - min b (juliaGet gap i))) :=
      List.map_congr_left (fun k hk => allocation_cons_of_ne perm b
        (fun hik => hnd.1 (hik ▸ hk)))
    have hR : 0 ≤ (perm.map (juliaGet gap)).sum :=
      List.sum_nonneg (by simpa using fun k _ => juliaGet_nonneg h k)
    have hhead : allocation gap (i :: perm) b i = min b (juliaGet gap i) := by
      simp [allocation]
    rw [List.map_cons, List.sum_cons, htail, hhead,
      sum_allocation h perm _ (budget_sub_min_nonneg b _) hnd.2, List.map_cons, List.sum_cons]
    have hg := juliaGet_nonneg h i
    rw [min_def b (juliaGet gap i)]
    split_ifs with hbg
    · rw [sub_self, min_eq_left hR, min_eq_left (by linarith)]; ring
    · rw [min_def, min_def]; split_ifs <;> linarith

/-- `c` is a threshold of the greedy allocation along `perm` for the values `W`: indices with
`W > c` get their full gap, indices with `W < c` get nothing (complementary slackness of the
interval LP).

Julia counterpart: none (Lean-side proof device). -/
def IsThreshold (W gap : Fin n → ℝ) (perm : List ℕ) (b c : ℝ) : Prop :=
  ∀ k ∈ perm, (c < juliaGet W k → allocation gap perm b k = juliaGet gap k) ∧
    (juliaGet W k < c → allocation gap perm b k = 0)

/-- Along a permutation that sorts `W` descending, the greedy allocation has a threshold `c`, which
can be chosen below any upper bound `u` of the values of `W` on `perm`.

Julia counterpart: none (Lean-side proof device). -/
theorem exists_threshold {gap : Fin n → ℝ} (h : ∀ i, 0 ≤ gap i) (W : Fin n → ℝ) :
    ∀ (perm : List ℕ) (b u : ℝ), 0 ≤ b → perm.Nodup →
      (perm.map (juliaGet W)).Pairwise (SortedPerm.ordered true) →
      (∀ k ∈ perm, juliaGet W k ≤ u) → ∃ c ≤ u, IsThreshold W gap perm b c
  | [], _, u, _, _, _, _ => ⟨u, le_rfl, by simp [IsThreshold]⟩
  | i :: perm, b, u, hb, hnd, hsort, hu => by
    rw [List.nodup_cons] at hnd
    rw [List.map_cons, List.pairwise_cons] at hsort
    have hle : ∀ k ∈ perm, juliaGet W k ≤ juliaGet W i := fun k hk => by
      simpa [SortedPerm.ordered] using hsort.1 _ (List.mem_map_of_mem hk)
    have htail : ∀ k ∈ perm, allocation gap (i :: perm) b k =
        allocation gap perm (b - min b (juliaGet gap i)) k :=
      fun k hk => allocation_cons_of_ne perm b (fun hik => hnd.1 (hik ▸ hk))
    have hhead : allocation gap (i :: perm) b i = min b (juliaGet gap i) := by
      simp [allocation]
    have hb' := budget_sub_min_nonneg b (juliaGet gap i)
    by_cases h0 : b - min b (juliaGet gap i) ≤ 0
    · have hz : b - min b (juliaGet gap i) = 0 := le_antisymm h0 hb'
      refine ⟨juliaGet W i, hu i List.mem_cons_self, fun k hk => ?_⟩
      rcases List.mem_cons.1 hk with rfl | hk
      · exact ⟨fun hc => absurd hc (lt_irrefl _), fun hc => absurd hc (lt_irrefl _)⟩
      · refine ⟨fun hc => absurd (hle k hk) (not_le.2 hc), fun _ => ?_⟩
        rw [htail k hk, hz]
        exact allocation_zero h perm k
    · have hmin : min b (juliaGet gap i) = juliaGet gap i := by
        rcases le_total b (juliaGet gap i) with hbg | hbg
        · exact (h0 (by rw [min_eq_left hbg, sub_self])).elim
        · exact min_eq_right hbg
      obtain ⟨c, hc, hcs⟩ := exists_threshold h W perm _ (juliaGet W i) hb' hnd.2 hsort.2 hle
      refine ⟨c, hc.trans (hu i List.mem_cons_self), fun k hk => ?_⟩
      rcases List.mem_cons.1 hk with rfl | hk
      · exact ⟨fun _ => hhead.trans hmin, fun hlt => absurd hc (not_le.2 hlt)⟩
      · rw [htail k hk]
        exact hcs k hk

/-- A sum over a permutation vector of `1:n` is the sum over all Julia indices `toJulia i`.

Julia counterpart: none (Lean-side proof device). -/
theorem sum_perm_eq {s : SortedPerm n} (σ : Permutation s) (g : ℕ → ℝ) :
    (σ.perm.map g).sum = ∑ i : Fin n, g (toJulia i) := by
  rw [(σ.perm_perm.map g).sum_eq, ← map_toJulia_finRange, List.map_map, Fin.sum_univ_def]
  rfl

/-- A permutation vector of `1:n` has no duplicates.

Julia counterpart: none (Lean-side proof device). -/
theorem Permutation.nodup {s : SortedPerm n} (σ : Permutation s) : σ.perm.Nodup :=
  σ.perm_perm.nodup_iff.2 List.nodup_range'

/-- Every Julia index `toJulia i` occurs in a permutation vector of `1:n`.

Julia counterpart: none (Lean-side proof device). -/
theorem Permutation.mem {s : SortedPerm n} (σ : Permutation s) (i : Fin n) :
    toJulia i ∈ σ.perm := by
  rw [σ.perm_perm.mem_iff, ← map_toJulia_finRange]
  exact List.mem_map_of_mem (List.mem_finRange i)

/-- The greedy allocations total the budget: `∑ᵢ pᵢ = budget`, because `budget ≤ ∑ gap`.

Julia counterpart: none (Lean-side proof device). -/
theorem sum_allocation_eq_budget (A : IntervalAmbiguity (Fin n)) {s : SortedPerm n}
    (σ : Permutation s) :
    ∑ i : Fin n, allocation A.gap σ.perm A.budget (toJulia i) = A.budget := by
  have h := sum_allocation A.gap_nonneg σ.perm A.budget A.budget_eq.2 σ.nodup
  rw [sum_perm_eq, sum_perm_eq] at h
  simp only [juliaGet_toJulia] at h
  rw [h, min_eq_left A.budget_le_sum_gap]

/-- The greedy distribution of O-maximization: `p = lower + allocation`, where the allocation
`min(budget, gap[i])` (with the budget left when `i` is reached in `perm`) is the amount the loop
of `gap_value` adds on top of the lower bound (`0` for indices after the early exit).

Julia counterpart: the distribution implicit in `state_action_bellman(::DenseIntervalOMaxWorkspace,
…)`: `dot(V, lower)` plus the `p * V[i]` terms of `gap_value` (`src/bellman.jl`). -/
noncomputable def greedy (A : IntervalAmbiguity (Fin n)) {s : SortedPerm n} (σ : Permutation s) :
    ProbVec (Fin n) where
  toFun i := A.lower i + allocation A.gap σ.perm A.budget (toJulia i)
  nonneg i := add_nonneg (A.lower_nonneg i)
    (allocation_nonneg A.gap_nonneg _ _ _ A.budget_eq.2)
  sum_eq_one := by
    rw [Finset.sum_add_distrib, sum_allocation_eq_budget, IntervalAmbiguity.budget]
    ring

/-- The greedy distribution is `lower + allocation` pointwise.

Julia counterpart: none (Lean-side proof device). -/
theorem greedy_apply (A : IntervalAmbiguity (Fin n)) {s : SortedPerm n} (σ : Permutation s)
    (i : Fin n) : greedy A σ i = A.lower i + allocation A.gap σ.perm A.budget (toJulia i) := rfl

/-- The greedy distribution lies in `P(l, u)`.

Julia counterpart: none (Lean-side proof device). -/
theorem greedy_mem (A : IntervalAmbiguity (Fin n)) {s : SortedPerm n} (σ : Permutation s) :
    greedy A σ ∈ A.toSet := by
  intro i
  rw [greedy_apply]
  have h0 := allocation_nonneg A.gap_nonneg σ.perm A.budget (toJulia i) A.budget_eq.2
  have h1 := allocation_le_gap A.gap_nonneg σ.perm A.budget (toJulia i)
  rw [juliaGet_toJulia] at h1
  constructor
  · linarith
  · linarith [A.lower_add_gap i]

/-- `state_action_bellman` evaluates `⟨p, V⟩` at the greedy distribution `p`, for every
permutation vector `perm` of `1:n` (the early exit of `gap_value` does not matter).

Julia counterpart: `state_action_bellman(::DenseIntervalOMaxWorkspace, …)` (`src/bellman.jl`). -/
theorem stateActionBellman_eq_dot (A : IntervalAmbiguity (Fin n)) {s : SortedPerm n}
    (σ : Permutation s) (V : Fin n → ℝ) :
    stateActionBellman A V σ.perm = dot V (greedy A σ) := by
  rw [stateActionBellman, gapValue_eq_res_add_sum V A.gap_nonneg σ.perm A.budget 0 σ.nodup,
    zero_add, sum_perm_eq]
  simp only [dot, greedy_apply, juliaGet_toJulia, mul_add, Finset.sum_add_distrib, mul_comm]

/-- `juliaGet` commutes with negation.

Julia counterpart: none (Lean-side proof device). -/
theorem juliaGet_neg (V : Fin n → ℝ) (k : ℕ) : juliaGet (-V) k = -juliaGet V k := by
  unfold juliaGet; split_ifs <;> simp

/-- `⟨p, -V⟩ = -⟨p, V⟩`.

Julia counterpart: none (Lean-side proof device). -/
theorem dot_neg {S : Type*} [Fintype S] (V p : S → ℝ) : dot (-V) p = -dot V p := by
  simp [dot, Finset.sum_neg_distrib]

/-- Exchange inequality: along a permutation that sorts `W` descending, the greedy distribution
maximizes `⟨q, W⟩` over `q ∈ P(l, u)`.

Julia counterpart: none (Lean-side proof device). -/
theorem dot_le_greedy (A : IntervalAmbiguity (Fin n)) {s : SortedPerm n} (σ : Permutation s)
    (W : Fin n → ℝ) (hW : (σ.perm.map (juliaGet W)).Pairwise (SortedPerm.ordered true))
    {q : ProbVec (Fin n)} (hq : q ∈ A.toSet) : dot W q ≤ dot W (greedy A σ) := by
  have hu : ∀ k ∈ σ.perm, juliaGet W k ≤ ∑ i, |W i| := by
    intro k _
    unfold juliaGet; split_ifs
    · exact (le_abs_self _).trans
        (Finset.single_le_sum (fun i _ => abs_nonneg (W i)) (Finset.mem_univ _))
    · exact Finset.sum_nonneg (fun i _ => abs_nonneg (W i))
  obtain ⟨c, -, hc⟩ := exists_threshold A.gap_nonneg W σ.perm A.budget _ A.budget_eq.2
    σ.nodup hW hu
  have hterm : ∀ i, (q i - A.lower i) * (W i - c) ≤
      allocation A.gap σ.perm A.budget (toJulia i) * (W i - c) := by
    intro i
    have hcs := hc (toJulia i) (σ.mem i)
    rw [juliaGet_toJulia, juliaGet_toJulia] at hcs
    have hqi := hq i
    have hg := A.lower_add_gap i
    rcases lt_trichotomy c (W i) with hlt | heq | hgt
    · rw [hcs.1 hlt]
      exact mul_le_mul_of_nonneg_right (by linarith) (by linarith)
    · rw [heq, sub_self, mul_zero, mul_zero]
    · rw [hcs.2 hgt, zero_mul]
      exact mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
  have hsum := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hterm i)
  have hq1 := q.sum_eq_one
  have ha := sum_allocation_eq_budget A σ
  simp only [sub_mul, mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul] at hsum
  simp only [dot, greedy_apply, mul_add, Finset.sum_add_distrib]
  have hbc : A.budget * c = c - (∑ i, A.lower i) * c := by
    rw [IntervalAmbiguity.budget]; ring
  have e1 : ∑ i, W i * q i = ∑ i, q i * W i := by simp only [mul_comm]
  have e2 : ∑ i, W i * A.lower i = ∑ i, A.lower i * W i := by simp only [mul_comm]
  have e3 : ∑ i, W i * allocation A.gap σ.perm A.budget (toJulia i) =
      ∑ i, allocation A.gap σ.perm A.budget (toJulia i) * W i := by simp only [mul_comm]
  have hq1' : ∑ i, q.toFun i = 1 := hq1
  rw [hq1', ha, hbc] at hsum
  rw [e1, e2, e3]
  linarith

/-- For `upper_bound = true`, `state_action_bellman` with any descending permutation vector is the
greatest element of `{⟨p, V⟩ : p ∈ P(l, u)}`.

Julia counterpart: `state_action_bellman(::DenseIntervalOMaxWorkspace, …)` (`src/bellman.jl`). -/
theorem stateActionBellman_isGreatest (A : IntervalAmbiguity (Fin n)) {s : SortedPerm n}
    (σ : Permutation s) (h : s.upperBound = true) :
    IsGreatest (valueSet A s.V) (stateActionBellman A s.V σ.perm) := by
  rw [stateActionBellman_eq_dot]
  refine ⟨⟨greedy A σ, greedy_mem A σ, rfl⟩, ?_⟩
  rintro x ⟨q, hq, rfl⟩
  have hW := σ.sorted
  rw [h] at hW
  exact dot_le_greedy A σ s.V hW hq

/-- For `upper_bound = false`, `state_action_bellman` with any ascending permutation vector is the
least element of `{⟨p, V⟩ : p ∈ P(l, u)}`.

Julia counterpart: `state_action_bellman(::DenseIntervalOMaxWorkspace, …)` (`src/bellman.jl`). -/
theorem stateActionBellman_isLeast (A : IntervalAmbiguity (Fin n)) {s : SortedPerm n}
    (σ : Permutation s) (h : s.upperBound = false) :
    IsLeast (valueSet A s.V) (stateActionBellman A s.V σ.perm) := by
  rw [stateActionBellman_eq_dot]
  refine ⟨⟨greedy A σ, greedy_mem A σ, rfl⟩, ?_⟩
  rintro x ⟨q, hq, rfl⟩
  have hW := σ.sorted
  rw [h] at hW
  have hW' : (σ.perm.map (juliaGet (-s.V))).Pairwise (SortedPerm.ordered true) := by
    have heq : σ.perm.map (juliaGet (-s.V)) = (σ.perm.map (juliaGet s.V)).map Neg.neg := by
      simp only [List.map_map]
      exact List.map_congr_left (fun k _ => juliaGet_neg s.V k)
    rw [heq, List.pairwise_map]
    exact hW.imp (fun hab => by simpa [SortedPerm.ordered] using hab)
  have := dot_le_greedy A σ (-s.V) hW' hq
  rw [dot_neg, dot_neg] at this
  linarith

/-- O-maximization is attained: `omax` returns `⟨p, V⟩` for the greedy distribution
`p = lower + allocation`, and `p ∈ P(l, u)`. Holds for both `upper_bound` values (all modes).

Julia counterpart: `state_action_bellman(::DenseIntervalOMaxWorkspace, …)` and `gap_value(V, gap,
budget, perm)` (`src/bellman.jl`). -/
theorem omax_mem (A : IntervalAmbiguity (Fin n)) (s : SortedPerm n) :
    greedy A (stablePermutation s) ∈ A.toSet ∧
      omax A s = dot s.V (greedy A (stablePermutation s)) :=
  ⟨greedy_mem A _, stateActionBellman_eq_dot A (stablePermutation s) s.V⟩

/-- For `upper_bound = true` (`rev = true`, `V` sorted descending) dense O-maximization computes
the exact maximum: `omax = sup {⟨p, V⟩ : p ∈ P(l, u)}`. Julia sets
`upper_bound = isoptimistic(spec)` (`src/robust_value_iteration.jl`), so this covers the optimistic
satisfaction mode with both strategy modes (maximize / minimize act after O-max, on the actions).

Julia counterpart: `bellman_precomputation!` + `state_action_bellman(::DenseIntervalOMaxWorkspace,
…)` with `upper_bound = true` (`src/bellman.jl`). -/
theorem omax_eq_sSup (A : IntervalAmbiguity (Fin n)) (s : SortedPerm n)
    (h : s.upperBound = true) : omax A s = sSup (valueSet A s.V) :=
  ((stateActionBellman_isGreatest A (stablePermutation s) h).csSup_eq).symm

/-- For `upper_bound = false` (`rev = false`, `V` sorted ascending) dense O-maximization computes
the exact minimum: `omax = inf {⟨p, V⟩ : p ∈ P(l, u)}`. This covers the pessimistic satisfaction
mode (`upper_bound = isoptimistic(spec) = false`, `src/robust_value_iteration.jl`) with both
strategy modes.

Julia counterpart: `bellman_precomputation!` + `state_action_bellman(::DenseIntervalOMaxWorkspace,
…)` with `upper_bound = false` (`src/bellman.jl`). -/
theorem omax_eq_sInf (A : IntervalAmbiguity (Fin n)) (s : SortedPerm n)
    (h : s.upperBound = false) : omax A s = sInf (valueSet A s.V) :=
  ((stateActionBellman_isLeast A (stablePermutation s) h).csInf_eq).symm

/-- The O-max value does not depend on how ties in `V` are ordered: any two permutation vectors
that sort `V` in the direction `rev = upper_bound` (e.g. a stable and an unstable `sortperm!`)
give the same `state_action_bellman` value. Holds for both `upper_bound` values.

Julia counterpart: `sortperm!(permutation(workspace), V; rev = upper_bound)` in
`bellman_precomputation!` and `state_action_bellman(::DenseIntervalOMaxWorkspace, …)`
(`src/bellman.jl`). -/
theorem omax_tie_invariant (A : IntervalAmbiguity (Fin n)) {s : SortedPerm n}
    (σ τ : Permutation s) :
    stateActionBellman A s.V σ.perm = stateActionBellman A s.V τ.perm := by
  cases h : s.upperBound
  · exact (stateActionBellman_isLeast A σ h).unique (stateActionBellman_isLeast A τ h)
  · exact (stateActionBellman_isGreatest A σ h).unique (stateActionBellman_isGreatest A τ h)

/-! ### Sparse O-maximization -/

/-- An interval ambiguity set `P(l, u)` whose gap `u - l` is stored as one column of a sparse
(CSC) matrix: the dense mathematical object `IntervalAmbiguity` together with its stored gap column
`gapCol` and the invariant that `gap[i]` is the stored value of row `i` (`0` off the support).
`lower` is kept as a function: `dot(V, lower)` of a sparse `lower` is the same sum.

Julia counterpart: `IntervalAmbiguitySet{R, <:SparseColumnView{R}}`, the column
`marginal[jₐ, jₛ]` of a sparse `IntervalAmbiguitySets`
(`src/probabilities/IntervalAmbiguitySets.jl`), with `support(ambiguity_set) = rowvals(gap)`. -/
structure SparseIntervalAmbiguity (n : ℕ) extends IntervalAmbiguity (Fin n) where
  /-- The stored gap column (Julia `gap(ambiguity_set)`, a `SparseColumnView`; its `rowvals` are
  `support(ambiguity_set)` and its `nonzeros` the stored gaps). -/
  gapCol : SparseCol n
  /-- The stored column is the gap `u - l`: `gap[i] = getindex(gap(ambiguity_set), i)`. -/
  gap_eq : toIntervalAmbiguity.gap = gapCol.getindex

/-- A result `Vp` that `sort!(Vp_workspace; rev = upper_bound, by = first)` may produce from the
pairs `zip(V[support], nonzeros(gap))` (`valuesGaps`) with any sorting algorithm: `Vp` is a
permutation of these pairs and their first components `V[i]` are descending (`upper_bound = true`)
or ascending (`upper_bound = false`). Pairs with tied `V[i]` may be in any order. Julia's default
`sort!` is stable (since Julia 1.9); its result is the instance `stableValuesGaps`.

Julia counterpart: `workspace.values_gaps[1:supportsize(ambiguity_set)]` after the `zip` loop and
`sort!` in `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` (`src/bellman.jl`). -/
structure SortedValuesGaps (s : SortedPerm n) (c : SparseCol n) where
  /-- The sorted pairs `(V[i], gap[i])` (Julia `Vp_workspace` after `sort!`). -/
  Vp : List (ℝ × ℝ)
  /-- `Vp` is a permutation of `zip(V[support], nonzeros(gap))`. -/
  perm : Vp.Perm (valuesGaps s.V c)
  /-- `first.(Vp)` is sorted in direction `rev = upper_bound`. -/
  sorted : (Vp.map Prod.fst).Pairwise (SortedPerm.ordered s.upperBound)

/-- The Boolean comparison of two pairs `(v, p)` by their first component in direction `rev`:
`a` may precede `b` iff `b.1 ≤ a.1` (`rev = true`) or `a.1 ≤ b.1` (`rev = false`).

Julia counterpart: the order `rev = upper_bound, by = first` of `sort!(Vp_workspace; …)` in
`state_action_bellman(::SparseIntervalOMaxWorkspace, …)` (`src/bellman.jl`). -/
noncomputable def pairLe (rev : Bool) (a b : ℝ × ℝ) : Bool :=
  if rev then decide (b.1 ≤ a.1) else decide (a.1 ≤ b.1)

/-- The stable sort of `zip(V[support], nonzeros(gap))` by first component (`mergeSort` with
`pairLe`), as a `SortedValuesGaps`; this shows the structure is inhabited for every input.

Julia counterpart: `sort!(Vp_workspace; rev = upper_bound, by = first, scratch = …)` (stable
default algorithm) in `state_action_bellman(::SparseIntervalOMaxWorkspace, …)`
(`src/bellman.jl`). -/
noncomputable def stableValuesGaps (s : SortedPerm n) (c : SparseCol n) :
    SortedValuesGaps s c where
  Vp := (valuesGaps s.V c).mergeSort (pairLe s.upperBound)
  perm := List.mergeSort_perm _ _
  sorted := by
    rw [List.pairwise_map]
    have hsort := List.pairwise_mergeSort (le := pairLe s.upperBound) ?trans ?total
      (valuesGaps s.V c)
    · refine hsort.imp (fun {a b} hab => ?_)
      unfold pairLe at hab
      unfold SortedPerm.ordered
      split_ifs at hab ⊢ <;> simpa using hab
    case trans =>
      intro a b c hab hbc
      unfold pairLe at *
      split_ifs at * <;> simp only [decide_eq_true_eq] at * <;> linarith
    case total =>
      intro a b
      unfold pairLe
      split_ifs <;> simp only [Bool.or_eq_true, decide_eq_true_eq] <;> exact le_total _ _

/-- Literal transcription of the loop of `gap_value(Vp, budget)`: for each pair `(V, p)` of `Vp`
(in order), `p = min(budget, p)`, `res += p * V`, `budget -= p`, and `break` once `budget <= 0`;
the result is `res`. Julia starts with `res = zero(T)`, i.e. `gapValueSparse Vp budget 0`.

Julia counterpart: `gap_value(Vp::VP, budget)` (`src/bellman.jl`, lines 555–571). -/
noncomputable def gapValueSparse : List (ℝ × ℝ) → ℝ → ℝ → ℝ
  | [], _, res => res
  | (V, p) :: Vp, budget, res =>
    let p := min budget p
    let res := res + p * V
    let budget := budget - p
    if budget ≤ 0 then res else gapValueSparse Vp budget res

/-- Transcription of `state_action_bellman(::SparseIntervalOMaxWorkspace, V, ambiguity_set, budget,
upper_bound)`: `dot(V, lower(ambiguity_set)) + gap_value(Vp_workspace, budget)`, where
`Vp_workspace = σ.Vp` is `zip(V[support], nonzeros(gap))` after `sort!(…; rev = upper_bound,
by = first)` and `budget = 1 - ∑ lower` (precomputed in the workspace).

Julia counterpart: `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` and
`gap_value(Vp, budget)` (`src/bellman.jl`, lines 536–571). -/
noncomputable def omaxSparse (A : SparseIntervalAmbiguity n) {s : SortedPerm n}
    (σ : SortedValuesGaps s A.gapCol) : ℝ :=
  dot s.V A.lower + gapValueSparse σ.Vp A.budget 0

/-- A list that is a permutation of `l.map f` is the image under `f` of a permutation of `l`.

Julia counterpart: none (Lean-side proof device). -/
theorem exists_perm_map_eq {α β : Type*} [DecidableEq α] (f : α → β) :
    ∀ (l₁ : List β) (l : List α), l₁.Perm (l.map f) → ∃ l' : List α, l'.Perm l ∧ l'.map f = l₁
  | [], l, h => ⟨[], by simpa using h.symm, rfl⟩
  | b :: t, l, h => by
    have hb : b ∈ l.map f := h.subset List.mem_cons_self
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hb
    have hl : l.Perm (x :: l.erase x) := List.perm_cons_erase hx
    have h' : (f x :: t).Perm (f x :: (l.erase x).map f) := h.trans (by simpa using hl.map f)
    obtain ⟨t', ht', rfl⟩ := exists_perm_map_eq f t (l.erase x) (List.Perm.cons_inv h')
    exact ⟨x :: t', (ht'.cons x).trans hl.symm, rfl⟩

/-- The sparse loop over the pairs `(V[i], gap[i])` of rows `L` is the dense loop `gap_value(V, gap,
budget, perm)` along the Julia indices of `L`, with the column's `getindex` as gap.

Julia counterpart: `gap_value(Vp, budget)` vs `gap_value(V, gap, budget, perm)`
(`src/bellman.jl`). -/
theorem gapValueSparse_map (V : Fin n → ℝ) (c : SparseCol n) :
    ∀ (L : List (Fin n)) (b res : ℝ),
      gapValueSparse (L.map (c.valueGap V)) b res = gapValue V c.getindex (L.map toJulia) b res
  | [], _, _ => rfl
  | i :: L, b, res => by
    simp only [List.map_cons, gapValueSparse, gapValue, SparseCol.valueGap, juliaGet_toJulia]
    split_ifs
    · rfl
    · exact gapValueSparse_map V c L _ _

/-- With budget `0` and nonnegative gaps, the dense loop returns `res` unchanged.

Julia counterpart: none (Lean-side proof device). -/
theorem gapValue_zero_budget (V : Fin n → ℝ) {gap : Fin n → ℝ} (h : ∀ i, 0 ≤ gap i) :
    ∀ (l : List ℕ) (res : ℝ), gapValue V gap l 0 res = res
  | [], _ => rfl
  | i :: _, res => by
    simp [gapValue, min_eq_left (juliaGet_nonneg h i)]

/-- Inserting zero-gap indices into the loop order does not change `gap_value`: if `L` is a
sublist of the duplicate-free `M` and every index of `M` outside `L` has gap `0`, the loops along
`M` and `L` agree for every nonnegative budget.

Julia counterpart: none (Lean-side proof device). -/
theorem gapValue_sublist (V : Fin n → ℝ) {gap : Fin n → ℝ} (h : ∀ i, 0 ≤ gap i) {L M : List ℕ}
    (hLM : L.Sublist M) (hnd : M.Nodup) (hz : ∀ k ∈ M, k ∉ L → juliaGet gap k = 0) :
    ∀ b res, 0 ≤ b → gapValue V gap M b res = gapValue V gap L b res := by
  induction hLM with
  | slnil => intro b res _; rfl
  | @cons L' M' a hsub ih =>
    intro b res hb
    rw [List.nodup_cons] at hnd
    have ha : juliaGet gap a = 0 :=
      hz a List.mem_cons_self (fun haL => hnd.1 (hsub.subset haL))
    have hz' : ∀ k ∈ M', k ∉ L' → juliaGet gap k = 0 :=
      fun k hk hkL => hz k (List.mem_cons_of_mem a hk) hkL
    simp only [gapValue, ha, min_eq_right hb, zero_mul, add_zero, sub_zero]
    split_ifs with hb0
    · rw [le_antisymm hb0 hb, gapValue_zero_budget V h]
    · exact ih hnd.2 hz' b res hb
  | @cons_cons L' M' a hsub ih =>
    intro b res hb
    rw [List.nodup_cons] at hnd
    have hz' : ∀ k ∈ M', k ∉ L' → juliaGet gap k = 0 :=
      fun k hk hkL => hz k (List.mem_cons_of_mem a hk) (fun h' => hkL ?_)
    · simp only [gapValue]
      split_ifs
      · rfl
      · exact ih hnd.2 hz' _ _ (budget_sub_min_nonneg _ _)
    · rcases List.mem_cons.1 h' with rfl | h'
      · exact absurd hk hnd.1
      · exact h'

/-- The comparison `SortedPerm.le` of `sortperm!` is transitive.

Julia counterpart: none (Lean-side proof device). -/
theorem sortedLe_trans (s : SortedPerm n) (a b c : Fin n) :
    s.le a b → s.le b c → s.le a c := by
  intro hab hbc
  unfold SortedPerm.le at *
  split_ifs at * <;> simp only [decide_eq_true_eq] at * <;> linarith

/-- The comparison `SortedPerm.le` of `sortperm!` is total.

Julia counterpart: none (Lean-side proof device). -/
theorem sortedLe_total (s : SortedPerm n) (a b : Fin n) : s.le a b || s.le b a := by
  unfold SortedPerm.le
  split_ifs <;> simp only [Bool.or_eq_true, decide_eq_true_eq] <;> exact le_total _ _

/-- A duplicate-free list of rows `L` sorted by `V` in direction `upper_bound` is a sublist of the
Julia indices of some full sorted permutation vector of `1:n` (stable `mergeSort` of `L` followed
by the remaining rows).

Julia counterpart: none (Lean-side proof device). -/
theorem exists_permutation_sublist (s : SortedPerm n) {L : List (Fin n)} (hnd : L.Nodup)
    (hsort : (L.map s.V).Pairwise (SortedPerm.ordered s.upperBound)) :
    ∃ σ : Permutation s, (L.map toJulia).Sublist σ.perm := by
  set base := L ++ (List.finRange n).filter (fun i => i ∉ L) with hbase
  have hperm : base.Perm (List.finRange n) := by
    rw [List.perm_ext_iff_of_nodup]
    · intro i
      simp only [hbase, List.mem_append, List.mem_filter, List.mem_finRange, true_and,
        decide_eq_true_eq]
      exact ⟨fun _ => trivial, fun _ => by tauto⟩
    · refine List.Nodup.append hnd ((List.nodup_finRange n).filter _) ?_
      intro a ha hb
      simp only [List.mem_filter, decide_eq_true_eq] at hb
      exact hb.2 ha
    · exact List.nodup_finRange n
  set M := base.mergeSort s.le with hM
  have hLle : L.Pairwise (fun a b => s.le a b = true) := by
    rw [List.pairwise_map] at hsort
    refine hsort.imp (fun {a b} hab => ?_)
    unfold SortedPerm.ordered at hab
    unfold SortedPerm.le
    split_ifs at hab ⊢ <;> simpa using hab
  have hsub : L.Sublist M :=
    List.sublist_mergeSort (sortedLe_trans s) (sortedLe_total s) hLle
      (List.sublist_append_left _ _)
  have hMsort := List.pairwise_mergeSort (le := s.le) (sortedLe_trans s) (sortedLe_total s) base
  refine ⟨⟨M.map toJulia, ?_, ?_⟩, hsub.map toJulia⟩
  · rw [← map_toJulia_finRange]
    exact ((List.mergeSort_perm _ _).trans hperm).map toJulia
  · have hcomp : juliaGet s.V ∘ toJulia = s.V := funext (juliaGet_toJulia s.V)
    rw [List.map_map, hcomp, List.pairwise_map]
    refine hMsort.imp (fun {a b} hab => ?_)
    unfold SortedPerm.le at hab
    unfold SortedPerm.ordered
    split_ifs at hab ⊢ <;> simpa using hab

/-- Sparse O-maximization equals dense O-maximization: for a sparse interval ambiguity set (CSC
gap column; rows outside the support have gap `0`) and any result of `sort!(…; rev = upper_bound,
by = first)` (stable or not), `omaxSparse = omax`. Holds for both `upper_bound = isoptimistic(spec)`
values, so all four satisfaction × strategy modes; with `omax_eq_sSup` / `omax_eq_sInf` it is the
exact maximum / minimum of `⟨p, V⟩` over `P(l, u)`.

Julia counterpart: `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` and
`gap_value(Vp, budget)` vs `state_action_bellman(::DenseIntervalOMaxWorkspace, …)`
(`src/bellman.jl`). -/
theorem omaxSparse_eq_omax (A : SparseIntervalAmbiguity n) (s : SortedPerm n)
    (σ : SortedValuesGaps s A.gapCol) :
    omaxSparse A σ = omax A.toIntervalAmbiguity s := by
  obtain ⟨hzip, hzero⟩ := sparse_zip_correct A.gapCol s.V
  have hp := σ.perm
  rw [hzip] at hp
  obtain ⟨L, hL, hLVp⟩ := exists_perm_map_eq (A.gapCol.valueGap s.V) σ.Vp _ hp
  have hsupp_nd : A.gapCol.supportRows.Nodup := (List.nodup_finRange n).filter _
  have hLnd : L.Nodup := hL.nodup_iff.2 hsupp_nd
  have hLsort : (L.map s.V).Pairwise (SortedPerm.ordered s.upperBound) := by
    have h := σ.sorted
    rw [← hLVp, List.map_map] at h
    exact h
  obtain ⟨τ, hτ⟩ := exists_permutation_sublist s hLnd hLsort
  have hz : ∀ k ∈ τ.perm, k ∉ L.map toJulia → juliaGet A.gap k = 0 := by
    intro k hk hkL
    have hkr : k ∈ List.range' 1 n := τ.perm_perm.subset hk
    rw [← map_toJulia_finRange] at hkr
    obtain ⟨i, -, rfl⟩ := List.mem_map.1 hkr
    rw [juliaGet_toJulia, A.gap_eq]
    apply hzero
    intro hi
    apply hkL
    have : i ∈ A.gapCol.supportRows := by
      simp only [SparseCol.supportRows, List.mem_filter, List.mem_finRange, true_and,
        decide_eq_true_eq]
      exact hi
    exact List.mem_map_of_mem (hL.symm.subset this)
  rw [omaxSparse, ← hLVp, gapValueSparse_map, ← A.gap_eq,
    ← gapValue_sublist s.V A.gap_nonneg hτ τ.nodup hz _ _ A.budget_eq.2]
  exact omax_tie_invariant A.toIntervalAmbiguity τ (stablePermutation s)

/-- Sparse O-maximization is exact: for `upper_bound = true` it is `sup {⟨p, V⟩ : p ∈ P(l, u)}`
and for `upper_bound = false` it is `inf {⟨p, V⟩ : p ∈ P(l, u)}` (all four modes).

Julia counterpart: `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` (`src/bellman.jl`). -/
theorem omaxSparse_exact (A : SparseIntervalAmbiguity n) (s : SortedPerm n)
    (σ : SortedValuesGaps s A.gapCol) :
    (s.upperBound = true → omaxSparse A σ = sSup (valueSet A.toIntervalAmbiguity s.V)) ∧
      (s.upperBound = false → omaxSparse A σ = sInf (valueSet A.toIntervalAmbiguity s.V)) := by
  rw [omaxSparse_eq_omax]
  exact ⟨omax_eq_sSup _ s, omax_eq_sInf _ s⟩

end IntervalMDP.OMax
