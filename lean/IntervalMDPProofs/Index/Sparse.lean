import IntervalMDPProofs.Index.Julia

/-!
# Sparse support pairing

A column of a `SparseMatrixCSC` (here: one column `gap(ambiguity_set)` of the sparse gap matrix of
`IntervalAmbiguitySets`) is stored as the 1-based row indices of its stored entries (`rowvals`,
which is `support(ambiguity_set)`) and their values (`nonzeros`). The sparse O-max Bellman update
`state_action_bellman(::SparseIntervalOMaxWorkspace, …)` in `src/bellman.jl` builds

    for (i, (v, p)) in enumerate(zip(V[support(ambiguity_set)], nonzeros(gap(ambiguity_set))))
        Vp_workspace[i] = (v, p)
    end

`sparse_zip_correct` proves that under the CSC column invariant (`SparseCol`: strictly increasing
row indices in range, `nzval` aligned to `rowval`) this pairs `V[i]` with `gap[i]` for exactly the
support rows, in increasing row order, and that every row outside the support has `gap[i] = 0`.
-/

namespace IntervalMDP.Index

/-- One column of a `SparseMatrixCSC` with `m` rows, with the CSC column invariant as fields:
the stored row indices `rowval` (1-based) are strictly increasing and lie in `1..m`, and the stored
values `nzval` are aligned to them (same length, `nzval[k]` belongs to row `rowval[k]`).

Julia counterpart: a column view `gap(ambiguity_set)` of the sparse gap matrix of
`IntervalAmbiguitySets` (`src/probabilities/IntervalAmbiguitySets.jl`), as read by
`state_action_bellman(::SparseIntervalOMaxWorkspace, …)` (`src/bellman.jl`). -/
structure SparseCol (m : ℕ) where
  /-- The 1-based row indices of the stored entries (Julia `rowvals(gap)` of the column, which is
  `support(ambiguity_set)`). -/
  rowval : List ℕ
  /-- The stored values, aligned to `rowval` (Julia `nonzeros(gap)` of the column). -/
  nzval : List ℝ
  /-- `nzval` is aligned to `rowval`: one stored value per stored row. -/
  length_eq : nzval.length = rowval.length
  /-- The row indices are strictly increasing (CSC invariant). -/
  rowval_sorted : rowval.Pairwise (· < ·)
  /-- Every row index is a valid Julia row index `1..m`. -/
  rowval_mem : ∀ r ∈ rowval, r ∈ juliaRange m

namespace SparseCol

variable {m : ℕ}

/-- The value `gap[i]` of the column at the row with Lean index `i` (Julia row `toJulia i`): the
stored value of that row, or `0` if the row is not stored.

Julia counterpart: `getindex` of a sparse column, `gap(ambiguity_set, i)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
def getindex (c : SparseCol m) (i : Fin m) : ℝ :=
  ((c.rowval.zip c.nzval).lookup (toJulia i)).getD 0

/-- The support rows of the column (rows whose Julia index is stored), in increasing order.

Julia counterpart: `support(ambiguity_set)` = `rowvals(gap)` of the column
(`src/probabilities/IntervalAmbiguitySets.jl`), as Lean indices. -/
def supportRows (c : SparseCol m) : List (Fin m) :=
  (List.finRange m).filter (fun i => toJulia i ∈ c.rowval)

/-- The pair `(V[i], gap[i])` of the value and the gap at row `i`.

Julia counterpart: an entry `(v, p)` of `workspace.values_gaps` in
`state_action_bellman(::SparseIntervalOMaxWorkspace, …)` (`src/bellman.jl`). -/
def valueGap (c : SparseCol m) (V : Fin m → ℝ) (i : Fin m) : ℝ × ℝ := (V i, c.getindex i)

end SparseCol

/-- The list `zip(V[support(ambiguity_set)], nonzeros(gap(ambiguity_set)))` that the sparse O-max
update writes into `Vp_workspace`; `V[support]` reads `V` at the stored Julia row indices.

Julia counterpart: the `zip` loop in `state_action_bellman(::SparseIntervalOMaxWorkspace, …)`
(`src/bellman.jl`). -/
def valuesGaps {m : ℕ} (V : Fin m → ℝ) (c : SparseCol m) : List (ℝ × ℝ) :=
  (c.rowval.map (juliaGet V)).zip c.nzval

/-- A key that does not occur in `ks` is not found in `ks.zip vs`.

Julia counterpart: none (Lean-side proof device). -/
theorem lookup_zip_eq_none {k : ℕ} {ks : List ℕ} (vs : List ℝ) (hk : k ∉ ks) :
    (ks.zip vs).lookup k = none := by
  rw [List.lookup_eq_none_iff]
  intro p hp
  have : p.1 ∈ ks := (List.of_mem_zip hp).1
  simp only [bne_iff_ne, ne_eq]
  rintro rfl
  exact hk this

/-- Zipping values along a list of rows with distinct Julia indices pairs each row with its
looked-up stored value.

Julia counterpart: none (Lean-side proof device). -/
theorem zip_map_eq {m : ℕ} (V : Fin m → ℝ) :
    ∀ (L : List (Fin m)) (nz : List ℝ), (L.map toJulia).Nodup → nz.length = L.length →
      (L.map V).zip nz = L.map (fun i => (V i, (((L.map toJulia).zip nz).lookup (toJulia i)).getD 0))
  | [], _, _, _ => by simp
  | _ :: _, [], _, h => by simp at h
  | j :: L, a :: nz, hnd, hlen => by
    simp only [List.map_cons, List.zip_cons_cons, List.lookup_cons_self, Option.getD_some,
      List.cons.injEq, true_and]
    rw [List.map_cons, List.nodup_cons] at hnd
    rw [zip_map_eq V L nz hnd.2 (by simpa using hlen)]
    apply List.map_congr_left
    intro i hi
    have hne : (toJulia i == toJulia j) = false := by
      rw [beq_eq_false_iff_ne]
      intro h
      exact hnd.1 (h ▸ List.mem_map_of_mem hi)
    simp [List.lookup_cons, hne]

/-- Under the CSC invariant, the stored row indices are exactly the Julia indices of the support
rows, in order: `rowval = map toJulia (supportRows c)`.

Julia counterpart: none (Lean-side proof device). -/
theorem SparseCol.rowval_eq {m : ℕ} (c : SparseCol m) :
    c.rowval = c.supportRows.map toJulia := by
  apply List.Pairwise.eq_of_mem_iff c.rowval_sorted
  · rw [List.pairwise_map]
    refine List.Pairwise.filter _ ?_
    refine List.Pairwise.imp ?_ (List.pairwise_lt_finRange m)
    intro a b hab
    simp only [toJulia]
    exact Nat.add_lt_add_right hab 1
  · intro r
    simp only [SparseCol.supportRows, List.mem_map, List.mem_filter, List.mem_finRange, true_and,
      decide_eq_true_eq]
    constructor
    · intro hr
      exact ⟨ofJulia r (c.rowval_mem r hr), by simpa using hr, by simp⟩
    · rintro ⟨i, hi, rfl⟩
      exact hi

/-- Sparse support pairing: `zip(V[support], nonzeros(gap))` is the list of pairs
`(V[i], gap[i])` over exactly the support rows `i` (in increasing order), and every row outside the
support has `gap[i] = 0`.

Julia counterpart: the `zip` loop of `state_action_bellman(::SparseIntervalOMaxWorkspace, …)`
(`src/bellman.jl`). -/
theorem sparse_zip_correct {m : ℕ} (c : SparseCol m) (V : Fin m → ℝ) :
    valuesGaps V c = c.supportRows.map (c.valueGap V) ∧
      ∀ i, toJulia i ∉ c.rowval → c.getindex i = 0 := by
  refine ⟨?_, fun i hi => ?_⟩
  · have hnd : (c.supportRows.map toJulia).Nodup := c.rowval_eq ▸ c.rowval_sorted.nodup
    have hlen : c.nzval.length = c.supportRows.length := by
      rw [c.length_eq, c.rowval_eq, List.length_map]
    have h := zip_map_eq V c.supportRows c.nzval hnd hlen
    simp only [valuesGaps]
    rw [c.rowval_eq, List.map_map]
    have hcomp : juliaGet V ∘ toJulia = V := funext (juliaGet_toJulia V)
    rw [hcomp, h]
    refine List.map_congr_left (fun i _ => ?_)
    simp only [SparseCol.valueGap, SparseCol.getindex, ← c.rowval_eq]
  · simp [SparseCol.getindex, lookup_zip_eq_none c.nzval hi]

end IntervalMDP.Index
