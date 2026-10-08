import IntervalMDPProofs.Index.Julia

/-!
# Column-major linear indices

Julia stores multi-dimensional arrays in column-major order: for a dims vector
`dims = (n₁, …, n_d)`, `LinearIndices(dims)[I₁, …, I_d] = 1 + ∑ₖ (Iₖ - 1) * strideₖ` with
`strideₖ = n₁ * ⋯ * nₖ₋₁`, and `CartesianIndices(dims)` is its inverse. Factored states, the value
arrays `V[I]` and the update sequences of IntervalMDP.jl use this layout.

`linear` is that formula, with the 1-based coordinates `Iₖ = toJulia (I k)`. We prove that it is
a bijection from the Cartesian indices onto `1..∏ dims` (`linear_bijective`) and that the first
dimension varies fastest (`linear_succ_first`). Julia evaluates the formula in `Int`; both theorems
carry the overflow bound `∏ dims < 2 ^ (N - 1)` for `N`-bit integers and also state that the
`N`-bit value is exact.
-/

namespace IntervalMDP.Index

open Finset

/-- A Cartesian index into an array of size `dims`: one coordinate `I k : Fin (dims k)` per
dimension (Julia coordinate `toJulia (I k)`).

Julia counterpart: `CartesianIndex` ranging over `CartesianIndices(dims)`, e.g. the source states of
`FullUpdateSequence` (`src/update_sequence.jl`) and `V[I]` in `src/bellman.jl`. -/
abbrev CartesianIndex {d : ℕ} (dims : Fin d → ℕ) : Type := (k : Fin d) → Fin (dims k)

/-- The column-major stride of dimension `k`: `stride k = dims 1 * ⋯ * dims (k - 1)`, the product
of the sizes of all earlier dimensions.

Julia counterpart: the stride `L` accumulated by `Base._sub2ind` behind `LinearIndices(dims)`, as
used for the value arrays `V[I]` (`src/bellman.jl`). -/
def stride {d : ℕ} (dims : Fin d → ℕ) (k : Fin d) : ℕ :=
  ∏ j : Fin k, dims (Fin.castLE k.is_lt.le j)

/-- The 1-based column-major linear index of the Cartesian index `I`:
`linear dims I = 1 + ∑ₖ (Iₖ - 1) * stride k`, with the Julia coordinates `Iₖ = toJulia (I k)`.

Julia counterpart: `LinearIndices(dims)[I]` (inverse `CartesianIndices(dims)`), used for the value
arrays `V[I]` and the states of `FullUpdateSequence` (`src/bellman.jl`, `src/update_sequence.jl`). -/
def linear {d : ℕ} (dims : Fin d → ℕ) (I : CartesianIndex dims) : ℕ :=
  1 + ∑ k, (toJulia (I k) - 1) * stride dims k

/-- The value of `LinearIndices(dims)[I]` computed with `N`-bit integers (Julia `Int`, `N = 64`;
`Int32` index paths, `N = 32`).

Julia counterpart: `LinearIndices(dims)[I]` evaluated in `Int` arithmetic (`src/bellman.jl`). -/
def linearInt (N : ℕ) {d : ℕ} (dims : Fin d → ℕ) (I : CartesianIndex dims) : ℤ :=
  machineInt N (linear dims I)

/-- The Cartesian index `I` with its first coordinate incremented:
`CartesianIndex(I[1] + 1, I[2], …, I[d])`; requires `I[1] + 1 ≤ dims[1]`.

Julia counterpart: the next index of `CartesianIndices(dims)` iteration when the first coordinate
is not at its end, e.g. in `FullUpdateSequence` (`src/update_sequence.jl`). -/
def incFirst {d : ℕ} {dims : Fin (d + 1) → ℕ} (I : CartesianIndex dims)
    (h : toJulia (I 0) < dims 0) : CartesianIndex dims :=
  Function.update I 0 ⟨toJulia (I 0), h⟩

/-- `linear` is `toJulia` of Mathlib's mixed-radix equivalence `finPiFinEquiv`.

Julia counterpart: none (Lean-side proof device). -/
theorem linear_eq_toJulia {d : ℕ} (dims : Fin d → ℕ) (I : CartesianIndex dims) :
    linear dims I = toJulia (finPiFinEquiv I) := by
  simp [linear, toJulia, finPiFinEquiv_apply, stride, Nat.add_comm]

/-- Column-major linear indexing is a bijection from the Cartesian indices onto the Julia range
`1..∏ dims`, and if `∏ dims < 2 ^ (N - 1)` its `N`-bit value does not overflow.

Julia counterpart: `LinearIndices(dims)` / `CartesianIndices(dims)` (Base), as used for `V[I]` and
the states of `FullUpdateSequence` (`src/bellman.jl`, `src/update_sequence.jl`). -/
theorem linear_bijective {d N : ℕ} {dims : Fin d → ℕ} (hBound : ∏ k, dims k < 2 ^ (N - 1)) :
    Set.BijOn (linear dims) Set.univ (juliaRange (∏ k, dims k)) ∧
      ∀ I, linearInt N dims I = linear dims I := by
  have hbij : Set.BijOn (linear dims) Set.univ (juliaRange (∏ k, dims k)) := by
    have h := toJulia_bijOn.comp (finPiFinEquiv (n := dims)).bijective.bijOn_univ
    convert h using 1
    funext I
    exact linear_eq_toJulia dims I
  refine ⟨hbij, fun I => machineInt_eq_self ?_⟩
  exact lt_of_le_of_lt (hbij.mapsTo (Set.mem_univ I)).2 hBound

/-- The first dimension varies fastest: incrementing the first coordinate increments the linear
index by one (also for the `N`-bit value, under the overflow bound `∏ dims < 2 ^ (N - 1)`).

Julia counterpart: column-major `LinearIndices(dims)` (Base), as used for `V[I]`
(`src/bellman.jl`). -/
theorem linear_succ_first {d N : ℕ} {dims : Fin (d + 1) → ℕ} (hBound : ∏ k, dims k < 2 ^ (N - 1))
    (I : CartesianIndex dims) (h : toJulia (I 0) < dims 0) :
    linearInt N dims (incFirst I h) = linearInt N dims I + 1 := by
  have hnat : linear dims (incFirst I h) = linear dims I + 1 := by
    have hrest : ∀ k : Fin d, incFirst I h k.succ = I k.succ :=
      fun k => Function.update_of_ne (Fin.succ_ne_zero k) _ _
    have hfirst : incFirst I h 0 = ⟨toJulia (I 0), h⟩ := Function.update_self _ _ _
    have hs0 : stride dims 0 = 1 := by simp [stride]
    simp only [linear, Fin.sum_univ_succ, hrest, hfirst, hs0, toJulia]
    omega
  obtain ⟨_, hInt⟩ := linear_bijective (N := N) hBound
  rw [hInt, hInt, hnat]
  push_cast
  ring

end IntervalMDP.Index
