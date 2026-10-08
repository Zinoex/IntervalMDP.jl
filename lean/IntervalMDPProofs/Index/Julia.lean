import Mathlib

/-!
# Julia's 1-based indices

Julia arrays are 1-based; Lean's `Fin n` is 0-based. This file defines the conversion **once**:
`toJulia : Fin n → ℕ := (· + 1)` maps the Lean index `i` to the Julia index `i + 1`, and
`ofJulia` is its inverse on the Julia index range `1..n` (`juliaRange n`). Every index theorem of
the indexing layer (`Index/Linear.lean`, `Index/Sparse.lean`, `Index/Perm.lean`) is stated through
`toJulia`.

It also models Julia's fixed-width integers (`Int32`, `Int64`): `machineInt N x` is the value an
`N`-bit two's-complement integer holds after computing the natural number `x` with wrap-around
`+`/`*`. Index theorems carry the overflow bound `x < 2 ^ (N - 1)` as a hypothesis and conclude
`machineInt N x = x`.
-/

namespace IntervalMDP.Index

/-- The Julia (1-based) index of the Lean (0-based) index `i : Fin n`: `toJulia i = i + 1`.

Julia counterpart: every 1-based integer index into a Julia array, e.g. `V[i]`, `gap[i]` and
`perm[k]` in `gap_value` (`src/bellman.jl`). -/
def toJulia {n : ℕ} (i : Fin n) : ℕ := i.val + 1

/-- The Julia index range `1..n` of a vector of length `n` (Julia `1:n`, `eachindex(V)`).

Julia counterpart: `eachindex(V)` / `1:length(V)` for the value vector `V` (`src/bellman.jl`). -/
def juliaRange (n : ℕ) : Set ℕ := Set.Icc 1 n

/-- The Lean index of a Julia index `k ∈ 1..n`: `ofJulia k = k - 1`.

Julia counterpart: a valid (in-bounds) 1-based index `k` into a vector of length `n`, as used in
`gap_value` (`src/bellman.jl`). -/
def ofJulia {n : ℕ} (k : ℕ) (hk : k ∈ juliaRange n) : Fin n :=
  ⟨k - 1, by obtain ⟨h1, h2⟩ := hk; omega⟩

/-- Julia's `V[k]` for a 1-based index `k` into a vector `V` of length `n`; out-of-range indices
(where Julia throws a `BoundsError`) read as `0`. Under the invariants of the indexing layer every
index used is in range.

Julia counterpart: `getindex(V, k)`, e.g. `V[i]` and `gap[i]` in `gap_value` and
`V[support(ambiguity_set)]` in `state_action_bellman` (`src/bellman.jl`). -/
def juliaGet {n : ℕ} (V : Fin n → ℝ) (k : ℕ) : ℝ :=
  if hk : 1 ≤ k ∧ k ≤ n then V (ofJulia k hk) else 0

/-- The value an `N`-bit two's-complement integer (Julia `Int32` for `N = 32`, `Int` = `Int64`
for `N = 64`) holds after computing the natural number `x`. Julia's integer `+`, `-` and `*` wrap
around modulo `2 ^ N`, so any expression built from them evaluates to `x` reduced into the signed
range `[-2 ^ (N - 1), 2 ^ (N - 1))`, which is `Int.bmod x (2 ^ N)`.

Julia counterpart: `Int32`/`Int` index arithmetic, e.g. the `Vector{Int32}` permutation of the
O-max workspace (`src/workspace.jl`) and `LinearIndices` products. -/
def machineInt (N : ℕ) (x : ℕ) : ℤ := Int.bmod x (2 ^ N)

/-- `toJulia` lands in the Julia index range `1..n`.

Julia counterpart: none (Lean-side proof device). -/
theorem toJulia_mem {n : ℕ} (i : Fin n) : toJulia i ∈ juliaRange n := by
  simp only [toJulia, juliaRange, Set.mem_Icc]; omega

/-- `ofJulia` is a left inverse of `toJulia`.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem ofJulia_toJulia {n : ℕ} (i : Fin n) : ofJulia (toJulia i) (toJulia_mem i) = i := by
  ext; simp [ofJulia, toJulia]

/-- `toJulia` is a left inverse of `ofJulia` on `1..n`.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem toJulia_ofJulia {n : ℕ} (k : ℕ) (hk : k ∈ juliaRange n) :
    toJulia (ofJulia k hk) = k := by
  obtain ⟨h1, _⟩ := hk; simp [ofJulia, toJulia]; omega

/-- `toJulia` is injective.

Julia counterpart: none (Lean-side proof device). -/
theorem toJulia_injective {n : ℕ} : Function.Injective (toJulia : Fin n → ℕ) := by
  intro i j h; ext; simp only [toJulia] at h; omega

/-- `toJulia` is a bijection from `Fin n` onto the Julia index range `1..n`.

Julia counterpart: none (Lean-side proof device). -/
theorem toJulia_bijOn {n : ℕ} : Set.BijOn (toJulia : Fin n → ℕ) Set.univ (juliaRange n) :=
  ⟨fun i _ => toJulia_mem i, toJulia_injective.injOn,
    fun k hk => ⟨ofJulia k hk, trivial, toJulia_ofJulia k hk⟩⟩

/-- Reading `V` at the Julia index `toJulia i` gives `V i`.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem juliaGet_toJulia {n : ℕ} (V : Fin n → ℝ) (i : Fin n) :
    juliaGet V (toJulia i) = V i := by
  have hk : 1 ≤ toJulia i ∧ toJulia i ≤ n := toJulia_mem i
  simp only [juliaGet, hk, and_self, dite_true]
  exact congrArg V (ofJulia_toJulia i)

/-- The list of Julia indices `[toJulia 0, …, toJulia (n - 1)]` is `1:n` (`List.range' 1 n`).

Julia counterpart: none (Lean-side proof device). -/
theorem map_toJulia_finRange (n : ℕ) : (List.finRange n).map toJulia = List.range' 1 n := by
  apply List.ext_getElem <;> simp [toJulia, Nat.add_comm]

/-- No overflow: if `x < 2 ^ (N - 1)` then the `N`-bit integer holds exactly `x`.

Julia counterpart: none (Lean-side proof device). -/
theorem machineInt_eq_self {N x : ℕ} (hx : x < 2 ^ (N - 1)) : machineInt N x = x := by
  unfold machineInt
  rcases N with _ | N
  · simp at hx; subst hx; simp
  · simp only [Nat.add_sub_cancel] at hx
    have hx' : (x : ℤ) < 2 ^ N := by exact_mod_cast hx
    apply Int.bmod_eq_of_le
    · have : (0 : ℤ) ≤ x := Int.natCast_nonneg x
      push_cast
      have : (0 : ℤ) ≤ 2 ^ (N + 1) / 2 := by positivity
      omega
    · push_cast
      rw [pow_succ]
      omega

end IntervalMDP.Index
