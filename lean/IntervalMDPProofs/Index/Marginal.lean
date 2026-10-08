import IntervalMDPProofs.Index.Linear
import IntervalMDPProofs.Models.Factored

/-!
# Marginal indexing

A `Marginal` of a factored model stores one interval ambiguity set per value of the variables it
conditions on, as the columns of an `IntervalAmbiguitySets`. `sub2ind(p::Marginal, action,
source)` in `src/probabilities/Marginal.jl` turns the global action and source state into the
column index:

    ind = zero(T)
    for i in StepRange(N1, -1, 1)
        ind *= p.source_dims[i]
        ind += source[p.state_indices[i]] - one(T)
    end
    for i in StepRange(M1, -1, 1)
        ind *= p.action_vars[i]
        ind += action[p.action_indices[i]] - one(T)
    end
    return ind + one(T)

`marginalSub2ind` transcribes this loop. We prove that it is the column-major linear index of
`(action[action_indices]…, source[state_indices]…)` over the dims
`(action_vars…, source_dims…)`, actions first (`marginalSub2ind_eq_linear`), that it is a
bijection between the conditioning tuples and `1..∏ action_vars * ∏ source_dims`
(`marginalSub2ind_bijective`) and that it reads only the conditioning variables
(`marginalSub2ind_depends_only`).

`sub2ind(::IntervalAmbiguitySets, jₐ, jₛ) = jₛ[1]` (`src/probabilities/IntervalAmbiguitySets.jl`)
ignores the action. `intervalSub2ind_correct` proves that it agrees with the `Marginal` layout of a
single-action model that conditions on state variable 1 (the layout built by
`IntervalMarkovDecisionProcess`); `intervalSub2ind_wrong_multiAction` shows that it cannot agree
with any layout in which a conditioning action variable takes two values. No call site in `src/`
or `ext/` reaches it (all go through `Marginal`); see the inventory.

The indices are modeled over the Lean factored model `Marginal sv av i`
(`Models/Factored.lean`): the Julia field `source_dims` of the marginal is
`sv.dims (stateIndices j)` and `action_vars` is `av.dims (actionIndices k)`, which is what
`check_transition` (`src/models/FactoredRobustMarkovDecisionProcess.jl`) enforces. For a model with
terminal slices (`source_dims < state_vars`) take `sv` to be the source box `source_dims`; the
index arithmetic is the same. Julia evaluates the loop in `Int` (CPU) or `Int32` (CUDA); the
`N`-bit value is `marginalSub2indInt`, exact under the bound `∏ dims < 2 ^ (N - 1)`.
-/

namespace IntervalMDP.Index

open Finset

variable {n m : ℕ} {sv : StateVars n} {av : ActionVars m} {i : Fin n}

/-- The marginal's source shape: `source_dims[j]` is the size of conditioning state variable
`state_indices[j]`.

Julia counterpart: the field `source_dims` of `Marginal` (`src/probabilities/Marginal.jl`), checked
to equal `source_dims[state_indices]` by `check_transition`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def marginalSourceDims (Mg : Marginal sv av i) : Fin Mg.numStateIndices → ℕ :=
  fun j => sv.dims (Mg.stateIndices j)

/-- The marginal's action shape: `action_vars[k]` is the size of conditioning action variable
`action_indices[k]`.

Julia counterpart: the field `action_vars` of `Marginal` (`src/probabilities/Marginal.jl`), checked
to equal `action_vars[action_indices]` by `check_transition`
(`src/models/FactoredRobustMarkovDecisionProcess.jl`). -/
def marginalActionVars (Mg : Marginal sv av i) : Fin Mg.numActionIndices → ℕ :=
  fun k => av.dims (Mg.actionIndices k)

/-- The Julia values `(I[1], …, I[d])` of a Cartesian index, as the integer tuple Julia passes
around (`Tuple(I)`): coordinate `k` is `toJulia (I k)`.

Julia counterpart: `Tuple(jₐ)` / `Tuple(jₛ)` of a `CartesianIndex`, as in
`sub2ind(p::Marginal, action::CartesianIndex, source::CartesianIndex)`
(`src/probabilities/Marginal.jl`). -/
def juliaTuple {d : ℕ} {dims : Fin d → ℕ} (I : CartesianIndex dims) : Fin d → ℤ :=
  fun k => (toJulia (I k) : ℤ)

/-- Literal transcription of the `sub2ind(p::Marginal, action, source)` loop of
`src/probabilities/Marginal.jl`. `action` and `source` are the global Julia integer tuples
(1-based). The first loop runs `i = N1, N1 - 1, …, 1` over the conditioning state variables
(`StepRange(N1, -1, 1)`, here `(List.finRange N1).reverse`), the second `i = M1, …, 1` over the
conditioning action variables; each step does `ind *= dims[i]` then `ind += x - one(T)`, and the
result is `ind + one(T)`. The arithmetic is exact (`ℤ`); see `marginalSub2indInt` for `N`-bit
integers.

Julia counterpart: `sub2ind(p::Marginal, action::NTuple, source::NTuple)`
(`src/probabilities/Marginal.jl`). -/
def marginalSub2ind (Mg : Marginal sv av i) (action : Fin m → ℤ) (source : Fin n → ℤ) : ℤ :=
  -- ind = zero(T)
  let ind : ℤ := 0
  -- for i in StepRange(N1, -1, 1); ind *= p.source_dims[i]; ind += source[p.state_indices[i]] - 1
  let ind := (List.finRange Mg.numStateIndices).reverse.foldl
    (fun ind i => ind * (marginalSourceDims Mg i : ℤ) + (source (Mg.stateIndices i) - 1)) ind
  -- for i in StepRange(M1, -1, 1); ind *= p.action_vars[i]; ind += action[p.action_indices[i]] - 1
  let ind := (List.finRange Mg.numActionIndices).reverse.foldl
    (fun ind i => ind * (marginalActionVars Mg i : ℤ) + (action (Mg.actionIndices i) - 1)) ind
  -- return ind + one(T)
  ind + 1

/-- The value `sub2ind(p::Marginal, action, source)` holds when the loop runs in `N`-bit
two's-complement integers (`N = 64` for `Int` on the CPU, `N = 32` for `Int32` on CUDA). Wrap-around
`+`, `-`, `*` are reduction modulo `2 ^ N`, a ring homomorphism, so the wrapped loop result is the
exact result reduced into the signed range, `Int.bmod · (2 ^ N)` (as in `machineInt`).

Julia counterpart: `sub2ind(p::Marginal, action, source)` evaluated in `T <: Integer` arithmetic
(`src/probabilities/Marginal.jl`). -/
def marginalSub2indInt (N : ℕ) (Mg : Marginal sv av i) (action : Fin m → ℤ)
    (source : Fin n → ℤ) : ℤ :=
  Int.bmod (marginalSub2ind Mg action source) (2 ^ N)

/-- The column-major dims of the marginal's columns, actions first:
`(action_vars[1], …, action_vars[M1], source_dims[1], …, source_dims[N1])`.

Julia counterpart: the column layout of `ambiguity_sets` of a `Marginal`, of size
`prod(action_vars) * prod(source_dims)` (`checkindices`, `src/probabilities/Marginal.jl`). -/
def marginalDims (Mg : Marginal sv av i) : Fin (Mg.numActionIndices + Mg.numStateIndices) → ℕ :=
  Fin.append (marginalActionVars Mg) (marginalSourceDims Mg)

/-- The conditioning tuple `(action[action_indices]…, source[state_indices]…)` of a global action
and source state, as a Cartesian index over `marginalDims`.

Julia counterpart: the values read by `sub2ind(p::Marginal, action, source)`, i.e.
`action[p.action_indices[k]]` and `source[p.state_indices[j]]` (`src/probabilities/Marginal.jl`). -/
def marginalCartesian (Mg : Marginal sv av i) (a : av.Action) (s : sv.State) :
    CartesianIndex (marginalDims Mg) :=
  Fin.addCases (motive := fun t => Fin (marginalDims Mg t))
    (fun k => Fin.cast (by simp [marginalDims, marginalActionVars]) (Mg.actionOf a k))
    (fun j => Fin.cast (by simp [marginalDims, marginalSourceDims]) (Mg.sourceOf s j))

/-- Model of `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ) = T(jₛ[1])`: the column is the first
coordinate of the source state; the action `_jₐ` is ignored.

Julia counterpart: `sub2ind(::IntervalAmbiguitySets, jₐ::NTuple, jₛ::NTuple)`
(`src/probabilities/IntervalAmbiguitySets.jl`). -/
def intervalSub2ind {d : ℕ} (_jₐ : Fin m → ℤ) (jₛ : Fin (d + 1) → ℤ) : ℤ :=
  jₛ 0

/-! ### Horner form of the loops -/

/-- The loop `for i in StepRange(d, -1, 1); ind *= D[i]; ind += c[i]; end` started at
`ind = init`: each step multiplies by `D i` and adds `c i`, for `i = d, d - 1, …, 1`.

Julia counterpart: either loop of `sub2ind(p::Marginal, action, source)`
(`src/probabilities/Marginal.jl`), with `D = source_dims`, `c = source[state_indices] .- 1`
(first loop) or `D = action_vars`, `c = action[action_indices] .- 1` (second loop). -/
def hornerLoop {d : ℕ} (D c : Fin d → ℤ) (init : ℤ) : ℤ :=
  (List.finRange d).reverse.foldl (fun ind i => ind * D i + c i) init

/-- The loop `hornerLoop D c init` is `init * ∏ D + ∑ᵢ c i * stride D i` (mixed-radix Horner
evaluation).

Julia counterpart: the two loops of `sub2ind(p::Marginal, …)` (`src/probabilities/Marginal.jl`). -/
theorem foldl_horner {d : ℕ} (D c : Fin d → ℤ) (init : ℤ) :
    hornerLoop D c init =
      init * ∏ i, D i + ∑ i, c i * ∏ j : Fin i, D (Fin.castLE i.is_lt.le j) := by
  induction d generalizing init with
  | zero => simp [hornerLoop]
  | succ d ih =>
    simp only [hornerLoop] at ih ⊢
    rw [List.finRange_succ_last, List.reverse_append, List.reverse_singleton,
      List.singleton_append, List.foldl_cons, ← List.map_reverse, List.foldl_map,
      ih (fun i => D i.castSucc) (fun i => c i.castSucc), Fin.prod_univ_castSucc,
      Fin.sum_univ_castSucc]
    have hlast : ∏ j : Fin (Fin.last d), D (Fin.castLE (Fin.last d).is_lt.le j) =
        ∏ i : Fin d, D i.castSucc := Finset.prod_congr rfl fun _ _ => rfl
    have hsum : ∑ x : Fin d, c x.castSucc * ∏ j : Fin x, D (Fin.castLE x.is_lt.le j).castSucc =
        ∑ x : Fin d, c x.castSucc * ∏ j : Fin x.castSucc, D (Fin.castLE x.castSucc.is_lt.le j) :=
      rfl
    rw [hlast, hsum]
    ring

/-- `linear` is the Horner loop over the Julia coordinates minus one, plus one.

Julia counterpart: `LinearIndices(dims)[I]` (Base), the closed form of the `sub2ind` loops
(`src/probabilities/Marginal.jl`). -/
theorem linear_eq_foldl {d : ℕ} (dims : Fin d → ℕ) (I : CartesianIndex dims) :
    (linear dims I : ℤ) = hornerLoop (Nat.cast ∘ dims) (juliaTuple I - 1) 0 + 1 := by
  rw [foldl_horner]
  simp only [linear, stride, juliaTuple, toJulia, Function.comp_apply, Pi.sub_apply, Pi.one_apply]
  push_cast
  simp only [add_sub_cancel_right, zero_mul, zero_add]
  ring

/-- `finRange (a + b)` lists `castAdd` of `finRange a`, then `natAdd` of `finRange b`.

Julia counterpart: none (Lean-side proof device). -/
theorem finRange_add (a b : ℕ) :
    List.finRange (a + b) =
      (List.finRange a).map (Fin.castAdd b) ++ (List.finRange b).map (Fin.natAdd a) := by
  simp [List.finRange, List.ofFn_add]
  rfl

/-! ### Theorems -/

/-- `sub2ind(p::Marginal, action, source)` is the column-major linear index of the conditioning
tuple `(action[action_indices]…, source[state_indices]…)` over the dims
`(action_vars…, source_dims…)`, actions first; under `∏ dims < 2 ^ (N - 1)` the `N`-bit value is
the same.

Julia counterpart: `sub2ind(p::Marginal, action, source)` (`src/probabilities/Marginal.jl`) versus
`LinearIndices((action_vars..., source_dims...))[action[action_indices]...,
source[state_indices]...]` (`test/base/indexing_reference.jl`). -/
theorem marginalSub2ind_eq_linear {N : ℕ} (Mg : Marginal sv av i)
    (hBound : ∏ k, marginalDims Mg k < 2 ^ (N - 1)) (a : av.Action) (s : sv.State) :
    marginalSub2ind Mg (juliaTuple a) (juliaTuple s) =
        linear (marginalDims Mg) (marginalCartesian Mg a s) ∧
      marginalSub2indInt N Mg (juliaTuple a) (juliaTuple s) =
        linear (marginalDims Mg) (marginalCartesian Mg a s) := by
  have hexact : marginalSub2ind Mg (juliaTuple a) (juliaTuple s) =
      linear (marginalDims Mg) (marginalCartesian Mg a s) := by
    rw [linear_eq_foldl, hornerLoop, finRange_add, List.reverse_append, List.foldl_append,
      ← List.map_reverse, ← List.map_reverse, List.foldl_map, List.foldl_map]
    simp [marginalSub2ind, marginalDims, marginalCartesian, juliaTuple, Marginal.actionOf,
      Marginal.sourceOf, toJulia]
    rfl
  refine ⟨hexact, ?_⟩
  obtain ⟨hbij, hInt⟩ := linear_bijective (N := N) hBound
  rw [marginalSub2indInt, hexact]
  exact hInt _

/-- Every conditioning tuple `(a[action_indices]…, s[state_indices]…)` comes from some global
action and state (the conditioning variables are distinct and every variable has a value).

Julia counterpart: none (Lean-side proof device for `sub2ind(p::Marginal, …)`,
`src/probabilities/Marginal.jl`). -/
theorem marginalCartesian_surjective (Mg : Marginal sv av i)
    (J : CartesianIndex (marginalDims Mg)) :
    ∃ x : av.Action × sv.State, marginalCartesian Mg x.1 x.2 = J := by
  classical
  let a : av.Action := fun v =>
    if h : ∃ k, Mg.actionIndices k = v then
      ⟨J (Fin.castAdd _ h.choose), by
        have := (J (Fin.castAdd _ h.choose)).is_lt
        simpa [marginalDims, marginalActionVars, h.choose_spec] using this⟩
    else ⟨0, av.dims_pos v⟩
  let s : sv.State := fun v =>
    if h : ∃ j, Mg.stateIndices j = v then
      ⟨J (Fin.natAdd _ h.choose), by
        have := (J (Fin.natAdd _ h.choose)).is_lt
        simpa [marginalDims, marginalSourceDims, h.choose_spec] using this⟩
    else ⟨0, sv.dims_pos v⟩
  refine ⟨(a, s), funext fun t => ?_⟩
  refine Fin.addCases (fun k => ?_) (fun j => ?_) t
  · have h : ∃ k', Mg.actionIndices k' = Mg.actionIndices k := ⟨k, rfl⟩
    have hk : h.choose = k := Mg.actionIndices_strictMono.injective h.choose_spec
    apply Fin.ext
    simp only [marginalCartesian, Fin.addCases_left, Fin.val_cast, Marginal.actionOf, a,
      dif_pos h]
    exact congrArg (fun t => (J (Fin.castAdd _ t) : ℕ)) hk
  · have h : ∃ j', Mg.stateIndices j' = Mg.stateIndices j := ⟨j, rfl⟩
    have hj : h.choose = j := Mg.stateIndices_strictMono.injective h.choose_spec
    apply Fin.ext
    simp only [marginalCartesian, Fin.addCases_right, Fin.val_cast, Marginal.sourceOf, s,
      dif_pos h]
    exact congrArg (fun t => (J (Fin.natAdd _ t) : ℕ)) hj

/-- The `N`-bit column `sub2ind(p::Marginal, action, source)` of a global (action, source state)
pair `x = (a, s)`, with `a` and `s` passed as Julia integer tuples.

Julia counterpart: `sub2ind(p::Marginal, action, source)` (`src/probabilities/Marginal.jl`),
evaluated in `N`-bit integers. -/
def marginalColumn (N : ℕ) (Mg : Marginal sv av i) (x : av.Action × sv.State) : ℤ :=
  marginalSub2indInt N Mg (juliaTuple x.1) (juliaTuple x.2)

/-- `sub2ind(p::Marginal, ·, ·)` is a bijection between the conditioning tuples and
`1..∏ action_vars * ∏ source_dims`: under the overflow bound its `N`-bit values on global actions
and states are exactly that range, and equal values have equal conditioning tuples. (Injectivity of
the `N`-bit value implies injectivity of the exact value `marginalSub2ind`, since equal exact
values have equal `N`-bit reductions.)

Julia counterpart: `sub2ind(p::Marginal, action, source)` (`src/probabilities/Marginal.jl`) with
`checkindices` (`num_sets == prod(source_dims) * prod(action_vars)`). -/
theorem marginalSub2ind_bijective {N : ℕ} (Mg : Marginal sv av i)
    (hBound : ∏ k, marginalDims Mg k < 2 ^ (N - 1)) :
    Set.range (marginalColumn N Mg) = Nat.cast '' juliaRange (∏ k, marginalDims Mg k) ∧
      ∀ x y, marginalColumn N Mg x = marginalColumn N Mg y →
        marginalCartesian Mg x.1 x.2 = marginalCartesian Mg y.1 y.2 := by
  obtain ⟨hbij, -⟩ := linear_bijective (N := N) hBound
  refine ⟨?_, fun x y h => ?_⟩
  · ext z
    simp only [Set.mem_range, Set.mem_image, Prod.exists, marginalColumn]
    constructor
    · rintro ⟨a, s, rfl⟩
      exact ⟨_, hbij.mapsTo (Set.mem_univ (marginalCartesian Mg a s)),
        ((marginalSub2ind_eq_linear Mg hBound a s).2).symm⟩
    · rintro ⟨k, hk, rfl⟩
      obtain ⟨J, -, rfl⟩ := hbij.surjOn hk
      obtain ⟨⟨a, s⟩, rfl⟩ := marginalCartesian_surjective Mg J
      exact ⟨a, s, (marginalSub2ind_eq_linear Mg hBound a s).2⟩
  · rw [marginalColumn, marginalColumn, (marginalSub2ind_eq_linear Mg hBound x.1 x.2).2,
      (marginalSub2ind_eq_linear Mg hBound y.1 y.2).2] at h
    exact hbij.injOn (Set.mem_univ _) (Set.mem_univ _) (by exact_mod_cast h)

/-- `sub2ind(p::Marginal, action, source)` reads only the conditioning variables: global actions
and sources that agree on `action_indices` and `state_indices` get the same column (for any
integer tuples, so also for the `N`-bit value).

Julia counterpart: `sub2ind(p::Marginal, action, source)` and `getindex(p::Marginal, action,
source)` (`src/probabilities/Marginal.jl`). -/
theorem marginalSub2ind_depends_only (Mg : Marginal sv av i) {action action' : Fin m → ℤ}
    {source source' : Fin n → ℤ}
    (ha : ∀ k, action (Mg.actionIndices k) = action' (Mg.actionIndices k))
    (hs : ∀ j, source (Mg.stateIndices j) = source' (Mg.stateIndices j)) :
    marginalSub2ind Mg action source = marginalSub2ind Mg action' source' := by
  simp only [marginalSub2ind, ha, hs]

/-- `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ) = jₛ[1]` gives the same column as
`sub2ind(::Marginal, jₐ, jₛ)` for a marginal that conditions on state variable 1 only and whose
conditioning action variables have a single value (a single-action model, as built by
`IntervalMarkovDecisionProcess` with one action).

Julia counterpart: `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ)`
(`src/probabilities/IntervalAmbiguitySets.jl`) versus `sub2ind(::Marginal, jₐ, jₛ)`
(`src/probabilities/Marginal.jl`). -/
theorem intervalSub2ind_correct {d : ℕ} {sv : StateVars (d + 1)} {i : Fin (d + 1)}
    (Mg : Marginal sv av i) (hState : Set.range Mg.stateIndices = {0})
    (hAction : ∀ k, marginalActionVars Mg k = 1) (a : av.Action) (s : sv.State) :
    intervalSub2ind (juliaTuple a) (juliaTuple s) =
      marginalSub2ind Mg (juliaTuple a) (juliaTuple s) := by
  have hall : ∀ j, Mg.stateIndices j = 0 := fun j => by
    have : Mg.stateIndices j ∈ Set.range Mg.stateIndices := ⟨j, rfl⟩
    rwa [hState] at this
  obtain ⟨j₀, -⟩ : (0 : Fin (d + 1)) ∈ Set.range Mg.stateIndices := by rw [hState]; rfl
  have hval : ∀ j : Fin Mg.numStateIndices, j = j₀ := fun j =>
    Mg.stateIndices_strictMono.injective ((hall j).trans (hall j₀).symm)
  have hj0 : (j₀ : ℕ) = 0 := by
    have := hval ⟨0, by have := j₀.is_lt; omega⟩
    rw [← this]
  have hcard : Mg.numStateIndices = 1 := by
    have h1 := hval ⟨0, by have := j₀.is_lt; omega⟩
    by_contra hne
    have h2 := hval ⟨1, by have := j₀.is_lt; omega⟩
    have := congrArg Fin.val (h1.trans h2.symm)
    simp at this
  have hact : ∀ k, (a (Mg.actionIndices k) : ℕ) = 0 := fun k => by
    have := (a (Mg.actionIndices k)).is_lt
    have h := hAction k
    simp only [marginalActionVars] at h
    omega
  simp only [marginalSub2ind, intervalSub2ind, ← hornerLoop.eq_def]
  rw [foldl_horner, foldl_horner]
  have hA : ∀ k, (marginalActionVars Mg k : ℤ) = 1 := fun k => by rw [hAction k]; rfl
  simp only [hA, Finset.prod_const_one, mul_one, juliaTuple, toJulia, hact]
  simp only [zero_add, Nat.cast_one, sub_self, zero_mul, Finset.sum_const_zero,
    add_zero]
  rw [Fintype.sum_eq_single j₀ (fun j hj => absurd (hval j) hj)]
  have hstride : ∏ j : Fin j₀, (marginalSourceDims Mg (Fin.castLE j₀.is_lt.le j) : ℤ) = 1 := by
    have : IsEmpty (Fin (j₀ : ℕ)) := by rw [hj0]; infer_instance
    exact Finset.prod_of_isEmpty _
  rw [hstride, hall j₀]
  push_cast
  ring

/-- `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ) = jₛ[1]` cannot give the `Marginal` column for two
global actions that differ on a conditioning action variable: it ignores the action, while
`sub2ind(::Marginal, …)` separates distinct conditioning tuples (`marginalSub2ind_bijective`).
Observation O8 in the inventory: the method is wrong for multi-action layouts (not reached by
package code).

Julia counterpart: `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ)` and
`getindex(p::IntervalAmbiguitySets, jₐ, jₛ)` (`src/probabilities/IntervalAmbiguitySets.jl`). -/
theorem intervalSub2ind_wrong_multiAction {d : ℕ} {sv : StateVars (d + 1)} {i : Fin (d + 1)}
    (Mg : Marginal sv av i) (s : sv.State) {a a' : av.Action} (h : Mg.actionOf a ≠ Mg.actionOf a') :
    intervalSub2ind (juliaTuple a) (juliaTuple s) ≠
        marginalSub2ind Mg (juliaTuple a) (juliaTuple s) ∨
      intervalSub2ind (juliaTuple a') (juliaTuple s) ≠
        marginalSub2ind Mg (juliaTuple a') (juliaTuple s) := by
  by_contra hcon
  obtain ⟨h1, h2⟩ := not_or.mp hcon
  rw [not_not] at h1 h2
  have heq : marginalSub2ind Mg (juliaTuple a) (juliaTuple s) =
      marginalSub2ind Mg (juliaTuple a') (juliaTuple s) := by
    rw [← h1, ← h2]; rfl
  have hJ := (marginalSub2ind_bijective (N := (∏ k, marginalDims Mg k) + 2) Mg
    (by
      calc ∏ k, marginalDims Mg k < (∏ k, marginalDims Mg k) + 1 := Nat.lt_succ_self _
        _ ≤ 2 ^ ((∏ k, marginalDims Mg k) + 1) := (Nat.lt_two_pow_self).le
        _ = _ := by rfl)).2 (a, s) (a', s)
    (by simp only [marginalColumn, marginalSub2indInt, heq])
  apply h
  funext k
  have := congrFun hJ (Fin.castAdd _ k)
  simpa [marginalCartesian, Fin.ext_iff, Marginal.actionOf] using this

end IntervalMDP.Index
