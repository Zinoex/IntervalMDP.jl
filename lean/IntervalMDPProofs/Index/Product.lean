import IntervalMDPProofs.Index.Linear
import IntervalMDPProofs.Models.Factored
import IntervalMDPProofs.Models.Product

/-!
# Product-state indexing

The value arrays of a `ProductProcess` have shape `state_values(proc) =
(state_values(markov_process(proc))..., num_states(automaton(proc)))`
(`src/models/ProductProcess.jl`): the MDP state variables first, the DFA state **last**. The
product Bellman operator `_bellman_helper!` (`src/bellman.jl`) uses this array in two places, once
per DFA state `state` (Lean `q`):

    map!(W, CartesianIndices(state_values(mp))) do idx
        return V[idx, dfa[state, lf[idx]]]
    end
    bellman!(…, selectdim(Vres, ndims(Vres), state), W, mp, mp_states; …)

`W` (`workspace.intermediate_values`) is the value vector that the MDP Bellman operator `bellman!`
reads at the **successor** states `idx` (Lean `s'`), and `selectdim(Vres, ndims(Vres), state)` is
the slice of the result written for DFA state `q`.

Lean model, over Julia's integer indices (`Index/Julia.lean`, `Index/Linear.lean`):

* `productDims sv nQ` is `state_values(proc)` and `productCartesian s q` is the Cartesian index
  `(idx..., q)`; `productIndex sv nQ (s, q)` is its column-major linear index
  `LinearIndices(state_values(proc))[idx, q]`, i.e. the position read by `V[idx, q]`.
* `productIndex_bijective` (hypothesis `hBound`; mode-independent): `productIndex` is a bijection from `S × Q` onto
  the Julia range `1..(∏ state_values(mp)) * nQ`, its `N`-bit value does not overflow, and the DFA
  state is the last (slowest) dimension:
  `productIndex (s, q) = linear s + (∏ state_values(mp)) * (q - 1)` (1-based `q`).
  `productEquiv` is the same bijection as an equivalence `S × Q ≃ Fin (|S| * |Q|)`, and
  `selectdimIndex_bijOn` shows that the slice `selectdim(Vres, ndims(Vres), state)` is the
  contiguous block of the `|S|` positions of DFA state `q`.
* `product_read_successor` (hypothesis `hBound`; deterministic labelling; mode-independent):
  `W[s'] = V[s', dfa[q, lf[s']]]` is the value
  of the product state `(s', δ(q, L(s')))`; the DFA transition uses the label of the
  **successor** `s'`, not of the source state. `product_read_nextDFA` (hypotheses `hBound`, `hL`;
  deterministic labelling; mode-independent)
  identifies it with the expectation under `ProductProcess.nextDFA` of `Models/Product.lean` for a
  deterministic labelling.

Indexing is mode-independent: the index computation is the same for every satisfaction mode
(`upper_bound`) and strategy mode (`maximize`), so all four combinations are covered. Only the
deterministic labelling (`DeterministicLabelling`, the `_bellman_helper!` method that reads
`V[idx, dfa[state, lf[idx]]]`) is modelled here; the probabilistic-labelling method reads the same
positions `V[idx, dfa[state, label]]` for every label. Proof scope: `abstract` (integer index
arithmetic; values in `ℝ`).
-/

namespace IntervalMDP.Index

open Finset

variable {n : ℕ}

/-- The dims vector of a product value array: the MDP state variables `sv.dims` followed by the
number of DFA states `nQ` as the **last** dimension.

Julia counterpart: `state_values(proc::ProductProcess) = (state_values(markov_process(proc))...,
Int32(num_states(automaton(proc))))` (`src/models/ProductProcess.jl`), the shape of `V` and `Vres`
in `_bellman_helper!` (`src/bellman.jl`). -/
def productDims (sv : StateVars n) (nQ : ℕ) : Fin (n + 1) → ℕ :=
  Fin.snoc (α := fun _ => ℕ) sv.dims nQ

/-- The first `n` dimensions of `productDims` are the MDP state variables.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem productDims_castSucc (sv : StateVars n) (nQ : ℕ) (i : Fin n) :
    productDims sv nQ i.castSucc = sv.dims i := by
  simp [productDims]

/-- The last dimension of `productDims` is the number of DFA states.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem productDims_last (sv : StateVars n) (nQ : ℕ) :
    productDims sv nQ (Fin.last n) = nQ := by
  simp [productDims]

/-- The product Cartesian index `(idx..., q)`: the MDP state `s` (Julia `idx`) in the first `n`
coordinates and the DFA state `q` in the last one.

Julia counterpart: the index `[idx, q]` of `V[idx, dfa[state, lf[idx]]]` (with
`q = dfa[state, lf[idx]]`) and of `selectdim(Vres, ndims(Vres), state)[idx]` in
`_bellman_helper!` (`src/bellman.jl`). -/
def productCartesian {sv : StateVars n} {nQ : ℕ} (s : sv.State) (q : Fin nQ) :
    CartesianIndex (productDims sv nQ) :=
  Fin.snoc (α := fun k => Fin (productDims sv nQ k))
    (fun i => Fin.cast (productDims_castSucc sv nQ i).symm (s i))
    (Fin.cast (productDims_last sv nQ).symm q)

/-- The MDP coordinates of `productCartesian s q` are those of `s`.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem productCartesian_castSucc_val {sv : StateVars n} {nQ : ℕ} (s : sv.State)
    (q : Fin nQ) (i : Fin n) : (productCartesian s q i.castSucc).val = (s i).val := by
  simp [productCartesian]

/-- The last coordinate of `productCartesian s q` is the DFA state `q`.

Julia counterpart: none (Lean-side proof device). -/
@[simp] theorem productCartesian_last_val {sv : StateVars n} {nQ : ℕ} (s : sv.State)
    (q : Fin nQ) : (productCartesian s q (Fin.last n)).val = q.val := by
  simp [productCartesian]

/-- The uncurried `productCartesian`, `(s, q) ↦ (idx..., q)`.

Julia counterpart: the index `[idx, q]` into `V` in `_bellman_helper!` (`src/bellman.jl`). -/
def productCartesianPair {sv : StateVars n} {nQ : ℕ} (z : sv.State × Fin nQ) :
    CartesianIndex (productDims sv nQ) :=
  productCartesian z.1 z.2

/-- Split a product Cartesian index into its MDP state (first `n` coordinates) and DFA state (last
coordinate); inverse of `productCartesianPair`.

Julia counterpart: `(idx, state)` recovered from a `CartesianIndex` over `state_values(proc)`
(`src/bellman.jl`). -/
def productSplit {sv : StateVars n} {nQ : ℕ} (I : CartesianIndex (productDims sv nQ)) :
    sv.State × Fin nQ :=
  (fun i => Fin.cast (productDims_castSucc sv nQ i) (I i.castSucc),
    Fin.cast (productDims_last sv nQ) (I (Fin.last n)))

/-- `(s, q) ↦ (idx..., q)` is an equivalence between product states and the Cartesian indices of
the product value array.

Julia counterpart: `CartesianIndices(state_values(proc))` for a `ProductProcess`
(`src/models/ProductProcess.jl`, `src/bellman.jl`). -/
def productCartesianEquiv (sv : StateVars n) (nQ : ℕ) :
    sv.State × Fin nQ ≃ CartesianIndex (productDims sv nQ) where
  toFun := productCartesianPair
  invFun := productSplit
  left_inv z := by
    obtain ⟨s, q⟩ := z
    ext i
    · simp [productSplit, productCartesianPair]
    · simp [productSplit, productCartesianPair]
  right_inv I := by
    funext k
    refine Fin.lastCases ?_ (fun i => ?_) k
    · apply Fin.ext
      simp [productSplit, productCartesianPair]
    · apply Fin.ext
      simp [productSplit, productCartesianPair]

/-- The 1-based linear position of the product state `(s, q)` in a product value array: the
column-major index `LinearIndices(state_values(proc))[idx, q]` of `productCartesian s q`.

Julia counterpart: the position read by `V[idx, dfa[state, lf[idx]]]` and written through
`selectdim(Vres, ndims(Vres), state)[idx]` in `_bellman_helper!` (`src/bellman.jl`). -/
def productIndex (sv : StateVars n) (nQ : ℕ) (z : sv.State × Fin nQ) : ℕ :=
  linear (productDims sv nQ) (productCartesianPair z)

/-- `productIndex` computed with `N`-bit integers (Julia `Int`, `N = 64`).

Julia counterpart: the `Int` linear index computed by `getindex(V, idx, q)` for the product value
array in `_bellman_helper!` (`src/bellman.jl`). -/
def productIndexInt (N : ℕ) (sv : StateVars n) (nQ : ℕ) (z : sv.State × Fin nQ) : ℤ :=
  machineInt N (productIndex sv nQ z)

/-- The size of a product value array is `(∏ state_values(mp)) * nQ`.

Julia counterpart: `prod(state_values(proc))` for a `ProductProcess`
(`src/models/ProductProcess.jl`). -/
theorem prod_productDims (sv : StateVars n) (nQ : ℕ) :
    ∏ k, productDims sv nQ k = (∏ i, sv.dims i) * nQ := by
  simp [Fin.prod_univ_castSucc]

/-- The product state `(s, q)` as a 0-based position `Fin ((∏ state_values(mp)) * nQ)` in the
product value array (column-major, DFA state last): the equivalence `S × Q ≃ Fin (|S| * |Q|)`.

Julia counterpart: `LinearIndices(state_values(proc))` for a `ProductProcess`
(`src/models/ProductProcess.jl`, `src/bellman.jl`). -/
def productEquiv (sv : StateVars n) (nQ : ℕ) :
    sv.State × Fin nQ ≃ Fin ((∏ i, sv.dims i) * nQ) :=
  (productCartesianEquiv sv nQ).trans (finPiFinEquiv.trans (finCongr (prod_productDims sv nQ)))

/-- `productIndex` is `toJulia` of `productEquiv`.

Julia counterpart: none (Lean-side proof device). -/
theorem productIndex_eq_toJulia (sv : StateVars n) (nQ : ℕ) (z : sv.State × Fin nQ) :
    productIndex sv nQ z = toJulia (productEquiv sv nQ z) := by
  rw [productIndex, linear_eq_toJulia]
  simp [productEquiv, productCartesianEquiv, toJulia]

/-- The stride of an MDP dimension of the product array is its stride in the MDP array.

Julia counterpart: none (Lean-side proof device). -/
theorem stride_productDims_castSucc (sv : StateVars n) (nQ : ℕ) (i : Fin n) :
    stride (productDims sv nQ) i.castSucc = stride sv.dims i := by
  simp only [stride]
  refine Fintype.prod_equiv (finCongr (Fin.val_castSucc i)) _ _ (fun j => ?_)
  have h : (Fin.castLE (i.castSucc).is_lt.le j : Fin (n + 1)) =
      (Fin.castLE i.is_lt.le (finCongr (Fin.val_castSucc i) j)).castSucc := Fin.ext (by simp)
  rw [h, productDims_castSucc]

/-- The stride of the DFA dimension (the last one) is the number of MDP states
`∏ state_values(mp)`.

Julia counterpart: none (Lean-side proof device). -/
theorem stride_productDims_last (sv : StateVars n) (nQ : ℕ) :
    stride (productDims sv nQ) (Fin.last n) = ∏ i, sv.dims i := by
  simp only [stride]
  refine Fintype.prod_equiv (finCongr (Fin.val_last n)) _ _ (fun j => ?_)
  have h : (Fin.castLE (Fin.last n).is_lt.le j : Fin (n + 1)) =
      (finCongr (Fin.val_last n) j).castSucc := Fin.ext (by simp)
  rw [h, productDims_castSucc]

/-- The DFA state is the last (slowest) dimension:
`productIndex (s, q) = LinearIndices(state_values(mp))[s] + (∏ state_values(mp)) * (q - 1)` with
the 1-based `q`, i.e. `linear sv.dims s + (∏ sv.dims) * q.val` with the 0-based `q.val`.

Julia counterpart: column-major `LinearIndices(state_values(proc))[idx, q]` with the DFA dimension
last (`src/models/ProductProcess.jl`, `src/bellman.jl`). -/
theorem productIndex_eq (sv : StateVars n) (nQ : ℕ) (s : sv.State) (q : Fin nQ) :
    productIndex sv nQ (s, q) = linear sv.dims s + (∏ i, sv.dims i) * q.val := by
  simp only [productIndex, productCartesianPair, linear, Fin.sum_univ_castSucc,
    stride_productDims_castSucc, stride_productDims_last, toJulia, productCartesian_castSucc_val,
    productCartesian_last_val, Nat.add_sub_cancel]
  ring

/-- `productIndex` is a bijection from `S × Q` onto the Julia range `1..(∏ state_values(mp)) * nQ`
(exact natural-number indices).

Julia counterpart: `LinearIndices(state_values(proc))` for a `ProductProcess`
(`src/models/ProductProcess.jl`, `src/bellman.jl`). -/
theorem productIndex_bijOn (sv : StateVars n) (nQ : ℕ) :
    Set.BijOn (productIndex sv nQ) Set.univ (juliaRange ((∏ i, sv.dims i) * nQ)) := by
  have h := toJulia_bijOn.comp (productEquiv sv nQ).bijective.bijOn_univ
  convert h using 1
  funext z
  exact productIndex_eq_toJulia sv nQ z

/-- Under the overflow bound `hBound : (∏ state_values(mp)) * nQ < 2 ^ (N - 1)`, the `N`-bit value
of `productIndex` is exact.

Julia counterpart: the `Int` linear index of `getindex(V, idx, q)` (`src/bellman.jl`). -/
theorem productIndexInt_eq {N : ℕ} {sv : StateVars n} {nQ : ℕ}
    (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1)) (z : sv.State × Fin nQ) :
    productIndexInt N sv nQ z = productIndex sv nQ z :=
  machineInt_eq_self (lt_of_le_of_lt ((productIndex_bijOn sv nQ).mapsTo (Set.mem_univ z)).2 hBound)

/-- **Product-state indexing is a bijection with the DFA state last.** Under the overflow bound
`hBound : (∏ state_values(mp)) * nQ < 2 ^ (N - 1)`:

* `productIndex` is a bijection from the product states `S × Q` onto the Julia range
  `1..(∏ state_values(mp)) * nQ` (as an equivalence: `productEquiv`);
* its `N`-bit value (Julia `Int`, `N = 64`) is exact;
* the DFA state is the last (slowest, column-major) dimension:
  `productIndex (s, q) = linear sv.dims s + (∏ sv.dims) * q.val`.

This is mode-independent: the index does not depend on the satisfaction mode (`upper_bound`) or the
strategy mode (`maximize`), so all four modes are covered. Scope `abstract`.

Julia counterpart: the product value arrays `V[idx, q]` and `selectdim(Vres, ndims(Vres), state)`
of shape `state_values(proc)` in `_bellman_helper!` (`src/bellman.jl`,
`src/models/ProductProcess.jl`); hypothesis `hBound`, mode-independent. -/
theorem productIndex_bijective {N : ℕ} {sv : StateVars n} {nQ : ℕ}
    (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1)) :
    Set.BijOn (productIndex sv nQ) Set.univ (juliaRange ((∏ i, sv.dims i) * nQ)) ∧
      (∀ z, productIndexInt N sv nQ z = productIndex sv nQ z) ∧
      ∀ s q, productIndex sv nQ (s, q) = linear sv.dims s + (∏ i, sv.dims i) * q.val :=
  ⟨productIndex_bijOn sv nQ, productIndexInt_eq hBound, productIndex_eq sv nQ⟩

/-- The position written for MDP state `s` through the slice of DFA state `q`:
`selectdim(Vres, ndims(Vres), q)[s]` is `Vres[s, q]`, at `productIndex (s, q)`.

Julia counterpart: `selectdim(Vres, ndims(Vres), state)` in `_bellman_helper!` (`src/bellman.jl`),
the result slice that `bellman!` writes for DFA state `state`. -/
def selectdimIndex (sv : StateVars n) (nQ : ℕ) (q : Fin nQ) (s : sv.State) : ℕ :=
  productIndex sv nQ (s, q)

/-- The slice `selectdim(Vres, ndims(Vres), q)` is the contiguous block of positions
`(∏ sv.dims) * q.val + 1 .. (∏ sv.dims) * q.val + ∏ sv.dims` of the product array, covered
bijectively by the MDP states (exact natural-number indices, so no overflow bound is needed;
mode-independent).

Julia counterpart: `selectdim(Vres, ndims(Vres), state)` in `_bellman_helper!` (`src/bellman.jl`). -/
theorem selectdimIndex_bijOn (sv : StateVars n) (nQ : ℕ) (q : Fin nQ) :
    Set.BijOn (selectdimIndex sv nQ q) Set.univ
      (Set.Icc ((∏ i, sv.dims i) * q.val + 1) ((∏ i, sv.dims i) * q.val + ∏ i, sv.dims i)) := by
  have hlin : Set.BijOn (linear sv.dims) Set.univ (juliaRange (∏ i, sv.dims i)) := by
    have h := toJulia_bijOn.comp (finPiFinEquiv (n := sv.dims)).bijective.bijOn_univ
    convert h using 1
    funext I
    exact linear_eq_toJulia sv.dims I
  have hfun : selectdimIndex sv nQ q = (· + (∏ i, sv.dims i) * q.val) ∘ linear sv.dims := by
    funext s
    simp [selectdimIndex, productIndex_eq]
  rw [hfun]
  refine Set.BijOn.comp ?_ hlin
  refine ⟨fun k hk => ?_, fun a _ b _ h => by simpa using h, fun k hk => ?_⟩
  · simp only [juliaRange, Set.mem_Icc] at hk ⊢; omega
  · simp only [Set.mem_Icc] at hk
    refine ⟨k - (∏ i, sv.dims i) * q.val, ?_, ?_⟩
    · simp only [juliaRange, Set.mem_Icc]; omega
    · simp only; omega

/-- Julia's `V[idx, q]` on the product value array `V`, stored column-major as the vector `V` of
length `(∏ state_values(mp)) * nQ`: the entry at the linear index `productIndexInt N` (`N`-bit
`Int` arithmetic; a wrapped-around index reads as an out-of-range `0`).

Julia counterpart: `getindex(V, idx, q)` for the product value array in `_bellman_helper!`
(`src/bellman.jl`). -/
def productGet (N : ℕ) {sv : StateVars n} {nQ : ℕ} (V : Fin ((∏ i, sv.dims i) * nQ) → ℝ)
    (s : sv.State) (q : Fin nQ) : ℝ :=
  juliaGet V (productIndexInt N sv nQ (s, q)).toNat

/-- The value vector `W` built for DFA state `state` (Lean `q`): a transcription of the closure
`idx -> V[idx, dfa[state, lf[idx]]]` mapped over `CartesianIndices(state_values(mp))`, i.e.
`W[idx] = V[idx, δ(state, L(idx))]` for every successor `idx` (Lean `s'`).

Julia counterpart: `map!(W, CartesianIndices(state_values(mp))) do idx; return V[idx, dfa[state,
lf[idx]]]; end` with `W = workspace.intermediate_values` in `_bellman_helper!` for a
`DeterministicLabelling` (`src/bellman.jl`); `dfa[state, l]` is `DFA.δ` (`src/models/DFA.jl`) and
`lf[idx]` is `Labelling.map` (`src/probabilities/DeterministicLabelling.jl`). -/
def intermediateValues (N : ℕ) {sv : StateVars n} {nQ : ℕ} {Λ : Type*} (dfa : DFA (Fin nQ) Λ)
    (lf : Labelling sv.State Λ) (V : Fin ((∏ i, sv.dims i) * nQ) → ℝ) (state : Fin nQ)
    (idx : sv.State) : ℝ :=
  productGet N V idx (dfa.δ state (lf.map idx))

/-- The product value array storing the product value function `Vprod : S × Q → ℝ` in Julia's
layout: the entry at `productEquiv (s, q)` is `Vprod (s, q)`.

Julia counterpart: the value array `V` of shape `state_values(proc)` passed to `_bellman_helper!`
(`src/bellman.jl`). -/
def productArray {sv : StateVars n} {nQ : ℕ} (Vprod : sv.State × Fin nQ → ℝ) :
    Fin ((∏ i, sv.dims i) * nQ) → ℝ :=
  Vprod ∘ (productEquiv sv nQ).symm

/-- Under the overflow bound `hBound`, `V[s, q]` reads the value of the product state `(s, q)`.

Julia counterpart: `getindex(V, idx, q)` on the product value array (`src/bellman.jl`). -/
theorem productGet_productArray {N : ℕ} {sv : StateVars n} {nQ : ℕ}
    (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1)) (Vprod : sv.State × Fin nQ → ℝ)
    (s : sv.State) (q : Fin nQ) : productGet N (productArray Vprod) s q = Vprod (s, q) := by
  rw [productGet, productIndexInt_eq hBound (s, q), Int.toNat_natCast,
    productIndex_eq_toJulia, juliaGet_toJulia]
  simp [productArray]

/-- **The product read uses the successor's label.** Under the overflow bound
`hBound : (∏ state_values(mp)) * nQ < 2 ^ (N - 1)`, the entry `W[s'] = V[s', dfa[q, lf[s']]]` of the
value vector built for DFA state `q` is the value of the product state `(s', δ(q, L(s')))`: the DFA
transition reads the label `L(s')` of the **successor** `s'` (the index `W` is read at by
`bellman!`), not the label of the source state.

This is mode-independent: the read is the same for every satisfaction mode (`upper_bound`) and
strategy mode (`maximize`), so all four modes are covered. Restriction: deterministic labelling
(`DeterministicLabelling`). Scope `abstract`.

Julia counterpart: `V[idx, dfa[state, lf[idx]]]` in `_bellman_helper!` (`src/bellman.jl`). -/
theorem product_read_successor {N : ℕ} {sv : StateVars n} {nQ : ℕ} {Λ : Type*}
    (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1)) (dfa : DFA (Fin nQ) Λ)
    (lf : Labelling sv.State Λ) (Vprod : sv.State × Fin nQ → ℝ) (q : Fin nQ) (s' : sv.State) :
    intermediateValues N dfa lf (productArray Vprod) q s' = Vprod (s', dfa.δ q (lf.map s')) :=
  productGet_productArray hBound Vprod s' _

/-- The value-level product kernel of `Models/Product.lean`: the expectation of `Vprod (s', ·)`
under the next-DFA-state distribution `ProductProcess.nextDFA q s'`,
`∑_{q'} nextDFA q s' q' · Vprod (s', q')`.

Julia counterpart: the value `W[idx]` that `_bellman_helper!` (`src/bellman.jl`) computes for
successor `idx` (deterministic: `V[idx, dfa[state, lf[idx]]]`; probabilistic: the sum over
`enumerate(lf[idx])`). -/
def nextDFAValue {sv : StateVars n} {nQ : ℕ} {A Λ : Type*} [Fintype Λ] [DecidableEq Λ]
    (P : ProductProcess sv.State A (Fin nQ) Λ) (Vprod : sv.State × Fin nQ → ℝ) (q : Fin nQ)
    (s' : sv.State) : ℝ :=
  ∑ q', P.nextDFA q s' q' * Vprod (s', q')

/-- For a product process with a deterministic labelling `hL : P.labelling = .deterministic lf`
and under the overflow bound `hBound`, Julia's `W[s'] = V[s', dfa[q, lf[s']]]` equals the
expectation `nextDFAValue` of the model's next-DFA-state distribution `ProductProcess.nextDFA`
(`Models/Product.lean`). Restriction: deterministic labelling (`hL`); mode-independent. Scope
`abstract`.

Julia counterpart: `V[idx, dfa[state, lf[idx]]]` in `_bellman_helper!` (`src/bellman.jl`) for a
`ProductProcess` with a `DeterministicLabelling` (`src/models/ProductProcess.jl`). -/
theorem product_read_nextDFA {N : ℕ} {sv : StateVars n} {nQ : ℕ} {A Λ : Type*} [Fintype Λ]
    [DecidableEq Λ] (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (P : ProductProcess sv.State A (Fin nQ) Λ) {lf : Labelling sv.State Λ}
    (hL : P.labelling = .deterministic lf) (Vprod : sv.State × Fin nQ → ℝ) (q : Fin nQ)
    (s' : sv.State) :
    intermediateValues N P.dfa lf (productArray Vprod) q s' = nextDFAValue P Vprod q s' := by
  rw [intermediateValues, productGet_productArray hBound]
  simp [nextDFAValue, ProductProcess.nextDFA, hL, ProbVec.map_dirac]

end IntervalMDP.Index
