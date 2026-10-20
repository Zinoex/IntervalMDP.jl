import IntervalMDPProofs.Index.Product
import IntervalMDPProofs.VI.Safety

/-!
# The product Bellman operator and DFA reachability / safety

The Bellman operator of a `ProductProcess` (`bellman!` → `_bellman_helper!`, `src/bellman.jl`)
loops over the DFA states and, for each DFA state `state` (Lean `q`), calls the Bellman operator of
the underlying Markov process on a value vector `W` built from the product value array:

    for dfa_state_ci in states.dfa_states
        state = Int32(dfa_state_ci[1])
        if !isnothing(prop) && state ∈ terminal(prop)
            continue
        end
        map!(W, CartesianIndices(state_values(mp))) do idx
            return V[idx, dfa[state, lf[idx]]]
        end
        bellman!(…, selectdim(Vres, ndims(Vres), state), W, mp, mp_states; upper_bound, maximize)
    end

`Product.T` transcribes this loop (`bellmanLoop`, in the order of `states.dfa_states`, which is
`1..nQ` for the default `ProductUpdateSequence`), reusing the 6a index layer
(`Index/Product.lean`): `intermediateValues` for `W`, `productIndexInt` for the positions of
`selectdim(Vres, ndims(Vres), state)` (`writeSlice`). The inner `bellman!(…, mp, …)` is the robust
Bellman operator `Bellman.T` of the underlying `RMDP` (Phase 2).

## Results

* `T_eq_flat` (hypothesis `hBound`; deterministic labelling `hL`; all four modes): the product
  Bellman `T` on the product array storing `V` equals the Bellman operator
  `Bellman.T P.toRMDP sat strat V` of the flattened product `ProductProcess.toRMDP` on every
  DFA state outside `terminal` (Julia's `terminal(prop)`; empty for `prop = nothing`), and leaves
  the slices of the terminal DFA states unchanged (`skipTerminal`). The proof reads `W` with
  `Index.product_read_successor` (the DFA step uses the label of the **successor**) and identifies
  the expectation with the product ambiguity set (`T_toRMDP_apply`, any labelling).
* `dfaReach_eq` (hypothesis `hBound`; deterministic labelling `hL`; all four modes; time-invariant
  model): the arrays that
  Julia's `_value_iteration!` computes for `AbstractDFAReachability` (`initialize!` and
  `step_postprocess_value_function!`: `selectdim(current, ndims(current), reach(prop)) .= 1.0`;
  no final postprocessing) are the reachability iterates `VI.reachIter` of the flattened product
  with target `S × reach` (`dfaReachProperty`); for `AbstractDFASafety` (`.= -1.0` on
  `avoid(prop)`, final `current .+= 1.0`) the returned arrays are the exact-time reach-avoid
  iterates of the flattened product with avoid set `S × avoid` (`dfaSafetyProperty`, via
  `VI.safety_shift_eq`).

Restrictions (stated in each theorem): deterministic labelling (`DeterministicLabelling`; the
`ProbabilisticLabelling` method of `_bellman_helper!` is not transcribed, Observation O19), the
optimizing / no strategy cache (`Bellman.T`), a time-invariant model (`select_model(mp, k) = mp`),
the default full update sequence. Both theorems hold for all four satisfaction × strategy modes.
Proof scope: `abstract` (values in `ℝ`; integer index arithmetic under the overflow bound `hBound`).
-/

namespace IntervalMDP.Product

open Finset Index Bellman VI

variable {n : ℕ} {sv : StateVars n} {nQ : ℕ} {A Λ : Type*}

/-- A product value array: a column-major vector of length `(∏ state_values(mp)) * nQ` (DFA state
last), indexed 0-based.

Julia counterpart: the value arrays `V`, `Vres` (`value_function.current` / `previous`) of shape
`state_values(proc)` for a `ProductProcess` (`src/models/ProductProcess.jl`, `src/bellman.jl`). -/
abbrev ProductArray (sv : StateVars n) (nQ : ℕ) : Type :=
  Fin ((∏ i, sv.dims i) * nQ) → ℝ

/-! ### The flattened product, for any labelling -/

section Flat

variable [Fintype Λ] [DecidableEq Λ] (P : ProductProcess sv.State A (Fin nQ) Λ)

/-- The expectation of `V` under a lifted vector is the expectation of the next-DFA-state value
`nextDFAValue P V q` under the original vector: `⟨V, lift q x⟩ = ⟨nextDFAValue V q, x⟩`.

Julia counterpart: none (Lean-side proof device for `_bellman_helper!`, `src/bellman.jl`). -/
theorem dot_liftLinear (V : sv.State × Fin nQ → ℝ) (q : Fin nQ) (x : sv.State → ℝ) :
    OMax.dot V (P.liftLinear q x) = OMax.dot (nextDFAValue P V q) x := by
  simp only [OMax.dot, ProductProcess.liftLinear, LinearMap.coe_mk, AddHom.coe_mk, nextDFAValue,
    Fintype.sum_prod_type, Finset.sum_mul]
  exact Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun q' _ => by ring

/-- The expected values over the product ambiguity set of `(s, q)` are those over `Γ_{s,a}` of the
next-DFA-state value `nextDFAValue P V q`.

Julia counterpart: none (Lean-side proof device for `_bellman_helper!`, `src/bellman.jl`). -/
theorem expectations_toRMDP (V : sv.State × Fin nQ → ℝ) (s : sv.State) (q : Fin nQ) (a : A) :
    expectations (P.toRMDP.ambiguity (s, q) a) V =
      expectations (P.mdp.ambiguity s a) (nextDFAValue P V q) := by
  rw [ProductProcess.toRMDP_ambiguity, expectations, expectations,
    AmbiguitySet.vecs_map (P.coe_lift q), Set.image_image]
  congr 1
  funext x
  exact dot_liftLinear P V q x

/-- The Bellman operator of the flattened product at `(s, q)` is the Bellman operator of the
underlying RMDP at `s`, applied to the next-DFA-state value `nextDFAValue P V q` (any labelling,
all four satisfaction × strategy modes).

Julia counterpart: one iteration `state = q` of the DFA loop of `_bellman_helper!`
(`src/bellman.jl`): `bellman!(…, selectdim(Vres, ndims(Vres), state), W, mp, …)` with
`W[idx] = ∑ₗ prob · V[idx, dfa[state, l]]`. -/
theorem T_toRMDP_apply (sat : SatisfactionMode) (strat : StrategyMode)
    (V : sv.State × Fin nQ → ℝ) (s : sv.State) (q : Fin nQ) :
    Bellman.T P.toRMDP sat strat V (s, q) = Bellman.T P.mdp sat strat (nextDFAValue P V q) s := by
  have h : stateActionBellman P.toRMDP sat V (s, q) =
      stateActionBellman P.mdp sat (nextDFAValue P V q) s := by
    funext a
    cases sat <;> simp only [stateActionBellman, innerOpt, expectations_toRMDP P]
  simp only [Bellman.T, h]
  rfl

end Flat

/-! ### Deterministic labelling: the successor read -/

/-- The value `V (s', δ(q, L(s')))` that Julia reads for successor `s'` in DFA state `q`: the
product state reached when the MDP moves to `s'` and the DFA reads the label `L(s')` of the
successor.

Julia counterpart: `V[idx, dfa[state, lf[idx]]]` in `_bellman_helper!` (`src/bellman.jl`) for a
`DeterministicLabelling`, as a function of `idx` (Lean `s'`). -/
def successorValue (dfa : DFA (Fin nQ) Λ) (lf : Labelling sv.State Λ)
    (V : sv.State × Fin nQ → ℝ) (q : Fin nQ) (s' : sv.State) : ℝ :=
  V (s', dfa.δ q (lf.map s'))

/-- Under the overflow bound `hBound`, the vector `W` built for DFA state `q` from the product
array storing `V` is `successorValue` (from `Index.product_read_successor`; deterministic labelling;
mode-independent).

Julia counterpart: `map!(W, CartesianIndices(state_values(mp))) do idx; return V[idx,
dfa[state, lf[idx]]]; end` in `_bellman_helper!` (`src/bellman.jl`). -/
theorem intermediateValues_productArray {N : ℕ} (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (dfa : DFA (Fin nQ) Λ) (lf : Labelling sv.State Λ) (V : sv.State × Fin nQ → ℝ) (q : Fin nQ) :
    intermediateValues N dfa lf (productArray V) q = successorValue dfa lf V q :=
  funext (product_read_successor hBound dfa lf V q)

/-- For a deterministic labelling (`hL`), the next-DFA-state value is the successor read.

Julia counterpart: `V[idx, dfa[state, lf[idx]]]` in `_bellman_helper!` (`src/bellman.jl`) for a
`DeterministicLabelling`. -/
theorem nextDFAValue_deterministic [Fintype Λ] [DecidableEq Λ]
    (P : ProductProcess sv.State A (Fin nQ) Λ) {lf : Labelling sv.State Λ}
    (hL : P.labelling = .deterministic lf) (V : sv.State × Fin nQ → ℝ) (q : Fin nQ) :
    nextDFAValue P V q = successorValue P.dfa lf V q := by
  funext s'
  simp [nextDFAValue, successorValue, ProductProcess.nextDFA, hL, ProbVec.map_dirac]

/-! ### The product Bellman loop -/

open Classical in
/-- Writing the vector `x` into the slice `selectdim(Vres, ndims(Vres), state)`: the entry at the
Julia (1-based, `N`-bit `Int`) position `productIndexInt N (s, state)` of every MDP state `s`
becomes `x s`, all other entries are unchanged. The inner `bellman!` writes every `s` of the full
`mp_states` sweep; the positions are disjoint (`Index.selectdimIndex_bijOn`), so their order is
immaterial.

Julia counterpart: the writes `Vres[jₛ] = …` of `bellman!(…, selectdim(Vres, ndims(Vres), state),
W, mp, mp_states; …)` in `_bellman_helper!` (`src/bellman.jl`). -/
noncomputable def writeSlice (N : ℕ) (Vres : ProductArray sv nQ) (state : Fin nQ)
    (x : sv.State → ℝ) : ProductArray sv nQ :=
  fun i => if h : ∃ s, (productIndexInt N sv nQ (s, state)).toNat = toJulia i then x h.choose
    else Vres i

/-- Under the overflow bound `hBound`, the `Int` position of `(s, q)` is the 1-based position of
`productEquiv (s, q)`.

Julia counterpart: `LinearIndices(state_values(proc))[idx, q]` (`src/bellman.jl`). -/
theorem productIndexInt_toNat {N : ℕ} (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (z : sv.State × Fin nQ) :
    (productIndexInt N sv nQ z).toNat = toJulia (productEquiv sv nQ z) := by
  rw [productIndexInt_eq hBound, Int.toNat_natCast, productIndex_eq_toJulia]

/-- Under the overflow bound `hBound`, `writeSlice N Vres state x` at the position of `(s, q)` is
`x s` for `q = state` and `Vres` there otherwise.

Julia counterpart: `selectdim(Vres, ndims(Vres), state)[idx] = x[idx]` (`src/bellman.jl`). -/
theorem writeSlice_apply {N : ℕ} (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (Vres : ProductArray sv nQ) (state : Fin nQ) (x : sv.State → ℝ) (z : sv.State × Fin nQ) :
    writeSlice N Vres state x (productEquiv sv nQ z) =
      if z.2 = state then x z.1 else Vres (productEquiv sv nQ z) := by
  have key : ∀ s, (productIndexInt N sv nQ (s, state)).toNat = toJulia (productEquiv sv nQ z) ↔
      (s, state) = z := fun s => by
    rw [productIndexInt_toNat hBound, toJulia_injective.eq_iff, (productEquiv sv nQ).apply_eq_iff_eq]
  obtain ⟨s, q⟩ := z
  unfold writeSlice
  by_cases hq : q = state
  · subst hq
    have hex : ∃ s', (productIndexInt N sv nQ (s', q)).toNat = toJulia (productEquiv sv nQ (s, q)) :=
      ⟨s, (key s).2 rfl⟩
    rw [dif_pos hex, if_pos rfl]
    have hs := (key _).1 hex.choose_spec
    simp only [Prod.mk.injEq, and_true] at hs
    rw [hs]
  · have hex : ¬ ∃ s', (productIndexInt N sv nQ (s', state)).toNat =
        toJulia (productEquiv sv nQ (s, q)) := by
      rintro ⟨s', hs'⟩
      have h := (key s').1 hs'
      simp only [Prod.mk.injEq] at h
      exact hq h.2.symm
    rw [dif_neg hex, if_neg hq]

/-- The DFA loop of `_bellman_helper!`, a transcription over the list `states` of DFA states (Julia
`states.dfa_states`) in order: a DFA state in `terminal` is skipped (`continue`); otherwise `W` is
built by `intermediateValues` (the `map!` closure) and the Bellman operator `Bellman.T mp sat strat`
of the underlying Markov process is written into the slice `selectdim(Vres, ndims(Vres), state)`
(`writeSlice`). `V` is only read and `Vres` only written.

Julia counterpart: `for dfa_state_ci in states.dfa_states … end` in `_bellman_helper!(workspace,
strategy_cache, Vres, V, dfa, lf::DeterministicLabelling, mp, states; upper_bound, maximize, prop)`
(`src/bellman.jl`), with `upper_bound` ↔ `sat` and `maximize` ↔ `strat`. -/
noncomputable def bellmanLoop (N : ℕ) (mp : RMDP sv.State A)
    (dfa : DFA (Fin nQ) Λ) (lf : Labelling sv.State Λ) (sat : SatisfactionMode)
    (strat : StrategyMode) (terminal : Finset (Fin nQ)) (V : ProductArray sv nQ) :
    ProductArray sv nQ → List (Fin nQ) → ProductArray sv nQ
  | Vres, [] => Vres
  | Vres, state :: states =>
    if state ∈ terminal then bellmanLoop N mp dfa lf sat strat terminal V Vres states
    else bellmanLoop N mp dfa lf sat strat terminal V
      (writeSlice N Vres state (Bellman.T mp sat strat (intermediateValues N dfa lf V state)))
      states

/-- **The product Bellman operator** `Vres ← T V`: the DFA loop `bellmanLoop` over all DFA states
`1..nQ` in order (`List.finRange nQ`). `terminal` is Julia's `terminal(prop)` (`reach(prop)` for
`AbstractDFAReachability`, `avoid(prop)` for `AbstractDFASafety`), and `∅` for `prop = nothing`.

Julia counterpart: `bellman!(workspace, strategy_cache, Vres, V, model::ProductProcess; upper_bound,
maximize, prop)` → `_bellman_helper!(…, dfa, lf::DeterministicLabelling, mp, states; …)`
(`src/bellman.jl`) with `states = default_update_sequence(model)` (`src/update_sequence.jl`), whose
`dfa_states` is the full sweep `1..nQ`, and an optimizing or no strategy cache. -/
noncomputable def T (N : ℕ) (mp : RMDP sv.State A) (dfa : DFA (Fin nQ) Λ)
    (lf : Labelling sv.State Λ) (sat : SatisfactionMode) (strat : StrategyMode)
    (terminal : Finset (Fin nQ)) (Vres V : ProductArray sv nQ) : ProductArray sv nQ :=
  bellmanLoop N mp dfa lf sat strat terminal V Vres (List.finRange nQ)

/-- After the DFA loop over a duplicate-free list `states`, the entry of `(s, q)` is the Bellman
value `Bellman.T mp sat strat W_q s` if `q` was visited and not terminal, and unchanged otherwise
(hypothesis `hBound`).

Julia counterpart: the DFA loop of `_bellman_helper!` (`src/bellman.jl`). -/
theorem bellmanLoop_apply {N : ℕ} (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (mp : RMDP sv.State A) (dfa : DFA (Fin nQ) Λ) (lf : Labelling sv.State Λ)
    (sat : SatisfactionMode) (strat : StrategyMode) (terminal : Finset (Fin nQ))
    (V : ProductArray sv nQ) (states : List (Fin nQ)) (hnd : states.Nodup)
    (Vres : ProductArray sv nQ) (z : sv.State × Fin nQ) :
    bellmanLoop N mp dfa lf sat strat terminal V Vres states (productEquiv sv nQ z) =
      if z.2 ∈ states ∧ z.2 ∉ terminal then
        Bellman.T mp sat strat (intermediateValues N dfa lf V z.2) z.1
      else Vres (productEquiv sv nQ z) := by
  induction states generalizing Vres with
  | nil => simp [bellmanLoop]
  | cons state states ih =>
    rw [List.nodup_cons] at hnd
    by_cases ht : state ∈ terminal
    · rw [bellmanLoop, if_pos ht, ih hnd.2]
      by_cases hz : z.2 = state
      · simp [hz, ht]
      · simp [hz]
    · rw [bellmanLoop, if_neg ht, ih hnd.2, writeSlice_apply hBound]
      by_cases hz : z.2 = state
      · subst hz
        simp [ht, hnd.1]
      · simp [hz]

/-- The flattened-product value with the slices of the terminal DFA states left at `Vres`:
`(s, q) ↦ Vres (s, q)` for `q ∈ terminal`, `W (s, q)` otherwise.

Julia counterpart: `if !isnothing(prop) && state ∈ terminal(prop); continue; end` in
`_bellman_helper!` (`src/bellman.jl`), which leaves `selectdim(Vres, ndims(Vres), state)` unwritten
for terminal DFA states. -/
def skipTerminal (terminal : Finset (Fin nQ)) (Vres W : sv.State × Fin nQ → ℝ)
    (z : sv.State × Fin nQ) : ℝ :=
  if z.2 ∈ terminal then Vres z else W z

/-- **The product Bellman operator is the Bellman operator of the flattened product.** Under the
overflow bound `hBound : (∏ state_values(mp)) * nQ < 2 ^ (N - 1)` and for a deterministic labelling
(`hL : P.labelling = .deterministic lf`), Julia's per-DFA-state loop `T` applied to the product
arrays storing `Vres` and `V` gives the product array storing `Bellman.T P.toRMDP sat strat V` on
every DFA state outside `terminal`, and `Vres` on the terminal ones (`skipTerminal`; with
`terminal = ∅`, i.e. `prop = nothing`, the result is exactly the flattened `Bellman.T`,
`T_eq_flat_empty`). The successor read uses `Index.product_read_successor`: the DFA step reads the
label of the successor. The arrays are in Julia's layout: `writeSlice` writes at the `Int` index
`productIndexInt`, and `productArray` is pinned by the explicit formula of
`Index.productIndex_bijective` (`productIndex (s, q) = linear sv.dims s + (∏ sv.dims) * q.val`);
`T_read` states the result as Julia reads it (`productGet`).

Covers all four satisfaction × strategy modes; restriction: deterministic labelling. Scope
`abstract`.

Julia counterpart: `bellman!(…, Vres, V, model::ProductProcess; upper_bound, maximize, prop)` and
`_bellman_helper!` for a `DeterministicLabelling` (`src/bellman.jl`). -/
theorem T_eq_flat {N : ℕ} [Fintype Λ] [DecidableEq Λ]
    (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1)) (P : ProductProcess sv.State A (Fin nQ) Λ)
    {lf : Labelling sv.State Λ} (hL : P.labelling = .deterministic lf) (sat : SatisfactionMode)
    (strat : StrategyMode) (terminal : Finset (Fin nQ)) (Vres V : sv.State × Fin nQ → ℝ) :
    T N P.mdp P.dfa lf sat strat terminal (productArray Vres) (productArray V) =
      productArray (skipTerminal terminal Vres (Bellman.T P.toRMDP sat strat V)) := by
  funext i
  obtain ⟨z, rfl⟩ := (productEquiv sv nQ).surjective i
  rw [T, bellmanLoop_apply hBound _ _ _ _ _ _ _ _ (List.nodup_finRange nQ),
    intermediateValues_productArray hBound, ← nextDFAValue_deterministic P hL,
    ← T_toRMDP_apply P]
  by_cases hz : z.2 ∈ terminal <;> simp [skipTerminal, productArray, hz]

/-- With `terminal = ∅` (`prop = nothing`), the product Bellman operator is exactly the Bellman
operator of the flattened product (hypothesis `hBound`; deterministic labelling `hL`; all four
modes).

Julia counterpart: `bellman!(…, Vres, V, model::ProductProcess; upper_bound, maximize)` without
`prop` (`src/bellman.jl`). -/
theorem T_eq_flat_empty {N : ℕ} [Fintype Λ] [DecidableEq Λ]
    (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1)) (P : ProductProcess sv.State A (Fin nQ) Λ)
    {lf : Labelling sv.State Λ} (hL : P.labelling = .deterministic lf) (sat : SatisfactionMode)
    (strat : StrategyMode) (Vres V : sv.State × Fin nQ → ℝ) :
    T N P.mdp P.dfa lf sat strat ∅ (productArray Vres) (productArray V) =
      productArray (Bellman.T P.toRMDP sat strat V) := by
  rw [T_eq_flat hBound P hL]
  rfl

/-- Read form of `T_eq_flat` (hypothesis `hBound`; deterministic labelling `hL`; all four modes),
pinned to Julia's layout: Julia's `Vres[s, q]` after the product Bellman update (the `Int` linear index
`productIndexInt`, i.e. `linear sv.dims s + (∏ sv.dims) * q.val + 1`) is
`Bellman.T P.toRMDP sat strat V (s, q)` for `q ∉ terminal` and `Vres (s, q)` for `q ∈ terminal`
(all four modes).

Julia counterpart: `getindex(Vres, idx, q)` after `bellman!(…, Vres, V, model::ProductProcess; …)`
(`src/bellman.jl`). -/
theorem T_read {N : ℕ} [Fintype Λ] [DecidableEq Λ]
    (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1)) (P : ProductProcess sv.State A (Fin nQ) Λ)
    {lf : Labelling sv.State Λ} (hL : P.labelling = .deterministic lf) (sat : SatisfactionMode)
    (strat : StrategyMode) (terminal : Finset (Fin nQ)) (Vres V : sv.State × Fin nQ → ℝ)
    (s : sv.State) (q : Fin nQ) :
    productGet N (T N P.mdp P.dfa lf sat strat terminal (productArray Vres) (productArray V)) s q =
      skipTerminal terminal Vres (Bellman.T P.toRMDP sat strat V) (s, q) := by
  rw [T_eq_flat hBound P hL, productGet_productArray hBound]

/-! ### DFA reachability and safety -/

open Classical in
/-- Setting the slices of the DFA states `states` to the constant `c`: every entry at the Julia
(1-based, `N`-bit `Int`) position `productIndexInt N (s, q)` with `q ∈ states` becomes `c`.

Julia counterpart: `selectdim(value_function.current, ndims(value_function.current), reach(prop))
.= 1.0` (and `avoid(prop)`, `-1.0`) in `initialize!` / `step_postprocess_value_function!` for
`AbstractDFAReachability` / `AbstractDFASafety` (`src/specification.jl`). -/
noncomputable def setDFAStates (N : ℕ) (states : Finset (Fin nQ)) (c : ℝ)
    (V : ProductArray sv nQ) : ProductArray sv nQ :=
  fun i => if ∃ s, ∃ q ∈ states, (productIndexInt N sv nQ (s, q)).toNat = toJulia i then c
    else V i

/-- Under the overflow bound `hBound`, `setDFAStates N states c V` at the position of `(s, q)` is
`c` for `q ∈ states` and `V` there otherwise.

Julia counterpart: `selectdim(current, ndims(current), states) .= c` (`src/specification.jl`). -/
theorem setDFAStates_apply {N : ℕ} (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (states : Finset (Fin nQ)) (c : ℝ) (V : ProductArray sv nQ) (z : sv.State × Fin nQ) :
    setDFAStates N states c V (productEquiv sv nQ z) =
      if z.2 ∈ states then c else V (productEquiv sv nQ z) := by
  have key : (∃ s, ∃ q ∈ states, (productIndexInt N sv nQ (s, q)).toNat =
      toJulia (productEquiv sv nQ z)) ↔ z.2 ∈ states := by
    simp only [productIndexInt_toNat hBound, toJulia_injective.eq_iff,
      (productEquiv sv nQ).apply_eq_iff_eq]
    constructor
    · rintro ⟨s, q, hq, rfl⟩
      exact hq
    · intro h
      exact ⟨z.1, z.2, h, rfl⟩
  unfold setDFAStates
  simp only [key]

/-- The target of DFA reachability on the flattened product: the product states whose DFA state is
in `reach`, `S × reach`.

Julia counterpart: `selectdim(V, ndims(V), reach(prop))` for `AbstractDFAReachability`
(`src/specification.jl`), the product states that `initialize!` and
`step_postprocess_value_function!` set to `1.0`. -/
def productTarget (reach : Finset (Fin nQ)) : Finset (sv.State × Fin nQ) :=
  univ ×ˢ reach

/-- DFA reachability as a reachability property of the flattened product: reach `S × reach`.

Julia counterpart: `FiniteTimeDFAReachability(reach, time_horizon)` /
`InfiniteTimeDFAReachability(reach, convergence_eps)` (`src/specification.jl`); time horizon and
`convergence_eps` only enter the termination criterion. -/
def dfaReachProperty (reach : Finset (Fin nQ)) :
    ReachProperty (sv.State × Fin nQ) :=
  .reachability (productTarget reach)

/-- DFA safety as a safety property of the flattened product: avoid `S × avoid`.

Julia counterpart: `FiniteTimeDFASafety(avoid, time_horizon)` /
`InfiniteTimeDFASafety(avoid, convergence_eps)` (`src/specification.jl`). -/
def dfaSafetyProperty (avoid : Finset (Fin nQ)) :
    SafetyProperty (sv.State × Fin nQ) :=
  ⟨productTarget avoid⟩

/-- The value-iteration arrays for DFA reachability, a transcription of the loop of
`_value_iteration!`: `dfaReachIter 0` is the zero array (`ValueFunction(problem)`) with the slices
of `reach` set to `1` (`initialize!`); `dfaReachIter (k + 1)` is the product Bellman `T` with
`terminal = reach` (`prop = system_property(spec)`), `Vres = current` and `V = previous`, both
equal to `dfaReachIter k` after `nextiteration!`, followed by `step_postprocess_value_function!`
(the slices of `reach` set to `1`). `postprocess_value_function!` is the identity. The model is
time-invariant (`select_model(mp, k) = mp`).

Julia counterpart: `_value_iteration!` / `step!` (`src/robust_value_iteration.jl`) with an
`AbstractDFAReachability` property (`initialize!`, `step_postprocess_value_function!`,
`postprocess_value_function!`, `src/specification.jl`). -/
noncomputable def dfaReachIter (N : ℕ) (mp : RMDP sv.State A)
    (dfa : DFA (Fin nQ) Λ) (lf : Labelling sv.State Λ) (sat : SatisfactionMode)
    (strat : StrategyMode) (reach : Finset (Fin nQ)) : ℕ → ProductArray sv nQ
  | 0 => setDFAStates N reach 1 0
  | k + 1 => setDFAStates N reach 1
      (T N mp dfa lf sat strat reach (dfaReachIter N mp dfa lf sat strat reach k)
        (dfaReachIter N mp dfa lf sat strat reach k))

/-- The shifted value-iteration arrays for DFA safety, a transcription of the loop of
`_value_iteration!`: `dfaSafetyIter 0` is the zero array with the slices of `avoid` set to `-1`
(`initialize!`); `dfaSafetyIter (k + 1)` is the product Bellman `T` with `terminal = avoid`,
`Vres = V = dfaSafetyIter k`, followed by `step_postprocess_value_function!` (the slices of `avoid`
set to `-1`). The value Julia returns is `dfaSafetyPostprocess (dfaSafetyIter k)`.

Julia counterpart: `_value_iteration!` / `step!` (`src/robust_value_iteration.jl`) with an
`AbstractDFASafety` property (`src/specification.jl`). -/
noncomputable def dfaSafetyIter (N : ℕ) (mp : RMDP sv.State A)
    (dfa : DFA (Fin nQ) Λ) (lf : Labelling sv.State Λ) (sat : SatisfactionMode)
    (strat : StrategyMode) (avoid : Finset (Fin nQ)) : ℕ → ProductArray sv nQ
  | 0 => setDFAStates N avoid (-1) 0
  | k + 1 => setDFAStates N avoid (-1)
      (T N mp dfa lf sat strat avoid (dfaSafetyIter N mp dfa lf sat strat avoid k)
        (dfaSafetyIter N mp dfa lf sat strat avoid k))

/-- The final postprocessing of DFA safety, undoing the shift: every entry `+ 1`.

Julia counterpart: `postprocess_value_function!(value_function, ::AbstractDFASafety)`,
`value_function.current .+= 1.0` (`src/specification.jl`). -/
def dfaSafetyPostprocess (V : ProductArray sv nQ) : ProductArray sv nQ :=
  fun i => V i + 1

variable [Fintype Λ] [DecidableEq Λ]

/-- DFA reachability (hypothesis `hBound`; deterministic labelling `hL`; all four modes): the
`k`-th Julia array is the product array of the `k`-th reachability iterate `VI.reachIter` of the
flattened product with target `S × reach`.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) with
`AbstractDFAReachability` (`src/specification.jl`). -/
theorem dfaReachIter_eq {N : ℕ} (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (P : ProductProcess sv.State A (Fin nQ) Λ) {lf : Labelling sv.State Λ}
    (hL : P.labelling = .deterministic lf) (sat : SatisfactionMode) (strat : StrategyMode)
    (reach : Finset (Fin nQ)) (k : ℕ) :
    dfaReachIter N P.mdp P.dfa lf sat strat reach k =
      productArray (reachIter P.toRMDP sat strat (dfaReachProperty reach) k) := by
  induction k with
  | zero =>
    funext i
    obtain ⟨z, rfl⟩ := (productEquiv sv nQ).surjective i
    rw [dfaReachIter, setDFAStates_apply hBound]
    simp [productArray, reachIter, initializeValueFunction, dfaReachProperty, ReachProperty.reach,
      productTarget]
  | succ k ih =>
    funext i
    obtain ⟨z, rfl⟩ := (productEquiv sv nQ).surjective i
    rw [dfaReachIter, setDFAStates_apply hBound, ih, T_eq_flat hBound P hL]
    by_cases hz : z.2 ∈ reach <;>
      simp [productArray, reachIter, step, stepPostprocessValueFunction, dfaReachProperty,
        productTarget, skipTerminal, hz]

/-- DFA safety, shifted form (hypothesis `hBound`; deterministic labelling `hL`; all four modes):
the `k`-th Julia array is the product array of the `k`-th shifted safety iterate `VI.safetyIter`
of the flattened product with avoid set `S × avoid`.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`) with `AbstractDFASafety`
(`src/specification.jl`), before `postprocess_value_function!`. -/
theorem dfaSafetyIter_eq {N : ℕ} (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (P : ProductProcess sv.State A (Fin nQ) Λ) {lf : Labelling sv.State Λ}
    (hL : P.labelling = .deterministic lf) (sat : SatisfactionMode) (strat : StrategyMode)
    (avoid : Finset (Fin nQ)) (k : ℕ) :
    dfaSafetyIter N P.mdp P.dfa lf sat strat avoid k =
      productArray (safetyIter P.toRMDP sat strat (dfaSafetyProperty avoid) k) := by
  induction k with
  | zero =>
    funext i
    obtain ⟨z, rfl⟩ := (productEquiv sv nQ).surjective i
    rw [dfaSafetyIter, setDFAStates_apply hBound]
    simp [productArray, safetyIter, SafetyProperty.initializeValueFunction, dfaSafetyProperty,
      productTarget]
  | succ k ih =>
    funext i
    obtain ⟨z, rfl⟩ := (productEquiv sv nQ).surjective i
    rw [dfaSafetyIter, setDFAStates_apply hBound, ih, T_eq_flat hBound P hL]
    by_cases hz : z.2 ∈ avoid <;>
      simp [productArray, safetyIter, safetyStep, SafetyProperty.stepPostprocessValueFunction,
        dfaSafetyProperty, productTarget, skipTerminal, hz]

/-- DFA safety, returned value (hypothesis `hBound`; deterministic labelling `hL`; all four modes):
after the final `+ 1`, the `k`-th Julia array is the product array of the `k`-th iterate
`VI.reachIter` of the exact-time reach-avoid property of the flattened product with avoid set
`S × avoid` and reach set its complement (`SafetyProperty.toReachProperty`), via
`VI.safety_shift_eq`.

Julia counterpart: the `value_function.current` returned by `_value_iteration!`
(`src/robust_value_iteration.jl`) for `AbstractDFASafety` (`src/specification.jl`). -/
theorem dfaSafety_eq {N : ℕ} (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (P : ProductProcess sv.State A (Fin nQ) Λ) {lf : Labelling sv.State Λ}
    (hL : P.labelling = .deterministic lf) (sat : SatisfactionMode) (strat : StrategyMode)
    (avoid : Finset (Fin nQ)) (k : ℕ) :
    dfaSafetyPostprocess (dfaSafetyIter N P.mdp P.dfa lf sat strat avoid k) =
      productArray (reachIter P.toRMDP sat strat (dfaSafetyProperty avoid).toReachProperty k) := by
  rw [← safety_shift_eq, dfaSafetyIter_eq hBound P hL]
  funext i
  simp [dfaSafetyPostprocess, productArray, SafetyProperty.postprocessValueFunction]

/-- **DFA reachability and safety postprocessing.** Under the overflow bound
`hBound : (∏ state_values(mp)) * nQ < 2 ^ (N - 1)` and for a deterministic labelling
(`hL : P.labelling = .deterministic lf`), for every iteration count `k`:

* reachability: the array Julia's `_value_iteration!` holds after `k` steps for an
  `AbstractDFAReachability` property with `reach(prop) = reach` (slices of `reach` set to `1` by
  `initialize!` and every `step_postprocess_value_function!`, skipped by the product Bellman, no
  final postprocessing) is the product array of `VI.reachIter` of the flattened product
  `P.toRMDP` with target `S × reach` (`dfaReachProperty`);
* safety: the array Julia returns after `k` steps for an `AbstractDFASafety` property with
  `avoid(prop) = avoid` (slices of `avoid` set to `-1`, final `.+= 1.0`) is the product array of
  `VI.reachIter` of the exact-time reach-avoid property with avoid set `S × avoid` and reach set its
  complement (`(dfaSafetyProperty avoid).toReachProperty`).

Covers all four satisfaction × strategy modes; restrictions: deterministic labelling,
time-invariant model.
Scope `abstract`.

Julia counterpart: `_value_iteration!` / `step!` (`src/robust_value_iteration.jl`) with
`initialize!`, `step_postprocess_value_function!`, `postprocess_value_function!` for
`AbstractDFAReachability` / `AbstractDFASafety` (`src/specification.jl`) and `bellman!` for a
`ProductProcess` (`src/bellman.jl`). -/
theorem dfaReach_eq {N : ℕ} (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (P : ProductProcess sv.State A (Fin nQ) Λ) {lf : Labelling sv.State Λ}
    (hL : P.labelling = .deterministic lf) (sat : SatisfactionMode) (strat : StrategyMode)
    (reach avoid : Finset (Fin nQ)) (k : ℕ) :
    dfaReachIter N P.mdp P.dfa lf sat strat reach k =
        productArray (reachIter P.toRMDP sat strat (dfaReachProperty reach) k) ∧
      dfaSafetyPostprocess (dfaSafetyIter N P.mdp P.dfa lf sat strat avoid k) =
        productArray (reachIter P.toRMDP sat strat (dfaSafetyProperty avoid).toReachProperty k) :=
  ⟨dfaReachIter_eq hBound P hL sat strat reach k, dfaSafety_eq hBound P hL sat strat avoid k⟩

/-- Read form of `dfaReach_eq` for reachability (hypothesis `hBound`; deterministic labelling `hL`;
all four modes; time-invariant model), pinned to Julia's layout: Julia's `current[s, q]` after `k` steps (`Int` linear index
`productIndexInt`) is `VI.reachIter` of the flattened product at `(s, q)` (all four modes).

Julia counterpart: `getindex(value_function.current, idx, q)` in `_value_iteration!`
(`src/robust_value_iteration.jl`) for `AbstractDFAReachability` (`src/specification.jl`). -/
theorem dfaReach_read {N : ℕ} (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (P : ProductProcess sv.State A (Fin nQ) Λ) {lf : Labelling sv.State Λ}
    (hL : P.labelling = .deterministic lf) (sat : SatisfactionMode) (strat : StrategyMode)
    (reach : Finset (Fin nQ)) (k : ℕ) (s : sv.State) (q : Fin nQ) :
    productGet N (dfaReachIter N P.mdp P.dfa lf sat strat reach k) s q =
      reachIter P.toRMDP sat strat (dfaReachProperty reach) k (s, q) := by
  rw [dfaReachIter_eq hBound P hL, productGet_productArray hBound]

/-- Read form of `dfaReach_eq` for safety (hypothesis `hBound`; deterministic labelling `hL`; all
four modes; time-invariant model), pinned to Julia's layout: Julia's returned `current[s, q]` after `k` steps is `VI.reachIter`
of the exact-time reach-avoid form of the flattened product at `(s, q)` (all four modes).

Julia counterpart: `getindex(value_function.current, idx, q)` returned by `_value_iteration!`
(`src/robust_value_iteration.jl`) for `AbstractDFASafety` (`src/specification.jl`). -/
theorem dfaSafety_read {N : ℕ} (hBound : (∏ i, sv.dims i) * nQ < 2 ^ (N - 1))
    (P : ProductProcess sv.State A (Fin nQ) Λ) {lf : Labelling sv.State Λ}
    (hL : P.labelling = .deterministic lf) (sat : SatisfactionMode) (strat : StrategyMode)
    (avoid : Finset (Fin nQ)) (k : ℕ) (s : sv.State) (q : Fin nQ) :
    productGet N (dfaSafetyPostprocess (dfaSafetyIter N P.mdp P.dfa lf sat strat avoid k)) s q =
      reachIter P.toRMDP sat strat (dfaSafetyProperty avoid).toReachProperty k (s, q) := by
  rw [dfaSafety_eq hBound P hL, productGet_productArray hBound]

end IntervalMDP.Product
