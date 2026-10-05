# Lean Proofs for Existing VI/Bellman Algorithms — Specification (Julia + Lean)

> Migration spec. IntervalMDP.jl has **no Lean project** today, so every existing VI/Bellman
> algorithm is a legacy verification gap. This spec covers the whole migration but is
> executed as **separate `/harness` runs, one per phase** (§ Phases). Set the line below
> before each run. Every agent scopes its mapping-table rows and its acceptance criteria to
> that phase. Earlier phases' theorems must stay green. A phase may not start until the
> previous phase has merged.

**Active phase: 0**

## Objective *(required)*

Create a Lean 4 project inside IntervalMDP.jl with three parts:

1. **Well-structured, readable models** of everything the algorithms operate on: ambiguity sets, IMDPs, factored IMDPs, DFAs, labellings, product processes, strategies, properties and specifications (§ Lean Model Structure & Readability).
2. **Correct indexing.** Machine-checked proofs that every index computation the Julia code uses selects the intended ambiguity set, value entry or successor (§ Indexing Correctness).
3. **Algorithm theorems** for the VI/Bellman algorithms that already exist in `src/`. For every algorithm that computes an approximation rather than the exact value, there is a **soundness theorem**: the approximation errs only in the conservative direction, and this survives value iteration end to end (§ Approximation Soundness).

The goal is to turn the legacy gaps in `harness/specs/inventory-intervalmdp.md` into `proved` entries, at **abstract** proof scope (exact real arithmetic), with traceability to the Julia code.

Adds or semantically changes a VI/Bellman algorithm: **no**. The Julia code is not changed. If a proof shows that the Julia behavior differs from its documented claim, the Dev agent reports it as a finding and stops that row. It does not change the Julia code or weaken the theorem to make it pass (§ Findings policy).

Model families: interval MDP (IMDP), factored interval MDP (fIMDP), product of IMDP × DFA. Definitions must be written over a general ambiguity set where the proof allows (a nonempty closed set of distributions — **not** necessarily convex, see § Factored ambiguity is not convex), and specialised to intervals only for the interval-specific results. This keeps L1-MDPs and mixtures open for later.

## Commands / Toolchain *(required)*

- Target root: `/home/fresen/.julia/dev/IntervalMDP`
- Julia version / compat: `[compat] julia = "1.11"`; local `julia` is 1.13.1 (satisfies compat).
- Julia instantiate: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- Julia test: `julia --project=. -e 'using Pkg; Pkg.test()'`
- Lean root: `<root>/lean` (new; lake package `IntervalMDPProofs`, library root `IntervalMDPProofs.lean`).
- Lean toolchain: **Phase 0 sets the pin.** Use Mathlib, and copy `lean-toolchain` verbatim from the Mathlib release tag that `lakefile.toml` pins. Only `leanprover/lean4:v4.33.0-rc2` is installed locally (`~/.elan`, which is not on `PATH`; call `~/.elan/bin/lake`). Fetching a different toolchain or the Mathlib cache (`lake exe cache get`) needs network access, and this spec approves that **once, in Phase 0 only**. After Phase 0, the toolchain must never change silently.
- Lean build: `lake build` (run from `lean/`)
- Lean axiom check: `lake env lean lean/AxiomCheck.lean`. This file contains one `#print axioms <thm>` line per theorem in the mapping tables, and the verifier compares its output against the approved axioms.
- GPU test: not applicable (§ CPU / GPU Matrix).
- Benchmark: not in scope.

## Julia Behavior & Tests *(required)*

- No Julia source changes. `src/` must be byte-identical to the base ref. `test/` may only gain the optional cross-check files described below.
- **Optional cross-check tests** (these show the Lean model describes the Julia code; they are not a substitute for the proofs):
  - Phase 1: add `test/base/omax_reference.jl`. It compares `IntervalMDP.bellman` on small dense and sparse IMDPs against a brute-force LP over the interval polytope (HiGHS/JuMP are already dependencies), for both `upper_bound` values, Float64, with tolerance `1e-9`. Use at least these cases:
    - degenerate interval (`l = u`), where the budget is 0;
    - ties in `V`;
    - all of the budget in a single successor;
    - a sparse column with an empty gap.
  - Phase 1: add `test/base/indexing_reference.jl`. For random small factored shapes, it checks that `sub2ind(marginal, a, s)` equals `LinearIndices((action_vars[action_indices]..., source_dims...))[a[action_indices]..., s[state_indices]...]`. This is the closed form that the Lean index theorems state.
  - Phase 5: add `test/base/approx_soundness_reference.jl`. On small fIMDPs, it checks that the McCormick and recursive O-max values lie on the conservative side of vertex enumeration (the exact value).
- No wall-clock thresholds.

## Phases

| Phase | Scope | Depends on |
|---|---|---|
| 0 | Lean scaffold and Mathlib pin, `AxiomCheck.lean`, inventory file, **all model definitions** (§ Lean Model Structure & Readability) with their well-formedness theorems, and the generic approximation-soundness lemmas | — |
| 1 | **Indexing**: Julia 1-based ↔ Lean `Fin`; `Marginal.sub2ind`; factored state ↔ linear index; sparse column ↔ support; sort permutation. **O-maximization** exactness (dense and sparse) | 0 |
| 2 | Robust Bellman operator: monotonicity, translation equivariance, sup-norm non-expansiveness; strategy extraction and policy evaluation | 1 |
| 3 | Robust VI per property: reachability, reach-avoid, safety (via the −1/+1 shift), reward, expected exit time. **Soundness** of stopping early, and of the synthesized strategy | 2 |
| 4 | Interval value iteration (IVI): **bracket soundness** of `V_lower` and `V_upper` | 3 |
| 5 | Factored IMDPs: marginal-product indexing, vertex enumeration exactness, **soundness** of recursive O-max and of LP McCormick, end-to-end VI soundness | 2, 3 |
| 6 | Product with DFA: **product-state indexing**, product Bellman equals the Bellman of the flattened product, DFA reachability/safety | 3 |

Phase 0 passes when:

- `lake build` succeeds with the pinned toolchain;
- every Phase 0 row of the mapping tables is proved;
- `harness/specs/inventory-intervalmdp.md` exists (from `TEMPLATE-onboarding-inventory.md`), listing every algorithm row with status `none` and every model, index and approximation row with its status.

## Lean Model Structure & Readability *(required)*

The models are the part a reviewer reads first. They must be understandable by someone who knows the IntervalMDP.jl docs but not Lean.

### Layout

One concept per file under `lean/IntervalMDPProofs/Models/`. Files build on each other in the order listed:

| Lean file | Lean structure / type | Julia counterpart | Invariants carried as structure fields |
|---|---|---|---|
| `Models/Distribution.lean` | `ProbVec S` (`S → ℝ` with `nonneg` and `sum_eq_one`) | probability columns in `IntervalAmbiguitySets` | nonnegative; sums to 1 |
| `Models/AmbiguitySet.lean` | `AmbiguitySet S` (a `Set (ProbVec S)`) and the predicate `AmbiguitySet.WellFormed` | `AbstractAmbiguitySet`, `PolytopicAmbiguitySet` | nonempty, closed (hence compact, as a subset of the simplex). Convexity is a separate predicate `AmbiguitySet.IsConvex`, required only by results that need it (for example the interval/O-max results) |
| `Models/IntervalAmbiguity.lean` | `IntervalAmbiguity S` {`lower`, `upper`} with `toSet`, `gap`, `budget` | `IntervalAmbiguitySet`, `IntervalAmbiguitySets` (`lower`, `gap`, `budget`) | `0 ≤ lower ≤ upper ≤ 1`, `Σ lower ≤ 1 ≤ Σ upper` (matches `checkprobabilities` in `IntervalAmbiguitySets.jl`) |
| `Models/RMDP.lean` | `RMDP S A` {`available`, `ambiguity`} | `FactoredRobustMarkovDecisionProcess` with one marginal (`IsIMDP` / `IsRMDP`), `AllAvailableActions` / `ListAvailableActions` / `TimeVaryingAvailableActions` | `available s` is a nonempty `Finset`; every `ambiguity s a` is `WellFormed` |
| `Models/IMDP.lean` | `IMDP S A` (extends `RMDP` with interval ambiguity), and `IMDP.toRMDP` | `IntervalMarkovDecisionProcess(...)`, `IntervalMarkovChain` | inherited, plus the interval invariants |
| `Models/Factored.lean` | `StateVars`/`ActionVars` (dimension vectors), `Marginal` {`stateIndices`, `actionIndices`, `sets`}, `FactoredIMDP` {`marginals`}, and `FactoredIMDP.productSet` (the literal product of the marginal sets) and `FactoredIMDP.toRMDP` whose ambiguity **is** `productSet` — not its convex hull | `Marginal`, `FactoredRobustMarkovDecisionProcess` (`IsFIMDP`) | dependency indices in range (matches `checkindices`) and strictly increasing (a Lean-side restriction that `checkindices` does not enforce; recorded in the inventory); one interval set per conditioning tuple |
| `Models/DFA.lean` | `DFA Q Σ` {`δ`, `q₀`, `accepting`}; `Labelling S Σ` (deterministic) and `ProbLabelling S Σ` | `DFA`, `TransitionFunction`, `DeterministicLabelling`, `ProbabilisticLabelling` | `δ` is total; probabilistic labelling rows are `ProbVec`s |
| `Models/Product.lean` | `ProductProcess` {`mdp`, `dfa`, `labelling`}, and `ProductProcess.toRMDP` | `ProductProcess` | — |
| `Models/Strategy.lean` | `StationaryStrategy S A`, `TimeVaryingStrategy S A` (horizon-indexed), plus validity w.r.t. `available` | `StationaryStrategy`, `TimeVaryingStrategy`, `GivenStrategyCache` | the chosen action is available |
| `Models/Specification.lean` | `inductive Property` (one constructor per Julia property struct), `SatisfactionMode` (`pessimistic`/`optimistic`), `StrategyMode` (`maximize`/`minimize`), `Specification` | `Property` subtypes, `SatisfactionMode`, `StrategyMode`, `Specification` in `src/specification.jl` | reach ∩ avoid = ∅; `0 < discount`; horizon ≥ 1 for finite-time properties (match `checkreward`/`checkdisjoint`) |

### Style rules (checked by the verifier and QE as acceptance criteria)

- **Invariants live in structures.** A model's validity conditions are fields of its structure, not hypotheses repeated in every theorem. Theorems take a `(M : IMDP S A)`, not `(lower upper : …) (h₁ : …) (h₂ : …) …`.
- **Names match Julia.** Field and definition names follow the Julia names (`lower`, `gap`, `budget`, `available`, `stateIndices`, `reach`, `avoid`, `discount`, `timeHorizon`). Where Lean conventions force a change (camelCase), the docstring says which Julia name it is.
- **Every definition and structure has a docstring.** The docstring gives the mathematical meaning (with the formula from the Julia docstring where one exists) and the Julia type or function plus file it models.
- **Small named definitions over inline lambdas.** Theorem statements use named definitions (for example `IntervalAmbiguity.toSet`, `Bellman.T`, `VI.reachIter`), so each statement reads as a sentence. No statement longer than about 6 lines after formatting. Introduce `notation` only where it appears in the Julia docs (for example `⟨p, V⟩`).
- **One model hierarchy.** IMDPs, fIMDPs and products all convert to `RMDP` via `toRMDP`, and the general Bellman/VI theorems are proved **once** for `RMDP`. No copy-pasted operator per model family.
- **Generic over finite types.** Use `[Fintype S] [DecidableEq S]`, never a hard-coded `Fin n`, except in the indexing layer (§ Indexing Correctness), which is exactly where Julia's integer indices are modeled.
- **Worked example.** `Models/Examples.lean` builds the 3-state IMDP from the `RobustValueIteration` docstring (`src/robust_value_iteration.jl`) and the 2-variable fIMDP from the `FactoredRobustMarkovDecisionProcess` docstring, with every invariant discharged (for example by `norm_num`). This shows the models can express the package's own examples.
- **Overview document.** `lean/README.md` gives the model hierarchy as a diagram or list, a Julia↔Lean glossary (type ↔ structure, function ↔ definition), the file map, and how to build. Keep it current every phase.
- `lake build` emits no linter warnings in `IntervalMDPProofs/` (unused variables, deprecated names). Mathlib's `docBlame` linter is turned on for the library.

## Factored ambiguity is not convex *(required)*

The ambiguity set of a factored IMDP is the set of product distributions `{⊗ᵢ pᵢ : pᵢ ∈ Pᵢ}`. This set is **not convex** in general: with two binary variables, the point masses on `(0,0)` and `(1,1)` are products, but their midpoint is not. This is known in the literature (arXiv:2411.11803 by the package author; arXiv:2508.00707 by Schnitzer et al.), so the Lean models must reflect it rather than work around it:

- `FactoredIMDP.toRMDP` uses the literal `productSet`. Do **not** replace it by its closed convex hull. A hull may appear only as a proof device inside a lemma, never in a model definition.
- `IntervalMDP.FactoredIMDP.productSet_not_convex` records the counterexample as a theorem.
- The general `RMDP` results (Phase 2 onwards: `T_mono`, `T_add_const`, `T_nonexpansive`, VI and the soundness lift) must therefore hold **without** convexity. They need only nonempty and closed sets of distributions, so each `opt` over a set is attained.
- The exact value `V*` for a factored IMDP, which A2 and A3 compare against, is defined with `productSet`.

The factored algorithms follow published results. Each Lean proof must follow the cited argument, and its docstring must cite the paper (arXiv ID and the result number):

| Julia algorithm | Claim | Source |
|---|---|---|
| Vertex enumeration (`FactoredVertexIteratorWorkspace`) | **exact**: the opt over `productSet` equals the opt over products of marginal vertices | Schnitzer et al., arXiv:2508.00707 |
| LP McCormick relaxation (`FactoredIntervalMcCormickWorkspace`, binary tree) | **sound** (conservative w.r.t. the exact value) | Schnitzer et al., arXiv:2508.00707 |
| Recursive O-max (`FactoredIntervalOMaxWorkspace`) | **sound**: the tree reduction is conservative | arXiv:2411.11803 (package author) |

If a Lean proof shows that the Julia code departs from the paper's algorithm (for example in the reduction order or tree shape), that is a Finding, not a reason to change the theorem.

## Indexing Correctness *(required)*

Julia computes indices by hand in several places. Lean models these index functions exactly, over `Fin`, and proves they are correct. Julia is 1-based and Lean's `Fin n` is 0-based. The conversion is defined **once**, in `Index/Julia.lean` (`toJulia : Fin n → ℕ := (· + 1)` and its inverse on `1..n`), and every index theorem is stated through it.

Index obligations (all in the mapping table):

- **Linear ↔ Cartesian (column-major).** `LinearIndices`/`CartesianIndices` over a dims vector is a bijection, with the first dimension varying fastest. Factored states and value arrays `V[I]` use this.
- **`Marginal.sub2ind`.** The loop in `src/probabilities/Marginal.jl` equals the column-major linear index of `(action[actionIndices]…, source[stateIndices]…)`, actions first. As a consequence:
  - it is a bijection onto `1..Π action_vars × Π source_dims`;
  - it depends only on the variables the marginal conditions on (two global states/actions that agree on those variables get the same ambiguity set);
  - it matches the column layout documented in the `FactoredRobustMarkovDecisionProcess` docstring.
- **`IntervalAmbiguitySets.sub2ind`** returns `jₛ[1]` and ignores the action. Prove it is only reached where that is correct, by showing that every call site goes through `Marginal`, or for single-action models. If it can be reached with more than one action, that is a **finding**.
- **Sparse support alignment.** Under the `SparseMatrixCSC` column invariant (strictly increasing row indices, with `nzval` aligned to `rowval`), `zip(V[support], nonzeros(gap))` pairs `V[i]` with `gap[i]` for exactly the support rows. Entries outside the support have `gap = 0`.
- **Sort permutation.** The `perm` from `sortperm!(…; rev = upper_bound)` is a permutation of `1..n` and orders `V` correctly. The greedy loop visits every index exactly once, and stopping early only skips indices whose allocation would be 0.
- **Product-state indexing.** `V[idx, dfa[state, lf[idx]]]` in the product Bellman (`src/bellman.jl`) reads the value at the successor product state `(s', δ(q, L(s')))`. The DFA transition uses the *successor's* label. `selectdim(Vres, ndims(Vres), state)` writes the slice for DFA state `q`. Together these make a bijection `S × Q ≃ Fin (|S|·|Q|)` with the DFA state as the last (slowest) dimension.
- **Strategy indexing.** `CartesianIndex(strategy_cache[jₛ])` returns an action that is in `available(model, jₛ)`, for strategies that pass validation.

## Approximation Soundness *(required)*

Definition, stated once in `Approx/Sound.lean`. A computed value `Ṽ` is **sound** for a specification with satisfaction mode `m` when `Ṽ ≤ V*` for `pessimistic` and `Ṽ ≥ V*` for `optimistic`, where `V*` is the exact robust value of the same specification, in pointwise order on states. Proofs must use this definition `Sound m Ṽ V*`, not a re-derived inequality in each file.

Every place where IntervalMDP.jl returns something other than the exact value, and the theorem required:

| # | Approximation | Where | Required theorem |
|---|---|---|---|
| A1 | **Generic lift**: a sound one-step operator stays sound through value iteration | Phase 0, `Approx/Lift.lean` | `IntervalMDP.Approx.iter_sound`: if `T'` is sound w.r.t. `T` pointwise and `T` (or `T'`) is monotone, then every iterate of `T'` is sound w.r.t. the matching iterate of `T`, and so are their limits or fixed points when they exist. All per-algorithm soundness results must go through this lemma rather than re-proving induction each time. |
| A2 | Recursive O-max / tree reduction (fIMDP) | Phase 5 | one step is sound w.r.t. the exact opt over `productSet` (arXiv:2411.11803); end-to-end VI is sound via A1 |
| A3 | LP McCormick relaxation (fIMDP) | Phase 5 | one step (mathematical LP optimum) is sound w.r.t. the exact opt over `productSet` (arXiv:2508.00707); end-to-end VI is sound via A1 |
| A4 | Stopping infinite-horizon reachability / reach-avoid / exit time at a finite k | Phase 3 | the iterates from below are sound lower bounds: `V_k ≤ lfp` for every k, including the k the termination criterion stops at. For optimistic mode, state precisely which direction holds. Where `V_k ≤ V*` is not conservative for the mode, say so in the inventory as a limitation; do not claim it. |
| A5 | Stopping infinite-horizon reward at the ε-criterion | Phase 3 | the explicit error bound `‖V_k − V*‖∞ ≤ ν/(1−ν)·‖V_k − V_{k−1}‖∞` (an interval around `V_k` that contains `V*`) |
| A6 | IVI bounds | Phase 4 | `V_lower_k ≤ V* ≤ V_upper_k` under the strategy coupling in `ivi_step!`, for every mode. Stopping on the initial-state gap gives `V*` within `ε` on the initial states. |
| A7 | Synthesized strategy | Phase 3 | finite horizon: the returned time-varying strategy *achieves* the computed `V_K` (evaluating it gives `V_K`). Infinite horizon: the returned stationary strategy's value is sound w.r.t. the computed value, **or** a counterexample is filed under Findings (greedy stationary strategies are known to fail this for maximizing reachability with end components). |
| A8 | Given-strategy verification (`NonOptimizingStrategyCache`) | Phase 2 | policy evaluation `T_π` is exact for the given strategy, and is sound w.r.t. the optimal `T` in the strategy-mode direction |

## Algorithm ↔ Theorem Mapping *(required)*

The verifier checks exactly these theorem names for the phase being run. Names are fixed by this spec. If a name needs to change, update this spec before the run. Every Lean file lives under `lean/IntervalMDPProofs/`.

Notation used below:

- `P(l,u) = {p ∈ ProbVec S : l ≤ p ≤ u}` (`IntervalAmbiguity.toSet`).
- `opt` is max when `upper_bound = true` (optimistic) and min otherwise.
- `T V s = opt_strat_{a ∈ available s} opt_{p ∈ ambiguity s a} ⟨p,V⟩`, where `opt_strat` is max if `maximize` and min otherwise.

### Models (Phase 0)

| Model | Julia | Lean definition | Lean theorem(s) |
|---|---|---|---|
| Interval ambiguity set | `IntervalAmbiguitySet` — `src/probabilities/IntervalAmbiguitySets.jl` | `IntervalMDP.IntervalAmbiguity.toSet` — `Models/IntervalAmbiguity.lean` | `IntervalMDP.IntervalAmbiguity.toSet_wellFormed` (nonempty because `Σl ≤ 1 ≤ Σu`; closed); `IntervalMDP.IntervalAmbiguity.toSet_convex`; `IntervalMDP.IntervalAmbiguity.budget_eq` (`budget = 1 − Σ lower ≥ 0`) |
| IMDP | `IntervalMarkovDecisionProcess` — `src/models/IntervalMarkovDecisionProcess.jl` | `IntervalMDP.IMDP.toRMDP` — `Models/IMDP.lean` | `IntervalMDP.IMDP.toRMDP_wellFormed` |
| Factored IMDP | `FactoredRobustMarkovDecisionProcess`, `Marginal` — `src/models/FactoredRobustMarkovDecisionProcess.jl`, `src/probabilities/Marginal.jl` | `IntervalMDP.FactoredIMDP.toRMDP` — `Models/Factored.lean` | `IntervalMDP.FactoredIMDP.toRMDP_wellFormed` (every product of marginal distributions is a `ProbVec` on the joint state; `productSet` is nonempty and closed); `IntervalMDP.FactoredIMDP.productSet_not_convex` |
| Product process | `ProductProcess` — `src/models/ProductProcess.jl` | `IntervalMDP.ProductProcess.toRMDP` — `Models/Product.lean` | `IntervalMDP.ProductProcess.toRMDP_wellFormed` (for both deterministic and probabilistic labelling) |
| Examples | docstrings of `RobustValueIteration`, `FactoredRobustMarkovDecisionProcess` | `IntervalMDP.Examples.docIMDP`, `IntervalMDP.Examples.docFIMDP` — `Models/Examples.lean` | (definitions that typecheck; no theorem) |
| Generic soundness lift | — | `IntervalMDP.Approx.Sound` — `Approx/Sound.lean` | `IntervalMDP.Approx.iter_sound` — `Approx/Lift.lean` (A1) |

### Indexing (Phase 1, except where tagged)

| Ph | Index computation | Julia (function — file) | Lean definition | Lean theorem(s) |
|---|---|---|---|---|
| 1 | Column-major linear index | `LinearIndices`, `CartesianIndices` (Base) as used for `V[I]` and states | `IntervalMDP.Index.linear` — `Index/Linear.lean` | `IntervalMDP.Index.linear_bijective`; `IntervalMDP.Index.linear_succ_first` (the first dimension varies fastest) |
| 1 | Marginal → ambiguity-set column | `sub2ind(::Marginal, action, source)` — `src/probabilities/Marginal.jl` | `IntervalMDP.Index.marginalSub2ind` — `Index/Marginal.lean` (a literal transcription of the Julia loop) | `IntervalMDP.Index.marginalSub2ind_eq_linear` (equals `linear` on `(action[actionIndices]…, source[stateIndices]…)`); `IntervalMDP.Index.marginalSub2ind_bijective`; `IntervalMDP.Index.marginalSub2ind_depends_only` (agreement on the conditioned variables gives the same index) |
| 1 | Non-factored set lookup | `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ) = jₛ[1]` — `src/probabilities/IntervalAmbiguitySets.jl` | `IntervalMDP.Index.intervalSub2ind` — `Index/Marginal.lean` | `IntervalMDP.Index.intervalSub2ind_correct` (correct under the conditions in which it is reachable; see § Indexing Correctness) |
| 1 | Sparse support pairing | `state_action_bellman(::SparseIntervalOMaxWorkspace, …)` zip over `support`/`nonzeros` — `src/bellman.jl` | `IntervalMDP.Index.SparseCol` (CSC column invariant) — `Index/Sparse.lean` | `IntervalMDP.Index.sparse_zip_correct` |
| 1 | Sort permutation | `sortperm!(perm, V; rev = upper_bound)` — `src/bellman.jl` | `IntervalMDP.Index.SortedPerm` — `Index/Perm.lean` | `IntervalMDP.Index.sortedPerm_bijective`; `IntervalMDP.Index.greedy_visits_once` |
| 2 | Strategy lookup | `CartesianIndex(strategy_cache[jₛ])` — `src/bellman.jl`, `src/strategy_cache.jl` | `IntervalMDP.Index.strategyAction` — `Index/Strategy.lean` | `IntervalMDP.Index.strategyAction_available` |
| 5 | Factored successor index | `CartesianIndices(num_target.(ambiguity_sets))` in vertex enumeration and the factored O-max loops — `src/bellman.jl` | reuses `IntervalMDP.Index.linear` | `IntervalMDP.Index.factored_successor_eq` (the product-of-marginals index `I` addresses `V[I]` for the joint successor state) |
| 6 | Product state | `V[idx, dfa[state, lf[idx]]]`, `selectdim(Vres, ndims(Vres), state)` — `src/bellman.jl` | `IntervalMDP.Index.productIndex` — `Index/Product.lean` | `IntervalMDP.Index.productIndex_bijective` (`S × Q ≃` the Julia array, DFA state last); `IntervalMDP.Index.product_read_successor` (the read is at `(s', δ(q, L(s')))`) |

### Algorithms

| Ph | Algorithm | Julia entry point (function — file) | Lean definition (name — file) | Lean theorem(s) | Correspondence note |
|---|---|---|---|---|---|
| 1 | O-maximization, dense | `state_action_bellman(::DenseIntervalOMaxWorkspace, …)`, `gap_value(V, gap, budget, perm)`, `bellman_precomputation!` — `src/bellman.jl` | `IntervalMDP.OMax.omax` — `OMax.lean` (takes an `IntervalAmbiguity` and a `SortedPerm`; returns `⟨lower,V⟩ + greedy(gap, budget, perm)`) | `IntervalMDP.OMax.omax_mem` (the greedy `p` is in `toSet`); `IntervalMDP.OMax.omax_eq_sSup` (descending `perm` gives `sSup`); `IntervalMDP.OMax.omax_eq_sInf` (ascending gives `sInf`); `IntervalMDP.OMax.omax_tie_invariant` | The Julia loop breaks early at `budget ≤ 0`; Lean models this as `p_i = min(budget_i, gap_i)` and uses `greedy_visits_once`. |
| 1 | O-maximization, sparse | `state_action_bellman(::SparseIntervalOMaxWorkspace, …)`, `gap_value(Vp, budget)` — `src/bellman.jl` | `IntervalMDP.OMax.omaxSparse` — `OMax.lean` | `IntervalMDP.OMax.omaxSparse_eq_omax` | Uses `sparse_zip_correct`. |
| 2 | Robust Bellman operator | `bellman!` → `state_bellman!` with `OptimizingStrategyCache` — `src/bellman.jl` | `IntervalMDP.Bellman.T` — `Bellman.lean` (on `RMDP`) | `IntervalMDP.Bellman.T_mono`; `IntervalMDP.Bellman.T_add_const`; `IntervalMDP.Bellman.T_nonexpansive`; `IntervalMDP.Bellman.T_interval_eq_omax` (for `IMDP.toRMDP`, agrees with `omax` looked up via `marginalSub2ind`) | Proved for general `RMDP`, then specialised. |
| 2 | Strategy extraction and evaluation | `extract_strategy!` — `src/strategy_cache.jl`; `NonOptimizingStrategyCache` path of `state_bellman!` | `IntervalMDP.Bellman.argoptAction`, `IntervalMDP.Bellman.Tπ` — `Bellman.lean` | `IntervalMDP.Bellman.argopt_attains`; `IntervalMDP.Bellman.policy_eval_sound` (A8) | — |
| 3 | Robust VI: reachability / reach-avoid | `_value_iteration!`, `step!`, `initialize!`/`step_postprocess_value_function!` for `AbstractReachability`, `AbstractReachAvoid`, `ExactTimeReach*` — `src/robust_value_iteration.jl`, `src/specification.jl` | `IntervalMDP.VI.reachIter` — `VI/Reach.lean` | `IntervalMDP.VI.reachIter_mem_unit`; `IntervalMDP.VI.reachIter_mono`; `IntervalMDP.VI.reachIter_tendsto_lfp`; `IntervalMDP.VI.reachIter_sound` (A4) | Without A4/A6, no error bound is claimed for the reachability ε-criterion. |
| 3 | Robust VI: safety | `AbstractSafety` initialize/postprocess — `src/specification.jl` | `IntervalMDP.VI.safetyIter` — `VI/Safety.lean` | `IntervalMDP.VI.safety_shift_eq` | Uses `T_add_const`. |
| 3 | Robust VI: reward | `AbstractReward` initialize/postprocess — `src/specification.jl` | `IntervalMDP.VI.rewardIter` — `VI/Reward.lean` | `IntervalMDP.VI.rewardIter_succ`; `IntervalMDP.VI.reward_contracting` (`0 < ν < 1`, Mathlib `ContractingWith`); `IntervalMDP.VI.reward_error_bound` (A5) | — |
| 3 | Robust VI: expected exit time | `ExpectedExitTime` initialize/postprocess — `src/specification.jl` | `IntervalMDP.VI.exitIter` — `VI/ExitTime.lean` | `IntervalMDP.VI.exitIter_mono`; `IntervalMDP.VI.exitIter_succ`; `IntervalMDP.VI.exitIter_sound` (A4) | Values may diverge. No convergence is claimed. |
| 3 | Synthesized strategy | `TimeVaryingStrategyCache`, `StationaryStrategyCache` — `src/strategy_cache.jl` | `IntervalMDP.VI.synthesizedStrategy` — `VI/Strategy.lean` | `IntervalMDP.VI.timeVarying_attains`; `IntervalMDP.VI.stationary_sound` (A7; or a Finding) | — |
| 4 | Interval value iteration | `ivi_step!`, `initialize_ivi!`, `IVIInitialGapCriteria` — `src/interval_value_iteration.jl`, `src/specification.jl` | `IntervalMDP.IVI.step` — `IVI.lean` | `IntervalMDP.IVI.lower_le_upper`; `IntervalMDP.IVI.bracket` (A6); `IntervalMDP.IVI.gap_stop_sound` (A6) | Dev must check `bracket` under the strategy coupling for all four modes before claiming it (§ Findings policy). |
| 5 | Vertex enumeration (fIMDP) | `state_action_bellman(::FactoredVertexIteratorWorkspace, …)` — `src/bellman.jl` | `IntervalMDP.Factored.vertexValue` — `Factored.lean` | `IntervalMDP.Factored.vertices_complete`; `IntervalMDP.Factored.vertexValue_eq_opt` (opt over `productSet` = opt over products of marginal vertices; exact, so this is the reference value for A2/A3) | Follows arXiv:2508.00707. |
| 5 | Recursive O-max (fIMDP) | `state_action_bellman(::FactoredIntervalOMaxWorkspace, …)` — `src/bellman.jl` | `IntervalMDP.Factored.recursiveOMax` — `Factored.lean` | `IntervalMDP.Factored.recursiveOMax_sound` (A2, one step); `IntervalMDP.Factored.recursiveOMax_vi_sound` (A2, end to end via `iter_sound`) | Tree reduction as in arXiv:2411.11803. Soundness only; exactness is not claimed. |
| 5 | LP McCormick relaxation (fIMDP) | `state_action_bellman(::FactoredIntervalMcCormickWorkspace, …)` — `src/bellman.jl` | `IntervalMDP.Factored.mcCormickLP` — `Factored.lean` | `IntervalMDP.Factored.mcCormick_sound` (A3, one step); `IntervalMDP.Factored.mcCormick_vi_sound` (A3, end to end) | Binary-tree McCormick relaxation as in arXiv:2508.00707. Lean reasons about the LP's mathematical optimum; whether HiGHS is correct is out of scope. |
| 6 | Product process (IMDP × DFA) | `bellman!` for `ProductProcess`, `AbstractDFAReachability`/`AbstractDFASafety` postprocess — `src/bellman.jl`, `src/specification.jl` | `IntervalMDP.Product.T` — `Product.lean` | `IntervalMDP.Product.T_eq_flat` (uses `product_read_successor`); `IntervalMDP.Product.dfaReach_eq` | — |

`T_add_const`, `T_nonexpansive` and `T_mono` must be proved for the **general** `RMDP`, not only for intervals.

## Proof Obligations & Limitations *(required)*

- **Domain:** exactly the structure invariants in § Lean Model Structure & Readability. A theorem may add a hypothesis only when its row says so (for example `ν < 1`).
- **Objectives:** every algorithm theorem is stated for all four combinations of satisfaction mode × strategy mode, or states explicitly which ones it covers. Soundness theorems are stated with `Sound m`.
- **Proof scope:** `abstract` for every row (values in `ℝ`, semantics = dynamic-programming recursion). Index theorems are about integer index arithmetic. They hold exactly for the Julia code, provided the `Int32`/`Int` index products do not overflow; that condition is stated as a hypothesis (`Π dims < 2^31` for `Int32` paths).
- **Limitations** (to be copied into the inventory for every row):
  - floating-point rounding, including the early-exit test `budget ≤ 0` and accumulated error in `budget -= p`;
  - overflow and underflow (beyond the stated index bound);
  - CUDA kernels (`ext/`) and threaded execution (`@threadstid`) are not modeled;
  - the Lean↔Julia correspondence is argued via literal transcription plus the cross-check tests, not proved;
  - DP value ↔ path-measure semantics (`ℙ^{π,η}[…]`) is not proved;
  - external LP solver correctness (McCormick) is not proved.
- Legacy gaps touched: every row above (all are `none` at the start of Phase 0).

## Findings policy

If a theorem as stated is false for the Julia behavior:

1. Dev writes a minimal counterexample. It should be a concrete Julia snippet if possible, and a Lean `example` that disproves the claim.
2. Dev records it under **Findings** in the inventory and marks the row `partial` with the strongest statement that *was* proved.
3. Dev does not edit `src/`, and does not quietly weaken the theorem's statement under its original name.
4. The run ends with QE as FAIL for that criterion and goes back to the user. It does not loop.

The rows most likely to trigger this policy are:

- `intervalSub2ind_correct` (Phase 1);
- `stationary_sound` (Phase 3);
- IVI `bracket` (Phase 4);
- a mismatch between the Julia factored algorithms and the cited papers (Phase 5).

## Proof Policy *(required)*

- Approved axioms: `propext`, `Classical.choice`, `Quot.sound`.
- Forbidden:
  - `sorry`, `admit` and `stop`;
  - placeholder statements (for example `: True`, or a theorem whose conclusion is a hypothesis);
  - new `axiom` declarations;
  - `native_decide` and `decide := true` on non-trivial goals;
  - `@[implemented_by]` and `unsafe` in proof dependencies.
- Mathlib is approved as a dependency. Mathlib lemmas count as proofs. Mathlib's axioms reduce to the approved set.
- Statements may not be specialised to small concrete instances (for example `Fin 3`) unless the table says so. `Models/Examples.lean` is the only place where concrete instances appear.
- Soundness theorems must go through `IntervalMDP.Approx.iter_sound` (A1), not re-prove the induction.

## CPU / GPU Matrix *(required)*

| Check | Backend | Required? | Command | Notes |
|---|---|---|---|---|
| Package tests | CPU | yes | Julia test command | Must stay green. With no `src/` change this is a regression guard, plus the optional cross-check tests. |
| GPU tests | CUDA | no | — | No GPU code changes. CUDA kernels are a stated proof limitation. |

## Performance Evidence

Not in scope.

## Acceptance Criteria *(QE verifies, for the phase being run)*

- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` passes on CPU. `git diff <base> -- src/` is empty.
- [ ] `lake build` in `lean/` succeeds with the pinned toolchain and no linter warnings in `IntervalMDPProofs/`. `lean-toolchain` matches the pinned Mathlib release, and has not changed since Phase 0 unless this spec was amended.
- [ ] Every theorem tagged with this phase in the Models, Indexing and Algorithms tables exists under its exact name, with a complete proof and no forbidden constructs.
- [ ] `lean/AxiomCheck.lean` lists every theorem from this phase and every earlier phase. Its output contains only approved axioms.
- [ ] Each theorem statement matches its row: the right hypotheses (structure invariants only, plus the ones the row names), all four modes or an explicit restriction, and no vacuous hypotheses.
- [ ] **Model readability:**
  - the § Layout files and structures exist with invariants as fields;
  - every definition, structure and theorem has a docstring naming its Julia counterpart and file;
  - names follow the Julia names;
  - the IMDP, fIMDP and product models all go through `toRMDP`, with no duplicated Bellman operator;
  - `Models/Examples.lean` builds the docstring examples;
  - `lean/README.md` has the hierarchy, the Julia↔Lean glossary and the file map, and is current for this phase.
- [ ] **Indexing:** the index theorems for this phase are proved. Every Lean index function that transcribes a Julia loop says so in its docstring and keeps the Julia loop's structure (same loop order and the same `- 1`/`+ 1` steps), so a reviewer can compare them line by line.
- [ ] **Approximation soundness:** every A-row due in this phase is proved, or recorded as a Finding, and is stated with `Sound m`. Every end-to-end soundness theorem uses `iter_sound`.
- [ ] `harness/specs/inventory-intervalmdp.md` is updated: status, theorem links, scope `abstract`, limitations, and Findings for each model, index, approximation and algorithm row.
- [ ] Cross-check tests added in this phase, if any, pass and cover the listed cases.

## File List

- `lean/lean-toolchain`, `lean/lakefile.toml`, `lean/lake-manifest.json`, `lean/README.md`, `lean/IntervalMDPProofs.lean`, `lean/AxiomCheck.lean`
- `lean/IntervalMDPProofs/Models/{Distribution,AmbiguitySet,IntervalAmbiguity,RMDP,IMDP,Factored,DFA,Product,Strategy,Specification,Examples}.lean`
- `lean/IntervalMDPProofs/Index/{Julia,Linear,Marginal,Sparse,Perm,Strategy,Product}.lean`
- `lean/IntervalMDPProofs/Approx/{Sound,Lift}.lean`
- `lean/IntervalMDPProofs/{OMax,Bellman,IVI,Factored,Product}.lean`, `lean/IntervalMDPProofs/VI/{Reach,Safety,Reward,ExitTime,Strategy}.lean`
- `harness/specs/inventory-intervalmdp.md`
- Optional tests: `test/base/{omax_reference,indexing_reference,approx_soundness_reference}.jl`, and their includes in the test runner
- `.gitignore`: add `lean/.lake/`

## Out of Scope

- Any change to `src/` or `ext/`, including fixes for findings (those are follow-up specs).
- Floating-point / implementation-scope proofs, and Lean→Julia code extraction.
- Path-measure semantics (relating DP values to `ℙ^{π,η}`), and MEC deflation for IVI on reachability/safety.
- Topological VI (only a TODO in `src/algorithms.jl`); L1-MDPs and mixtures (not yet implemented in Julia; the `RMDP`/`AmbiguitySet` layer must make them possible to add later).
- Lean CI in GitHub Actions (a separate Ops task once Phase 1 has merged).
