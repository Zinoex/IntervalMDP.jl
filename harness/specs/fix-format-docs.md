# Fix repository formatting and the documentation build — Specification (Julia + Lean)

> One `/harness` run. Branch `fix/format-docs` from a freshly fetched `origin/main`; one PR, opened ready for review
> (not draft). Base ref for every diff below: `origin/main` at the time Dev starts (`668a27d` when this spec was written;
> Dev records the actual sha).

## Objective *(required)*

Both non-required CI checks fail on `main`, and have done so for every PR since Phase 0:

1. **format-check** (`.github/workflows/FormatCheck.yml`, JuliaFormatter 2.1.6, `.JuliaFormatter.toml`) reports 28
   unformatted files (list in § Julia Behavior & Tests).
2. **Documentation** (`.github/workflows/documentation.yml`, `julia --project=docs/ docs/make.jl`) stops before rendering
   with `makedocs encountered errors [:missing_docs, :cross_references]`: 3 unresolved `@ref`s and 5 docstrings missing
   from the manual.

Make both checks pass, with **no behavior change**: formatting is layout only, and documentation changes touch only
docstrings and `docs/`.

3. **CUDA test items** (`:cuda` tag, excluded from CI) fail on `main` as well: 1095 pass, 3 fail, 13 error. They call an
   outdated `_bellman_helper!` signature and hard-code `CUDA.DeviceMemory` in `show` strings. Bring them up to date
   (amended 2026-10-08 by operator decision: CUDA test repair is in scope, tests only).
4. **CUDA large-sparse Bellman crash** (Finding exposed by the repaired tests, already on `main`):
   `try_large_sparse_bellman!` in `ext/cuda/bellman/sparse.jl` does not pass `states` to `_large_sparse_bellman_helper!`,
   which uses an undefined `states`, so every sparse CUDA Bellman update that takes the large-sparse kernel path throws
   `UndefVarError`. Fix it (amended 2026-10-08 by operator decision: this one `ext/` fix, E1, is in scope).

This change does **not** add or semantically change a VI/Bellman algorithm. It does touch Julia files that existing Lean
proofs are mapped to (`src/update_sequence.jl`, `src/interval_value_iteration.jl`, `ext/cuda/bellman/factored.jl`,
`src/probabilities/IntervalAmbiguitySets.jl`), so the existing Lean build and axiom check must stay green, and any
`file:line` citation of these files in `lean/` or the inventory must still point at the same code.

Model family: all (formatting is repository-wide); no model semantics change.

## Commands / Toolchain *(required)*

- Target root: `/home/fresen/.julia/dev/IntervalMDP`
- Julia version / compat: `[compat] julia = "1.11"`; local `julia` is 1.13.1 (satisfies compat).
- Julia instantiate: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- Julia test: `julia --project=. -e 'using Pkg; Pkg.test()'`; afterwards `git checkout -- test/data/multiObj_robotIMDP.nc` (see `harness/LEARNING.md`).
- Format (fix, Dev only): `python3 harness/tools/harness/julia_format.py fix . --files <file> ...` with the 28 files listed below.
- Format check (repository-wide, CI-equivalent): in a scratch `git worktree` of the branch head, run
  `julia --project=@juliaformatter -e 'using JuliaFormatter; format("."; verbose = false)'` (JuliaFormatter 2.1.6, as in
  `FormatCheck.yml`; `julia_format.py` installs that version into `@juliaformatter`), then `git -C <worktree> diff --name-only`
  must be empty. Never run this in the target checkout.
- Format check (touched files): `python3 harness/tools/harness/julia_format.py check . --base <base-ref>` → `RESULT: PASS`.
- Docs build: `julia --project=docs/ -e 'using Pkg; Pkg.instantiate()'` then `julia --project=docs/ docs/make.jl > <scratch>/docs.log 2>&1`.
  Deployment (`deploydocs`) only acts on CI; locally it is a no-op. Read only `grep -nE 'Error|Warning' <scratch>/docs.log`.
- Lean root: `<root>/lean`; pinned toolchain `leanprover/lean4:v4.33.0-rc2`, called directly, never the elan proxy:
  - build (from `lean/`): `~/.elan/toolchains/leanprover--lean4---v4.33.0-rc2/bin/lake build`
  - axiom check (from `lean/`): `~/.elan/toolchains/leanprover--lean4---v4.33.0-rc2/bin/lake env lean AxiomCheck.lean`
  - Julia reference check: `python3 lean/scripts/check_julia_refs.py`
- GPU test: the `:cuda`-tagged test items (`test/runtests.jl` with `:cuda` removed from `EXCLUDED_TAGS`, not committed),
  classified through `python3 harness/tools/harness/gpu_check.py run . --cmd "<cmd>"`; probe:
  `python3 harness/tools/harness/gpu_check.py probe .`.
- Benchmark: not in scope.

## Julia Behavior & Tests *(required)*

No behavior change. No new tests are required; `Pkg.test()` is the regression guard, plus the GPU tests because CUDA
files are reformatted and the CUDA test items are repaired (§ CUDA test repair).

### Formatting

Format exactly these 28 files (the list CI reports on `main`) with JuliaFormatter 2.1.6 and `.JuliaFormatter.toml`, and
no other file:

- `benchmark/`: `analyze.jl`, `cases/fimdp.jl`, `cases/imdp.jl`, `cases/product.jl`, `cases/real.jl`,
  `cases/synthesis.jl`, `compare.jl`, `lib/environment.jl`, `lib/measure.jl`, `lib/reference.jl`, `profile.jl`, `run.jl`,
  `stream.jl`
- `ext/cuda/bellman/factored.jl`
- `src/interval_value_iteration.jl`, `src/update_sequence.jl`
- `test/base/`: `bellman.jl`, `factored.jl`, `indexing_reference.jl`, `ivi.jl`, `mixture.jl`, `omax_reference.jl`,
  `specification.jl`
- `test/cuda/dense/bellman.jl`, `test/cuda/sparse/bellman.jl`, `test/data/prism.jl`, `test/sparse/bellman.jl`,
  `test/sparse/imdp.jl`

If the CI-equivalent check on the base ref reports a different list (e.g. `main` moved), use that list and say so.

Formatting must be **semantics-preserving**. For every formatted file, the parsed code before and after must be equal
once line-number nodes are removed. Check with a script that, for each file, compares
`Base.remove_linenums!(Meta.parseall(read(old, String)))` with the same for the new file (old = `git show <base-ref>:<file>`).
Commit the script to the scratch directory, not to the repository. Any file whose ASTs differ must be explained (for
example a docstring edit in the same file) or the formatting change reverted.

Formatting `src/`, `ext/` and `test/` shifts line numbers. Every `file:line` or `file:line–line` citation of a formatted
file in `lean/` (docstrings, `lean/README.md`) and in `harness/specs/inventory-intervalmdp.md` must be updated to the new
lines of the same code. Known instance: `ext/cuda/bellman/factored.jl:451` in the inventory (Observation O8). Find the rest
with `grep -rnoE '(src|ext|test)/[A-Za-z_/]+\.jl:[0-9]+' lean/ harness/specs/inventory-intervalmdp.md`.

### CUDA test repair (tests only)

Failures on the base ref (`gpu_check.py run` with `:cuda` enabled; identical on the branch before this amendment):

| # | Where | Failure | Fix |
|---|---|---|---|
| C1 | `test/cuda/dense/bellman.jl:29,46`, `test/cuda/sparse/bellman.jl:71,97` (N = Float32, Float64) | `MethodError: no method matching _bellman_helper!(::Cu…OMaxWorkspace, ::NoStrategyCache, …)` | Call the current API. Prefer the public `bellman`/`bellman!` entry points (as the CPU tests in `test/base/bellman.jl`, `test/sparse/bellman.jl` do) over the internal `_bellman_helper!`; if a test needs the internal function, use its current signature in `src/bellman.jl`. |
| C2 | `test/cuda/sparse/bellman.jl:136,179` and the following "large matrices" items | Same `MethodError`, on the CPU reference computation (`SparseIntervalOMaxWorkspace`, `Vector{Float64}`) | Same as C1. |
| C3 | `test/cuda/dense/factored.jl:28,124` (`show 1d`, `show 3d`) | expects `CuArray{Float64, 2, CUDA.DeviceMemory}`; CUDA now prints `CUDACore.DeviceMemory` | Build the expected string from the value's type (`"… (dense, $(typeof(ambiguity_sets_array)))"` or `string(nameof(…))`), not a hard-coded module name — see the `CartesianIndex` printing lesson in `harness/LEARNING.md`. |

Rules:
- Tests only: `src/` and `ext/` are not changed for this (beyond the docstrings of D1–D3). If a CUDA failure turns out to
  come from wrong `ext/` behavior rather than an outdated test, stop and report it as a Finding with the failing case;
  do not fix `ext/` here.
- Do not weaken tests: no deleted `@test`/`@testitem`, no `@test_broken`/`@test_skip`, no loosened tolerances, no tag
  changes. Each repaired test checks the same quantity against the same reference as before. Report `@test` counts per
  repaired file before and after.
- `test/runtests.jl` stays unchanged (`:cuda` remains excluded by default; enabling it for the run is temporary).

### CUDA large-sparse fix (E1, the only `ext/` code change)

| # | Where | Failure | Fix |
|---|---|---|---|
| E1 | `ext/cuda/bellman/sparse.jl`: call in `try_large_sparse_bellman!` (≈ 542–552) and signature of `_large_sparse_bellman_helper!` (≈ 555–564); `states` used at ≈ 568, 584, 608 | `UndefVarError: states not defined in IntervalMDPCudaExt` in "large matrices — many/more/even more/most non-zeros"; "too many non-zeros" gets the `UndefVarError` instead of `OutOfSharedMemory` | Pass `states` through the call and add a `states::IntervalMDP.AbstractUpdateSequence` parameter to `_large_sparse_bellman_helper!`, in the same position as in the other CUDA helpers of this file. No other `ext/` change. |

Rules:
- The diff of `ext/` against the base is exactly E1 (plus the formatting of `ext/cuda/bellman/factored.jl`): the added
  argument and parameter, nothing else. The AST check reports `ext/cuda/bellman/sparse.jl` as "FIXED (E1)" with exactly
  those differences.
- The repaired "large matrices" test items are the regression tests: they must error on the base ref and pass after E1.
  Record both outcomes.
- E1 adds lines, so `file:line` citations of `ext/cuda/bellman/sparse.jl` in `harness/specs/inventory-intervalmdp.md`
  (Observation O8: `ext/cuda/bellman/sparse.jl:339, 380, 762, 800`) must be updated to the same code.
- CUDA kernels are a stated limitation of the Lean proofs (L3); E1 adds no Lean obligation.

### Known issue K1 (accepted, not fixed here)

Amended 2026-10-09 by operator decision. After E1, the CUDA item "large matrices — most non-zeros"
(`test/cuda/sparse/bellman.jl`, `nnz_per_column = 8000`) throws `OutOfSharedMemory`. The test sizes this case for the
`(Int32, Nothing)` large-sparse kernel, which is commented out in `_bellman_helper!` (`ext/cuda/bellman/sparse.jl:84`,
"DISABLED because the new GPUSparseDeviceMatrixCSC implementation fails getindex"): `getindex` is missing from CUDA.jl's
`GPUSparseDeviceMatrixCSC` (the operator's upstream issue may simply not have been resolved yet). The remaining kernels
need 64–128 KB of shared memory, above the 48 KB default launch limit. This is pre-existing on `main` (hidden by the E1
`UndefVarError`). It stays as one known error; re-enabling the kernel is a separate spec. Side observation for that spec:
the size check and the error message use `CUDA.limit(LIMIT_SHMEM_SIZE)` (the opt-in maximum, 101376 B on the test GPU)
although kernels launch within the 48 KB default, so the reported "available" figure is misleading.

### Documentation build

Fix every `Error` the docs build reports on the base ref:

| # | Error on the base ref | Fix |
|---|---|---|
| D1 | `Cannot resolve @ref for md"[Marginal](@ref)"` (docstring of `IntervalAmbiguitySets`, `src/probabilities/IntervalAmbiguitySets.jl:5`) | Use a code link: `` [`Marginal`](@ref) `` (the docstring of `Marginal` is already in `docs/src/reference/systems.md`). |
| D2 | `Cannot resolve @ref for` `` [`VerificationSolution`](@ref) `` (docstring of `solve(…, ::IntervalValueIteration)`, `src/interval_value_iteration.jl:22`) | Add a docstring to `struct VerificationSolution` (`src/problem.jl`): what it stores (value function, residual, number of iterations, and the IVI-specific fields if any) and its accessors (`value_function`, `residual`, `num_iterations`). List it in the `docs/src/reference/solve.md` `@docs` block (as `IntervalMDP.VerificationSolution` if it stays unexported). |
| D3 | Same for `` [`ControlSynthesisSolution`](@ref) `` | Same as D2 for `struct ControlSynthesisSolution`, including the strategy and `strategy(::ControlSynthesisSolution)`. |
| D4 | Docstring of `IntervalMDP.IntervalValueIteration` not in the manual | Add it to `docs/src/reference/solve.md`, under "VI-like Algorithms" next to `RobustValueIteration`. |
| D5 | `IntervalMDP.AbstractUpdateSequence`, `FullUpdateSequence`, `ProductUpdateSequence` not in the manual | Add them to the reference page that documents algorithm options (`solve.md`, a short "Update sequences" subsection). |
| D6 | `IntervalMDP.source_shape :: Tuple{DFA}` not in the manual | Add `source_shape(dfa::DFA)` to the `@docs` block where the DFA is documented (`docs/src/reference/systems.md`). |

Do not silence errors: no `warnonly`, no changing `checkdocs`, no `@docs` entries with `canonical=false` to hide a
docstring, no removing a docstring to make `missing_docs` pass. Exporting a previously unexported name is an API change
and out of scope.

| D7 | Appears once D5 is fixed: `Cannot resolve @ref for` `` [`IntervalMarkovProcess`](@ref) `` (docstring of `FullUpdateSequence`) | List the existing docstring `IntervalMDP.IntervalMarkovProcess` in the "System representation" `@docs` block of `docs/src/reference/systems.md`. |

Optional (not gating): the `DocumenterCitations` warning about the unmodified `assets/citations.css` copy may be fixed by
deleting the asset entry and file, as the warning suggests. Any other warning must not be new.

## Algorithm ↔ Theorem Mapping *(required)*

none — no algorithm is added or changed. The existing mapping stays as is; the verifier checks that every theorem listed
in `lean/AxiomCheck.lean` (40 entries on the base ref) is still present and proved.

## Proof Obligations & Limitations

No new obligations. Existing proofs must stay green: `lake build` clean with the pinned toolchain, `AxiomCheck.lean`
output only `propext`, `Classical.choice`, `Quot.sound`, `check_julia_refs.py` passes. No Lean file changes except line
citations (and the docstrings that carry them). Proof scope stays `abstract`.

## Proof Policy

Unchanged: approved axioms `propext`, `Classical.choice`, `Quot.sound`; no `sorry`/`admit`/`stop`, no new `axiom`.

## CPU / GPU Matrix *(required)*

| Check | Backend | Required? | Command | Notes |
|---|---|---|---|---|
| Package tests | CPU | yes | Julia test command | regression guard |
| GPU tests | CUDA | yes | `:cuda` test items via `gpu_check.py run` | `ext/cuda/bellman/factored.jl` and `test/cuda/*` are reformatted and repaired; 0 fail, and the only error is K1; UNAVAILABLE = unmet criterion |
| Lean build + axiom check | — | yes | see Commands | existing proofs stay green |
| Docs build | — | yes | see Commands | must finish without errors |

## Performance Evidence

Not in scope.

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] `Pkg.test()` passes on CPU. The `:cuda` test items run on the GPU with 0 failures and exactly one error, the known issue K1 ("large matrices — most
      non-zeros", `OutOfSharedMemory`); any other failure or error is unmet. `gpu_check.py run` reports FAIL because of K1;
      that verdict is accepted only when K1 is the sole error. UNAVAILABLE is unmet.
- [ ] CUDA test repair C1–C3 is tests-only and does not weaken any test: no `@test`/`@testitem` removed, no
      `@test_broken`/`@test_skip`, tolerances and tags unchanged, each repaired test compares the same quantity against
      the same reference; `@test` counts per repaired file are unchanged. `test/runtests.jl` is unchanged.
- [ ] Repository-wide format check (CI-equivalent, in a scratch worktree of the branch head): JuliaFormatter 2.1.6
      `format(".")` changes no file (`git diff --name-only` empty). `julia_format.py check . --base <base-ref>` → `RESULT: PASS`.
- [ ] Formatting is semantics-preserving: for every formatted file, the line-number-stripped `Meta.parseall` ASTs at the
      base ref and the branch head are equal, except files that also carry a docstring edit from D1–D3 (differences are
      exactly those docstrings) and the CUDA test files repaired under C1–C3 (differences are exactly those repairs).
- [ ] Only the 28 listed files (or the base ref's actual CI list, if it differs), the docstring files of D1–D3, the
      CUDA test files of C1–C3, `docs/`, and the line citations in `lean/` and the inventory are changed. `git diff <base-ref> --stat` shows nothing else
      (no `Project.toml`/`Manifest.toml`, no `.JuliaFormatter.toml`, no `.github/` change).
- [ ] Docs build: `julia --project=docs/ docs/make.jl` finishes, writing `docs/build/`, with no `Error` lines. Every
      error D1–D7 is gone. No new `Warning` compared with the base ref. No error is silenced (no `warnonly`, no `checkdocs`
      change, no docstring removed).
- [ ] D2/D3: `VerificationSolution` and `ControlSynthesisSolution` have docstrings that match the structs' fields and
      accessors, and appear in `docs/src/reference/solve.md`. D4–D6 docstrings appear in the reference pages named above.
- [ ] E1: `git diff <base-ref> -- ext/` is exactly the E1 change in `ext/cuda/bellman/sparse.jl` plus the formatting of
      `ext/cuda/bellman/factored.jl`. The "large matrices" CUDA items error on the base ref (with C1–C3 applied) and pass
      after E1, except "most non-zeros", which now fails only with K1's `OutOfSharedMemory`. Citations of `ext/cuda/bellman/sparse.jl` in the inventory point at the same code.
- [ ] No public API change: no `export` added or removed, no signature or default changed
      (`git diff <base-ref> -- src/ ext/ | grep -E '^[+-].*\bexport\b'` empty).
- [ ] Lean: `lake build` clean with the pinned toolchain; `AxiomCheck.lean` lists the same 40 theorems as the base ref,
      all with approved axioms only; `check_julia_refs.py` passes; every `file:line` citation of a formatted file in `lean/`
      and the inventory points at the same code as before (QE spot-checks each one).

## File List

- The 28 formatted files listed above (tracked; commit).
- `src/problem.jl` (D2, D3 docstrings), `src/probabilities/IntervalAmbiguitySets.jl` (D1), `src/interval_value_iteration.jl`
  (formatting; D2/D3 links stay).
- `ext/cuda/bellman/sparse.jl` (E1 only).
- `test/cuda/dense/bellman.jl`, `test/cuda/sparse/bellman.jl`, `test/cuda/dense/factored.jl` (C1–C3), and any other
  `test/cuda/**` file with the same outdated call.
- `docs/src/reference/solve.md`, `docs/src/reference/systems.md` (D2–D7); optionally `docs/make.jl` and
  `docs/src/assets/citations.css` (citations warning).
- `lean/IntervalMDPProofs/**/*.lean`, `lean/README.md`, `harness/specs/inventory-intervalmdp.md` (line citations only).

## Out of Scope

- Any behavior change, bug fix or refactor of `src/`/`ext/` other than E1; exporting new names. (Repairing outdated CUDA
  *tests* is in scope, § CUDA test repair; the E1 crash fix is in scope.)
- Changing the CI workflows, the JuliaFormatter version or `.JuliaFormatter.toml`.
- Formatting files under `harness/` (ignored by `.JuliaFormatter.toml` on purpose).
- The `:mixture` test items (disabled; formatting `test/base/mixture.jl` is in scope, fixing its API is not).
- New Lean theorems.
