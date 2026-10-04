# <Feature name> — Specification (Julia + Lean)

> Template for `/harness` tasks on Julia targets with Lean proofs (e.g. IntervalMDP.jl).
> Copy to `harness/specs/<feature>.md`, fill every section, and delete guidance quotes.
> Sections marked *(required)* are read by the orchestrator to decide gates.

## Objective *(required)*

One paragraph: what changes and why. State whether this change **adds or semantically changes a VI/Bellman algorithm** (yes/no). If "no", say which Lean/proof dependencies (if any) it touches.

Model family: `<MDP | interval MDP | L1-MDP | mixture | factored | other>` — obligations below are specific to this family; do not assume interval models.

## Commands / Toolchain *(required; overrides harness.config.toml and discovery)*

> Leave a line out to fall back to `harness.config.toml`, then discovery
> (`python3 harness/tools/harness/discover.py <target-root>`). If a needed value can't be
> inferred, the harness reports a blocker.

- Target root: `<absolute path to target clone>`
- Julia version / compat: `<e.g. 1.10; must satisfy Project.toml [compat] julia (IntervalMDP.jl: 1.9+)>`
- Julia instantiate: `julia --project=<root> -e 'using Pkg; Pkg.instantiate()'`
- Julia test: `julia --project=<root> -e 'using Pkg; Pkg.test()'`
- Lean root: `<root>/<lean dir>` — pinned toolchain from `lean-toolchain`: `<leanprover/lean4:vX.Y.Z>`
- Lean build: `lake build`
- Lean test (optional): `<e.g. lake test>`
- GPU test (if applicable): `<command, e.g. env var that enables CUDA tests>`
- Benchmark (performance-scoped only): `<command>`

## Julia Behavior & Tests *(required)*

- Public API / entry points affected (module, function, file).
- Exact behavior: inputs, outputs, edge cases (absorbing states, empty supports, tie-breaking, convergence tolerance, iteration limit, …).
- Test cases: small deterministic examples with known mathematical results; independent reference calculations; regression cases. Element types / storage / execution paths **only where promised** (dense/sparse, Float32/Float64, CPU/CUDA).
- No wall-clock thresholds in unit tests.

## Algorithm ↔ Theorem Mapping *(required when an algorithm is added/changed; otherwise write "none")*

| Algorithm | Julia entry point (function — file) | Lean definition (name — file) | Lean theorem(s) (fully qualified name — file) | Correspondence note |
|---|---|---|---|---|
| `<name>` | `<f>` — `src/<file>.jl` | `<Ns.def>` — `lean/<File>.lean` | `<Ns.thm>` — `lean/<File>.lean` | `<how the Lean model matches the Julia code>` |

The verifier checks exactly the theorem names in this table.

## Proof Obligations & Limitations *(required when the mapping table is non-empty)*

- Assumptions and input domain (e.g. probabilities in [0,1] summing to 1; interval bounds `l ≤ u`; discount `0 ≤ γ < 1`).
- Objective: pessimistic/optimistic, maximize/minimize.
- Properties promised and proved (only what the spec promises): Bellman update/recurrence; value bound; contraction/convergence; optimality; …
- **Proof scope**: `abstract` (mathematical model, exact/real arithmetic) or `implementation` (only with a verified Lean↔Julia/float correspondence).
- **Limitations** (must be listed unless proved): floating-point rounding; overflow/underflow; GPU execution; correspondence between Lean semantics and Julia code. Do not claim a "verified floating-point implementation".
- Legacy gaps touched: `<entries from harness/specs/inventory-<target>.md, or none>`.

## Proof Policy *(required when the mapping table is non-empty)*

- Approved axioms: `propext`, `Classical.choice`, `Quot.sound` `<add others only with justification>`.
- Forbidden: `sorry`, `admit`, `stop`, placeholder statements (e.g. `: True`), new `axiom` declarations, `native_decide` (unless `Lean.ofReduceBool` is approved above).
- Existing project axioms may not discharge a new obligation without explicit review here.

## CPU / GPU Matrix *(required)*

| Check | Backend | Required? | Command | Notes |
|---|---|---|---|---|
| Package tests | CPU | yes | Julia test command | always |
| GPU tests | CUDA | `<yes if GPU code changes / no>` | GPU test command | outcome PASS / FAIL / UNAVAILABLE; UNAVAILABLE on a required row = unmet criterion |

## Performance Evidence *(only for explicitly performance-scoped changes; otherwise "not in scope")*

- Benchmark command and harness (e.g. BenchmarkTools), problem sizes.
- Environment to record: Julia version, threads, CPU model, GPU model/driver.
- Baseline ref vs changed ref, same command; report median time and allocations.
- Correctness tests and the proof gate pass first. No timing thresholds in unit tests.

## Acceptance Criteria *(required — QE verifies one by one)*

- [ ] Julia test command passes on CPU (`Pkg.test()`), including the new/updated tests.
- [ ] `<behavioral criteria …>`
- [ ] Lean project builds with the pinned toolchain (`lake build`); each theorem in the mapping table is present with a complete proof; no `sorry`/`admit`/placeholder/unapproved axiom (`#print axioms` clean).
- [ ] Julia↔Lean traceability documented; proof scope and limitations documented.
- [ ] `<GPU row: GPU tests PASS on CUDA hardware — UNAVAILABLE is unmet>`
- [ ] `<performance: reproducible benchmark evidence baseline vs change>`

## File List

- `src/...`, `test/...`, `lean/...`, `harness/specs/inventory-<target>.md` (if proof status changes)

## Out of Scope

- `<explicit non-goals>`
