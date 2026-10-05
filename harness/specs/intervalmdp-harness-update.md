# IntervalMDP.jl Harness Adaptation — Specification

## Objective

Update this harness so it can safely develop and verify Julia scientific-computing code for `IntervalMDP.jl`, while requiring Lean theorem statements and machine-checked proofs for every value-iteration (VI) / Bellman algorithm. The harness must prioritize mathematical and implementation correctness before performance, support CPU and optional GPU testing, and keep the verification approach extensible to additional robust decision-process models.

This work changes the harness and its documentation/configuration. It does **not** implement or change an algorithm in `IntervalMDP.jl`.

## Project Context and Constraints

- Target: [`Zinoex/IntervalMDP.jl`](https://github.com/Zinoex/IntervalMDP.jl), a Julia package for robust/interval and factored decision processes, including CPU and CUDA value iteration.
- Julia is the implementation language to preserve low-level performance control and enable GPU/HPC execution; do not introduce a Rust or C++ implementation path.
- Correctness is the primary gate. Optimization must preserve tested semantics and be supported by measurements when performance is part of a change.
- Every VI/Bellman algorithm must have a corresponding Lean theorem statement and proof. A Julia test or benchmark is not a substitute for that proof.
- Long-term goal: formally verified floating-point implementation, with Lean serving verification and Julia serving optimized execution. Until a sound Lean-to-Julia/float correspondence is established, the harness must not claim the executable floating-point code itself is formally verified. It must clearly distinguish proofs about the mathematical/abstract algorithm from proofs about its concrete floating-point implementation.
- Future model coverage includes MDPs, L1-MDPs, mixtures, and factored variants. Do not design harness rules that hard-code interval models as the only possible model family.

## Scope

Adapt the existing Dev → QE → Ops harness to recognize Julia and Lean projects, execute their configured validation commands, and enforce a formal-proof gate for algorithmic changes. Preserve telemetry, lessons learned, autonomous stage progression, failure loop-back, and Ops gating.

The harness must identify the target repository root independently of the harness repository and must use the target project's declared toolchain/configuration rather than assuming the Node.js sample project's commands or directory layout.

## Workflow Requirements

### Stages and gates

1. **Dev** implements the scoped change and adds/updates Julia tests and, for any new or changed VI/Bellman algorithm, its Lean theorem and proof. It reports changed artifacts and exact Julia/Lean commands and results.
2. **Formal Verification** independently checks the relevant Lean project and proof obligations before QE. Add a dedicated `verifier` agent/stage (or an equivalently isolated, independently prompted verification gate); it must not merely trust Dev's report. This gate verifies that the expected theorem is present, Lean accepts the proof, and the proof contains no admitted/sorry placeholder or unapproved axiom.
3. **QE** independently runs the target project's Julia test suite and acceptance checks. It checks CPU behavior for all applicable algorithm changes and CUDA/GPU behavior when the change affects GPU code and a configured GPU runner is available. A required GPU check may not silently pass as skipped: unavailable hardware must be reported as a blocker or explicit unmet criterion according to the spec.
4. **Ops** runs only after Dev, Formal Verification, and QE pass. It retains the existing safe branch/commit/PR and learning-update behavior.

A failure in Formal Verification or QE returns to Dev with specific failure evidence and follows the existing maximum of two Dev-to-gate cycles. Ops must never run if any required gate fails. Record each new stage, gate, loop-back, and terminal state in telemetry using the existing event conventions, with stage-specific events for the verifier.

### Project/toolchain discovery

- Determine the target root and discover Julia configuration from `Project.toml` (and `Manifest.toml` when present), and Lean configuration from the target's declared Lean project files/toolchain.
- Use the Julia version declared by the target project; for the current IntervalMDP.jl target, honor its documented Julia 1.9+ requirement unless the repository declares a stricter version.
- Instantiate dependencies only through the project's Julia environment. Run the package tests using the repository's documented/configured command (normally `Pkg.test()`); do not assume npm, Jest, HTTP endpoints, or `curl` are relevant.
- Run Lean with the project-pinned toolchain and documented command (normally `lake build` and any project test command). Do not download or silently switch toolchain versions.
- If a required toolchain/configuration is missing or a command cannot be inferred safely, report a blocker rather than inventing a passing result.
- Make commands and paths configurable or discoverable so future repositories can use the harness without hard-coding IntervalMDP.jl internals.

## Formal Verification Contract

For every VI/Bellman algorithm introduced or semantically changed by a task:

- The change must identify the algorithm and its Julia entry point(s), mathematical assumptions, input domain, objective (for example, pessimistic/optimistic and maximize/minimize where applicable), and the relevant Lean theorem(s).
- Lean must contain a precise theorem statement and a complete machine-checked proof of the algorithm's stated correctness properties. The statement must cover the behavior promised by the spec, such as the Bellman update/recurrence and any claimed value, bound, convergence, or optimality property; do not demand unrelated theorems.
- The spec/change must provide traceability between the Lean definition/theorem and the Julia implementation (names/paths and a concise correspondence explanation). Do not claim that proving a mathematical model automatically proves the Julia implementation correct.
- The independent verification gate must build the Lean project and reject `sorry`/admitted goals, placeholder proofs, or new axioms unless explicitly justified and approved by the task's proof policy. Existing axioms may not be treated as proof of a new obligation without explicit review.
- For changes that do not add or alter VI/Bellman semantics, no new algorithm theorem is required; existing proofs and tests must remain green if the change touches their dependencies.
- Floating-point rounding, overflow/underflow, GPU execution, and correspondence between Lean semantics and Julia code must be documented as proof limitations unless the task supplies and verifies those obligations. No silent upgrade of the claim to “verified floating-point implementation.”

For target onboarding, the harness must also establish a baseline inventory of existing VI/Bellman algorithms and their Lean proof links/status. Report any legacy algorithms that do not yet have proofs as explicit verification gaps; do not describe the package as fully verified while gaps remain. A task that touches an unproved legacy algorithm must supply its theorem/proof before that algorithm change can pass the formal gate. Completing proofs for every existing algorithm is a separate migration effort, not work to hide inside this harness-only change.

## Julia Correctness and Performance Requirements

- Tests must exercise the public/package-level behavior and algorithm edge cases specified by each feature spec. Prefer deterministic small examples with known mathematical results, independent reference calculations where practical, and regression cases for prior defects.
- Check relevant supported element types and storage/execution paths only where the target project or feature spec promises them (for example dense/sparse, Float32/Float64, and CUDA); do not accidentally narrow existing support.
- Correctness tests and the Lean proof gate must pass before performance comparisons are considered.
- For an explicitly performance-oriented change, include a reproducible benchmark, baseline/comparison context, and measured results. Avoid brittle wall-clock thresholds in ordinary unit tests. A speedup claim must not replace correctness evidence.
- GPU tests must be conditional on the project's supported CUDA setup and hardware availability, clearly distinguish skipped/unavailable from passed, and never hide a failure in a GPU-specific change.
- Changes must remain compatible with Julia package conventions and avoid unnecessary allocations or regressions when performance is in scope.

## Harness Interfaces and Documentation

- Update the `/harness` orchestrator instructions to collect the target root, read the task spec and `harness/LEARNING.md`, delegate the new verification gate, pass concrete commands/criteria to every agent, record telemetry for every stage/gate, and enforce the expanded order.
- Add a verifier agent definition with a least-privilege tool set sufficient to inspect Lean sources and run the configured proof command. It reports theorem/proof coverage, exact command output, and pass/fail per obligation; it does not edit proofs to make its own gate pass.
- Update Dev, QE, Planner (where applicable), and Ops instructions to remove assumptions that every project uses Node/npm and to require Julia/Lean behavior described here.
- Update README and the spec-writing guidance with a Julia/Lean feature-spec template: Julia behavior and tests, algorithm/theorem mapping, proof obligations and limitations, commands/toolchain, CPU/GPU matrix, performance evidence when applicable, acceptance checkboxes, and out-of-scope items.
- Preserve the existing Node.js sample as a harness fixture/demo unless a separate task explicitly removes it. Its tests must continue to work; Julia support must not break existing supported workflows.
- Preserve telemetry as the audit record and `harness/LEARNING.md` as durable lessons. Document and test the new verifier-stage event names and gate decisions consistently with the current telemetry schema.

## Test Specification

Add or update harness-level validation (automated tests or documented reproducible fixture checks) covering at least:

1. A Julia-only, non-algorithm change discovers and runs the configured Julia test command without invoking npm or requiring a Lean proof.
2. A Julia algorithm change with a valid theorem runs the Julia tests and Lean build and reaches QE only after both Dev and Formal Verification pass.
3. A missing theorem, failed Lean build, admitted/sorry proof, or unapproved axiom fails the Formal Verification gate and prevents QE/Ops progression until the retry policy is exhausted.
4. A Julia test failure fails QE and prevents Ops.
5. A GPU-affecting change distinguishes GPU pass, GPU failure, and unavailable hardware; unavailable hardware cannot be recorded as a pass.
6. An explicitly performance-scoped change requests benchmark evidence without enforcing fragile timing thresholds in unit tests.
7. Existing Node.js sample workflow behavior and telemetry lifecycle remain intact.
8. Failure loop-back, two-cycle maximum, terminal failure telemetry, and the “Ops only after all gates pass” invariant work with the added gate.

## Acceptance Criteria

- [ ] Harness instructions support a target root containing a Julia package and use its declared Julia environment and test command; no Node/npm assumptions are applied to Julia targets.
- [ ] A distinct, independently prompted Formal Verification gate exists and runs the pinned Lean project command before QE for algorithmic changes.
- [ ] Every new or semantically changed VI/Bellman algorithm is required to map to a Lean theorem statement and complete proof; missing/failed/admitted proofs fail the gate.
- [ ] Target onboarding produces an inventory of existing VI/Bellman algorithms and linked Lean proof status, clearly identifying any legacy proof gaps without claiming complete verification.
- [ ] The harness explicitly distinguishes mathematical-algorithm proof from proof of the concrete Julia floating-point/GPU implementation.
- [ ] Julia correctness testing is mandatory for applicable changes; GPU requirements and unavailable hardware are reported honestly and cannot be silently marked passed.
- [ ] Performance-oriented specs require reproducible benchmark evidence while correctness remains the first gate and ordinary unit tests avoid brittle timing limits.
- [ ] The workflow, telemetry, loop-back limit, and Ops gate cover the added verification stage; Ops cannot run after any failed required gate.
- [ ] The updated README/spec guidance documents how to write Julia + Lean specs and how to configure target-specific commands/toolchains.
- [ ] Existing Node.js sample checks continue to pass unchanged.
- [ ] The harness changes themselves have tests or reproducible fixture evidence for all cases in the Test Specification.

## Expected File List

Update the existing harness command, agent instructions, README, and spec guidance; add a verifier agent and harness fixtures/tests as needed. Expected areas include:

- `.claude/commands/harness.md`
- `.claude/agents/dev.md`
- `.claude/agents/qe.md`
- `.claude/agents/ops.md`
- `.claude/agents/planner.md` (if its instructions cover task planning)
- `.claude/agents/verifier.md` (new)
- `README.md`
- `harness/specs/` (Julia/Lean spec template or example)
- Harness test/fixture files for Julia, Lean, and retained Node behavior
- `harness/tools/telemetry-mcp/` only if required to represent or validate the additional stage

Do not add Lean proofs for IntervalMDP.jl algorithms as part of this harness-only change; the harness must be ready to require them in subsequent algorithm work.

## Out of Scope

- Implementing or optimizing IntervalMDP.jl algorithms, model types, or GPU kernels.
- Claiming full floating-point/GPU implementation verification without a verified correspondence proof.
- Formalizing every future model family in this harness change; keep the process extensible, while individual algorithm specs define their mathematical obligations.
- Replacing Julia with Rust/C++, rewriting the telemetry store, or removing the existing Node.js fixture.
- Adding arbitrary Lean axioms or treating tests/benchmarks as formal proofs.
- Requiring a specific Lean formalization architecture or theorem library before inspecting the target repository; the selected project layout/toolchain must be documented and pinned by the implementation spec.
