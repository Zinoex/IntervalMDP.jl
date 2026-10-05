# Agent harness (vendored)

A gated **Dev → Formal Verification → QE → Ops** workflow harness built on Claude Code subagents. You hand it a spec and a target repository; it implements, independently verifies proofs, tests, and ships a pull request, recording every step to a local SQLite telemetry DB and accumulating lessons in `harness/LEARNING.md`.

It drives **Julia** scientific-computing packages (primary target: [`IntervalMDP.jl`](https://github.com/Zinoex/IntervalMDP.jl)) with **Lean** machine-checked proofs for every value-iteration (VI) / Bellman algorithm, and it keeps the original **Node.js** sample workflow.

## How it works

The `/harness` slash command runs in the main thread as an **orchestrator**. It does not edit code itself. It identifies the target root, discovers the target's toolchain, delegates each stage to a dedicated subagent with a self-contained prompt, then enforces a gate before advancing.

Stages:

1. **Dev** (`.claude/agents/dev.md`) — implements the spec in the target root and writes/runs tests (`Pkg.test()` for Julia, `npm test` for Node). For any new or semantically changed VI/Bellman algorithm it also writes the Lean theorem + complete proof, runs `lake build`, and documents Julia↔Lean traceability and proof limitations. Gate: all tests pass (and the Lean build when proofs are required).
2. **Formal Verification** (`.claude/agents/verifier.md`, new) — an independent, read-only verifier (tools: Read, Glob, Grep, Bash, telemetry; **no Write/Edit**). It builds the Lean project with the **pinned** toolchain, checks every expected theorem is present by name, and rejects `sorry`, `admit`, placeholder proofs, and unapproved axioms (static pre-check + `#print axioms`). It never edits proofs and never downloads or switches toolchains; a missing toolchain is a blocker, not a pass. It reports pass/fail per obligation and the **proof scope** (abstract/mathematical vs concrete implementation). Gate: every obligation passes. When a change adds/alters no VI/Bellman semantics and touches no proof dependency, the gate is recorded as `skip` with a reason (never as `pass`).
3. **QE** (`.claude/agents/qe.md`) — independently re-runs the Julia test suite (CPU always; CUDA/GPU when the change affects GPU code), lints unit tests for brittle timing thresholds, checks benchmark evidence for performance-scoped specs, and for Node targets runs the original `npm test` + live `curl` checks. GPU outcomes are **PASS / FAIL / UNAVAILABLE**; UNAVAILABLE on a required GPU check is an unmet criterion, never a pass. Gate: every criterion green.
4. **Ops** (`.claude/agents/ops.md`) — runs **only** if Dev, Formal Verification (when required) and QE all passed in the same cycle. Branches, commits, opens a PR via `gh`, and appends lessons to `harness/LEARNING.md`. If the target is not a git repo, has no remote, or `gh` is missing, Ops reports a blocker instead of improvising.

A failure in Dev, Formal Verification or QE loops back to Dev with the specific failure evidence. Maximum **2** Dev → gate cycles in total before the run aborts as `harness_failed`. Environment blockers (required toolchain/GPU UNAVAILABLE, command not inferable) end the run immediately. Ops never runs after a failed gate.

Planner (`.claude/agents/planner.md`) is an optional helper for breaking down work; on **target onboarding** it produces the verification inventory (see below).

Persistence:
- **Telemetry** — `harness/tools/telemetry-mcp` is a local MCP server that appends events (`harness_started`, `delegation_start/end`, `verify_started/finished`, `gate_decision`, `loopback`, `harness_completed`, …) to a SQLite DB. Records *what happened*. If `node` is unavailable, `harness/tools/harness/record_event.py` writes the same events to the same schema.
- **harness/LEARNING.md** — durable, human-readable lessons appended at the end of each run. Records *what was learned*.

### ASCII flow

```
                 ┌────────────────────────────────────────────────┐
                 │  /harness <spec.md> [target-root]              │
                 │  (orchestrator — main thread)                  │
                 └────────────────────────────────────────────────┘
                                       │ reads harness/LEARNING.md + spec,
                                       │ discovers target toolchain
                                       ▼
        ┌──────────────────────────────────────────────────────────────────┐
        ▼                                                                  │ loopback
   ┌─────────┐ pass ┌──────────────┐ pass/skip ┌─────────┐ pass ┌───────┐ │ (max 2 cycles)
   │   Dev   │─────▶│ Formal Verif. │─────────▶│   QE    │─────▶│  Ops  │ │
   │ subagent│      │ verifier (RO) │          │ subagent│      │       │ │
   └─────────┘      └──────────────┘           └─────────┘      └───────┘ │
        │ fail             │ fail                   │ fail          │      │
        └──────────────────┴────────────┬───────────┘               ▼      │
                                        ▼                      gh PR url   │
                          ┌───────────────────────────────┐                │
                          │  re-spawn Dev w/ failure ctx  │────────────────┘
                          └───────────────────────────────┘
          blocker (toolchain / GPU UNAVAILABLE) ──▶ harness_failed (no Ops)

   every transition ──▶  mcp__telemetry__recordTelemetry  ──▶  telemetry.db
   end of run        ──▶  append lessons                  ──▶  harness/LEARNING.md
```

Operator status lines: `▶ [1/4] Dev`, `▶ [2/4] Formal Verification` (or `– [2/4] Formal Verification — not required (<reason>)`), `▶ [3/4] QE`, `▶ [4/4] Ops`, `↻ [1/4] Dev — remediating … (cycle 2/2)`.

## Repository layout

Vendored into the IntervalMDP.jl repository (paths relative to the repo root; run all commands from there):

```
.claude/
  agents/        dev.md  verifier.md  qe.md  ops.md  planner.md   (subagent definitions)
  commands/      harness.md                                     (the /harness slash command)
.mcp.json                             (telemetry MCP server → harness/tools/telemetry-mcp)
harness/
  README.md                           (this file)
  LEARNING.md                         (accumulated lessons)
  harness.config.example.toml         (per-target command/toolchain configuration)
  specs/
    TEMPLATE-julia-lean.md            (Julia + Lean feature-spec template)
    TEMPLATE-onboarding-inventory.md  (VI/Bellman algorithm ↔ Lean proof inventory template)
    intervalmdp-harness-update.md     (spec for this harness adaptation)
    hello-world-api.md                (Node.js sample spec)
  tools/
    telemetry-mcp/                    (local MCP server, SQLite events; needs node)
    harness/
      discover.py                     (target root → kinds, commands, toolchain, blockers)
      proof_hygiene.py                (verifier static pre-check + lake/#print axioms pipeline)
      gpu_check.py                    (GPU probe/classifier: PASS / FAIL / UNAVAILABLE)
      timing_lint.py                  (flags wall-clock thresholds in Julia unit tests)
      gate_sim.py                     (executable model of the orchestrator gates)
      record_event.py                 (telemetry fallback writer, same schema, no node)
  tests/harness/                      (harness tests + fixtures: Julia, Lean, Node)
src/ test/ ext/ ...                   (the IntervalMDP.jl package — the default target root)
```

The package's own `Pkg.test()` only runs `test/`; the harness suite is separate: `sh harness/tests/harness/run.sh`. A target-specific `harness.config.toml` goes at the target root.

> The Node sample *target project* (`hello-world-api/`) is **not** checked into this directory; only its spec (`harness/specs/hello-world-api.md`) is. A `/harness harness/specs/hello-world-api.md <empty-dir>` run creates it. The previously referenced `specs/goodbye-endpoint.md` is not present either. A discovery-only Node fixture lives in `harness/tests/harness/fixtures/node-sample/`.

## Configuring a Julia + Lean target

### Target root

The harness is vendored into IntervalMDP.jl, so by default the target root is this repository's root (`/harness harness/specs/<feature>.md`). To target another checkout, pass it explicitly: `/harness harness/specs/<feature>.md /abs/path/to/IntervalMDP.jl`, or put `Target root:` in the spec's *Commands / Toolchain* section, or `[target] root` in a config file. `harness/` and its fixtures are never the target (except harness-only specs, which run `sh harness/tests/harness/run.sh`).

### Commands and toolchains — precedence

1. **Spec** — the *Commands / Toolchain* section of the task spec (passed as `--set key=value`).
2. **Config** — `harness.config.toml` (`--config PATH`, `$HARNESS_CONFIG`, or `<target-root>/harness.config.toml`). See `harness/harness.config.example.toml`; keep it in the harness repo if you don't want to modify the upstream target.
3. **Discovery** — `python3 harness/tools/harness/discover.py <target-root> [--require julia,lean,gpu,node]`:
   - Julia: `Project.toml` (+ `Manifest.toml`), Julia version must satisfy `[compat] julia` (IntervalMDP.jl: 1.9+ unless stricter); commands `julia --project=<root> -e 'using Pkg; Pkg.instantiate()'` and `julia --project=<root> -e 'using Pkg; Pkg.test()'`. No npm/Jest/curl for Julia targets.
   - Lean: `lean-toolchain` (pinned; never switched or downloaded) + `lakefile.lean`/`lakefile.toml`, at the root or a subdirectory (≤2 levels); command `lake build` from the Lean root, plus a configured `lean.test` (e.g. `lake test`). Approved axioms default to `propext`, `Classical.choice`, `Quot.sound`.
   - GPU: CUDA/AMDGPU/Metal/KernelAbstractions in `[deps]`/`[weakdeps]`/`ext/` marks GPU code; the GPU **test runner is never inferred** — configure `gpu.test`. `gpu.probe` is `harness/tools/harness/gpu_check.py probe <root>`, which probes CUDA in a **temporary** Julia environment (develops the target + adds CUDA from the local depot, offline), so it works when CUDA is only a weak dependency / test extra (IntervalMDP.jl layout). Probe `NOT_FUNCTIONAL` → GPU UNAVAILABLE; probe `ERROR` is **not** evidence of "no GPU" — it is a FAIL / blocker to investigate.
   - Lean: if the pinned toolchain is already installed under `$ELAN_HOME/toolchains` (default `~/.elan`), `lean.toolchain_installed=yes` and `lean.toolchain_bin` points at its own `bin/` (use that `lake` directly — it can never trigger a download, unlike the elan proxy).
   - Node: `package.json` → `npm install`, `npm test`, `npm start`.
4. **Blocker** — if a required value can't be inferred (no project files, ambiguous Lean roots, no pinned toolchain, Julia version outside compat, required tool missing), discovery prints `blocker=…` and exits 2.

### CPU / GPU matrix

| Check | When | Outcomes | Gate effect |
|---|---|---|---|
| Julia tests on CPU | every applicable change | pass / fail | fail → loopback |
| Julia tests on CUDA | change affects GPU code (spec matrix says required) | PASS / FAIL / UNAVAILABLE, classified by `harness/tools/harness/gpu_check.py run <root> --cmd "<gpu.test>"` — exit code alone never yields PASS; PASS needs evidence the GPU testset ran and passed | FAIL → loopback; UNAVAILABLE on a required check → blocker (never pass) |
| Lean `lake build` + obligations | new/changed VI/Bellman algorithm, or proof dependency touched | pass / fail / blocked | fail → loopback; missing toolchain → blocker |
| Benchmark evidence | explicitly performance-scoped spec only | present / missing | missing → QE criterion fails; no timing thresholds in unit tests |

### Proof scope

Proofs about the abstract/mathematical algorithm are **not** proofs of the Julia floating-point/GPU implementation. Specs and reports must list floating-point rounding, overflow/underflow, GPU execution, and Lean↔Julia correspondence as limitations unless the task proves them. The harness never claims a "verified floating-point implementation".

### Target onboarding

On a new target, run the Planner to produce `harness/specs/inventory-<target>.md` from `harness/specs/TEMPLATE-onboarding-inventory.md`: every existing VI/Bellman algorithm, its Lean proof link/status, and an explicit list of legacy verification gaps. A task touching an unproved legacy algorithm must supply its proof before the formal gate passes. The model family column keeps the process open to MDPs, L1-MDPs, mixtures and factored variants.

## Writing your own spec

- **Julia + Lean work**: copy `harness/specs/TEMPLATE-julia-lean.md` (Objective; Commands / Toolchain; Julia behavior & tests; Algorithm ↔ Theorem mapping; Proof obligations & limitations; Proof policy; CPU/GPU matrix; Performance evidence; Acceptance criteria; File list; Out of scope).
- **Node work**: follow `harness/specs/hello-world-api.md` — Objective, Module Contracts, Endpoint / Behavior Specification, Test Specification, Acceptance Criteria (checkboxes QE verifies one by one), File List, Out of Scope.

Then: `/harness harness/specs/<your-spec>.md <target-root>`.

## Quickstart

### Prereqs

- [Claude Code](https://docs.claude.com/claude-code) CLI, `python3` ≥ 3.11 (harness helpers)
- Node.js ≥ 18 + `npm` — for the telemetry MCP server (`cd harness/tools/telemetry-mcp && npm install`) and Node targets
- Julia (version per target `[compat]`, e.g. via juliaup) — Julia targets
- elan/Lean with the target's pinned toolchain **pre-installed** — Lean proof gate
- CUDA-capable GPU + configured `gpu.test` — only for required GPU checks
- `gh` CLI authenticated and a git repo with a remote — for Ops to open a PR

### Julia + Lean target

```bash
git clone https://github.com/Zinoex/IntervalMDP.jl /abs/path/IntervalMDP.jl
python3 harness/tools/harness/discover.py /abs/path/IntervalMDP.jl --require julia   # inspect commands/blockers
claude
/harness harness/specs/<feature>.md /abs/path/IntervalMDP.jl
```

Expected status lines:
```
▶ [1/4] Dev — delegating to dev agent ...
✓ Dev passed
▶ [2/4] Formal Verification — delegating to verifier agent ...
✓ Formal Verification passed
▶ [3/4] QE — delegating to qe agent ...
✓ QE passed
▶ [4/4] Ops — delegating to ops agent ...
✓ Ops — PR: https://github.com/<you>/.../pull/<n>
✓ harness_completed — PR <url>
```

### Node sample

```bash
mkdir -p hello-world-api
claude
/harness harness/specs/hello-world-api.md /abs/path/hello-world-api
```
Formal Verification is recorded as `skip` (no VI/Bellman algorithm); QE runs `npm test` and the `curl` checks from the spec.

## Harness tests

```bash
sh harness/tests/harness/run.sh
```
Result line / exit code: `RESULT: PASS` (exit 0) only when every check ran and passed; `RESULT: FAIL` (exit 1) on any failure; `RESULT: PASS-INCOMPLETE (n UNAVAILABLE …)` (exit **2**) when nothing failed but n checks could not run — that is **not** a full pass.

Real runs use temporary copies of the fixtures (no Manifest.toml / `.lake` written into `harness/tests/harness/fixtures`):
- **GPU**: when `julia` exists the suite probes CUDA (`gpu_check.py probe`, temp env, offline) and, if functional, runs the configured `gpu.test` on `julia-gpu` (expects classifier **PASS**), `julia-gpu-failing` (expects **FAIL**) and `julia-gpu` with `CUDA_VISIBLE_DEVICES=-1` (expects **UNAVAILABLE**); it also shows that a CPU-only command exiting 0 is classified FAIL, never GPU PASS. If the probe is not functional those checks are UNAVAILABLE with the probe's reason.
- **Lean**: `lake build` + `#print axioms` (via `proof_hygiene.lean_pipeline`) run only with the fixture's **pinned** toolchain already installed, using the toolchain's own `bin/lake` (never the elan proxy, never a download); otherwise UNAVAILABLE. Opt-in `HARNESS_LEAN_TOOLCHAIN_OVERRIDE=<installed toolchain>` (e.g. `leanprover/lean4:v4.33.0-rc2`) rewrites the pin in the temp copies to exercise the real pipeline — the runner labels that **NON-PINNED evidence**; it does not show acceptance under the pinned toolchain.
- **Node**: with `node`/`npm` present, `test_case7_node_sample_tests` is only a smoke check that the unchanged `harness/tools/telemetry-mcp/telemetry.js` loads; the hello-world `npm test` + `curl` workflow stays QE's job.
Covers all eight Test Specification cases of `harness/specs/intervalmdp-harness-update.md`: static checks that the instruction files encode the rules, discovery against Julia/Lean/Node fixtures, the proof-hygiene scanner against valid/sorry/admit/axiom/missing/placeholder Lean fixtures plus adversarial ones (sorry hidden between `'"'` char literals, `axiom propext` inside a namespace, multi-line `True` placeholder, kernel bypass via `debug.skipKernelTC`/`implemented_by`/`extern`/`unsafe`, decl name on the next line), the gate state machine (order, loopback max 2, blockers, Ops-only-after-all-gates), GPU PASS/FAIL/UNAVAILABLE mapping and classifier (synthetic + real CUDA runs), timing-threshold lint, real `Pkg.test()` runs of the Julia fixtures (when `julia` exists), and the telemetry schema. Checks that need the pinned Lean toolchain, `node`/`npm` or a functional GPU are reported as **UNAVAILABLE** (distinct from PASS) when missing.

## Telemetry

Events stream to a SQLite DB (`harness/tools/telemetry-mcp/telemetry.db`, table `events(id, ts, event_name, details)`) via the `mcp__telemetry__recordTelemetry` tool exposed by `harness/tools/telemetry-mcp`, or via `python3 harness/tools/harness/record_event.py <event> '<json>'` when node is unavailable. Inspect runs with `sqlite3` (or `python3 -c "import sqlite3; ..."`).

## Notes

- Ops detects a missing git repo / remote / `gh` and reports a blocker rather than failing mid-stage.
- The harness runs autonomously between gates — no per-stage prompts. Only blockers and terminal states are surfaced.
- harness/LEARNING.md is read at the start of every run and re-consulted on stage failure; lessons feed forward into subsequent runs.
