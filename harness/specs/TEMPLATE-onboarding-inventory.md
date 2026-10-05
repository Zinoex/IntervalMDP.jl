# <Target> — Verification Inventory (onboarding baseline)

> Produced by the Planner on target onboarding (see `.claude/agents/planner.md`)
> and saved as `harness/specs/inventory-<target>.md`. Dev keeps it current whenever a
> proof is added or an algorithm changes. This inventory is a status report,
> **not** a verification claim.

- Target: `<repo URL>` @ `<commit sha>`
- Target root: `<abs path>`
- Julia compat: `<[compat] julia>`; Lean root / pinned toolchain: `<path>` / `<lean-toolchain>` (or "no Lean project yet")
- Inventory date: `<YYYY-MM-DD>`

## VI / Bellman algorithms

| # | Algorithm | Model family | Julia entry point (function — file) | CPU / GPU paths | Lean theorem(s) (name — file) | Proof status (`proved` / `partial` / `none`) | Proof scope (`abstract` / `implementation`) | Limitations |
|---|---|---|---|---|---|---|---|---|
| 1 | `<name>` | `<MDP / interval / L1 / mixture / factored>` | `<f>` — `src/<file>.jl` | `<dense, sparse, CUDA>` | `<Ns.thm>` — `lean/<File>.lean` or — | `none` | — | float rounding, overflow, GPU, Lean↔Julia correspondence |

## Legacy verification gaps

> Every row above whose status is not `proved`. A task touching one of these
> must supply its theorem + proof before it can pass the Formal Verification gate.

- `<algorithm>` — no Lean theorem yet (status `none`).

## Summary

- Algorithms: `<n>`; proved: `<p>`; partial: `<q>`; none: `<r>`.
- Statement: the package is **not** fully verified while gaps remain; proofs (where present) cover the abstract algorithm unless stated otherwise.
