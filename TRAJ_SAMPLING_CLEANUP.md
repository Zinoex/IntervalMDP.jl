# Trajectory-sampling rework: cleanup list

Deferred cleanup from the `jy/traj-sampling-rework` branches (IntervalMDP.jl and the
thesis benchmark repo). Do these **after merging**, not before. Each item says what
still uses the old thing, so check that before removing it.

Paths: `lib/` = this repo, `bench/` =
`~/thesis/sampled-robust-value-iteration-thesis/experiments/benchmark`, `thesis/` =
`/mnt/d/Thesis/documents/msc-thesis-report`.

## Action selection (reworked 2026-10-10)

The new design is a hierarchy: `epsilon_greedy` → `action_epsilon`, or `boltzmann` →
schedule `fixed` (`action_temperature`) or `gap_based` (`action_temperature_k`,
`action_temperature_min`). Only the benchmark config enforces it so far.

### Library (`lib/`)

- [ ] **Restrict the action policy's schedule.** `TrajectorySampling(action_policy = ...)`
  still accepts any `TemperatureSchedule`, including `GapDecayTemperature` and
  `UpdateDecayTemperature`. Validate it in the constructor so only `FixedTemperature`
  and `GapBasedTemperature` are allowed.
- [ ] **Remove `GapDecayTemperature` and `UpdateDecayTemperature`** once nothing uses
  them. The benchmark no longer exposes them for either policy (state selection
  reworked 2026-10-11). Remaining users:
  - the priority-sweeping sampler (`src/prioritysweeping.jl` imports both, and its
    tests use them in `test/base/prioritysweeping.jl`);
  - the library's own trajectory tests ("gap-decay temperature", "update-decay
    temperature", and the backup-counter testset in `test/base/trajectorysampling.jl`).
- [ ] **When `UpdateDecayTemperature` goes, check whether the backup counter can go
  too.** `ss.updates` and `TemperatureContext.n` only exist to feed it
  (`src/trajectorysampling.jl`, see `_count_updates!`; and the same counter in
  `src/prioritysweeping.jl`).
- [ ] **Library docs that still describe the old schedules:**
  - the `TemperatureSchedule` docstring;
  - the `TemperatureContext` docstring ("keeps a decayed temperature inside
    `[t_min, t_max]`");
  - the "Temperature schedules" section and the `t_min` warning in the
    `TrajectorySampling` docstring;
  - the schedule list in `docs/src/reference/solve.md`.
- [ ] **Optional:** the `GapBasedTemperature` import in `src/prioritysweeping.jl` was
  added for symmetry and isn't used there. Keep it or drop it.

### Benchmark (`bench/`)

- [ ] **`_traj_policy` and `_traj_temperature` in `src/registry.jl`** are dead since
  the state-selection rework, so delete them (see that section below).
- [ ] **`src/analysis/report.jl`:** `_FACTOR_FAMILIES`, `_FACTOR_NAMES` and
  `_PARAM_SHORT` only know the state temperature keys. Add an
  `action_temperature_schedule` family (members: schedule, `action_temperature`,
  `action_temperature_k`, `action_temperature_min`) before any OFAT sweep varies the
  action temperature.

### Thesis (`thesis/chapters/sampling.tex`)

- [ ] **Replace the old `\subsection{Action Selection}`** with
  `\subsection{New Action Selection}`, and rename the new one to "Action Selection".
  The old one still uses `p_A`; the new one uses `\epsilon_A`.
- [ ] **`SampleTrajectory` algorithm:** the line
  `T_a = ComputeTemp(n, t_min, t_max, τ_a)` should become
  `T_a = ComputeTemp_A(t, k, t_min)`.
- [ ] **The general `ComputeTemp` equation** (fixed / termination-criteria decay /
  Bellman-update decay) is obsolete now that `T_s` is fixed too. Delete it, along with
  the `T_s = ComputeTemp(n, t_min, t_max, τ_s)` line in `SampleTrajectory` (see the
  state-selection section).

## State selection (reworked 2026-10-11)

The new design: `state_exploit` (U/L, both policies); `epsilon_greedy` → `state_epsilon`,
or `boltzmann` → `state_temperature` (fixed), `state_score = mixture | log_mixture`,
`state_explore`, and the weights α_e and α_p (`state_alpha_{e,p}_schedule = fixed |
power`, with `_k`, `_min` and `_max` under `power`). Library: `MixtureStateScore`,
`AlphaSchedule` (`FixedAlpha`, `PowerAlpha`, `CustomAlpha`, `ClippedAlpha`),
`_resolve_score`, `_initial_gap`.

### Library (`lib/`)

- [ ] **Remove the old successor scores** once the tests stop using them:
  `GreedyScore`, `ExplorationScore`, `GapWeightedExplorationScore`, `GapFunction`,
  `ExponentialGap`, `PolynomialGap` (and `_gap_weight`), `VPIScore` (and `_vpi`,
  `_vpi_gain`). Keep `LogStateScore` and every `ExplorationTerm`, since the mixture uses
  them. Current users: `test/base/trajectorysampling.jl` (the "successor scores f_S",
  "VPI successor score" and ε-greedy/Boltzmann testitems, and most solve-parity cases);
  the `src/prioritysweeping.jl` docstring that points at `ExplorationScore` (line ~159);
  `docs/src/reference/solve.md`.
- [ ] **Change the library default `state_score = GreedyScore()`** to
  `LogStateScore(MixtureStateScore())`, the power rule the benchmark defaults to under
  Boltzmann (2026-10-11). Under the default ε-greedy only the bound is read, so it
  behaves the same.
- [ ] **Restrict the state policy's schedule** in the `TrajectorySampling` constructor
  to `FixedTemperature`, to match the benchmark.
- [ ] **Rename "greedy" to "exploit":** `_greedy_bound` → `_exploit_bound`, and
  `f_greedy` → `f_exploit` in the docstrings (`StateScore`, `_sample_state`, the
  `state_policy` keyword note in `TrajectorySampling`).
- [ ] **Shrink `_ExploringScore`** to `MixtureStateScore` alone, or fold the hooks into
  its methods, once the old scores are gone. The four-argument `_state_scores`
  fallback for the union goes with it.
- [ ] **Stale "reach-avoid is negative on avoid states" comments.** It isn't (checked
  2026-10-11): reach-avoid initializes avoid states to `0` in both bounds, the
  per-step postprocess pins them to `0`, and the final postprocess is a no-op. The
  `-1` sentinel and the final `+1` shift belong to (DFA) safety, which GSRDP rejects.
  Keep the guards, but re-justify them (round-off below zero in `U − L`, negative
  `ActionUncertaintyExplore`, values from a caller-supplied value function):
  - `src/trajectorysampling.jl`: the `_categorical_sample` and `_safe_log`
    docstrings (the `TemperatureContext` one was rewritten 2026-10-11);
  - `test/base/trajectorysampling.jl`: the comments in the `_safe_log` and
    "TemperatureContext clamps the gap" testsets.
- [ ] **`GapDecayTemperature` docstring/tests:** since 2026-10-11 the rollout context's
  `diff` is the initial gap `δ = max_{s ∈ S₀} Diff(s)`, not `Diff(s₀)`, so the legacy
  schedules changed meaning too. Moot if they are removed (see the action section);
  otherwise re-check the "gap-decay temperature" solve case.
- [ ] **Module header comment** at the top of `src/trajectorysampling.jl`: drop "incl.
  VPI" and the per-policy annealed temperature schedule from the list of choices.

### Benchmark (`bench/`)

- [ ] **Delete the dead resolvers in `src/registry.jl`:** `_traj_policy`,
  `_traj_temperature`, `_traj_state_score` and `_traj_state_score_base`. There is a
  NOTE above them; `_traj_explore` is still used.
- [ ] **Banked `configs/grid-search-1/*.toml` no longer resolve.** Every trajectory arm
  uses removed keys (`state_temperature_schedule`, `state_bound`, `state_beta`,
  `gap_fn`, `tau`, ...), and the old scores have no equivalent under the mixture. They
  are records of past runs: archive them, add a header saying they need a pre-rework
  commit to re-run, or delete them.
- [ ] **`src/analysis/report.jl`:** the `state_temperature_schedule` and
  `state_score`/`state_beta`/`tau` entries in `_FACTOR_FAMILIES`, `_FACTOR_NAMES` and
  `_PARAM_SHORT`, and the `T∈[min,max]` special case in the label formatter, only
  describe the old keys. Keep them while banked results are being reported, then
  replace them with `state_alpha_e_schedule` and `state_alpha_p_schedule` families
  (members: schedule, α, `_k`, `_min`, `_max`) and `state_temperature`.

### Thesis (`thesis/chapters/sampling.tex`)

- [ ] **Replace the old `\subsection{State Selection}`** (the greedy / exploration /
  gap-weighted `f_S`, `G(δ)`, and the "Near Greedy States" `f_gap` paragraph) with
  `\subsection{New State Selection}`, and rename the new one to "State Selection". The
  old one uses `p_S`; the new one uses `\epsilon_S`.
- [ ] **Heuristics:** rename the "Greedy State" paragraph and `f_{\mathrm{greedy}}` to
  the exploit heuristic `f_{\mathrm{exploit}}`. The new subsection currently says that
  `f_exploit` "is the greedy state heuristic above", so fix that sentence too.
- [ ] **`SampleTrajectory` algorithm:** replace the `T_s = ComputeTemp(...)` line with
  the computation of `α_e(δ)` and `α_p(δ)` from `δ = max_{s ∈ S₀} Diff(s)` (`T_s` is a
  fixed parameter now), and pass them to `SampleState`.

## Gauss-Seidel batch size as `k_frac` (2026-10-10)

Trajectory sampling now takes `k_frac`, so batch size = `max(1, ceil(k_frac·|S|))`,
resolved per problem by `resolve_k_frac` in `bench/src/registry.jl`. It requires
`gauss_seidel = true`.

- [ ] **Decide whether to drop absolute `k` for `sampling = "trajectory"`** and
  require `k_frac`. Absolute `k` is still accepted in `resolve_trajectory`
  (`_require_k`). Banked configs use it, e.g. smoke-4 `k = 1` and the grid-search
  configs, so convert them first.
- [ ] **`k` without `gauss_seidel = true` is silently ignored** (`resolve_trajectory`
  falls back to `Int(get(params, "k", 1))`). Make it an error to match the
  hierarchical parameters, as `k_frac` already does.

## Preset removal (2026-10-10)

- [ ] **Optional, `bench/configs/smoke-4*.toml`:** `gsrdp_trajectory_gauss_seidel_k1`
  carries an explicit `label` only so it pools with banked results. Rename it freely
  once those banked runs no longer matter. (The `gap_weighted` and `boltzmann` rows
  were rewritten to the mixture in the state rework, with new labels
  `gsrdp_trajectory_mixture` and `gsrdp_trajectory_boltzmann_mixture`, so they no
  longer pool with the banked ones.)
- [ ] **One preset is back (2026-10-11): `trajectory_brtdp`** (`TRAJECTORY_PRESETS` and
  `_expand_preset` in `bench/src/registry.jl`). It is the closest point to slow-tail's
  preset, with no library changes. It differs from slow-tail in four ways, listed in
  `bench/docs/experiment-config.md`. If exact parity with slow-tail's brtdp is ever
  needed, port these from `jy/slow-tail`:
  - a `max(p_U, p_L)` probability source on the mixture, over the union support (from
    slow-tail's `GapContributionScore`);
  - `ExpectedGapStop(; max_adversary)`;
  - the strategy's `dedup` flag.
  Its DelftBlue run lives in `bench/configs/brtdp-rework/`, under the label
  `brtdp_rework`.

## Performance pass (2026-10-11)

Speedups with behaviour unchanged bit for bit. They touch `sampling.jl`,
`trajectorysampling.jl`, `gsrdp.jl`, `value.jl`, `workspace.jl` and
`bellman/{kernels,state}.jl`. They were developed in the worktree
`lib/.claude/worktrees/perf-sampling` and are now on `jy/traj-sampling-rework` (commits
`6c2794b` and `483170a`). The incremental gap/`previous` bookkeeping in the GSRDP loop
(`_IncrementalBounds`, 2026-10-12) came after and is not in `perf-sampling`.

- [x] **Bring the changes into `jy/traj-sampling-rework`.** Done. The benchmark env
  develops the main checkout, so DelftBlue needs this branch pulled and `bootstrap.jl`
  re-run.
- [ ] **Remove the worktrees** `lib/.claude/worktrees/agent-a06ec297a56ed9096` (the
  subagent's) and `lib/.claude/worktrees/perf-sampling`. Both are merged.
- [ ] **Callbacks must only read the value function.** `_IncrementalBounds` relies on it,
  and the `solve` docstring says so. Revisit if a callback or sampler ever needs to
  write the bounds.
- [ ] **The full test suite needs to run from `test/`.** Under `julia -e`,
  `@run_package_tests` scans `..` relative to the working directory, so from the repo root
  it scans `~/thesis`. Run it from `test/` with a filter relative to the repo, and skip
  `test/cuda`, which segfaults without a GPU. The suite also rewrites
  `test/data/multiObj_robotIMDP.nc`; restore it with `git checkout`.
- [ ] **Optional:** the generic `_omax_value_order`/`_omax_fill` are now used only by dense
  ambiguity sets and by tests. Keep them as the reference implementation.
