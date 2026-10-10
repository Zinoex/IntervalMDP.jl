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
  them. Current users:
  - the successor policy (`state_temperature_schedule = gap_decay | update_decay` in
    the benchmark);
  - the priority-sweeping sampler (`src/prioritysweeping.jl` imports both, and its
    tests use them in `test/base/prioritysweeping.jl`).
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

- [ ] **`_traj_policy` and `_traj_temperature` in `src/registry.jl`** are still generic
  over `kind`, but only the successor policy uses them now. Fold them into the
  state-selection rework, or rename them to `_traj_state_*`.
- [ ] **`docs/experiment-config.md`, temperature notes:** the bullets on `gap_decay`,
  `update_decay` and `state_temperature_min` only apply to the successor policy now.
  Rewrite them when the state side is reworked.
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
  Bellman-update decay) now only describes `T_s`. Keep it until the state-selection
  rework, then rewrite it to match.

## Preset removal (2026-10-10)

- [ ] **Optional, `bench/configs/smoke-{2,3,4}*.toml`:** the rows rewritten from
  presets carry explicit `label`s (`gsrdp_trajectory_gap_weighted`,
  `gsrdp_trajectory_boltzmann` and `gsrdp_trajectory_gauss_seidel_k1`) only so they
  pool with banked results. Rename them freely once those banked runs no longer
  matter.
