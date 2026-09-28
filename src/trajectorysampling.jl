###################################
# Trajectory-based sampling        #
###################################
#
# The general `TrajectorySamplingStrategy` category, and the O-max primitives
# every trajectory step is built from (`_omax_distribution`,
# `_omax_expectation`, `_is_source_state`, ...), live in `sampling.jl` — the
# priority-queue strategies share them, and `_gsrdp_sample` dispatches on the
# category. This file holds everything that is trajectory-specific: the
# reach/avoid and initial-state helpers, the categorical draw, and the
# configurable sampler itself.
#
# The sampler is one strategy parameterized by a product of orthogonal
# choices, rather than a family of subtypes each baking in one combination:
#
#   selection policy  (ε-greedy | Boltzmann)      — for actions and for states
#   × temperature schedule (fixed | annealed, per policy)
#   × score function  (f_A over actions, f_S over successors — incl. VPI,
#                      with a pluggable f_explore term)
#   × concrete transition (which bound, which adversary direction)
#   × Gauss-Seidel batching (whole trajectory per iteration, or `k` at a time)

###################################
# Reach/avoid, initial state       #
###################################
# `reach`/`avoid` (specification.jl) are only defined for
# AbstractReachability/AbstractReachAvoid/AbstractSafety — not every
# `Property`. These fall back to "no reach/avoid states" for anything else,
# so trajectory sampling still runs (just never stops via goal/obstacle).

_trajectory_reach_states(prop) = CartesianIndex[]
_trajectory_reach_states(prop::AbstractReachability) = reach(prop)   # AbstractReachAvoid IS-A AbstractReachability

_trajectory_avoid_states(prop) = CartesianIndex[]
_trajectory_avoid_states(prop::AbstractReachAvoid) = avoid(prop)
_trajectory_avoid_states(prop::AbstractSafety) = avoid(prop)

# `initial_states(mp)` elements aren't normalized to `CartesianIndex` at
# model-construction time (could be `Int`, `Tuple`, or `CartesianIndex`).
_to_state_index(s::CartesianIndex) = s
_to_state_index(s::Integer) = CartesianIndex(s)
_to_state_index(s::Tuple) = CartesianIndex(s)

# `AllStates()` (no restriction declared) samples uniformly over every
# state; a concrete initial-states vector samples uniformly among them.
function _trajectory_initial_state(model)
    init = initial_states(model)
    return init isa AllStates ? rand(CartesianIndices(source_shape(model))) :
           _to_state_index(rand(init))
end

# Hard safety cap, independent of the configured termination rules —
# guarantees a single `sample` call can't hang GSRDP if a strategy's own
# termination logic never fires (e.g. an absorbing-free transient region, or a
# configuration carrying only a user `PredicateStop`).
_trajectory_max_steps(model) = 1 * num_states(model)

"""
    _categorical_sample(weights::AbstractVector) -> Union{Int, Nothing}

Sample an index into `weights` proportionally to it (not required to be
normalized), via cumulative-sum search. Backs the Boltzmann draw, which passes
the exponentiated scores.

Returns `nothing` when `weights` carries no positive mass — every entry zero or
negative, or the vector empty. There is then no index the vector actually
selects, and the caller must decide what that means rather than being handed an
arbitrary one.

Only strictly positive entries are summed and drawn from, so a zero-weight
index is never selected even on the `rand() === 0.0` draw, and negative weights
(which `weights` should not contain, but which a reach-avoid value function
carries on avoid states before the first postprocess) can neither be selected
nor shrink the sampling range.
"""
function _categorical_sample(weights::AbstractVector)
    total = zero(eltype(weights))
    for w in weights
        w > zero(w) && (total += w)
    end
    total > zero(total) || return nothing

    u = rand() * total
    acc = zero(eltype(weights))
    last_positive = nothing
    for i in eachindex(weights)
        w = weights[i]
        w > zero(w) || continue
        acc += w
        acc >= u && return i
        last_positive = i
    end
    return last_positive   # floating-point fallback; not `nothing`, since total > 0
end

###################################
# The TrajectorySampling submodule #
###################################
#
# The configuration vocabulary (`EpsilonGreedy`, `Boltzmann`, `GreedyScore`,
# ...) lives in a submodule so these short, generic names don't land in the
# top-level `IntervalMDP` namespace, mirroring `PriorityQueueSampling`.
#
# NOTE: this module and the strategy type inside it are both named
# `TrajectorySampling` (Julia allows a type to share its enclosing module's
# name), but only one of the two may carry a docstring — Julia's docsystem
# can't disambiguate "the module" from "the same-named member inside it" —
# so this module has none; see the docstring on the type below.
module TrajectorySampling

using LinearAlgebra: dot

import ..IntervalMDP:
    IntervalMDP,
    TrajectorySamplingStrategy,
    SatisfactionMode,
    IntervalMode,
    Upper,
    Lower,
    Optimistic,
    Pessimistic,
    StateIterator,
    sample,
    reset_sampling_strategy!,
    available,
    num_states,
    num_actions,
    system_property,
    isoptimistic,
    _omax_marginal,
    _omax_distribution,
    _omax_expectation,
    _isoptimistic,
    _ismaximize,
    _is_source_state,
    _maybe_target_state,
    _action_uncertainty,
    _categorical_sample,
    _trajectory_reach_states,
    _trajectory_avoid_states,
    _trajectory_initial_state,
    _trajectory_max_steps

###################################
# 1. Temperatures and policies     #
###################################
#
# Action selection and successor selection are the same operation — draw one
# candidate given a score per candidate — so they share one policy type. What
# differs between them is only the score function (§2 and §4).
#
# A Boltzmann policy's temperature need not be a constant: it is a
# `TemperatureSchedule`, resolved to one number per rollout (§7) against the
# `TemperatureContext` that rollout starts from. That is what lets exploration
# anneal — as the bounds tighten, or as backups accumulate — instead of the
# caller having to pick one compromise temperature for the whole solve.

"""
    TemperatureSchedule

How a [`Boltzmann`](@ref) policy's temperature `T` is obtained at the start of
each rollout: [`FixedTemperature`](@ref) (a constant),
[`GapDecayTemperature`](@ref) (annealed by how far apart the bounds still are
at the rollout's initial state), or [`UpdateDecayTemperature`](@ref) (annealed
by how many Bellman updates the strategy has issued so far).

`_temperature(schedule, ctx)` evaluates one against a
[`TemperatureContext`](@ref), and `_resolve_policy` does that once per rollout,
so a single trajectory is always drawn at one temperature.
"""
abstract type TemperatureSchedule end

"""
    TemperatureContext(diff, n)

What a [`TemperatureSchedule`](@ref) is evaluated against at the start of a
rollout:

  * `diff` — `U(s₀) − L(s₀)`, the value-function gap at the state the rollout
    starts from, clamped to `[0, 1]`. The clamp keeps a fractional `τ` off a
    negative base — a reach-avoid value function carries negative values on
    avoid states until `postprocess_value_function!` runs — the same guard
    `_gap_weight` takes for [`PolynomialGap`](@ref), and it also keeps a
    decayed temperature inside `[t_min, t_max]`.
  * `n` — the number of Bellman updates the strategy has issued so far, i.e.
    `Σ length(batch) * (num_actions + 1)` over the previous `sample` calls,
    which is exactly the `bellman_updates` count GSRDP reports to its callback.
"""
struct TemperatureContext
    diff::Float64
    n::Int

    TemperatureContext(diff::Real, n::Integer) = new(clamp(Float64(diff), 0.0, 1.0), Int(n))
end

"""
    FixedTemperature(t)

`T = t` at every rollout — the constant temperature, and what a bare number
means: `Boltzmann(0.5)` is shorthand for `Boltzmann(FixedTemperature(0.5))`.
`t` must be positive.
"""
struct FixedTemperature <: TemperatureSchedule
    t::Float64

    function FixedTemperature(t::Real)
        t > 0 || throw(ArgumentError("t must be positive, got $t"))
        return new(Float64(t))
    end
end

"""
    GapDecayTemperature(t_min, t_max, tau)

`T = t_min + (t_max − t_min)·Diff(s₀)^τ`, the termination-criteria decay: the
temperature tracks how much uncertainty is left at the state the rollout starts
from, `Diff(s₀) = U(s₀) − L(s₀)` (clamped to `[0, 1]`, see
[`TemperatureContext`](@ref)).

A wide-open `s₀` (`Diff = 1`) is explored at `t_max`; a converged one
(`Diff = 0`) is exploited at `t_min`, since there is nothing left to learn
where the rollout began. Larger `τ` holds the temperature near `t_min` until
the gap is nearly closed; `τ = 1` interpolates linearly in the gap.

Requires `0 < t_min ≤ t_max` and `τ > 0`. `t_min` is an exploration floor, not
a formality — see the warning on
[`IntervalMDP.TrajectorySampling.TrajectorySampling`](@ref).
"""
struct GapDecayTemperature <: TemperatureSchedule
    t_min::Float64
    t_max::Float64
    tau::Float64

    function GapDecayTemperature(t_min::Real, t_max::Real, tau::Real)
        t_min > 0 || throw(ArgumentError("t_min must be positive, got $t_min"))
        t_max >= t_min ||
            throw(ArgumentError("t_max must be at least t_min = $t_min, got $t_max"))
        tau > 0 || throw(ArgumentError("tau must be positive, got $tau"))
        return new(Float64(t_min), Float64(t_max), Float64(tau))
    end
end

"""
    UpdateDecayTemperature(t_min, t_max, tau)

`T = max(t_min, t_max·τⁿ)`, the Bellman-update decay: the temperature falls
geometrically in `n`, the number of Bellman updates issued so far (see
[`TemperatureContext`](@ref)), and floors at `t_min`.

`n` counts *backups*, not iterations — one trajectory of 20 states on a
2-action model already contributes 60 — so a useful `τ` is very close to 1
(e.g. `0.999`).

Requires `0 < t_min ≤ t_max` and `0 < τ ≤ 1`: a `τ > 1` would make `t_max·τⁿ`
grow without bound, which is an annealing schedule run backwards. `t_min` is an
exploration floor, not a formality — see the warning on
[`IntervalMDP.TrajectorySampling.TrajectorySampling`](@ref).
"""
struct UpdateDecayTemperature <: TemperatureSchedule
    t_min::Float64
    t_max::Float64
    tau::Float64

    function UpdateDecayTemperature(t_min::Real, t_max::Real, tau::Real)
        t_min > 0 || throw(ArgumentError("t_min must be positive, got $t_min"))
        t_max >= t_min ||
            throw(ArgumentError("t_max must be at least t_min = $t_min, got $t_max"))
        0 < tau <= 1 || throw(
            ArgumentError(
                "tau must be in (0, 1] — tau > 1 grows the temperature rather than " *
                "decaying it, got $tau",
            ),
        )
        return new(Float64(t_min), Float64(t_max), Float64(tau))
    end
end

"""
    _temperature(schedule, ctx) -> Float64

The temperature `schedule` gives a rollout starting in context `ctx` —
`ComputeTemp(n, t_min, t_max, τ)`.
"""
_temperature(schedule::FixedTemperature, ::TemperatureContext) = schedule.t

_temperature(schedule::GapDecayTemperature, ctx::TemperatureContext) =
    schedule.t_min + (schedule.t_max - schedule.t_min) * ctx.diff^schedule.tau

_temperature(schedule::UpdateDecayTemperature, ctx::TemperatureContext) =
    max(schedule.t_min, schedule.t_max * schedule.tau^ctx.n)

"""
    SelectionPolicy

How a candidate is drawn given a score per candidate: [`EpsilonGreedy`](@ref)
or [`Boltzmann`](@ref). Used for both action selection (scored by an
[`ActionScore`](@ref)) and successor selection (scored by a
[`StateScore`](@ref)).
"""
abstract type SelectionPolicy end

"""
    EpsilonGreedy(p)

Take the highest-scoring candidate with probability `1 - p`, and a uniformly
random one with probability `p`. `p = 0` is deterministic argmax; `p = 1` is
uniform.
"""
struct EpsilonGreedy <: SelectionPolicy
    p::Float64

    function EpsilonGreedy(p::Real)
        0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1], got $p"))
        return new(Float64(p))
    end
end

"""
    Boltzmann(T)

Draw a candidate with probability proportional to `exp(score / T)`. Small `T`
concentrates on the argmax; large `T` approaches uniform.

`T` is a [`TemperatureSchedule`](@ref), and a positive number is shorthand for
[`FixedTemperature`](@ref) — `Boltzmann(0.5) == Boltzmann(FixedTemperature(0.5))`.
A decaying schedule ([`GapDecayTemperature`](@ref),
[`UpdateDecayTemperature`](@ref)) is resolved to one number at the top of each
rollout, so every draw within one trajectory shares a temperature; see
`_resolve_policy`.
"""
struct Boltzmann{S <: TemperatureSchedule} <: SelectionPolicy
    T::S
end

Boltzmann(T::Real) = Boltzmann(FixedTemperature(T))

"""
    _select(policy, candidates, scores) -> eltype(candidates)

Draw one of `candidates` under `policy`, given the parallel vector `scores`.
`candidates` must be non-empty.

A `-Inf` score (which [`LogScore`](@ref) produces for a non-positive inner
score) is never selected by either policy — unless *every* score is `-Inf`, in
which case there is no signal to act on and both policies fall back to uniform.
"""
function _select(policy::EpsilonGreedy, candidates, scores)
    rand() < policy.p && return rand(candidates)
    m = maximum(scores)
    isfinite(m) || return rand(candidates)   # all -Inf: nothing to prefer
    return candidates[argmax(scores)]
end

function _select(policy::Boltzmann{FixedTemperature}, candidates, scores)
    m = maximum(scores)
    # Shift by the max before exponentiating (log-sum-exp): `exp` of a raw
    # score overflows to Inf for large scores and underflows to 0 for very
    # negative ones, either of which destroys the relative weights.
    isfinite(m) || return rand(candidates)   # all -Inf: nothing to prefer
    weights = [exp((s - m) / policy.T.t) for s in scores]
    i = _categorical_sample(weights)
    # `weights` always has a 1.0 entry (the argmax), so this cannot be
    # `nothing` in practice; guard rather than propagate a silent failure.
    return i === nothing ? candidates[argmax(scores)] : candidates[i]
end

# A decaying schedule is not a number until a rollout gives it one, so drawing
# straight from it is a programming error rather than a missing method.
_select(policy::Boltzmann, candidates, scores) = throw(
    ArgumentError(
        "a Boltzmann policy carrying a $(typeof(policy.T)) has no temperature until " *
        "it is resolved against a rollout's TemperatureContext (`_resolve_policy`)",
    ),
)

"""
    _resolve_policy(policy, ctx) -> SelectionPolicy

The policy a rollout actually draws with: a [`Boltzmann`](@ref)'s schedule
collapsed to the single temperature [`TemperatureContext`](@ref) `ctx` gives
it, and any other policy unchanged.
"""
_resolve_policy(policy::SelectionPolicy, ::TemperatureContext) = policy

_resolve_policy(policy::Boltzmann, ctx::TemperatureContext) =
    Boltzmann(FixedTemperature(_temperature(policy.T, ctx)))

###################################
# 2. Action scores f_A             #
###################################

"""
    ActionScore

`f_A(s, a)`, the score an action carries at the current state, consumed by a
[`SelectionPolicy`](@ref). `U(s, a)` and `L(s, a)` denote the O-max Q-values of
`a` at `s` against the upper and lower bound respectively, both realized in the
same adversary direction as the strategy's [`ConcreteTransition`](@ref).
"""
abstract type ActionScore end

"`f_A(s, a) = U(s, a)`, the upper-bound (optimistic) Q-value."
struct UpperBoundScore <: ActionScore end

"`f_A(s, a) = L(s, a)`, the lower-bound (pessimistic) Q-value."
struct LowerBoundScore <: ActionScore end

"""
    WeightedAverageScore(beta)

`f_A(s, a) = L(s, a) + β·[U(s, a) − L(s, a)]`, the gap-uncertainty /
weighted-average score. `β = 0` is the lower bound, `β = 1` the upper bound,
and values in between interpolate — larger `β` weights an action's remaining
uncertainty more heavily. `β` must be in `[0, 1]`.
"""
struct WeightedAverageScore <: ActionScore
    beta::Float64

    function WeightedAverageScore(beta::Real)
        0 <= beta <= 1 || throw(ArgumentError("beta must be in [0, 1], got $beta"))
        return new(Float64(beta))
    end
end

"""
    LogScore(inner)

`f_A(s, a) = log(f'_A(s, a))`, the normalized score, where `f'_A` is any other
[`ActionScore`](@ref). Composed with [`Boltzmann`](@ref) at temperature `T`
this draws an action with probability proportional to `f'_A^(1/T)` — so
`Boltzmann(1.0)` over a `LogScore` samples proportionally to the inner score
itself, rather than softmax-ing it.

A non-positive inner score maps to `-Inf`, which `_select` never
selects (and gives weight `0` under Boltzmann). See `_safe_log`.
"""
struct LogScore{S <: ActionScore} <: ActionScore
    inner::S
end

"""
    _safe_log(x) -> Float64

`log(x)` for `x > 0`, and `-Inf` otherwise. Guards [`LogScore`](@ref) against
the two ways its inner score can leave the domain of `log`: an exactly-zero
score (an unreached state under the optimistic bound), and a negative one (a
reach-avoid value function carries negative values on avoid states until
`postprocess_value_function!` runs). `log` would give `-Inf` for the first and
`NaN` for the second; `NaN` would then poison `maximum`/`argmax`.
"""
_safe_log(x) = x > 0 ? log(Float64(x)) : -Inf

"""
    _action_score(score, ambiguity_set, U, L, dir) -> Float64

`f_A` for one action, given that action's `ambiguity_set` at the current state.
`dir` is the adversary direction (`true` = O-maximization) that
[`_omax_expectation`](@ref) realizes the Q-values under.
"""
_action_score(::UpperBoundScore, as, U, L, dir) = Float64(_omax_expectation(as, U, dir))
_action_score(::LowerBoundScore, as, U, L, dir) = Float64(_omax_expectation(as, L, dir))

function _action_score(score::WeightedAverageScore, as, U, L, dir)
    l = Float64(_omax_expectation(as, L, dir))
    u = Float64(_omax_expectation(as, U, dir))
    return l + score.beta * (u - l)
end

_action_score(score::LogScore, as, U, L, dir) =
    _safe_log(_action_score(score.inner, as, U, L, dir))

###################################
# 3. Concrete transition           #
###################################

"""
    ConcreteTransition(; bound = Upper, adversary = nothing)

Which concrete distribution `p(·|s,a)` the adversary realizes out of the
ambiguity set `Γ_{s,a}`:

    p_{s,a} = argopt_{p ∈ Γ_{s,a}} Σ_{s'} p(s')·V(s')

with `V` the `bound` value function (`Upper` or `Lower`, the
`IntervalMode` enum) and `argopt` being `argmax` for an optimistic
adversary and `argmin` for a pessimistic one, computed by O-maximization
(`_omax_distribution`).

`adversary` is a [`SatisfactionMode`](@ref) (`Optimistic` / `Pessimistic`), or
`nothing` — the default — to take the direction from the specification via
`isoptimistic(spec)`.

!!! warning
    `nothing` is the safe default for a reason. `bellman_update!` backs both
    bounds up in the `isoptimistic(spec)` direction, and a rollout realized
    under the *other* direction explores states the Bellman update is not
    actually driven toward, which can starve the trajectory of any path to the
    reach set. Override only deliberately.
"""
struct ConcreteTransition
    bound::IntervalMode
    adversary::Union{Nothing, SatisfactionMode}

    ConcreteTransition(; bound::IntervalMode = Upper, adversary = nothing) =
        new(bound, adversary)
end

_bound_values(::Val{Upper}, vf) = vf.upper.current
_bound_values(::Val{Lower}, vf) = vf.lower.current
_bound_values(bound::IntervalMode, vf) = _bound_values(Val(bound), vf)

_adversary_direction(::Nothing, spec) = _isoptimistic(spec)
_adversary_direction(mode::SatisfactionMode, spec) = isoptimistic(mode)

###################################
# 4. Successor scores f_S          #
###################################

"""
    GapFunction

`G(δ)`, the discount a successor's exploration term takes for its greedy
shortfall `δ = f_gap(s',s,a)` behind the best successor: `G(0) = 1` keeps the
full bonus for a competitive successor, and `G → 0` as it falls further
behind. See [`ExponentialGap`](@ref) and [`PolynomialGap`](@ref); `τ` must be
positive in both.
"""
abstract type GapFunction end

"`G(δ) = exp(-δ/τ)`."
struct ExponentialGap <: GapFunction
    tau::Float64

    function ExponentialGap(tau::Real)
        tau > 0 || throw(ArgumentError("tau must be positive, got $tau"))
        return new(Float64(tau))
    end
end

"`G(δ) = (1-δ)^τ`, with `1-δ` clamped at 0 so a fractional `τ` can't go complex."
struct PolynomialGap <: GapFunction
    tau::Float64

    function PolynomialGap(tau::Real)
        tau > 0 || throw(ArgumentError("tau must be positive, got $tau"))
        return new(Float64(tau))
    end
end

_gap_weight(g::ExponentialGap, delta) = exp(-delta / g.tau)
_gap_weight(g::PolynomialGap, delta) = max(1 - delta, 0.0)^g.tau

###################################
# 4a. Exploration terms f_explore  #
###################################
#
# The bonus half of `ExplorationScore` / `GapWeightedExplorationScore`. Each
# term is a function of the successor alone. The probability weighting and
# `β` are applied by the score.

"""
    ExplorationTerm

`f_explore(t)`, the per-successor exploration bonus an [`ExplorationScore`](@ref)
or [`GapWeightedExplorationScore`](@ref) adds on top of its greedy term:

  * [`GapExplore`](@ref): `U(t) − L(t)`, the default
  * [`UpperBoundExplore`](@ref): `U(t)`, the best-case reach probability
  * [`ActionUncertaintyExplore`](@ref): `U^{-a_L}(t) − L(t)`
  * [`BellmanErrorExplore`](@ref): `|U_{k-1}(t) − U_k(t)|` or
    `|L_{k-1}(t) − L_k(t)|`
  * [`MaxBellmanErrorExplore`](@ref): the larger of those two
  * [`VPIExplore`](@ref): `maxₐ VPI(t, a)`, the value of perfect information
    about the Q-values at `t`

[`VPIScore`](@ref) is a different VPI: `VPI_{s,a}(t)` measures how much
learning `V*(t)` would change the decision at the rollout's *current* state,
so it depends on the rollout's action and is a successor score of its own.
"""
abstract type ExplorationTerm end

"`f_explore(t) = U(t) − L(t)`, the value-function gap. The default."
struct GapExplore <: ExplorationTerm end

"""
    UpperBoundExplore()

`f_explore(t) = U(t)`, the best-case reach probability. Paired with a score
whose `bound` is `Upper`, this just scales the greedy term by `1 + β`.
"""
struct UpperBoundExplore <: ExplorationTerm end

"""
    ActionUncertaintyExplore()

`f_explore(t) = U^{-a_L}(t) − L(t)` (Fan et al., 2007): how wrong the
lower-bound-optimal action `a_L = argmaxₐ L(t, a)` at `t` might still be,

    L(t)        = L(t, a_L)
    U^{-a_L}(t) = max_{a ∈ A(t) ∖ {a_L}} U(t, a)

computed by `_action_uncertainty`. It is `0` at a state with a single action,
and at an implicit sink. This costs `2·|A(t)|` O-max expectations per successor
per step, so a step is about as expensive as a Bellman backup at every
successor. It needs the step context, so the four-argument `_state_scores`
throws.
"""
struct ActionUncertaintyExplore <: ExplorationTerm end

"""
    VPIExplore()

`f_explore(t) = maxₐ VPI(t, a)`, the myopic value of perfect information about
the Q-values at `t` (Dearden et al., 1998). It measures how much learning the
true `Q(t, a)` of some action could change which action is best at `t`.

With `q_a ~ Uniform[L(t,a), U(t,a)]` and midpoints `m_a = (U(t,a) + L(t,a))/2`,
let `a₁` be the action with the best midpoint and `a₂` the runner-up. Then

    VPI(t, a₁) = E[max(0, m_{a₂} − q_{a₁})]   (a₁ may be worse than it looks)
    VPI(t, a)  = E[max(0, q_a − m_{a₁})]      (a ≠ a₁ may be better)

"Best" is the largest midpoint under `Maximize` and the smallest under
`Minimize`. The Q-bounds are O-max expectations in the `isoptimistic(spec)`
direction, the same one `bellman_update!` backs both bounds up in. It is `0`
at a state with a single action, and at an implicit sink. Like
[`ActionUncertaintyExplore`](@ref) it costs `2·|A(t)|` O-max expectations per
successor per step, and the four-argument `_state_scores` throws.
"""
struct VPIExplore <: ExplorationTerm end

"""
    BackupTracker()

The last backup magnitude of every state, `|V_before − V_after|`, for both
bounds. `NaN` marks a state that has not been backed up yet. It has to be
measured rather than read off the value function: `_gsrdp!` calls
`nextiteration!` just before `sample`, so `current` and `previous` agree
whenever a strategy is asked for a sample.

`TrajectorySampling.sample` drives it through `_before_sample!` and
`_after_sample!`. The first snapshots the `(U, L)` of every state in the batch
it hands to GSRDP, and the second measures how far those values moved once the
next `sample` call comes around.
"""
struct BackupTracker
    dU::Base.RefValue{Vector{Float64}}
    dL::Base.RefValue{Vector{Float64}}
    pending::Base.RefValue{Vector{Int}}
    snapshot::Base.RefValue{Vector{Tuple{Float64, Float64}}}

    BackupTracker() = new(
        Ref(Float64[]),
        Ref(Float64[]),
        Ref(Int[]),
        Ref(Tuple{Float64, Float64}[]),
    )
end

function _reset!(tracker::BackupTracker)
    tracker.dU[] = Float64[]
    tracker.dL[] = Float64[]
    tracker.pending[] = Int[]
    tracker.snapshot[] = Tuple{Float64, Float64}[]
    return nothing
end

# Sized lazily: how many target states there are is only known once there is a
# value function in hand.
function _ensure_size!(tracker::BackupTracker, vf)
    n = length(vf.upper.current)
    if length(tracker.dU[]) != n
        tracker.dU[] = fill(NaN, n)
        tracker.dL[] = fill(NaN, n)
    end
    return nothing
end

function _absorb_backups!(tracker::BackupTracker, vf)
    _ensure_size!(tracker, vf)
    U, L = vec(vf.upper.current), vec(vf.lower.current)
    dU, dL = tracker.dU[], tracker.dL[]
    for (i, (u, l)) in zip(tracker.pending[], tracker.snapshot[])
        dU[i] = abs(Float64(U[i]) - u)
        dL[i] = abs(Float64(L[i]) - l)
    end
    tracker.pending[] = Int[]
    tracker.snapshot[] = Tuple{Float64, Float64}[]
    return nothing
end

function _snapshot_batch!(tracker::BackupTracker, batch, vf)
    U, L = vf.upper.current, vf.lower.current
    lin = LinearIndices(U)
    tracker.pending[] = [lin[s] for s in batch]
    tracker.snapshot[] = [(Float64(U[s]), Float64(L[s])) for s in batch]
    return nothing
end

"""
    BellmanErrorExplore(; bound = Upper)

`f_explore(t) = |V_{k-1}(t) − V_k(t)|` with `V = U` or `L` per `bound`: how far
the last backup at `t` moved that bound. A state still moving is probably not
converged. A state that has not been backed up yet has no reading, so it falls
back to its gap `U(t) − L(t)`. That keeps an unvisited successor as attractive
as [`GapExplore`](@ref) would make it.

`k` counts `t`'s own backups, not solver iterations. A state that was not in
the last batch keeps the reading from the last batch it was in.
"""
struct BellmanErrorExplore <: ExplorationTerm
    bound::IntervalMode
    tracker::BackupTracker

    BellmanErrorExplore(; bound::IntervalMode = Upper) = new(bound, BackupTracker())
end

"""
    MaxBellmanErrorExplore()

`f_explore(t) = max(|U_{k-1}(t) − U_k(t)|, |L_{k-1}(t) − L_k(t)|)`, the larger
of the two [`BellmanErrorExplore`](@ref) readings. Falls back to the gap in the
same way.
"""
struct MaxBellmanErrorExplore <: ExplorationTerm
    tracker::BackupTracker

    MaxBellmanErrorExplore() = new(BackupTracker())
end

"""
    _explore_values(term, supp, vf, model, spec) -> Vector{Float64}

`f_explore(t)` for every successor `t` in `supp`, parallel to `supp`. `model`
and `spec` may be `nothing` for terms that only read the value function.
"""
function _explore_values(::GapExplore, supp, vf, model, spec)
    U, L = vec(vf.upper.current), vec(vf.lower.current)
    return [Float64(U[i]) - Float64(L[i]) for i in supp]
end

function _explore_values(::UpperBoundExplore, supp, vf, model, spec)
    U = vec(vf.upper.current)
    return [Float64(U[i]) for i in supp]
end

_explore_values(::ActionUncertaintyExplore, supp, vf, ::Nothing, spec) = throw(
    ArgumentError(
        "ActionUncertaintyExplore needs the model; call the nine-argument `_state_scores`",
    ),
)

_explore_values(::ActionUncertaintyExplore, supp, vf, model, spec) = [
    Float64(_action_uncertainty(model, _maybe_target_state(model, i), vf, spec)) for
    i in supp
]

"""
    _uniform_excess(l, u, c) -> Float64

`E[max(0, q − c)]` for `q ~ Uniform[l, u]`, with `l ≤ u`:

    0                  if c ≥ u
    (l + u)/2 − c      if c ≤ l
    (u − c)² / 2(u−l)  otherwise

The third branch only runs when `l < c < u`, so it never divides by zero.
`E[max(0, c − q)]` is `_uniform_excess(-u, -l, -c)`.
"""
function _uniform_excess(l::Float64, u::Float64, c::Float64)
    c >= u && return 0.0
    c <= l && return (l + u) / 2 - c
    return (u - c)^2 / (2 * (u - l))
end

"""
    _state_vpi(model, t, vf, spec) -> Float64

`maxₐ VPI(t, a)` at the state `t`; see [`VPIExplore`](@ref).
"""
function _state_vpi(model, t, vf, spec)
    _is_source_state(model, t) || return 0.0

    marginal = _omax_marginal(model)
    dir = _isoptimistic(spec)
    U, L = vf.upper.current, vf.lower.current
    # Flip the sign under Minimize, so "best" is always the largest.
    σ = _ismaximize(spec) ? 1.0 : -1.0

    # σ-scaled Q-intervals [lo, hi] of every available action.
    intervals = map(collect(available(model, t))) do a
        as = marginal[a, t]
        qu = σ * Float64(_omax_expectation(as, U, dir))
        ql = σ * Float64(_omax_expectation(as, L, dir))
        return minmax(qu, ql)
    end
    length(intervals) < 2 && return 0.0

    mid = [(lo + hi) / 2 for (lo, hi) in intervals]
    best = argmax(mid)
    runner_up = maximum(mid[j] for j in eachindex(mid) if j != best)

    lo₁, hi₁ = intervals[best]
    vpi = _uniform_excess(-hi₁, -lo₁, -runner_up)
    for (j, (lo, hi)) in enumerate(intervals)
        j == best && continue
        vpi = max(vpi, _uniform_excess(lo, hi, mid[best]))
    end
    return vpi
end

_explore_values(::VPIExplore, supp, vf, ::Nothing, spec) = throw(
    ArgumentError("VPIExplore needs the model; call the nine-argument `_state_scores`"),
)

_explore_values(::VPIExplore, supp, vf, model, spec) =
    [_state_vpi(model, _maybe_target_state(model, i), vf, spec) for i in supp]

# The tracker's reading for bound `V`, or the gap where there is none yet.
function _tracked_or_gap(tracker::BackupTracker, pick, supp, vf)
    _ensure_size!(tracker, vf)
    U, L = vec(vf.upper.current), vec(vf.lower.current)
    dU, dL = tracker.dU[], tracker.dL[]
    return map(supp) do i
        d = pick(dU[i], dL[i])
        isnan(d) ? Float64(U[i]) - Float64(L[i]) : d
    end
end

_explore_values(term::BellmanErrorExplore, supp, vf, model, spec) = _tracked_or_gap(
    term.tracker,
    term.bound == Upper ? (u, l) -> u : (u, l) -> l,
    supp,
    vf,
)

# Both readings are written together, so they are either both NaN or neither.
_explore_values(term::MaxBellmanErrorExplore, supp, vf, model, spec) =
    _tracked_or_gap(term.tracker, max, supp, vf)

# Lifecycle hooks, driven by `TrajectorySampling.sample` through the successor
# score. Only the Bellman-error terms carry state.
_before_sample!(::ExplorationTerm, vf) = nothing
_after_sample!(::ExplorationTerm, batch, vf) = nothing
_reset_score!(::ExplorationTerm) = nothing

const _TrackedTerm = Union{BellmanErrorExplore, MaxBellmanErrorExplore}
_before_sample!(term::_TrackedTerm, vf) = _absorb_backups!(term.tracker, vf)
_after_sample!(term::_TrackedTerm, batch, vf) = _snapshot_batch!(term.tracker, batch, vf)
_reset_score!(term::_TrackedTerm) = _reset!(term.tracker)

"""
    StateScore

`f_S(s', s, a)`, the score a candidate successor carries, consumed by a
[`SelectionPolicy`](@ref). All three forms are built from

    f_greedy(s')  = U(s') or L(s'),  per the score's `bound`
    f_explore(s') = an [`ExplorationTerm`](@ref), by default U(s') − L(s')

weighted by the realized transition probability `p(s'|s,a)`. See
[`GreedyScore`](@ref), [`ExplorationScore`](@ref) and
[`GapWeightedExplorationScore`](@ref). [`VPIScore`](@ref) instead scores by
the myopic value of perfect information about `s'`, and
[`LogStateScore`](@ref) wraps any of them for proportional sampling.
"""
abstract type StateScore end

"""
    GreedyScore(; bound = Upper)

`f_S(s') = p(s'|s,a)·f_greedy(s')` — the expected contribution of `s'`, with no
exploration term.
"""
struct GreedyScore <: StateScore
    bound::IntervalMode

    GreedyScore(; bound::IntervalMode = Upper) = new(bound)
end

"""
    ExplorationScore(beta; bound = Upper, explore = GapExplore())

`f_S(s') = p(s'|s,a)·f_greedy(s') + β·p(s'|s,a)·f_explore(s')`, the greedy
term plus a bonus for successors that `explore` rates as worth visiting (an
[`ExplorationTerm`](@ref)). By default that means successors whose bounds are
still far apart. `β ≥ 0`; `β = 0` reduces to [`GreedyScore`](@ref).
"""
struct ExplorationScore{E <: ExplorationTerm} <: StateScore
    bound::IntervalMode
    beta::Float64
    explore::E

    function ExplorationScore(
        beta::Real;
        bound::IntervalMode = Upper,
        explore::E = GapExplore(),
    ) where {E <: ExplorationTerm}
        beta >= 0 || throw(ArgumentError("beta must be non-negative, got $beta"))
        return new{E}(bound, Float64(beta), explore)
    end
end

"""
    GapWeightedExplorationScore(beta, gap; bound = Upper, explore = GapExplore())

`f_S(s') = p·f_greedy(s') + β·p·f_explore(s')·G(f_gap(s',s,a))` — the
exploration score with its bonus additionally discounted by how far `s'` falls
behind the best successor,

    f_gap(s',s,a) = δ = max_{x ∈ supp(s,a)} p(x)·V(x) − p(s')·V(s')

with `V = f_greedy`'s bound. Both terms are probability-weighted, so `δ ≥ 0`
always, and `δ = 0` exactly at the best successor — "`s'` is the greedy best",
not "`s'` is converged". `gap` is the discount `G`, a [`GapFunction`](@ref),
and `explore` is `f_explore`, an [`ExplorationTerm`](@ref).
"""
struct GapWeightedExplorationScore{G <: GapFunction, E <: ExplorationTerm} <: StateScore
    bound::IntervalMode
    beta::Float64
    gap::G
    explore::E

    function GapWeightedExplorationScore(
        beta::Real,
        gap::G;
        bound::IntervalMode = Upper,
        explore::E = GapExplore(),
    ) where {G <: GapFunction, E <: ExplorationTerm}
        beta >= 0 || throw(ArgumentError("beta must be non-negative, got $beta"))
        return new{G, E}(bound, Float64(beta), gap, explore)
    end
end

_greedy_bound(score::StateScore) = score.bound

# Lifecycle hooks, forwarded down to the score's `ExplorationTerm` (see
# `BackupTracker`). A score with no term has nothing to track.
_before_sample!(::StateScore, vf) = nothing
_after_sample!(::StateScore, batch, vf) = nothing
_reset_score!(::StateScore) = nothing

const _ExploringScore = Union{ExplorationScore, GapWeightedExplorationScore}
_before_sample!(score::_ExploringScore, vf) = _before_sample!(score.explore, vf)
_after_sample!(score::_ExploringScore, batch, vf) = _after_sample!(score.explore, batch, vf)
_reset_score!(score::_ExploringScore) = _reset_score!(score.explore)

"""
    _state_scores(score, probs, supp, vf) -> Vector{Float64}

`f_S` for every successor in `supp` (linear target indices carrying positive
probability), parallel to `supp`.
"""
function _state_scores(score::GreedyScore, probs, supp, vf)
    Vg = vec(_bound_values(score.bound, vf))
    return [Float64(probs[i]) * Float64(Vg[i]) for i in supp]
end

# Without the step context. Only an `ExplorationTerm` that reads nothing but the
# value function can be evaluated this way.
_state_scores(score::_ExploringScore, probs, supp, vf) =
    _state_scores(score, probs, supp, vf, nothing, nothing, nothing, nothing, nothing)

function _state_scores(
    score::ExplorationScore,
    probs,
    supp,
    vf,
    s,
    a,
    model,
    spec,
    transition,
)
    Vg = vec(_bound_values(score.bound, vf))
    x = _explore_values(score.explore, supp, vf, model, spec)
    return [
        Float64(probs[i]) * (Float64(Vg[i]) + score.beta * x[n]) for
        (n, i) in enumerate(supp)
    ]
end

function _state_scores(
    score::GapWeightedExplorationScore,
    probs,
    supp,
    vf,
    s,
    a,
    model,
    spec,
    transition,
)
    Vg = vec(_bound_values(score.bound, vf))
    x = _explore_values(score.explore, supp, vf, model, spec)

    greedy = [Float64(probs[i]) * Float64(Vg[i]) for i in supp]
    best = maximum(greedy)

    return [
        greedy[n] +
        score.beta * Float64(probs[i]) * x[n] * _gap_weight(score.gap, best - greedy[n])
        for (n, i) in enumerate(supp)
    ]
end

"""
    _state_scores(score, probs, supp, vf, s, a, model, spec, transition) -> Vector{Float64}

`f_S` given the full step context: the current state `s`, the rollout's action
`a`, and the strategy's [`ConcreteTransition`](@ref). This is what
`_sample_state` calls. Scores that only need `probs` and the value function
fall back to the four-argument form; [`VPIScore`](@ref) needs the rest.
"""
_state_scores(score::StateScore, probs, supp, vf, s, a, model, spec, transition) =
    _state_scores(score, probs, supp, vf)

###################################
# 4b. Value of perfect information #
###################################
#
# VPI-RTDP (Sanner et al., 2009), restated for interval MDPs. The full
# derivation is in docs/design/vpi.md. In short: with a uniform belief
# v_t ~ U[L(t), U(t)] and m_a = (U(s,a) + L(s,a))/2, revealing v_t = m(t) + x
# moves the expected Q-value of `a` along the line m_a + p_a(t)·x. Averaging the
# gain of `a` over the rollout's action a*, max(0, Δ_a + δ_a·x), over
# x ∈ [-h, h] has a closed form, `_vpi_gain`.

"""
    _vpi_gain(Δ, k) -> Float64

`φ(Δ, k)`, the average of `max(0, Δ + δ·x)` over `x ∈ [-h, h]` with
`k = |δ|·h ≥ 0`:

    φ(Δ, k) = 0               if Δ ≤ -k   (the line never crosses zero)
            = Δ               if Δ ≥  k   (the line never drops below zero)
            = (Δ + k)² / (4k) otherwise   (the triangle above zero)

`φ` is continuous and continuously differentiable, non-negative, and
non-decreasing in both arguments. The third branch only runs when `k > |Δ|`, so
it never divides by zero. With `k = 0` (no uncertainty, or no difference in
transition probability) it reduces to `max(0, Δ)`.
"""
function _vpi_gain(Δ::Float64, k::Float64)
    Δ <= -k && return 0.0
    Δ >= k && return Δ
    return (Δ + k)^2 / (4k)
end

"""
    _vpi(s, a, probs, supp, vf, model, spec, transition) -> Vector{Float64}

`VPI_{s,a}(t)` for every successor `t` in `supp`, parallel to `supp`:

    VPI_{s,a}(t) = max_{a' ≠ a} φ(Δ_{a'}, k_{a'}(t))
    Δ_{a'}       = σ·(m_{a'} − m_a),  m_{a'} = (U(s,a') + L(s,a')) / 2
    k_{a'}(t)    = |p_{a'}(t) − p_a(t)| · (U(t) − L(t)) / 2

where `a` is the rollout's action, `σ = ±1` for `Maximize`/`Minimize`, and φ is
[`_vpi_gain`](@ref). `U(s,a')` and `L(s,a')` are the O-max Q-bounds in
`transition`'s adversary direction. `p_{a'}` is the distribution `transition`
realizes for `a'`, and `p_a = probs`, the distribution the rollout already
drew. All zeros when `a` is the only action available at `s`.
"""
function _vpi(s, a, probs, supp, vf, model, spec, transition)
    marginal = _omax_marginal(model)
    dir = _adversary_direction(transition.adversary, spec)
    σ = _ismaximize(spec) ? 1.0 : -1.0
    U, L = vf.upper.current, vf.lower.current
    Uv, Lv = vec(U), vec(L)

    # Midpoint of the Q-interval, plus the distribution `transition` realizes.
    function q_mid_and_p(a′)
        as = marginal[a′, s]
        pU = _omax_distribution(as, U, dir)
        pL = _omax_distribution(as, L, dir)
        m = (Float64(dot(pU, Uv)) + Float64(dot(pL, Lv))) / 2
        return m, (transition.bound == Upper ? pU : pL)
    end

    m_star, _ = q_mid_and_p(a)
    vpi = zeros(Float64, length(supp))
    for a′ in available(model, s)
        a′ == a && continue
        m, p = q_mid_and_p(a′)
        Δ = σ * (m - m_star)
        for (n, t) in enumerate(supp)
            h = (Float64(Uv[t]) - Float64(Lv[t])) / 2
            k = abs(Float64(p[t]) - Float64(probs[t])) * max(h, 0.0)
            vpi[n] = max(vpi[n], _vpi_gain(Δ, k))
        end
    end
    return vpi
end

"""
    VPIScore(; beta = 0.95, bound = Upper)

`f_S(t) = VPI_{s,a}(t)`, the myopic value of perfect information about the
successor `t` (VPI-RTDP, Sanner et al. 2009): how much learning `V*(t)` is
expected to improve on the rollout's action `a` at `s`. Unlike the gap-based
scores, it ignores successors whose open gap cannot change the decision at `s`.

With `v_t ~ Uniform[L(t), U(t)]` and `m_{a'} = (U(s,a') + L(s,a'))/2`,

    VPI_{s,a}(t) = max_{a' ≠ a} φ(Δ_{a'}, k_{a'}(t))
    Δ_{a'}       = σ·(m_{a'} − m_a)
    k_{a'}(t)    = |p_{a'}(t) − p_a(t)| · (U(t) − L(t)) / 2
    φ(Δ, k)      = 0 if Δ ≤ -k;  Δ if Δ ≥ k;  (Δ + k)²/(4k) otherwise

where `σ = ±1` for `Maximize`/`Minimize` and each `p_{a'}` is the distribution
the strategy's [`ConcreteTransition`](@ref) realizes. See `docs/design/vpi.md`
for the derivation.

The score follows the paper's `CHOOSENEXTSTATE-VPI` (Algorithm 3), minus its
termination branch. With `b(t) = p(t)·(U(t) − L(t))`:

 1. if `max_t b(t) > beta`, the score is `b`. VPI tells you little while the
    bounds are nearly vacuous, so this is BRTDP's rule. `beta` is absolute, and
    the paper's `0.95` assumes values in `[0, 1]`.
 2. otherwise, if some successor has positive VPI, the score is `VPI`.
 3. otherwise, the score is `b` again. Ending the rollout is left to
    `terminate`, e.g. [`ExpectedGapStop`](@ref).

Successors come from the support of the rollout's `p(·|s,a)`. The paper
considers every `t`, which could move the rollout to a state that is not a
successor of `(s, a)`.

The paper draws `t` with probability proportional to its score, which is
`state_policy = Boltzmann(1.0)` with `state_score = LogStateScore(VPIScore())`.
Like the other successor scores, `EpsilonGreedy` reads only `bound`.

!!! note
    `a` is the action the rollout sampled, as in the paper, not necessarily the
    one that is best by its midpoint `m_a`. If another action `a'` has a higher
    midpoint, `Δ_{a'} > 0`, and every successor then scores at least `Δ_{a'}`,
    even one whose bounds have already met.

Each step computes two O-max distributions per available action, the same
order of work as a Bellman backup at `s`.
"""
struct VPIScore <: StateScore
    bound::IntervalMode
    beta::Float64

    function VPIScore(; beta::Real = 0.95, bound::IntervalMode = Upper)
        beta >= 0 || throw(ArgumentError("beta must be non-negative, got $beta"))
        return new(bound, Float64(beta))
    end
end

_state_scores(score::VPIScore, probs, supp, vf) = throw(
    ArgumentError(
        "VPIScore needs the step context (s, a, model, spec, transition); " *
        "call the nine-argument `_state_scores`",
    ),
)

function _state_scores(score::VPIScore, probs, supp, vf, s, a, model, spec, transition)
    U, L = vec(vf.upper.current), vec(vf.lower.current)
    b = [Float64(probs[i]) * (Float64(U[i]) - Float64(L[i])) for i in supp]
    maximum(b) > score.beta && return b

    vpi = _vpi(s, a, probs, supp, vf, model, spec, transition)
    return sum(vpi) > 0 ? vpi : b
end

"""
    LogStateScore(inner)

`f_S(t) = log(f'_S(t))`, with `f'_S` any other [`StateScore`](@ref). This is
the successor-side counterpart of [`LogScore`](@ref). Under
[`Boltzmann`](@ref) at temperature `T` it draws `t` with probability
proportional to `f'_S(t)^(1/T)`, so `Boltzmann(1.0)` samples in proportion to
the inner score itself. A non-positive inner score maps to `-Inf` (see
`_safe_log`), and `_select` never picks it.
"""
struct LogStateScore{S <: StateScore} <: StateScore
    inner::S
end

_greedy_bound(score::LogStateScore) = _greedy_bound(score.inner)

_before_sample!(score::LogStateScore, vf) = _before_sample!(score.inner, vf)
_after_sample!(score::LogStateScore, batch, vf) = _after_sample!(score.inner, batch, vf)
_reset_score!(score::LogStateScore) = _reset_score!(score.inner)

# Both arities spelled out: a varargs method would be ambiguous against the
# nine-argument `StateScore` fallback.
_state_scores(score::LogStateScore, probs, supp, vf) =
    _safe_log.(_state_scores(score.inner, probs, supp, vf))

_state_scores(score::LogStateScore, probs, supp, vf, s, a, model, spec, transition) =
    _safe_log.(_state_scores(score.inner, probs, supp, vf, s, a, model, spec, transition))

###################################
# 5. Termination rules             #
###################################

"""
    TerminationRule

An extra stopping condition for a rollout, on top of reaching a goal/obstacle
state, leaving the source sub-box, and the unconditional
`_trajectory_max_steps` backstop. A strategy carries a list of them and
stops as soon as any one fires.

Rules are checked at one of two points, which is what the two hooks below
distinguish:

  * [`terminate_pre`](@ref) — at the top of a step, before an action is even
    selected. Cheapest; for criteria that are functions of the current state
    and the trajectory so far.
  * [`terminate_post`](@ref) — after the concrete transition distribution has
    been realized, for criteria that are functions of `probs` itself. Sees the
    distribution the step already paid for, so it costs no extra O-max work.

Both default to `false`, so a rule implements only the hook it needs.
"""
abstract type TerminationRule end

"""
    terminate_pre(rule, s, trajectory, i, value_function, model, spec) -> Bool

Stop before selecting an action at `s`. `trajectory` is the states visited so
far (not yet including `s`) and `i` is the step count. Defaults to `false`.
"""
terminate_pre(::TerminationRule, s, trajectory, i, value_function, model, spec) = false

"""
    terminate_post(rule, s, a, probs, trajectory, i, value_function, model, spec) -> Bool

Stop after realizing `probs = p(·|s,a)` but before drawing a successor from it,
ending the rollout with `s` as its last state. Defaults to `false`.
"""
terminate_post(::TerminationRule, s, a, probs, trajectory, i, value_function, model, spec) =
    false

"""
    MaxSteps(n = nothing)

Stop after `n` transitions. `nothing` — the default — means `num_states(model)`,
matching the unconditional backstop, so the default configuration is capped
exactly as trajectory sampling has always been.
"""
struct MaxSteps <: TerminationRule
    n::Union{Int, Nothing}

    function MaxSteps(n::Union{Int, Nothing} = nothing)
        (n === nothing || n >= 0) || throw(ArgumentError("n must be non-negative, got $n"))
        return new(n)
    end
end

terminate_pre(rule::MaxSteps, s, trajectory, i, value_function, model, spec) =
    i >= (rule.n === nothing ? _trajectory_max_steps(model) : rule.n)

"""
    ExpectedGapStop(tau)

BRTDP's expected-gap criterion (McMahan et al., 2005). With

    B = Σ_{s'} p(s'|s,a)·(U(s') − L(s'))

the trajectory ends at `s` as soon as

    B < (U(s) − L(s)) / tau

i.e. once the gap it can expect to shrink one step ahead has fallen a factor
`tau` below the gap still open at the state it is standing on. Larger `tau`
means longer trajectories (a stricter condition to stop); `tau` must be
positive.

A fully converged `s` (`U(s) == L(s)`) also ends the rollout: the threshold is
then 0, which `B >= 0` can never fall below, so without this the trajectory
would run on to the step cap through an already-tight region.
"""
struct ExpectedGapStop <: TerminationRule
    tau::Float64

    function ExpectedGapStop(tau::Real = 10.0)
        tau > 0 || throw(ArgumentError("tau must be positive, got $tau"))
        return new(Float64(tau))
    end
end

function terminate_post(
    rule::ExpectedGapStop,
    s,
    a,
    probs,
    trajectory,
    i,
    value_function,
    model,
    spec,
)
    gap = value_function.upper.current .- value_function.lower.current
    diff = gap[s]
    iszero(diff) && return true

    B = dot(probs, vec(gap))
    return B < diff / rule.tau
end

"""
    PredicateStop(f)

Stop when `f(s, trajectory, i, value_function, model, spec)` returns `true` —
an escape hatch for criteria the built-in rules don't cover. Checked at the
[`terminate_pre`](@ref) point.
"""
struct PredicateStop <: TerminationRule
    f::Function
end

terminate_pre(rule::PredicateStop, s, trajectory, i, value_function, model, spec) =
    rule.f(s, trajectory, i, value_function, model, spec)

###################################
# 6. The strategy                  #
###################################

"""
    TrajectorySampling.TrajectorySampling(; kwargs...)

Trajectory-based sampling: instead of sampling states independently, simulate a
trajectory through the model — from an initial state, repeatedly select an
action, realize a concrete transition distribution out of the ambiguity set,
and draw a successor from it — and relax the states visited along the way.

Each of the four decisions is configured independently:

# Keywords
- `action_policy::SelectionPolicy = EpsilonGreedy(0.1)`: how the action is
  drawn given `action_score`. A `Boltzmann` policy may carry an annealing
  [`TemperatureSchedule`](@ref) rather than a constant — see below.
- `action_score::ActionScore = UpperBoundScore()`: `f_A(s, a)`.
- `transition::ConcreteTransition = ConcreteTransition()`: which bound and
  adversary direction realize `p(·|s,a)`.
- `state_policy::SelectionPolicy = EpsilonGreedy(0.1)`: how the successor is
  drawn. **Note the asymmetry** documented under `_sample_state`:
  `EpsilonGreedy` always exploits on `p(s')·f_greedy(s')` and ignores
  `state_score`'s exploration terms; only `Boltzmann` honours the full `f_S`.
- `state_score::StateScore = GreedyScore()`: `f_S(s', s, a)`.
- `terminate = [MaxSteps()]`: extra [`TerminationRule`](@ref)s, ORed together.
- `gauss_seidel::Bool = false`: see below.
- `k::Int = 1`: batch size, used only when `gauss_seidel = true`.
- `reverse::Bool = true`: return each trajectory goal-first.

# Temperature schedules

A `Boltzmann` policy's temperature is resolved once per rollout, before the
first action is selected, and held for the whole trajectory — so the action
policy and the successor policy each get their own `T` from their own schedule
(`τ_a` and `τ_s` of the writeup):

* [`FixedTemperature`](@ref) — `T = t`, the constant.
* [`GapDecayTemperature`](@ref) — `T = t_min + (t_max − t_min)·Diff(s₀)^τ`,
  where `Diff(s₀) = U(s₀) − L(s₀)` is the gap at the state *this* rollout
  starts from. Exploration therefore fades where the bounds have already met.
* [`UpdateDecayTemperature`](@ref) — `T = max(t_min, t_max·τⁿ)`, where `n` is
  the number of Bellman updates the strategy has issued so far — the same
  `length(states) * (num_actions + 1)` per iteration GSRDP reports to its
  callback, accumulated across `sample` calls and reset by
  `reset_sampling_strategy!`.

In Gauss-Seidel mode the temperature belongs to the rollout, not the batch: a
trajectory doled out over several iterations keeps the temperature it was
sampled at.

!!! warning
    Keep `t_min` well away from zero. A cold Boltzmann policy is argmax in all
    but name, so annealing all the way down runs into exactly the hazard
    described under "Why the default ε is not zero" below: the rollout locks
    onto the states it already prefers, the rest keep their initial values, and
    the gap criterion never fires. `t_min` is the exploration floor the schedule
    can never anneal past — on a three-state IMDP, `t_min = 0.01` stalls where
    `t_min = 0.5` converges.

# Gauss-Seidel batching

With `gauss_seidel = false` the whole trajectory is returned from every
`sample` call, and GSRDP relaxes all of it in one (Jacobi) iteration — every
backup reads the *previous* iterate, so no value propagates along the
trajectory within that iteration.

With `gauss_seidel = true` one trajectory is kept active and doled out `k`
states at a time, one batch per solver iteration. Since GSRDP calls
`nextiteration!` between iterations, batch `n+1` reads the values batch `n`
wrote — which is what makes values actually propagate along the trajectory. A
fresh trajectory is sampled only once the active one is exhausted.

`reverse` therefore only has an observable effect in Gauss-Seidel mode: it puts
the goal-adjacent end of the trajectory in the first batch, so the newly-tight
values propagate backward along the remaining batches.

# Why the default ε is not zero

`EpsilonGreedy(0.0)` is pure exploitation: the rollout is a deterministic
function of the current bounds, so it can lock onto a cycle and relax the same
handful of states forever while the rest keep their initial values. GSRDP's gap
criterion then never fires and `solve` does not terminate — this is not
hypothetical, it is what a fully greedy configuration does on a three-state IMDP
whose greedy successor from the initial state is the initial state itself.

Any `ε > 0` makes every successor in the support, and every available action,
reachable with positive probability at every step, so every state reachable from
the initial set is visited infinitely often. That is what makes convergence an
expectation rather than a hope, so `0.1` — not `0` — is the default on both
policies. `EpsilonGreedy(0.0)` remains available for deliberate pure
exploitation; pair it with a strategy that covers the state space some other way
(e.g. `EpsilonGreedyMixture` with `RandomSubsetState`).

!!! note
    A goal or obstacle state is *not* part of the returned trajectory — the
    rollout stops on reaching one, without relaxing it. If every initial state
    is a goal state the trajectory is empty and `sample` returns an empty
    sequence; GSRDP would then iterate without making progress, since its gap
    criterion has no iteration cap. Declare initial states outside the reach
    set.

Only flat (non-factored) models are supported — the O-max realization
(`_omax_marginal`) requires a single `Marginal`.
"""
struct TrajectorySampling <: TrajectorySamplingStrategy
    action_policy::SelectionPolicy
    action_score::ActionScore
    transition::ConcreteTransition
    state_policy::SelectionPolicy
    state_score::StateScore
    terminate::Vector{TerminationRule}
    gauss_seidel::Bool
    k::Int
    reverse::Bool

    # Active-trajectory batching state. `Base.RefValue{Any}` because the
    # `CartesianIndex{N}` arity isn't known until there is a model in hand —
    # the same reason `RNDPriorityQueueSampling` holds its networks that way.
    # `nothing` means "no active trajectory".
    buffer::Base.RefValue{Any}
    cursor::Base.RefValue{Int}

    # Bellman updates issued so far — `n` for `UpdateDecayTemperature`. Tracked
    # here because `sample` is not handed the solver's iteration count; see the
    # increment in §8.
    updates::Base.RefValue{Int}

    function TrajectorySampling(;
        action_policy::SelectionPolicy = EpsilonGreedy(0.1),
        action_score::ActionScore = UpperBoundScore(),
        transition::ConcreteTransition = ConcreteTransition(),
        state_policy::SelectionPolicy = EpsilonGreedy(0.1),
        state_score::StateScore = GreedyScore(),
        terminate = TerminationRule[MaxSteps()],
        gauss_seidel::Bool = false,
        k::Int = 1,
        reverse::Bool = true,
    )
        rules = collect(TerminationRule, terminate)
        gauss_seidel &&
            k < 1 &&
            throw(ArgumentError("k must be positive when gauss_seidel = true, got $k"))
        return new(
            action_policy,
            action_score,
            transition,
            state_policy,
            state_score,
            rules,
            gauss_seidel,
            k,
            reverse,
            Ref{Any}(nothing),
            Ref(0),
            Ref(0),
        )
    end
end

function reset_sampling_strategy!(ss::TrajectorySampling)
    # Discard a partially-consumed trajectory: it was sampled against the
    # previous `solve`'s value function and says nothing about this one.
    ss.buffer[] = nothing
    ss.cursor[] = 0
    # The temperature schedules anneal over one solve, not over the lifetime of
    # the strategy object, so the backup count restarts with it.
    ss.updates[] = 0
    # Backup magnitudes measured in the previous `solve` say nothing about this
    # one's value function.
    _reset_score!(ss.state_score)
    return nothing
end

###################################
# 7. SampleTrajectory              #
###################################

"""
    _step_cap(strategy, model) -> Int

The hard bound on transitions per rollout. A rollout is always bounded, even if
a configuration's own rules never fire — e.g. one carrying only a
[`PredicateStop`](@ref) — so this defaults to
`_trajectory_max_steps(model)`.

It never *shortens* an explicit [`MaxSteps`](@ref), though: that is already a
finite bound the caller chose, and silently clamping `MaxSteps(1000)` to the
state count on a smaller model would ignore what they asked for.
"""
function _step_cap(ss::TrajectorySampling, model)
    cap = _trajectory_max_steps(model)
    for rule in ss.terminate
        rule isa MaxSteps && rule.n !== nothing && (cap = max(cap, rule.n))
    end
    return cap
end

_terminate_pre(ss::TrajectorySampling, s, trajectory, i, vf, model, spec) =
    any(r -> terminate_pre(r, s, trajectory, i, vf, model, spec), ss.terminate)

_terminate_post(ss::TrajectorySampling, s, a, probs, trajectory, i, vf, model, spec) =
    any(r -> terminate_post(r, s, a, probs, trajectory, i, vf, model, spec), ss.terminate)

"""
    _sample_action(ss, s, vf, model, spec, marginal, dir, policy = ss.action_policy) -> Union{CartesianIndex, Nothing}

Score every action available at `s` with `ss.action_score` and draw one under
`policy` — the strategy's action policy with its temperature already resolved
for this rollout (`_resolve_policy`). `nothing` if `s` has no available
actions.
"""
function _sample_action(
    ss::TrajectorySampling,
    s,
    vf,
    model,
    spec,
    marginal,
    dir,
    policy::SelectionPolicy = ss.action_policy,
)
    actions = collect(available(model, s))
    isempty(actions) && return nothing

    U, L = vf.upper.current, vf.lower.current
    scores = [_action_score(ss.action_score, marginal[a, s], U, L, dir) for a in actions]
    return _select(policy, actions, scores)
end

"""
    _sample_state(ss, s, a, probs, vf, model, spec, policy = ss.state_policy) -> Union{CartesianIndex, Nothing}

Draw a successor from the support of `probs` — the linear target indices
carrying positive mass. `nothing` if nothing does, which ends the rollout.

The support is read off `probs` rather than `support(ambiguity_set)`: the
latter returns the *full* target range for a dense `IntervalAmbiguitySets`, so
going by it would offer successors the realized distribution assigns no mass.

`policy` is the strategy's successor policy with its temperature already
resolved for this rollout (`_resolve_policy`).

`EpsilonGreedy` and `Boltzmann` score differently here, deliberately, following
the writeup: ε-greedy exploits on `p(x)·f_greedy(x)` alone, whatever
`ss.state_score` is, while Boltzmann uses the full `f_S`. So the exploration
and gap-weighting terms of an [`ExplorationScore`](@ref) /
[`GapWeightedExplorationScore`](@ref) only take effect under `Boltzmann` — an
ε-greedy configuration reads only the score's `bound`.
"""
function _sample_state(
    ss::TrajectorySampling,
    s,
    a,
    probs,
    vf,
    model,
    spec,
    policy::SelectionPolicy = ss.state_policy,
)
    supp = [i for i in eachindex(probs) if probs[i] > zero(eltype(probs))]
    isempty(supp) && return nothing

    if policy isa EpsilonGreedy
        Vg = vec(_bound_values(_greedy_bound(ss.state_score), vf))
        scores = [Float64(probs[i]) * Float64(Vg[i]) for i in supp]
    else
        scores =
            _state_scores(ss.state_score, probs, supp, vf, s, a, model, spec, ss.transition)
    end

    return _maybe_target_state(model, _select(policy, supp, scores))
end

"""
    _sample_trajectory(ss, model, value_function, spec) -> Vector{<:CartesianIndex}

One rollout, in chronological order. A state is appended at the *top* of each
step, so the state that ends the rollout — a goal state, an obstacle state, an
implicit sink, or one reached when a termination rule fires — is not part of
the result and is never relaxed.
"""
function _sample_trajectory(ss::TrajectorySampling, model, value_function, spec)
    prop = system_property(spec)
    reach_set = Set(_trajectory_reach_states(prop))
    avoid_set = Set(_trajectory_avoid_states(prop))

    marginal = _omax_marginal(model)
    dir = _adversary_direction(ss.transition.adversary, spec)
    Vt = _bound_values(ss.transition.bound, value_function)
    cap = _step_cap(ss, model)

    s = _trajectory_initial_state(model)

    # ComputeTemp, once per rollout: both policies are resolved against the gap
    # at s₀ and the backups issued so far, so the whole trajectory is drawn at
    # one temperature (lines 4-5 of SampleTrajectory).
    ctx = TemperatureContext(
        Float64(value_function.upper.current[s]) - Float64(value_function.lower.current[s]),
        ss.updates[],
    )
    action_policy = _resolve_policy(ss.action_policy, ctx)
    state_policy = _resolve_policy(ss.state_policy, ctx)

    trajectory = typeof(s)[]
    i = 0

    while !(s in reach_set) &&
              !(s in avoid_set) &&
              !_terminate_pre(ss, s, trajectory, i, value_function, model, spec) &&
              i < cap
        push!(trajectory, s)

        a = _sample_action(ss, s, value_function, model, spec, marginal, dir, action_policy)
        a === nothing && break

        probs = _omax_distribution(marginal[a, s], Vt, dir)
        _terminate_post(ss, s, a, probs, trajectory, i, value_function, model, spec) &&
            break

        sp = _sample_state(ss, s, a, probs, value_function, model, spec, state_policy)
        # `nothing` = no successor carries mass under this weighting, so there
        # is none the strategy can honestly pick — end the rollout rather than
        # move to an arbitrary state.
        sp === nothing && break

        s = sp
        # An implicit sink is absorbing and has no strategy-cache entry, so the
        # trajectory both ends here and excludes it — `bellman_v!` and the
        # (source_shape-sized) strategy cache can only index source states.
        _is_source_state(model, s) || break

        i += 1
    end

    return trajectory
end

###################################
# 8. TrajectorySampling (batching) #
###################################

"""
    _count_updates!(ss, model, batch) -> batch

Record the Bellman updates `batch` is about to cause, which is `n` for
[`UpdateDecayTemperature`](@ref). It mirrors GSRDP's own `_bellman_update_count`
exactly — the sampler hands back a `StateIterator`, on which
`project_to_state_sequence` is the identity, and every state in it gets a full
action sweep plus the state backup.

The count is taken *after* the rollout it belongs to, so a rollout's
temperature reads the backups issued before it.
"""
function _count_updates!(ss::TrajectorySampling, model, batch)
    ss.updates[] += length(batch) * (num_actions(model) + 1)
    return batch
end

function sample(ss::TrajectorySampling, model, strategy_cache, value_function, spec)
    # The previous batch has been backed up by now, so its backup magnitudes can
    # be measured before this call's rollout is scored.
    _before_sample!(ss.state_score, value_function)

    if !ss.gauss_seidel
        trajectory = _sample_trajectory(ss, model, value_function, spec)
        # A no-op under GSRDP's Jacobi backup, but applied here too so the
        # flag's meaning doesn't depend on `gauss_seidel`.
        ss.reverse && reverse!(trajectory)
        _count_updates!(ss, model, trajectory)
        _after_sample!(ss.state_score, trajectory, value_function)
        return StateIterator(trajectory)
    end

    buffer = ss.buffer[]
    if buffer === nothing
        buffer = _sample_trajectory(ss, model, value_function, spec)
        ss.reverse && reverse!(buffer)
        # An empty rollout (the initial state was already a goal state) leaves
        # no trajectory to activate; return empty rather than spin.
        if isempty(buffer)
            _after_sample!(ss.state_score, buffer, value_function)
            return StateIterator(buffer)
        end
        ss.buffer[] = buffer
        ss.cursor[] = 1
    end

    start = ss.cursor[]
    stop = min(start + ss.k - 1, length(buffer))
    batch = buffer[start:stop]
    ss.cursor[] = stop + 1
    _count_updates!(ss, model, batch)
    _after_sample!(ss.state_score, batch, value_function)

    if ss.cursor[] > length(buffer)
        ss.buffer[] = nothing   # exhausted: the next call samples afresh
        ss.cursor[] = 0
    end

    return StateIterator(batch)
end

end # module TrajectorySampling
