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
#   × score function  (f_A over actions; f_S over successors, the
#                      exploit/explore mixture with a pluggable f_explore term)
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
# The `initial_state = :random` start; `:max` is `_max_gap_initial_state`.
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
(which `weights` should not contain, but which a caller-supplied value function
or round-off can produce) can neither be selected nor shrink the sampling range.
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
# The configuration vocabulary (`EpsilonGreedy`, `Boltzmann`, `MixtureStateScore`,
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
    system_property,
    isoptimistic,
    _omax_marginal,
    _omax_distribution,
    _omax_expectation,
    _omax_realize!,
    _omax_expectations!,
    _OMaxScratch,
    _omax_scratch,
    _omax_scratch_matches,
    _positive_support,
    _RealizedDistribution,
    _isoptimistic,
    _ismaximize,
    _is_source_state,
    _maybe_target_state,
    _target_state,
    convergence_eps,
    restrict_to_initial,
    _action_uncertainty,
    _categorical_sample,
    _trajectory_reach_states,
    _trajectory_avoid_states,
    _trajectory_initial_state,
    _trajectory_max_steps,
    _to_state_index,
    initial_states,
    AllStates,
    source_shape

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
# `TemperatureContext` that rollout starts from. That is what lets the action
# policy follow the initial gap instead of the caller having to pick one
# compromise temperature for the whole solve.

"""
    TemperatureSchedule

How a [`Boltzmann`](@ref) policy's temperature `T` is obtained at the start of
each rollout: [`FixedTemperature`](@ref) (a constant) or
[`GapBasedTemperature`](@ref) (the odds of the initial gap). The successor
policy only takes the first; see [`IntervalMDP.TrajectorySampling.TrajectorySampling`](@ref).

`_temperature(schedule, ctx)` evaluates one against a
[`TemperatureContext`](@ref), and `_resolve_policy` does that once per rollout,
so a single trajectory is always drawn at one temperature.
"""
abstract type TemperatureSchedule end

"""
    TemperatureContext(diff)

What a [`TemperatureSchedule`](@ref) is evaluated against at the start of a
rollout: `diff`, the initial gap `δ = max_{s ∈ S₀} (U(s) − L(s))` over the
declared initial states `S₀` (every state when the model declares none),
clamped to `[0, 1]` (`_initial_gap`). The clamp keeps the odds `δ / (1 − δ)` of
[`GapBasedTemperature`](@ref) in `[0, ∞]` should round-off push `U − L` below
zero, or a caller-supplied value function carry a gap above one.
"""
struct TemperatureContext
    diff::Float64

    TemperatureContext(diff::Real) = new(clamp(Float64(diff), 0.0, 1.0))
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
    GapBasedTemperature(k, t_min = 0.1)

`T = max(t_min, k·δ / (1 − δ))`: the temperature is `k` times the odds of the
initial gap `δ = max_{s ∈ S₀} Diff(s)` (clamped to `[0, 1]`, see
[`TemperatureContext`](@ref)), floored at `t_min`.

`δ = ½` gives `T = k`. Wide-open initial states heat without bound (`δ → 1`,
`T → ∞`, uniform), and converged ones cool to the floor (`δ → 0`,
`T → t_min`). The odds alone reach `T ≈ k·ε` once the gap is down to `ε`, which
is argmax in all but name; `t_min` is the exploration floor that keeps the
policy from getting there — see the warning on
[`IntervalMDP.TrajectorySampling.TrajectorySampling`](@ref).

`T` is also clamped to the representable positive range, so a rollout at
`δ = 1` draws uniformly, and with `t_min = 0` (the bare odds, no floor) one at
`δ = 0` takes the argmax (uniform among ties).

Requires `k > 0` and `t_min ≥ 0`.
"""
struct GapBasedTemperature <: TemperatureSchedule
    k::Float64
    t_min::Float64

    function GapBasedTemperature(k::Real, t_min::Real = 0.1)
        k > 0 || throw(ArgumentError("k must be positive, got $k"))
        t_min >= 0 || throw(ArgumentError("t_min must be non-negative, got $t_min"))
        return new(Float64(k), Float64(t_min))
    end
end

"""
    _temperature(schedule, ctx) -> Float64

The temperature `schedule` gives a rollout starting in context `ctx` —
`ComputeTemp_A(δ, k, t_min)` for [`GapBasedTemperature`](@ref).
"""
_temperature(schedule::FixedTemperature, ::TemperatureContext) = schedule.t

# `Diff = 1` gives T = ∞, and `Diff = 0` with `t_min = 0` gives T = 0, neither of
# which `FixedTemperature` (nor the log-sum-exp in `_select`) accepts; the clamp
# turns them into uniform and argmax.
_temperature(schedule::GapBasedTemperature, ctx::TemperatureContext) = clamp(
    max(schedule.t_min, schedule.k * ctx.diff / (1 - ctx.diff)),
    floatmin(Float64),
    floatmax(Float64),
)

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
A varying schedule ([`GapBasedTemperature`](@ref)) is resolved to one number at
the top of each rollout, so every draw within one trajectory shares a temperature; see
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
_select(policy::EpsilonGreedy, candidates, scores) = _select_lazy(policy, candidates, () -> scores)

"""
    _select_lazy(policy, candidates, scores_fn)

`_select(policy, candidates, scores_fn())`, but under [`EpsilonGreedy`](@ref) the
scores are only computed when the greedy branch needs them: the exploration draw
comes first and never reads them. The random draws are the same either way.
"""
_select_lazy(policy::SelectionPolicy, candidates, scores_fn) =
    _select(policy, candidates, scores_fn())

function _select_lazy(policy::EpsilonGreedy, candidates, scores_fn)
    rand() < policy.p && return rand(candidates)
    scores = scores_fn()
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

# A varying schedule is not a number until a rollout gives it one, so drawing
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
[`SelectionPolicy`](@ref), which prefers high scores. `U(s, a)` and `L(s, a)` denote
the O-max Q-values of `a` at `s` against the upper and lower bound respectively, both
realized in the same adversary direction as the strategy's [`ConcreteTransition`](@ref).

Scores are oriented to the controller. Under `Maximize` they read the bracket as is.
Under `Minimize` they read its complement, `U'(s, a) = 1 − L(s, a)` and
`L'(s, a) = 1 − U(s, a)` — the bracket of the dual maximization problem — so
"upper" keeps meaning *controller-optimistic*: [`UpperBoundScore`](@ref) picks
`argmin L` for a minimizer, as the termination argument requires (the optimistic
action of a minimizer is its lowest plausible value).
"""
abstract type ActionScore end

"`f_A(s, a) = U(s, a)`, the controller-optimistic Q-value (`1 − L(s, a)` under `Minimize`)."
struct UpperBoundScore <: ActionScore end

"`f_A(s, a) = L(s, a)`, the controller-pessimistic Q-value (`1 − U(s, a)` under `Minimize`)."
struct LowerBoundScore <: ActionScore end

"""
    WeightedAverageScore(beta)

`f_A(s, a) = L(s, a) + β·[U(s, a) − L(s, a)]`, the gap-uncertainty /
weighted-average score (on the complemented bracket under `Minimize`, see
[`ActionScore`](@ref)). `β = 0` is the lower bound, `β = 1` the upper bound,
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

`log(x)` for `x > 0`, and `-Inf` otherwise. Guards [`LogScore`](@ref) and
[`LogStateScore`](@ref) against the two ways an inner score can leave the domain
of `log`: an exactly-zero score (an unreached state under the optimistic bound,
or a successor a mixture term gives no weight), and a negative one (`U − L`
dipping below zero by round-off, or values from a caller-supplied value
function). `log` would give `-Inf` for the first and `NaN` for the second; `NaN`
would then poison `maximum`/`argmax`.
"""
_safe_log(x) = x > 0 ? log(Float64(x)) : -Inf

"""
    _action_score(score, ambiguity_set, U, L, dir; maximize = true) -> Float64

`f_A` for one action, given that action's `ambiguity_set` at the current state.
`dir` is the adversary direction (`true` = O-maximization) that
[`_omax_expectation`](@ref) realizes the Q-values under. `maximize = false` scores a
minimizing controller on the complemented bracket (see [`ActionScore`](@ref)).
"""
_action_score(::UpperBoundScore, as, U, L, dir; maximize::Bool = true) =
    maximize ? _q(as, U, dir) : 1 - _q(as, L, dir)
_action_score(::LowerBoundScore, as, U, L, dir; maximize::Bool = true) =
    maximize ? _q(as, L, dir) : 1 - _q(as, U, dir)

function _action_score(score::WeightedAverageScore, as, U, L, dir; maximize::Bool = true)
    l = _action_score(LowerBoundScore(), as, U, L, dir; maximize)
    u = _action_score(UpperBoundScore(), as, U, L, dir; maximize)
    return l + score.beta * (u - l)
end

_action_score(score::LogScore, as, U, L, dir; maximize::Bool = true) =
    _safe_log(_action_score(score.inner, as, U, L, dir; maximize))

_q(as, V, dir) = Float64(_omax_expectation(as, V, dir))

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
# A branch rather than `Val(bound)`: the bound is a run-time value, and a `Val` built
# from it is a dynamic dispatch on every call.
_bound_values(bound::IntervalMode, vf) =
    bound == Upper ? _bound_values(Val(Upper), vf) : _bound_values(Val(Lower), vf)

_adversary_direction(::Nothing, spec) = _isoptimistic(spec)
_adversary_direction(mode::SatisfactionMode, spec) = isoptimistic(mode)

###################################
# 4. Successor scores f_S          #
###################################

###################################
# 4a. Exploration terms f_explore  #
###################################
#
# The explore half of `MixtureStateScore`. Each term is a function of the
# successor alone. The probability weighting and normalization are applied by
# the score.

"""
    ExplorationTerm

`f_explore(t)`, the per-successor exploration term a [`MixtureStateScore`](@ref)
mixes with its exploit term:

  * [`GapExplore`](@ref): `U(t) − L(t)`, the default
  * [`UpperBoundExplore`](@ref): `U(t)`, the best-case reach probability
  * [`ActionUncertaintyExplore`](@ref): `U^{-a_L}(t) − L(t)`
  * [`BellmanErrorExplore`](@ref): `|U_{k-1}(t) − U_k(t)|` or
    `|L_{k-1}(t) − L_k(t)|`
  * [`MaxBellmanErrorExplore`](@ref): the larger of those two
  * [`VPIExplore`](@ref): `maxₐ VPI(t, a)`, the value of perfect information
    about the Q-values at `t`
"""
abstract type ExplorationTerm end

"`f_explore(t) = U(t) − L(t)`, the value-function gap. The default."
struct GapExplore <: ExplorationTerm end

"""
    UpperBoundExplore()

`f_explore(t) = U(t)`, the best-case reach probability. Paired with a
[`MixtureStateScore`](@ref) whose `exploit` is `Upper`, the explore term equals
the exploit term, so the mixture is the exploit term whatever `α_e` is.
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
successor. It needs the model, so `_state_scores` throws when handed
`nothing` for it.
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
successor per step, and needs the model.
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
        "ActionUncertaintyExplore needs the model; `_state_scores` was given `nothing`",
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
    ArgumentError("VPIExplore needs the model; `_state_scores` was given `nothing`"),
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
[`SelectionPolicy`](@ref). [`MixtureStateScore`](@ref) is built from

    f_exploit(s') = U(s') or L(s'),  per the score's `exploit`
    f_explore(s') = an [`ExplorationTerm`](@ref), by default U(s') − L(s')

weighted by the realized transition probability `p(s'|s,a)`, normalized, and
mixed with weights that can follow the initial gap. [`LogStateScore`](@ref)
wraps it for proportional sampling.

A score implements `_state_scores` and `_exploit_bound`, the bound an
[`EpsilonGreedy`](@ref) successor policy reads instead of the full score.
"""
abstract type StateScore end

###################################
# 4b. Exploit/explore mixture      #
###################################

"""
    AlphaSchedule

`α(δ)`, a mixing weight in `[0, 1]` of a [`MixtureStateScore`](@ref), as a
function of the initial gap `δ = max_{s ∈ S₀} Diff(s)`, clamped to `[0, 1]`
(`S₀` is every state when the model declares no initial states). Resolved once
per rollout, like a [`TemperatureSchedule`](@ref):

  * [`FixedAlpha`](@ref): `α(δ) = α`
  * [`PowerAlpha`](@ref): `α(δ) = δᵏ`
  * [`CustomAlpha`](@ref): `α(δ) = f(δ)`
  * [`ClippedAlpha`](@ref): any of the above, clipped to `[α_min, α_max]`
"""
abstract type AlphaSchedule end

"`α(δ) = α`, the constant. `α ∈ [0, 1]`."
struct FixedAlpha <: AlphaSchedule
    alpha::Float64

    function FixedAlpha(alpha::Real)
        0 <= alpha <= 1 || throw(ArgumentError("alpha must be in [0, 1], got $alpha"))
        return new(Float64(alpha))
    end
end

"""
    PowerAlpha(k)

`α(δ) = δᵏ`: `α = 1` (uniform) while the initial gap is wide open and `α → 0`
as it closes. Larger `k` drops `α` sooner. `k > 0`.
"""
struct PowerAlpha <: AlphaSchedule
    k::Float64

    function PowerAlpha(k::Real)
        k > 0 || throw(ArgumentError("k must be positive, got $k"))
        return new(Float64(k))
    end
end

"""
    CustomAlpha(f)

`α(δ) = f(δ)` for a user mapping `f: [0, 1] → [0, 1]`. The result is clamped to
`[0, 1]`.
"""
struct CustomAlpha{F} <: AlphaSchedule
    f::F
end

"""
    ClippedAlpha(inner, alpha_min, alpha_max)

`α(δ) = clip(α_min, inner(δ), α_max)`: keeps some exploitation (`α_max < 1`)
or some exploration (`α_min > 0`) whatever the gap. `0 ≤ α_min ≤ α_max ≤ 1`.
"""
struct ClippedAlpha{S <: AlphaSchedule} <: AlphaSchedule
    inner::S
    alpha_min::Float64
    alpha_max::Float64

    function ClippedAlpha(inner::S, alpha_min::Real, alpha_max::Real) where {S <: AlphaSchedule}
        0 <= alpha_min <= alpha_max <= 1 || throw(
            ArgumentError(
                "need 0 ≤ alpha_min ≤ alpha_max ≤ 1, got [$alpha_min, $alpha_max]",
            ),
        )
        return new{S}(inner, Float64(alpha_min), Float64(alpha_max))
    end
end

_alpha(schedule::FixedAlpha, delta) = schedule.alpha
_alpha(schedule::PowerAlpha, delta) = Float64(delta)^schedule.k
_alpha(schedule::CustomAlpha, delta) = clamp(Float64(schedule.f(delta)), 0.0, 1.0)
_alpha(schedule::ClippedAlpha, delta) =
    clamp(_alpha(schedule.inner, delta), schedule.alpha_min, schedule.alpha_max)

"""
    MixtureStateScore(; exploit = Upper, explore = GapExplore(),
                        alpha_explore = FixedAlpha(0.5), alpha_prob = FixedAlpha(0.0))

`f_S(s') = (1 − α_e)·f'_exploit(s') + α_e·f'_explore(s')`, a mixture of two
distributions over the support of `p(·|s,a)`:

    f'_exploit(s') = p'(s')·f_exploit(s') / Σ_t p'(t)·f_exploit(t)
    f'_explore(s') = p'(s')·f_explore(s') / Σ_t p'(t)·f_explore(t)
    p'(s')         = (1 − α_p)·p(s') + α_p / |supp(s,a)|

with `f_exploit = U` or `L` per `exploit`, and `f_explore` an
[`ExplorationTerm`](@ref). `α_e` moves the score from exploiting the successors
with the highest expected reach probability (`0`) to exploring (`1`); `α_p`
flattens the transition probabilities towards uniform. Both are
[`AlphaSchedule`](@ref)s (a bare number means [`FixedAlpha`](@ref)), resolved
once per rollout against the initial gap.

Each term is clamped at zero before normalizing: [`ActionUncertaintyExplore`](@ref)
is negative once the lower-bound action dominates every alternative, and `U − L`
can dip below zero by round-off. A term that is zero across the whole support
is taken as uniform, which shifts every score by the same constant and so leaves
the draw to the other term.

Use it wrapped in [`LogStateScore`](@ref) under [`Boltzmann`](@ref): that is the
power rule `P ∝ f_S^(1/T)`, where `T = 1` draws exactly from `f_S`, larger `T`
flattens towards uniform and smaller sharpens towards argmax. Because `f_S` sums
to one, its values scale like `1/|supp|`, and the `1/|supp|` cancels under the
power rule, so `T` means the same at every support size. Unwrapped, `Boltzmann(T)`
is `softmax(f_S / T)`, whose effective temperature is `|supp|·T`: the same `T` is
near-argmax on a small support and near-uniform on a large one.
"""
struct MixtureStateScore{E <: ExplorationTerm, AE <: AlphaSchedule, AP <: AlphaSchedule} <:
       StateScore
    exploit::IntervalMode
    explore::E
    alpha_explore::AE
    alpha_prob::AP

    function MixtureStateScore(;
        exploit::IntervalMode = Upper,
        explore::E = GapExplore(),
        alpha_explore::Union{Real, AlphaSchedule} = FixedAlpha(0.5),
        alpha_prob::Union{Real, AlphaSchedule} = FixedAlpha(0.0),
    ) where {E <: ExplorationTerm}
        ae = alpha_explore isa Real ? FixedAlpha(alpha_explore) : alpha_explore
        ap = alpha_prob isa Real ? FixedAlpha(alpha_prob) : alpha_prob
        return new{E, typeof(ae), typeof(ap)}(exploit, explore, ae, ap)
    end
end

"""
    _exploit_bound(score) -> IntervalMode

The bound `f_exploit` reads, which is all an [`EpsilonGreedy`](@ref) successor
policy uses of `score` (see `_sample_state`).
"""
_exploit_bound(score::MixtureStateScore) = score.exploit

# Lifecycle hooks, forwarded down to the score's `ExplorationTerm` (see
# `BackupTracker`). A score with no term has nothing to track.
_before_sample!(::StateScore, vf) = nothing
_after_sample!(::StateScore, batch, vf) = nothing
_reset_score!(::StateScore) = nothing

_before_sample!(score::MixtureStateScore, vf) = _before_sample!(score.explore, vf)
_after_sample!(score::MixtureStateScore, batch, vf) = _after_sample!(score.explore, batch, vf)
_reset_score!(score::MixtureStateScore) = _reset_score!(score.explore)

# `w` as a distribution: clamped at zero, then normalized; uniform if nothing is
# left, which shifts every score of a mixture by the same constant.
function _normalize_weights(w::Vector{Float64})
    w = max.(w, 0.0)
    total = sum(w)
    return total > 0 ? w ./ total : fill(1 / length(w), length(w))
end

_resolved_alpha(schedule::FixedAlpha) = schedule.alpha
_resolved_alpha(::AlphaSchedule) = throw(
    ArgumentError(
        "a MixtureStateScore with a non-fixed AlphaSchedule must be resolved against " *
        "the initial gap first (`_resolve_score`)",
    ),
)

"""
    _state_scores(score, probs, supp, vf, s, a, model, spec, transition) -> Vector{Float64}

`f_S` for every successor in `supp` (linear target indices carrying positive
probability), parallel to `supp`, given the step context: the current state `s`,
the rollout's action `a`, and the strategy's [`ConcreteTransition`](@ref). An
[`ExplorationTerm`](@ref) that reads only the value function accepts `nothing`
for `model` and `spec`.
"""
function _state_scores(
    score::MixtureStateScore,
    probs,
    supp,
    vf,
    s,
    a,
    model,
    spec,
    transition,
)
    alpha_e = _resolved_alpha(score.alpha_explore)
    alpha_p = _resolved_alpha(score.alpha_prob)

    Vg = vec(_bound_values(score.exploit, vf))
    x = _explore_values(score.explore, supp, vf, model, spec)

    m = length(supp)
    pp = [(1 - alpha_p) * Float64(probs[i]) + alpha_p / m for i in supp]
    exploit = _normalize_weights([pp[n] * Float64(Vg[i]) for (n, i) in enumerate(supp)])
    explore = _normalize_weights([pp[n] * Float64(x[n]) for n in 1:m])

    return (1 - alpha_e) .* exploit .+ alpha_e .* explore
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

_exploit_bound(score::LogStateScore) = _exploit_bound(score.inner)

_before_sample!(score::LogStateScore, vf) = _before_sample!(score.inner, vf)
_after_sample!(score::LogStateScore, batch, vf) = _after_sample!(score.inner, batch, vf)
_reset_score!(score::LogStateScore) = _reset_score!(score.inner)

_state_scores(score::LogStateScore, probs, supp, vf, s, a, model, spec, transition) =
    _safe_log.(_state_scores(score.inner, probs, supp, vf, s, a, model, spec, transition))

"""
    _initial_gap(model, vf) -> Float64

`δ = max_{s ∈ S₀} (U(s) − L(s))`, clamped to `[0, 1]`, over the declared initial
states, or over every state when the model declares none (`AllStates`).
"""
function _initial_gap(model, vf)
    init = initial_states(model)
    states = init isa AllStates ? CartesianIndices(source_shape(model)) :
             (_to_state_index(s) for s in init)
    U, L = vf.upper.current, vf.lower.current
    delta = maximum(Float64(U[s]) - Float64(L[s]) for s in states)
    return clamp(delta, 0.0, 1.0)
end

"""
    _max_gap_initial_state(model, vf) -> CartesianIndex

The state of `S₀` (every state when the model declares none) with the widest gap
`U(s) − L(s)`, ties broken uniformly at random. The `initial_state = :max` start.
"""
function _max_gap_initial_state(model, vf)
    init = initial_states(model)
    states = init isa AllStates ? CartesianIndices(source_shape(model)) :
             (_to_state_index(s) for s in init)
    U, L = vf.upper.current, vf.lower.current
    best = first(states)
    best_gap, ties = -Inf, 0
    for s in states
        gap = Float64(U[s]) - Float64(L[s])
        if gap > best_gap
            best, best_gap, ties = s, gap, 1
        elseif gap == best_gap
            # Reservoir sampling: the i-th tie replaces the pick with probability
            # 1/i, so each of the tied states ends up the pick with equal odds.
            ties += 1
            rand(1:ties) == 1 && (best = s)
        end
    end
    return best
end

"""
    _resolve_score(score, delta) -> StateScore

`score` with its [`AlphaSchedule`](@ref)s evaluated at the initial gap `delta`
(`_initial_gap`, the same `δ` as the rollout's [`TemperatureContext`](@ref)),
once per rollout, so every step of the rollout mixes with the same weights. The
resolved score shares `score`'s exploration term, so the lifecycle hooks, which
run on the strategy's own score, still reach it.
"""
_resolve_score(score::StateScore, delta) = score

_resolve_score(score::LogStateScore, delta) = LogStateScore(_resolve_score(score.inner, delta))

_resolve_score(score::MixtureStateScore{E, FixedAlpha, FixedAlpha}, delta) where {E} = score

function _resolve_score(score::MixtureStateScore, delta)
    return MixtureStateScore(;
        exploit = score.exploit,
        explore = score.explore,
        alpha_explore = _alpha(score.alpha_explore, delta),
        alpha_prob = _alpha(score.alpha_prob, delta),
    )
end

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
    U, L = value_function.upper.current, value_function.lower.current
    diff = U[s] - L[s]
    iszero(diff) && return true

    # Summed over the support rather than via a full `U .- L` vector: this runs at
    # every step of every rollout, and the allocation was O(|S|) per step.
    return _expected_gap(probs, vec(U), vec(L)) < diff / rule.tau
end

# `B = Σ p·(U − L)` over the positive entries of `probs`, in ascending index order.
function _expected_gap(probs::AbstractVector, Uv, Lv)
    B = zero(Float64)
    for j in eachindex(probs)
        p = probs[j]
        p > 0 && (B += Float64(p) * (Float64(Uv[j]) - Float64(Lv[j])))
    end
    return B
end

# The same sum over the indices where a realized distribution can be nonzero, which
# are ascending, so the terms are added in the same order.
function _expected_gap(probs::_RealizedDistribution, Uv, Lv)
    B = zero(Float64)
    for j in probs.candidates
        p = probs[j]
        p > 0 && (B += Float64(p) * (Float64(Uv[j]) - Float64(Lv[j])))
    end
    return B
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
  drawn given `action_score`. A `Boltzmann` policy carries a
  [`FixedTemperature`](@ref) or a [`GapBasedTemperature`](@ref) — see below.
- `action_score::ActionScore = UpperBoundScore()`: `f_A(s, a)`.
- `transition::ConcreteTransition = ConcreteTransition()`: which bound and
  adversary direction realize `p(·|s,a)`.
- `state_policy::SelectionPolicy = EpsilonGreedy(0.1)`: how the successor is
  drawn. A `Boltzmann` policy carries a [`FixedTemperature`](@ref) only.
  **Note the asymmetry** documented under `_sample_state`: `EpsilonGreedy`
  always exploits on `p(s')·f_exploit(s')` and ignores `state_score`'s
  exploration term; only `Boltzmann` honours the full `f_S`.
- `state_score::StateScore = LogStateScore(MixtureStateScore())`: `f_S(s', s, a)`,
  by default the power rule over the exploit/explore mixture. A
  [`MixtureStateScore`](@ref)'s weights `α_e`, `α_p` are resolved once per
  rollout against the initial gap, like a temperature (below).
- `exclude_visited::Bool = false`: never move to a state this rollout has
  already visited. See "Successor filters" below.
- `exclude_converged::Bool = false`: never move to a successor whose gap
  `U(s') − L(s')` is below `converged_eps`. See "Successor filters" below.
- `converged_eps = nothing`: the threshold for `exclude_converged`; `nothing`
  means the property's `convergence_eps`, the tolerance GSRDP itself terminates
  on. Only allowed with `exclude_converged = true`.
- `dedup::Bool = false`: drop repeated states from each batch (the whole
  trajectory when `gauss_seidel = false`), keeping the first occurrence. Every
  backup in a batch reads the previous iterate, so a second backup of the same
  state in one batch recomputes the same value and only costs a Bellman update.
  Unlike `exclude_visited` it does not change where rollouts go.
- `dedup_scope::Symbol = :batch`: what `dedup` removes repeats from. `:batch`
  (each Gauss-Seidel batch, as above); `:rollout` (the whole rollout, before it
  is doled out, keeping each state's first occurrence in the returned order --
  the goal-end one when `reverse`; with `k = 1` that is Gauss-Seidel over the
  rollout's distinct states); `:consecutive` (only runs of the same state, i.e.
  self-loop steps, so states revisited through a cycle are backed up again).
- `terminate = [MaxSteps()]`: extra [`TerminationRule`](@ref)s, ORed together.
- `gauss_seidel::Bool = false`: see below.
- `k::Int = 1`: batch size, used only when `gauss_seidel = true`.
- `reverse::Bool = true`: return each trajectory goal-first.
- `initial_state::Symbol = :random`: where each rollout starts. `:random` draws
  uniformly from the declared initial states `S₀` (every state when the model
  declares none); `:max` takes the state of `S₀` with the widest gap
  `U(s) − L(s)`, breaking ties uniformly at random.

# Temperature schedules

A `Boltzmann` policy's temperature is resolved once per rollout, before the
first action is selected, and held for the whole trajectory. The action policy
takes either schedule; the successor policy only a fixed one, since its
exploration follows the initial gap through the mixture weights instead:

* [`FixedTemperature`](@ref) — `T = t`, the constant (`T_s`, and `T_A` under the
  fixed schedule).
* [`GapBasedTemperature`](@ref) — `T = max(t_min, k·δ / (1 − δ))`, the odds of
  the initial gap `δ = max_{s ∈ S₀} Diff(s)` over the declared initial states
  (every state when the model declares none), scaled by `k` and floored at
  `t_min`. This is `ComputeTemp_A`, and the same `δ` a
  [`MixtureStateScore`](@ref)'s weights follow.

In Gauss-Seidel mode the temperature belongs to the rollout, not the batch: a
trajectory doled out over several iterations keeps the temperature it was
sampled at.

!!! warning
    Keep [`GapBasedTemperature`](@ref)'s `t_min` well away from zero. Its odds
    alone reach `T ≈ k·ε` once the initial gap is down to `ε`, and a cold
    Boltzmann policy is argmax in all but name, so it runs into exactly the
    hazard described under "Why the default ε is not zero" below: the rollout
    locks onto the states it already prefers, the rest keep their initial
    values, and the gap criterion never fires. `t_min` is the exploration floor
    the schedule never cools past; `t_min = 0` removes it.

# Successor filters

`exclude_visited` and `exclude_converged` remove successors from the support of
`p(·|s,a)` before anything is scored, so they hold under every policy and score:
ε-greedy's argmax and its uniform branch, the softmax and the power rule alike,
with a [`MixtureStateScore`](@ref) renormalizing over the successors that remain.
The transition is still realized over the full ambiguity set, and the Bellman
backups still use the full model, so neither filter affects correctness, only
which states the rollouts reach.

* `exclude_visited` keeps the states of the current rollout, the one being
  stood on included, and drops them from every later draw of that rollout, so a
  trajectory never revisits a state (no self-loops, no cycles). The set starts
  empty for each rollout.
* `exclude_converged` drops a successor `s'` with `U(s') − L(s') < converged_eps`.
  Terminal states are exempt: goal and avoid states and implicit sinks have a
  zero gap, but stepping onto one is how a rollout ends, so it still can.

A rollout whose successors are all filtered out ends at the current state, as
when no successor carries mass.

!!! warning
    Pair `exclude_converged` with a property that sets `restrict_to_initial`
    (and a model that declares its initial states); `sample` warns once per
    solve otherwise.
    A state's gap can fall below `converged_eps` while one of its successors'
    gaps is still above it (that successor carries little probability), and
    once every route to an open state passes through such a converged state, no
    rollout reaches it again. Converging on the initial states alone is
    unaffected, but GSRDP's default criterion asks every state's gap to close,
    and those cut-off states never do: on `linear_stochastically_switched_S10`
    and `car_parking_S10_I3` the solve stalls with a handful of open states that
    no rollout can reach.

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
    exclude_visited::Bool
    exclude_converged::Bool
    converged_eps::Union{Nothing, Float64}
    dedup::Bool
    dedup_scope::Symbol
    initial_state::Symbol

    # Active-trajectory batching state. `Base.RefValue{Any}` because the
    # `CartesianIndex{N}` arity isn't known until there is a model in hand —
    # the same reason `RNDPriorityQueueSampling` holds its networks that way.
    # `nothing` means "no active trajectory".
    buffer::Base.RefValue{Any}
    cursor::Base.RefValue{Int}

    # Whether this solve has already warned about `exclude_converged` without
    # `restrict_to_initial` (`_warn_converged_scope!`).
    warned_scope::Base.RefValue{Bool}
    # The `_OMaxScratch` rollouts realize their transition distributions into
    # (`_rollout_scratch`), or `nothing` before the first rollout of a solve.
    omax_scratch::Base.RefValue{Any}

    function TrajectorySampling(;
        action_policy::SelectionPolicy = EpsilonGreedy(0.1),
        action_score::ActionScore = UpperBoundScore(),
        transition::ConcreteTransition = ConcreteTransition(),
        state_policy::SelectionPolicy = EpsilonGreedy(0.1),
        state_score::StateScore = LogStateScore(MixtureStateScore()),
        terminate = TerminationRule[MaxSteps()],
        gauss_seidel::Bool = false,
        k::Int = 1,
        reverse::Bool = true,
        exclude_visited::Bool = false,
        exclude_converged::Bool = false,
        converged_eps::Union{Nothing, Real} = nothing,
        dedup::Bool = false,
        dedup_scope::Symbol = :batch,
        initial_state::Symbol = :random,
    )
        rules = collect(TerminationRule, terminate)
        _check_schedule(action_policy, "action_policy", (FixedTemperature, GapBasedTemperature))
        _check_schedule(state_policy, "state_policy", (FixedTemperature,))
        dedup_scope in (:batch, :rollout, :consecutive) || throw(
            ArgumentError("dedup_scope must be :batch, :rollout or :consecutive, got :$dedup_scope"),
        )
        dedup_scope === :batch || dedup || throw(
            ArgumentError("dedup_scope = :$dedup_scope only applies with dedup = true"),
        )
        initial_state in (:random, :max) || throw(
            ArgumentError("initial_state must be :random or :max, got :$initial_state"),
        )
        gauss_seidel &&
            k < 1 &&
            throw(ArgumentError("k must be positive when gauss_seidel = true, got $k"))
        if converged_eps !== nothing
            exclude_converged || throw(
                ArgumentError("converged_eps only applies with exclude_converged = true"),
            )
            converged_eps >= 0 || throw(
                ArgumentError("converged_eps must be non-negative, got $converged_eps"),
            )
        end
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
            exclude_visited,
            exclude_converged,
            converged_eps === nothing ? nothing : Float64(converged_eps),
            dedup,
            dedup_scope,
            initial_state,
            Ref{Any}(nothing),
            Ref(0),
            Ref(false),
            Ref{Any}(nothing),
        )
    end
end

# Which `TemperatureSchedule`s a policy keyword accepts; anything but `Boltzmann`
# carries none.
_check_schedule(::SelectionPolicy, name, allowed) = nothing

function _check_schedule(policy::Boltzmann, name, allowed)
    any(T -> policy.T isa T, allowed) && return nothing
    throw(
        ArgumentError(
            "$name: a Boltzmann policy here takes a " *
            join(nameof.(allowed), " or ") * ", got $(typeof(policy.T))",
        ),
    )
end

function reset_sampling_strategy!(ss::TrajectorySampling)
    # Discard a partially-consumed trajectory: it was sampled against the
    # previous `solve`'s value function and says nothing about this one.
    ss.buffer[] = nothing
    ss.cursor[] = 0
    ss.warned_scope[] = false
    ss.omax_scratch[] = nothing
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

# `ss.terminate` holds abstractly typed rules, so the built-in ones are split out by
# hand to avoid a dynamic dispatch per rule and step. Same rules, same order, same
# short-circuit as `any`.
function _terminate_pre(ss::TrajectorySampling, s, trajectory, i, vf, model, spec)
    for r in ss.terminate
        hit = if r isa MaxSteps
            terminate_pre(r, s, trajectory, i, vf, model, spec)
        elseif r isa ExpectedGapStop
            terminate_pre(r, s, trajectory, i, vf, model, spec)
        else
            terminate_pre(r, s, trajectory, i, vf, model, spec)
        end
        hit && return true
    end
    return false
end

function _terminate_post(ss::TrajectorySampling, s, a, probs, trajectory, i, vf, model, spec)
    for r in ss.terminate
        hit = if r isa MaxSteps
            terminate_post(r, s, a, probs, trajectory, i, vf, model, spec)
        elseif r isa ExpectedGapStop
            terminate_post(r, s, a, probs, trajectory, i, vf, model, spec)
        else
            terminate_post(r, s, a, probs, trajectory, i, vf, model, spec)
        end
        hit && return true
    end
    return false
end

"""
    _sample_action(ss, s, vf, model, spec, marginal, dir, policy = ss.action_policy, score = ss.action_score, scratch = nothing) -> Union{CartesianIndex, Nothing}

Score every action available at `s` with `score` (the strategy's `action_score`), oriented to the
specification's strategy mode (see [`ActionScore`](@ref)), and draw one under
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
    score::ActionScore = ss.action_score,
    scratch = nothing,
)
    actions = collect(available(model, s))
    isempty(actions) && return nothing

    U, L = vf.upper.current, vf.lower.current
    maximize = _ismaximize(spec)
    scores_fn = () -> _action_scores(score, scratch, marginal, s, actions, U, L, dir, maximize)
    return _select_lazy(policy, actions, scores_fn)
end

"""
    _action_scores(score, scratch, marginal, s, actions, U, L, dir, maximize) -> Vector{Float64}

`_action_score` for every action at `s`. For the plain bound scores, with a rollout
scratch and more than one action, the Q-values come from one shared sort
(`_omax_expectations!`); the values are the same.
"""
_action_scores(score, scratch, marginal, s, actions, U, L, dir, maximize) =
    [_action_score(score, marginal[a, s], U, L, dir; maximize) for a in actions]

function _action_scores(
    score::Union{UpperBoundScore, LowerBoundScore},
    scratch::_OMaxScratch,
    marginal,
    s,
    actions,
    U,
    L,
    dir,
    maximize,
)
    length(actions) > 1 ||
        return [_action_score(score, marginal[a, s], U, L, dir; maximize) for a in actions]
    # UpperBoundScore: maximize ? _q(U) : 1 - _q(L); LowerBoundScore the other way.
    V = (score isa UpperBoundScore) == maximize ? U : L
    q = _omax_expectations!(Vector{eltype(V)}(undef, length(actions)), scratch, marginal, s, actions, V, dir)
    return maximize ? Float64[Float64(x) for x in q] : Float64[1 - Float64(x) for x in q]
end

"""
    _filter_successors!(supp, model, vf, visited, converged_eps, terminal = nothing) -> supp

Drop from `supp` (linear target indices) every successor whose gap
`U − L` is below `converged_eps`, and every one in `visited`, a set of state
indices. `nothing` turns either filter off.

Terminal successors are exempt from the convergence filter: the states in
`terminal` (the property's reach and avoid states) and the implicit sinks. Their
gap is zero, but drawing one is how a rollout ends, so dropping them would leave
every rollout running to its step cap.
"""
function _filter_successors!(supp, model, vf, visited, converged_eps, terminal = nothing)
    if converged_eps !== nothing
        Uv, Lv = vec(vf.upper.current), vec(vf.lower.current)
        filter!(supp) do i
            Float64(Uv[i]) - Float64(Lv[i]) >= converged_eps && return true
            t = _target_state(model, i)
            return (terminal !== nothing && t in terminal) || !_is_source_state(model, t)
        end
    end
    if visited !== nothing
        filter!(i -> !(_target_state(model, i) in visited), supp)
    end
    return supp
end

"""
    _sample_state(ss, s, a, probs, vf, model, spec, policy = ss.state_policy, score = ss.state_score;
                  visited = nothing, converged_eps = nothing, terminal = nothing) -> Union{CartesianIndex, Nothing}

Draw a successor from the support of `probs` — the linear target indices
carrying positive mass. `nothing` if nothing does, which ends the rollout.

The support is read off `probs` rather than `support(ambiguity_set)`: the
latter returns the *full* target range for a dense `IntervalAmbiguitySets`, so
going by it would offer successors the realized distribution assigns no mass.

`policy` and `score` are the strategy's successor policy and score, resolved
for this rollout (`_resolve_policy`, `_resolve_score`). `visited` and
`converged_eps` are the successor filters, applied before scoring
(`_filter_successors!`); `nothing` turns each off. `terminal`, the reach and
avoid states, are exempt from the convergence filter, as are implicit sinks.

`EpsilonGreedy` and `Boltzmann` score differently here, deliberately, following
the writeup: ε-greedy exploits on `p(x)·f_exploit(x)` alone, whatever
`ss.state_score` is, while Boltzmann uses the full `f_S`. So the exploration
term and the mixture weights of a [`MixtureStateScore`](@ref) only take effect
under `Boltzmann` — an ε-greedy configuration reads only the score's `exploit`
bound (`_exploit_bound`).
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
    score::StateScore = ss.state_score;
    visited = nothing,
    converged_eps = nothing,
    terminal = nothing,
)
    supp = _positive_support(probs)
    _filter_successors!(supp, model, vf, visited, converged_eps, terminal)
    isempty(supp) && return nothing

    if policy isa EpsilonGreedy
        Vg = vec(_bound_values(_exploit_bound(score), vf))
        scores = [Float64(probs[i]) * Float64(Vg[i]) for i in supp]
    else
        scores = _state_scores(score, probs, supp, vf, s, a, model, spec, ss.transition)
    end

    return _maybe_target_state(model, _select(policy, supp, scores))
end

"""
    _rollout_context(ss, model, value_function) -> TemperatureContext

What a rollout's schedules are resolved against: the initial gap over all of
`S₀`, whichever initial state the rollout happens to start from.
"""
_rollout_context(ss::TrajectorySampling, model, value_function) =
    TemperatureContext(_initial_gap(model, value_function))

"""
    _sample_trajectory(ss, model, value_function, spec) -> Vector{<:CartesianIndex}

One rollout, in chronological order. A state is appended at the *top* of each
step, so the state that ends the rollout — a goal state, an obstacle state, an
implicit sink, or one reached when a termination rule fires — is not part of
the result and is never relaxed.
"""
_sample_trajectory(ss::TrajectorySampling, model, value_function, spec) =
    _sample_trajectory(ss, model, value_function, spec, _rollout_scratch(ss, model))

"""
    _rollout_scratch(ss, model) -> Union{_OMaxScratch, Nothing}

The scratch the rollout realizes each step's transition distribution into
(`_omax_realize!`), kept on the strategy so its per-column budget cache lasts the
solve. `nothing` for models without sparse interval ambiguity sets.
"""
function _rollout_scratch(ss::TrajectorySampling, model)
    marginal = _omax_marginal(model)
    scratch = ss.omax_scratch[]
    _omax_scratch_matches(scratch, marginal) && return scratch
    scratch = _omax_scratch(marginal)
    ss.omax_scratch[] = scratch
    return scratch
end

function _sample_trajectory(ss::TrajectorySampling, model, value_function, spec, scratch)
    prop = system_property(spec)
    reach_set = Set(_trajectory_reach_states(prop))
    avoid_set = Set(_trajectory_avoid_states(prop))

    marginal = _omax_marginal(model)
    dir = _adversary_direction(ss.transition.adversary, spec)
    Vt = _bound_values(ss.transition.bound, value_function)
    cap = _step_cap(ss, model)

    s = ss.initial_state === :max ? _max_gap_initial_state(model, value_function) :
        _trajectory_initial_state(model)

    # ComputeTemp_A, once per rollout: both policies are resolved against the
    # initial gap δ = max_{s ∈ S₀} Diff(s), so the whole trajectory is drawn at
    # one temperature (SampleTrajectory). The mixture weights α_e, α_p follow the same δ; ε-greedy
    # reads only the score's bound, so it skips that resolve.
    ctx = _rollout_context(ss, model, value_function)
    action_policy = _resolve_policy(ss.action_policy, ctx)
    state_policy = _resolve_policy(ss.state_policy, ctx)
    state_score =
        state_policy isa EpsilonGreedy ? ss.state_score : _resolve_score(ss.state_score, ctx.diff)

    trajectory = typeof(s)[]
    # The successor filters, fixed for the rollout.
    visited = ss.exclude_visited ? Set{typeof(s)}() : nothing
    converged_eps =
        ss.exclude_converged ? something(ss.converged_eps, Float64(convergence_eps(prop))) :
        nothing
    terminal = ss.exclude_converged ? union(reach_set, avoid_set) : nothing

    # A function barrier: the resolved policies and scores are only known at run
    # time, so the loop is compiled for their concrete types.
    return _rollout!(
        trajectory,
        s,
        ss,
        model,
        value_function,
        spec,
        scratch,
        reach_set,
        avoid_set,
        marginal,
        dir,
        Vt,
        cap,
        ss.action_score,
        action_policy,
        state_policy,
        state_score,
        visited,
        converged_eps,
        terminal,
    )
end

function _rollout!(
    trajectory,
    s,
    ss,
    model,
    value_function,
    spec,
    scratch,
    reach_set,
    avoid_set,
    marginal,
    dir,
    Vt,
    cap,
    action_score,
    action_policy,
    state_policy,
    state_score,
    visited,
    converged_eps,
    terminal,
)
    i = 0

    while !(s in reach_set) &&
              !(s in avoid_set) &&
              !_terminate_pre(ss, s, trajectory, i, value_function, model, spec) &&
              i < cap
        push!(trajectory, s)
        visited === nothing || push!(visited, s)

        a = _sample_action(
            ss,
            s,
            value_function,
            model,
            spec,
            marginal,
            dir,
            action_policy,
            action_score,
            scratch,
        )
        a === nothing && break

        probs = _omax_realize!(scratch, marginal[a, s], Vt, dir)
        _terminate_post(ss, s, a, probs, trajectory, i, value_function, model, spec) &&
            break

        sp = _sample_state(
            ss,
            s,
            a,
            probs,
            value_function,
            model,
            spec,
            state_policy,
            state_score;
            visited,
            converged_eps,
            terminal,
        )
        # `nothing` = no successor carries mass under this weighting, or the
        # filters left none, so there is none the strategy can honestly pick —
        # end the rollout rather than move to an arbitrary state.
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
    _warn_converged_scope!(ss, model, spec)

Warn, once per solve, when `exclude_converged` is on but the solve converges on
every state: the property does not set `restrict_to_initial`, or the model
declares no initial states, so the restriction has nothing to restrict to.
Trajectory sampling is designed for convergence on the initial states, and
there a converged state can cut rollouts off from open states the criterion
still waits on (see the warning on [`TrajectorySampling`](@ref)).
"""
function _warn_converged_scope!(ss::TrajectorySampling, model, spec)
    (ss.exclude_converged && !ss.warned_scope[]) || return nothing
    prop = system_property(spec)
    restrict_to_initial(prop) && !(initial_states(model) isa AllStates) && return nothing
    ss.warned_scope[] = true
    @warn "TrajectorySampling: exclude_converged = true, but the solve converges on every " *
          "state (the property does not set restrict_to_initial, or the model declares " *
          "no initial states). A converged state can cut rollouts off from open states, " *
          "and the solve can stall. Trajectory sampling is designed for " *
          "restrict_to_initial = true."
    return nothing
end

# Remove repeats from a whole rollout per `dedup_scope` (`:batch` leaves it to `sample`).
function _dedup_rollout!(ss::TrajectorySampling, trajectory)
    ss.dedup || return trajectory
    if ss.dedup_scope === :rollout
        unique!(trajectory)
    elseif ss.dedup_scope === :consecutive
        n = length(trajectory)
        n <= 1 && return trajectory
        j = 1
        @inbounds for i in 2:n
            if trajectory[i] != trajectory[j]
                j += 1
                trajectory[j] = trajectory[i]
            end
        end
        resize!(trajectory, j)
    end
    return trajectory
end

function sample(ss::TrajectorySampling, model, strategy_cache, value_function, spec)
    _warn_converged_scope!(ss, model, spec)

    # The previous batch has been backed up by now, so its backup magnitudes can
    # be measured before this call's rollout is scored.
    _before_sample!(ss.state_score, value_function)

    if !ss.gauss_seidel
        trajectory = _sample_trajectory(ss, model, value_function, spec)
        # A no-op under GSRDP's Jacobi backup, but applied here too so the
        # flag's meaning doesn't depend on `gauss_seidel`.
        ss.reverse && reverse!(trajectory)
        _dedup_rollout!(ss, trajectory)
        ss.dedup && ss.dedup_scope === :batch && unique!(trajectory)
        _after_sample!(ss.state_score, trajectory, value_function)
        return StateIterator(trajectory)
    end

    buffer = ss.buffer[]
    if buffer === nothing
        buffer = _sample_trajectory(ss, model, value_function, spec)
        ss.reverse && reverse!(buffer)
        _dedup_rollout!(ss, buffer)
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
    ss.dedup && ss.dedup_scope === :batch && unique!(batch)
    ss.cursor[] = stop + 1
    _after_sample!(ss.state_score, batch, value_function)

    if ss.cursor[] > length(buffer)
        ss.buffer[] = nothing   # exhausted: the next call samples afresh
        ss.cursor[] = 0
    end

    return StateIterator(batch)
end

end # module TrajectorySampling
