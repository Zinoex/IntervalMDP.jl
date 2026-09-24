"""
    GeneralizedSamplingbasedRobustDynamicProgramming(bellman_alg; sampling_strategy, term_criteria)

Generalized sampling-based robust dynamic programming. Drives an
`IntervalValueFunction` — i.e. simultaneous lower and upper
bounds on the value — and terminates when the gap `V_upper - V_lower`
falls below `convergence_eps(prop)`.

Restrictions:

* Only infinite-horizon (convergence-based) properties are accepted.
  Finite-horizon properties have a fixed-iteration termination criterion
  that's incompatible with gap-based convergence.

The optimal action at each visited state is picked from the upper
bound. The chosen action is then applied to the lower bound through 
a `NonOptimizingStrategyCache` so both bounds track the same policy.

`sampling_strategy` controls which states (or `(a, s)` pairs) are
relaxed each iteration; defaults to [`ExhaustiveState`](@ref). State-action
samplers are projected to their unique state set via
`project_to_state_sequence` — visited states get a full action
sweep, unvisited states retain `V_prev`.

`term_criteria` overrides the default termination criterion derived from
the property. When omitted, the algorithm uses the gap-based criterion
from `convergence_eps(prop)`.

`solve` accepts a `callback` keyword, invoked once per iteration (plus once
before any update, with a count of zero) at either of two arities:

    callback(value_function::IntervalValueFunction, bellman_updates::Int)
    callback(value_function::IntervalValueFunction, bellman_updates::Int, state_seq)

The three-argument form additionally receives the states relaxed in that
iteration — exactly the sequence the Bellman update swept, duplicates
included — and `nothing` on the pre-update fire. See `_invoke_callback`.
"""
struct GeneralizedSamplingbasedRobustDynamicProgramming{B <: BellmanAlgorithm} <:
       ModelCheckingAlgorithm
    bellman_alg::B
    sampling_strategy::SamplingStrategy

    function GeneralizedSamplingbasedRobustDynamicProgramming(
        bellman_alg::B;
        sampling_strategy::Union{Nothing, SamplingStrategy} = nothing,
    ) where {B <: BellmanAlgorithm}
        new{B}(
            bellman_alg,
            isnothing(sampling_strategy) ? ExhaustiveState() : sampling_strategy,
        )
    end
end

bellman_algorithm(alg::GeneralizedSamplingbasedRobustDynamicProgramming) = alg.bellman_alg
sampling_strategy(alg::GeneralizedSamplingbasedRobustDynamicProgramming) =
    alg.sampling_strategy

construct_value_function(::GeneralizedSamplingbasedRobustDynamicProgramming, problem) =
    IntervalValueFunction(
        StateValueFunction(problem, Lower),
        StateValueFunction(problem, Upper),
    )

"""
    GapTerminationCriteria(tol)

Terminates when `maximum(gap) < tol` over the elementwise gap
`V_upper - V_lower`. Used by
[`GeneralizedSamplingbasedRobustDynamicProgramming`](@ref).
"""
struct GapTerminationCriteria{T <: Real} <: TerminationCriteria
    tol::T
end
# `gap_residual` is the signed `upper - lower` bracket width, which `gap` has already
# checked to be non-negative, so no `abs` is needed (and taking one would mask an
# inverted bracket as a small gap).
(f::GapTerminationCriteria)(_, _, gap_residual) = maximum(gap_residual) < f.tol

function termination_criteria(
    ::GeneralizedSamplingbasedRobustDynamicProgramming,
    spec::Specification,
    mp,
)
    prop = system_property(spec)
    if isfinitetime(prop)
        throw(
            ArgumentError(
                "GeneralizedSamplingbasedRobustDynamicProgramming requires an " *
                "infinite-horizon property; got $(typeof(prop)).",
            ),
        )
    end
    return apply_initial_restriction(
        GapTerminationCriteria(convergence_eps(prop)),
        prop,
        mp,
    )
end

function solve(
    problem::VerificationProblem,
    alg::GeneralizedSamplingbasedRobustDynamicProgramming;
    kwargs...,
)
    V, k, res, _ = _gsrdp!(problem, alg; kwargs...)
    return VerificationSolution(V, res, k)
end

# Sampling dispatcher for GSRDP.
# Dispatches on `sampling_context_requirement(ss)` (a trait, since the
# sampling-strategy category hierarchy is flat — see `sampling.jl`) to call
# either the richest `sample(ss, mp, strategy_cache, value_function, spec)`
# signature, or the plain `sample(ss, mp, strategy_cache)`.
function _gsrdp_sample(ss::SamplingStrategy, mp, strategy_cache, value_function, spec)
    return _gsrdp_sample(
        sampling_context_requirement(ss),
        ss,
        mp,
        strategy_cache,
        value_function,
        spec,
    )
end

_gsrdp_sample(::NeedsModelOnly, ss, mp, strategy_cache, value_function, spec) =
    sample(ss, mp, strategy_cache)

_gsrdp_sample(::NeedsValueFunctionAndSpec, ss, mp, strategy_cache, value_function, spec) =
    sample(ss, mp, strategy_cache, value_function, spec)

function solve(
    problem::ControlSynthesisProblem,
    alg::GeneralizedSamplingbasedRobustDynamicProgramming;
    kwargs...,
)
    V, k, res, strategy_cache = _gsrdp!(problem, alg; kwargs...)
    strategy = cachetostrategy(strategy_cache)

    return ControlSynthesisSolution(strategy, V, res, k)
end

function _gsrdp!(
    problem::AbstractIntervalMDPProblem,
    alg::GeneralizedSamplingbasedRobustDynamicProgramming;
    callback = nothing,
)
    mp = system(problem)
    spec = specification(problem)
    prop = system_property(spec)

    workspace = construct_workspace(mp, bellman_algorithm(alg))
    strategy_cache = _gsrdp_strategy_cache(problem)
    term_criteria = termination_criteria(alg, spec, mp)
    sampling_strat = sampling_strategy(alg)
    reset_sampling_strategy!(sampling_strat)

    value_function = construct_value_function(alg, problem)
    _gsrdp_initialize!(value_function, prop)
    nextiteration!(value_function)
    bellman_updates = 0

    # Initial callback before any updates for debug. Nothing has been sampled yet,
    # so there is no update sequence to report.
    _invoke_callback(callback, value_function, bellman_updates, nothing)

    update_sequence = _gsrdp_sample(
        sampling_strat,
        mp,
        select_strategy_cache(strategy_cache, 0),
        value_function,
        spec,
    )
    state_seq = project_to_state_sequence(update_sequence)
    bellman_updates += _bellman_update_count(state_seq, mp)
    bellman_update!(
        alg,
        workspace,
        strategy_cache,
        state_seq,
        value_function,
        0,
        mp,
        spec,
    )
    k = 1

    _invoke_callback(callback, value_function, bellman_updates, state_seq)

    while !term_criteria(value_function, k, gap(value_function))
        nextiteration!(value_function)

        update_sequence = _gsrdp_sample(
            sampling_strat,
            mp,
            select_strategy_cache(strategy_cache, k),
            value_function,
            spec,
        )
        state_seq = project_to_state_sequence(update_sequence)
        bellman_updates += _bellman_update_count(state_seq, mp)
        bellman_update!(
            alg,
            workspace,
            strategy_cache,
            state_seq,
            value_function,
            k,
            mp,
            spec,
        )

        _invoke_callback(callback, value_function, bellman_updates, state_seq)

        k += 1
    end

    postprocess_value_function!(value_function.lower, prop)
    postprocess_value_function!(value_function.upper, prop)

    return value_function.lower.current, k, gap(value_function), strategy_cache
end

_bellman_update_count(update_sequence, mp) =
    length(project_to_state_sequence(update_sequence)) * (num_actions(mp) + 1)

# Invoke a user callback at whichever arity it supports.
#
#   callback(value_function, bellman_updates)             # the long-standing contract
#   callback(value_function, bellman_updates, state_seq)  # additionally observes WHICH
#                                                         # states this iteration relaxed
#
# `state_seq` is exactly the collection `bellman_v!` iterates (`for jₛ in
# update_sequence`), so a per-state occurrence count of it IS the number of backups that
# state received this iteration. Duplicates are meaningful and are preserved: a
# trajectory sampler that revisits a state really does relax it twice. It is `nothing`
# on the pre-loop fire, which happens before anything has been sampled.
#
# Arity is probed rather than required so every existing two-argument callback keeps
# working untouched. Note a callback declared with a `(args...)` splat is `applicable` at
# both arities and will therefore receive three arguments.
@inline function _invoke_callback(callback, value_function, bellman_updates, state_seq)
    isnothing(callback) && return nothing
    if applicable(callback, value_function, bellman_updates, state_seq)
        callback(value_function, bellman_updates, state_seq)
    else
        callback(value_function, bellman_updates)
    end
    return nothing
end

# Pessimistic returns the lower bound, Optimistic returns the upper bound —
# matching `RobustValueIteration`'s single-bound output for parity.
_solution_value(V::IntervalValueFunction, spec) =
    ispessimistic(spec) ? V.lower.current : V.upper.current

# Initialise both bounds from the property. Reachability splits upper/lower
# differently (lower starts at 0/1, upper starts at 1 everywhere); other
# properties initialise both bounds identically.
function _gsrdp_initialize!(V::IntervalValueFunction, prop::AbstractReachability)
    initialize!(V.lower, prop, Val(false))
    initialize!(V.upper, prop, Val(true))
end

# Any other property would silently fall through to a `MethodError` deep inside
# the solve; fail at the point the restriction actually applies instead.
_gsrdp_initialize!(::IntervalValueFunction, prop) = throw(
    ArgumentError(
        "GeneralizedSamplingbasedRobustDynamicProgramming currently supports only " *
        "reachability and reach-avoid properties; got $(typeof(prop)).",
    ),
)

# GSRDP always allocates a fresh `StationaryStrategyCache`. The algorithm
# is infinite-horizon, so the optimal policy under a fixed model is
# stationary; using the same cache type for both `VerificationProblem`
# and `ControlSynthesisProblem` keeps the upper/lower coordination
# uniform. For verification, the cache is scratch and discarded; for
# synthesis, `cachetostrategy` extracts the final policy.
function _gsrdp_strategy_cache(problem::AbstractIntervalMDPProblem)
    mp = system(problem)
    N = length(action_values(mp))
    strategy_arr = arrayfactory(mp, NTuple{N, Int32}, source_shape(mp))
    strategy_arr .= (ntuple(_ -> 0, N),)
    return StationaryStrategyCache(strategy_arr)
end

# Wrap the upper bellman call's strategy cache as a non-optimizing
# follower for the lower bellman call. After `bellman_v!` with the
# stationary cache, `cache.strategy` holds the chosen action per state;
# we expose it as an `ActiveGivenStrategyCache` for the lower call.
_follow_strategy_cache(cache::StationaryStrategyCache) =
    ActiveGivenStrategyCache(cache.strategy)

function bellman_update!(
    ::GeneralizedSamplingbasedRobustDynamicProgramming,
    workspace,
    strategy_cache,
    update_sequence,
    value_function::IntervalValueFunction,
    k,
    mp,
    spec,
)
    state_seq = project_to_state_sequence(update_sequence)
    model = select_model(mp, k)

    #TODO: upper drives action selection; lower follows.
    # Upper bound drives optimistic action selection, lower bound follows. 
    upper, lower = value_function.upper, value_function.lower

    # `upper_bound` is the *adversary's* direction inside the ambiguity set
    # (`true` = O-maximization), not a tag for which bracket is being written.
    # `value_function.lower` and `value_function.upper` bracket the *same* fixed
    # point from below and above, so both calls must use the direction the
    # satisfaction mode dictates - the same value `RobustValueIteration` passes.
    # Keep these two `upper_bound` arguments identical; that is what makes the
    # bracket valid.
    upper_sc = select_strategy_cache(strategy_cache, k)
    bellman_v!(
        workspace,
        upper_sc,
        StateValueArray(upper.current),
        StateValueArray(upper.previous),
        model,
        state_seq;
        upper_bound = isoptimistic(spec),
        maximize = ismaximize(spec),
        prop = system_property(spec),
    )

    lower_sc = _follow_strategy_cache(upper_sc)
    bellman_v!(
        workspace,
        lower_sc,
        StateValueArray(lower.current),
        StateValueArray(lower.previous),
        model,
        state_seq;
        upper_bound = isoptimistic(spec),
        maximize = ismaximize(spec),
        prop = system_property(spec),
    )

    step_postprocess_value_function!(value_function.lower, spec)
    step_postprocess_value_function!(value_function.upper, spec)
    step_postprocess_strategy_cache!(strategy_cache)
end
