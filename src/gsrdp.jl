"""
    GeneralizedSamplingbasedRobustDynamicProgramming(bellman_alg, sampling_strategy, config)

Generalized sampling-based robust dynamic programming. Drives an
[`IntervalValueFunction`](@ref) — i.e. simultaneous lower and upper
bounds on the value — and terminates when the gap `V_upper - V_lower`
falls below `convergence_eps(prop)`.

Restrictions:

* Only infinite-horizon (convergence-based) properties are accepted.
  Finite-horizon properties have a fixed-iteration termination criterion
  that's incompatible with gap-based convergence.

The optimal action at each visited state is picked from the *primary*
bound (lower for `Pessimistic`, upper for `Optimistic`) using the
specification's `Maximize`/`Minimize` mode. The chosen action is then
applied to the *secondary* bound through a `NonOptimizingStrategyCache`
so both bounds track the same policy.

`sampling_strategy` controls which states (or `(a, s)` pairs) are
relaxed each iteration. State-action samplers are projected to their
unique state set via [`project_to_state_sequence`](@ref) — visited states
get a full action sweep, unvisited states retain `V_prev`.

`config` stores optional solver overrides such as a custom sampling
strategy or termination criteria. When a field is left as `nothing`, the
algorithm falls back to the default behavior derived from the problem.
"""
struct GeneralizedSamplingbasedRobustDynamicProgramming{B <: BellmanAlgorithm, S} <:
       ModelCheckingAlgorithm
    bellman_alg::B
    sampling_strategy::S
    config::Union{Config, Nothing}

    function GeneralizedSamplingbasedRobustDynamicProgramming(
        bellman_alg::B,
        sampling_strategy::S,
        config::Union{Config, Nothing} = nothing,
    ) where {B <: BellmanAlgorithm, S}
        new{B, S}(bellman_alg, sampling_strategy, config)
    end
end
GeneralizedSamplingbasedRobustDynamicProgramming(bellman_alg::BellmanAlgorithm) =
    GeneralizedSamplingbasedRobustDynamicProgramming(bellman_alg, AllStatesSweep())

GeneralizedSamplingbasedRobustDynamicProgramming(
    bellman_alg::BellmanAlgorithm,
    sampling_strategy,
) = GeneralizedSamplingbasedRobustDynamicProgramming(bellman_alg, sampling_strategy, nothing)

bellman_algorithm(alg::GeneralizedSamplingbasedRobustDynamicProgramming) = alg.bellman_alg

construct_value_function(::GeneralizedSamplingbasedRobustDynamicProgramming, problem) =
    IntervalValueFunction(
        StateValueFunction(problem, Lower),
        StateValueFunction(problem, Upper),
    )

"""
    GapTerminationCriteria(tol)

Terminates when `maximum(abs, gap) < tol` over the elementwise gap
`V_upper - V_lower`. Used by
[`GeneralizedSamplingbasedRobustDynamicProgramming`](@ref).
"""
struct GapTerminationCriteria{T <: Real} <: TerminationCriteria
    tol::T
end
(f::GapTerminationCriteria)(_, _, gap_residual) = maximum(abs, gap_residual) < f.tol

"""
    GapTerminationCriteriaInitial(tol)

Terminates when `maximum(abs, gap) < tol` over the elementwise gap
`V_upper - V_lower`. Used by
[`GeneralizedSamplingbasedRobustDynamicProgramming`](@ref).
"""
struct GapTerminationCriteriaInitial{T <: Real, I} <: TerminationCriteria
    tol::T
    initial::I
end
(f::GapTerminationCriteriaInitial)(_, _, gap_residual) =
    maximum(abs, gap_residual[f.initial]) < f.tol

function _termination_criteria(spec::Specification)
    prop = system_property(spec)
    if isfinitetime(prop)
        throw(
            ArgumentError(
                "GeneralizedSamplingbasedRobustDynamicProgramming requires an " *
                "infinite-horizon property; got $(typeof(prop)).",
            ),
        )
    end
    return termination_criteria(prop)
end

function termination_criteria(
    alg::GeneralizedSamplingbasedRobustDynamicProgramming,
    spec::Specification,
)
    if !isnothing(alg.config) && !isnothing(alg.config.term_criteria)
        return alg.config.term_criteria
    end

    return _termination_criteria(spec)
end

termination_criteria(prop::InfiniteTimeReachAvoidInitial) =
    GapTerminationCriteriaInitial(convergence_eps(prop), initial(prop))

termination_criteria(prop) = GapTerminationCriteria(convergence_eps(prop))

sampling_strategy(alg::GeneralizedSamplingbasedRobustDynamicProgramming) =
    isnothing(alg.config) || isnothing(alg.config.sampling_strategy) ?
    alg.sampling_strategy : alg.config.sampling_strategy
    
function solve(
    problem::VerificationProblem,
    alg::GeneralizedSamplingbasedRobustDynamicProgramming;
    kwargs...,
)
    V, k, res, _ = _gsrdp!(problem, alg; kwargs...)
    return VerificationSolution(V, res, k)
end

# Sampling dispatcher for GSRDP.
# Prefer samplers that accept a `value_function` argument (4-arg `sample`).
# Fall back to older 3-arg or 2-arg `sample` signatures for backwards compatibility.
function _gsrdp_sample(ss::ValueBasedSamplingStrategy, mp, strategy_cache, value_function)
    return sample(ss, mp, strategy_cache, value_function)
end

function _gsrdp_sample(ss::SamplingStrategy, mp, strategy_cache, value_function)
    return sample(ss, mp, strategy_cache)
end

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
    term_criteria = termination_criteria(alg, spec)
    sampling_strat = sampling_strategy(alg)

    value_function = construct_value_function(alg, problem)
    _gsrdp_initialize!(value_function, prop)
    nextiteration!(value_function)
    bellman_updates = 0

    # Initial callback before any updates for debug
    if !isnothing(callback)
        callback(value_function, bellman_updates)
    end

    update_sequence = _gsrdp_sample(
        sampling_strat,
        mp,
        select_strategy_cache(strategy_cache, 0),
        value_function,
    )
    bellman_updates += _bellman_update_count(update_sequence, mp)
    bellman_update!(
        alg,
        workspace,
        strategy_cache,
        update_sequence,
        value_function,
        0,
        mp,
        spec,
    )
    k = 1

    if !isnothing(callback)
        callback(value_function, bellman_updates)
    end

    while !term_criteria(value_function, k, gap(value_function))
        nextiteration!(value_function)

        update_sequence = _gsrdp_sample(
            sampling_strat,
            mp,
            select_strategy_cache(strategy_cache, k),
            value_function,
        )
        bellman_updates += _bellman_update_count(update_sequence, mp)
        bellman_update!(
            alg,
            workspace,
            strategy_cache,
            update_sequence,
            value_function,
            k,
            mp,
            spec,
        )

        if !isnothing(callback)
            callback(value_function, bellman_updates)
        end

        k += 1
    end

    postprocess_value_function!(value_function.lower, prop)
    postprocess_value_function!(value_function.upper, prop)

    return _solution_value(value_function, spec), k, gap(value_function), strategy_cache
end

_bellman_update_count(update_sequence, mp) =
    length(project_to_state_sequence(update_sequence)) * (num_actions(mp) + 1)

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

# Wrap the primary bellman call's strategy cache as a non-optimizing
# follower for the secondary bellman call. After `bellman_v!` with the
# stationary cache, `cache.strategy` holds the chosen action per state;
# we expose it as an `ActiveGivenStrategyCache` for the secondary call.
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

    # Primary drives action selection; secondary follows.
    primary, secondary = if ispessimistic(spec)
        value_function.lower, value_function.upper
    else
        value_function.upper, value_function.lower
    end

    primary_sc = select_strategy_cache(strategy_cache, k)
    bellman_v!(
        workspace,
        primary_sc,
        StateValueArray(primary.current),
        StateValueArray(primary.previous),
        model,
        state_seq;
        upper_bound = ispessimistic(spec),
        maximize = ismaximize(spec),
        prop = system_property(spec),
    )

    secondary_sc = _follow_strategy_cache(primary_sc)
    bellman_v!(
        workspace,
        secondary_sc,
        StateValueArray(secondary.current),
        StateValueArray(secondary.previous),
        model,
        state_seq;
        upper_bound = ispessimistic(spec),
        maximize = ismaximize(spec),
        prop = system_property(spec),
    )

    step_postprocess_value_function!(value_function.lower, spec)
    step_postprocess_value_function!(value_function.upper, spec)
    step_postprocess_strategy_cache!(strategy_cache)
end

