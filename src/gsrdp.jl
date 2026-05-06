
"""
    GeneralizedSamplingbasedRobustDynamicProgramming(bellman_alg, sampling_strategy)

Generalized sampling-based robust dynamic programming. The `sampling_strategy`
field controls which state (or state-action) pairs are updated each iteration.
Defaults to `AllSampling()` for parity with [`RobustValueIteration`](@ref).
"""
struct GeneralizedSamplingbasedRobustDynamicProgramming{B <: BellmanAlgorithm, S} <:
       ModelCheckingAlgorithm
    bellman_alg::B
    sampling_strategy::S
end
GeneralizedSamplingbasedRobustDynamicProgramming(bellman_alg::BellmanAlgorithm) =
    GeneralizedSamplingbasedRobustDynamicProgramming(bellman_alg, AllStatesSweep())
bellman_algorithm(alg::GeneralizedSamplingbasedRobustDynamicProgramming) = alg.bellman_alg
termination_criteria(::GeneralizedSamplingbasedRobustDynamicProgramming, spec) =
    termination_criteria(spec)
construct_value_function(::GeneralizedSamplingbasedRobustDynamicProgramming, problem) =
    StateValueFunction(problem)

function solve(
    problem::VerificationProblem,
    alg::GeneralizedSamplingbasedRobustDynamicProgramming;
    kwargs...,
)
    V, k, res, _ = _gsrdp!(problem, alg; kwargs...)
    return VerificationSolution(V, res, k)
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
    term_criteria = termination_criteria(alg, spec)

    # It is more efficient to use allocate first and reuse across iterations
    workspace = construct_workspace(mp, bellman_algorithm(alg))
    strategy_cache = construct_strategy_cache(problem)
    sampling_strat = sampling_strategy(alg)

    value_function = construct_value_function(alg, problem)
    initialize!(value_function, spec)
    nextiteration!(value_function)

    update_sequence = sample(sampling_strat, mp, select_strategy_cache(strategy_cache, 0))
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
        callback(value_function.current, k)
    end

    while !term_criteria(value_function.current, k, lastdiff!(value_function))
        nextiteration!(value_function)

        update_sequence =
            sample(sampling_strat, mp, select_strategy_cache(strategy_cache, k))
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
            callback(value_function.current, k)
        end

        k += 1
    end

    postprocess_value_function!(value_function, spec)

    # `lastdiff!` was last called inside the `while` condition above and
    # has stored `V_prev - V_cur` into `value_function.previous`. Reuse it
    # as the residual return value, matching the contract used by
    # `_robust_value_iteration!`.
    return value_function.current, k, value_function.previous, strategy_cache
end

function bellman_update!(
    ::GeneralizedSamplingbasedRobustDynamicProgramming,
    workspace,
    strategy_cache,
    update_sequence,
    value_function::StateValueFunction,
    k,
    mp,
    spec,
)
    # Until the `IntervalValueFunction{StateActionValueFunction}` redesign
    # (Problem 2) lands, gsrdp drives a `StateValueFunction` and calls
    # `bellman_v!`. State-action samplers are projected to their unique
    # state set (semantics weakened: visited states get a full action
    # sweep, unvisited states retain `V_prev`).
    state_seq = project_to_state_sequence(update_sequence)

    bellman_v!(
        workspace,
        select_strategy_cache(strategy_cache, k),
        StateValueArray(value_function.current),
        StateValueArray(value_function.previous),
        select_model(mp, k),
        state_seq;
        upper_bound = isoptimistic(spec),
        maximize = ismaximize(spec),
    )

    step_postprocess_value_function!(value_function, spec)
    step_postprocess_strategy_cache!(strategy_cache)
end
