"""
    solve(problem::AbstractIntervalMDPAlgorithm, alg::IntervalIteration; callback=nothing)

Solve minimizes/maximizes optimistic/pessimistic specification problems using interval iteration for interval Markov processes. 

It is possible to provide a callback function that will be called at each iteration with the current value function and
iteration count. The callback function should have the signature `callback(V_primal::AbstractArray, V_dual::AbstractArray, k::Int)`.

`solve` can be called without specifying the algorithm, in which case it defaults to [`IntervalIteration`](@ref).

### Examples

```jldoctest
prob1 = IntervalProbabilities(;
    lower = [
        0.0 0.5
        0.1 0.3
        0.2 0.1
    ],
    upper = [
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ],
)

prob2 = IntervalProbabilities(;
    lower = [
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ],
    upper = [
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ],
)

prob3 = IntervalProbabilities(;
    lower = [0.0; 0.0; 1.0],
    upper = [0.0; 0.0; 1.0]
)

transition_probs = [prob1, prob2, prob3]
initial_state = 1
mdp = IntervalMarkovDecisionProcess(transition_probs, initial_state)

terminal_states = [3]
time_horizon = 10
prop = FiniteTimeReachability(terminal_states, time_horizon)
spec = Specification(prop, Pessimistic, Maximize)

### Verification
problem = VerificationProblem(mdp, spec)
sol = solve(problem, IntervalIteration(); callback = (V_primal, V_dual, k) -> println("Iteration ", k))
V, k, res = sol  # or `value_function(sol), num_iterations(sol), residual(sol)`
V_dual = dual_value_function(sol)

# Control synthesis
problem = ControlSynthesisProblem(mdp, spec)
sol = solve(problem, IntervalIteration(); callback = (VV_primal, V_dual, k) -> println("Iteration ", k))
σ, V, k, res = sol # or `strategy(sol), value_function(sol), num_iterations(sol), residual(sol)`
V_dual = dual_value_function(sol)
```
"""
function solve(problem::VerificationProblem, alg::IntervalIteration; kwargs...)
    V, k, res, _, dual = _interval_iteration(problem, alg; kwargs...)
    return VerificationSolution(V, res, k, Dict("dual" => dual))
end

function dual_value_function(sol::VerificationSolution)
    return sol.additional_data["dual"]
end

function solve(problem::ControlSynthesisProblem, alg::IntervalIteration; kwargs...)
    V, k, res, strategy_cache = _interval_iteration(problem, alg; kwargs...)
    strategy = cachetostrategy(strategy_cache)

    return ControlSynthesisSolution(strategy, V, res, k, Dict("dual" => dual))
end

function dual_value_function(sol::ControlSynthesisSolution)
    return sol.additional_data["dual"]
end

function _interval_iteration(problem::AbstractIntervalMDPProblem, alg::IntervalIteration; callback = nothing)
    mp = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(spec)
    upper_bound = isoptimistic(spec)
    maximize = ismaximize(spec)

    # It is more efficient to use allocate first and reuse across iterations
    workspace = construct_workspace(mp)
    strategy_cache = construct_strategy_cache(problem)

    # TODO: Think about how the value functions should be initialized for interval iteration
    # in particular for (discounted) reward problems and expected exit time problems. Options:
    # 1. restrict Interval Iteration to logic problems only and add a lower bound/upper bound initialize.
    primary_value_function = ValueFunction(problem)
    initialize!(primary_value_function, spec)
    nextiteration!(primary_value_function)

    dual_value_function = ValueFunction(problem)
    initialize!(dual_value_function, spec)
    nextiteration!(dual_value_function)

    residual = zero(primary_value_function.current)

    # Step primal
    step!(
        workspace,
        strategy_cache,
        primary_value_function,
        0,
        mp;
        upper_bound = upper_bound,
        maximize = maximize,
    )
    step_specification!(primary_value_function, spec)
    step_strategy_cache!(strategy_cache)

    # Step dual
    dual_strategy_cache = ActiveGivenStrategyCache(current_strategy(strategy_cache, 0))
    step!(
        workspace,
        dual_strategy_cache,
        dual_value_function,
        0,
        mp;
        upper_bound = !upper_bound
    )
    step_specification!(dual_value_function, spec)

    k = 1
    maybe_callback(callback, primary_value_function.current, dual_value_function.current, k)

    while !term_criteria(value_function.current, k, diff!(residual, primary_value_function, dual_value_function))
        # Step primal
        nextiteration!(primary_value_function)
        step!(
            workspace,
            strategy_cache,
            primary_value_function,
            k,
            mp;
            upper_bound = upper_bound,
            maximize = maximize,
        )
        step_specification!(primary_value_function, spec)
        step_strategy_cache!(strategy_cache)

        # Step dual
        nextiteration!(dual_value_function)
        dual_strategy_cache = ActiveGivenStrategyCache(current_strategy(strategy_cache, 0))
        step!(
            workspace,
            dual_strategy_cache,
            dual_value_function,
            k,
            mp;
            upper_bound = !upper_bound
        )
        step_specification!(dual_value_function, spec)

        k += 1
        maybe_callback(callback, primary_value_function.current, dual_value_function.current, k)
    end

    postprocess_value_function!(value_function, spec)

    return value_function.current, k, residual, strategy_cache, dual_value_function.current
end
