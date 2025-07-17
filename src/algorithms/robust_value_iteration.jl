

"""
    solve(problem::AbstractIntervalMDPAlgorithm, alg::RobustValueIteration; callback=nothing)

Solve minimizes/maximizes optimistic/pessimistic specification problems using value iteration for interval Markov processes. 

It is possible to provide a callback function that will be called at each iteration with the current value function and
iteration count. The callback function should have the signature `callback(V::AbstractArray, k::Int)`.

`solve` can be called without specifying the algorithm, in which case it defaults to [`RobustValueIteration`](@ref).

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
sol = solve(problem, RobustValueIteration(); callback = (V, k) -> println("Iteration ", k))
V, k, res = sol  # or `value_function(sol), num_iterations(sol), residual(sol)`

# Control synthesis
problem = ControlSynthesisProblem(mdp, spec)
sol = solve(problem, RobustValueIteration(); callback = (V, k) -> println("Iteration ", k))
σ, V, k, res = sol # or `strategy(sol), value_function(sol), num_iterations(sol), residual(sol)`
```
"""
function solve(problem::VerificationProblem, alg::RobustValueIteration; kwargs...)
    V, k, res, _ = _value_iteration(problem, alg; kwargs...)
    return VerificationSolution(V, res, k)
end

function solve(problem::ControlSynthesisProblem, alg::RobustValueIteration; kwargs...)
    V, k, res, strategy_cache = _value_iteration(problem, alg; kwargs...)
    strategy = cachetostrategy(strategy_cache)

    return ControlSynthesisSolution(strategy, V, res, k)
end

function _value_iteration(problem::AbstractIntervalMDPProblem, alg::RobustValueIteration; callback = nothing)
    mp = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(spec)
    upper_bound = isoptimistic(spec)
    maximize = ismaximize(spec)

    # It is more efficient to use allocate first and reuse across iterations
    workspace = construct_workspace(mp)
    strategy_cache = construct_strategy_cache(problem)

    value_function = ValueFunction(problem)
    initialize!(value_function, spec)
    nextiteration!(value_function)

    step!(
        workspace,
        strategy_cache,
        value_function,
        0,
        mp;
        upper_bound = upper_bound,
        maximize = maximize,
    )
    step_specification!(value_function, spec)
    step_strategy_cache!(strategy_cache)

    k = 1
    maybe_callback(callback, value_function.current, k)

    while !term_criteria(value_function.current, k, lastdiff!(value_function))
        nextiteration!(value_function)

        step!(
            workspace,
            strategy_cache,
            value_function,
            k,
            mp;
            upper_bound = upper_bound,
            maximize = maximize,
        )
        step_specification!(value_function, spec)
        step_strategy_cache!(strategy_cache)
        
        k += 1
        maybe_callback(callback, value_function.current, k)
    end

    postprocess_value_function!(value_function, spec)

    # lastdiff! uses previous to store the latest difference
    # and it is already computed from the condition in the loop
    return value_function.current, k, value_function.previous, strategy_cache
end

function step!(
    workspace,
    strategy_cache::C,
    value_function,
    k,
    mp;
    upper_bound,
    maximize,
) where {C <: AbstractStrategyCache}
    bellman!(
        workspace,
        strategy_cache,
        value_function.current,
        value_function.previous,
        mp;
        upper_bound = upper_bound,
        maximize = maximize,
    )
end

function step!(
    workspace,
    strategy_cache::GivenStrategyCache,
    value_function,
    k,
    mp;
    upper_bound,
    maximize,
)
    bellman!(
        workspace,
        strategy_cache[time_length(strategy_cache) - k],
        value_function.current,
        value_function.previous,
        mp;
        upper_bound = upper_bound,
        maximize = maximize,
    )
end