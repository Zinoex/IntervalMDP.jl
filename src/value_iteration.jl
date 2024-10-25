abstract type TerminationCriteria end
function termination_criteria(spec::Specification)
    prop = system_property(spec)
    return termination_criteria(prop, Val(isfinitetime(prop)))
end

struct FixedIterationsCriteria{T <: Integer} <: TerminationCriteria
    n::T
end
(f::FixedIterationsCriteria)(V, k, u) = k >= f.n
termination_criteria(prop, finitetime::Val{true}) =
    FixedIterationsCriteria(time_horizon(prop))

struct CovergenceCriteria{T <: AbstractFloat} <: TerminationCriteria
    tol::T
end
(f::CovergenceCriteria)(V, k, u) = maximum(u) < f.tol
termination_criteria(prop, finitetime::Val{false}) =
    CovergenceCriteria(convergence_eps(prop))

"""
    value_iteration(problem::Problem; callback=nothing)

Solve minimizes/mazimizes optimistic/pessimistic specification problems using value iteration for interval Markov processes. 

It is possible to provide a callback function that will be called at each iteration with the current value function and
iteration count (starting from zero). The callback function should have the signature `callback(V::AbstractArray, k::Int)`.

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
problem = Problem(mdp, spec)
V, k, residual = value_iteration(problem)
```

"""
function value_iteration(problem::Problem; callback=nothing)
    strategy_config = whichstrategyconfig(problem)
    V, k, res, _ = _value_iteration!(strategy_config, problem; callback=callback)

    return V, k, res
end
whichstrategyconfig(::Problem{S, F, <:NoStrategy}) where {S, F} = NoStrategyConfig()
whichstrategyconfig(::Problem{S, F, <:AbstractStrategy}) where {S, F} =
    GivenStrategyConfig()

function _value_iteration!(strategy_config::AbstractStrategyConfig, problem::Problem; callback=nothing)
    mp = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(spec)
    upper_bound = isoptimistic(spec)
    maximize = ismaximize(spec)

    # It is more efficient to use allocate first and reuse across iterations
    workspace = construct_workspace(mp)
    strategy_cache = construct_strategy_cache(mp, strategy_config, strategy(problem))

    value_function = ValueFunction(problem)
    initialize!(value_function, spec)

    step!(
        workspace,
        strategy_cache,
        value_function,
        0,
        mp;
        upper_bound = upper_bound,
        maximize = maximize,
    )
    step_postprocess_value_function!(value_function, spec)
    step_postprocess_strategy_cache!(strategy_cache)

    if !isnothing(callback)
        callback(value_function.current, 0)
    end

    k = 1

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
        step_postprocess_value_function!(value_function, spec)
        step_postprocess_strategy_cache!(strategy_cache)

        if !isnothing(callback)
            callback(value_function.current, 0)
        end

        k += 1
    end

    postprocess_value_function!(value_function, spec)

    # lastdiff! uses previous to store the latest difference
    # and it is already computed from the condition in the loop
    return value_function.current, k, value_function.previous, strategy_cache
end

struct ValueFunction{R, A <: AbstractArray{R}}
    previous::A
    current::A
end

function ValueFunction(previous::A) where {R, A <: AbstractArray{R}}
    current = copy(previous)
    return ValueFunction{R, A}(previous, current)
end

function ValueFunction(problem::Problem)
    mp = system(problem)
    pns = product_num_states(mp)

    return ValueFunction(mp, pns)
end

ValueFunction(mp::IntervalMarkovProcess, num_states) =
    ValueFunction(transition_prob(mp), num_states)
ValueFunction(prob::MixtureIntervalProbabilities, num_states) =
    ValueFunction(first(prob), num_states)
ValueFunction(prob::OrthogonalIntervalProbabilities, num_states) =
    ValueFunction(first(prob), num_states)
ValueFunction(prob::IntervalProbabilities, num_states) =
    ValueFunction(gap(prob), num_states)
ValueFunction(::MR, num_states) where {R, MR <: AbstractMatrix{R}} =
    ValueFunction(zeros(R, num_states))

function lastdiff!(V)
    # Reuse prev to store the latest difference
    V.previous .-= V.current
    rmul!(V.previous, -1.0)

    return V.previous
end

function nextiteration!(V)
    copy!(V.previous, V.current)

    return V
end

function step!(
    workspace,
    strategy_cache::OptimizingStrategyCache,
    value_function,
    k,
    mp::IntervalMarkovProcess;
    upper_bound,
    maximize,
)
    bellman!(
        workspace,
        strategy_cache,
        value_function.current,
        value_function.previous,
        transition_prob(mp),
        stateptr(mp);
        upper_bound = upper_bound,
        maximize = maximize,
    )
end

function step!(
    workspace,
    strategy_cache::NonOptimizingStrategyCache,
    value_function,
    k,
    mp::IntervalMarkovProcess;
    upper_bound,
    maximize,
)
    bellman!(
        workspace,
        strategy_cache[time_length(strategy_cache) - k],
        value_function.current,
        value_function.previous,
        transition_prob(mp),
        stateptr(mp);
        upper_bound = upper_bound,
        maximize = maximize,
    )
end
