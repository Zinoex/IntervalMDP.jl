abstract type TerminationCriteria end
function termination_criteria(spec::Specification)
    prop = system_property(spec)
    ft = isfinitetime(prop)
    return termination_criteria(prop, Val(ft))
end

struct FixedIterationsCriteria{T <: Integer} <: TerminationCriteria
    n::T
end
(f::FixedIterationsCriteria)(V, k, u) = k >= f.n
termination_criteria(prop, finitetime::Val{true}) =
    FixedIterationsCriteria(time_horizon(prop))

struct CovergenceCriteria{T <: Real} <: TerminationCriteria
    tol::T
end
(f::CovergenceCriteria)(V, k, u) = maximum(abs, u) < f.tol
termination_criteria(prop, finitetime::Val{false}) =
    CovergenceCriteria(convergence_eps(prop))

"""
    solve(problem::AbstractIntervalMDPProblem, alg::RobustValueIteration; callback=nothing)

Solve minimizes/maximizes optimistic/pessimistic specification problems using value iteration for interval Markov processes. 

It is possible to provide a callback function that will be called at each iteration with the current value function and
iteration count. The callback function should have the signature `callback(V::AbstractArray, k::Int)`.

`solve` can be called without specifying the algorithm, in which case it defaults to [`RobustValueIteration`](@ref).

### Examples

```jldoctest robust_vi
using IntervalMDP

prob1 = IntervalAmbiguitySets(;
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

prob2 = IntervalAmbiguitySets(;
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

prob3 = IntervalAmbiguitySets(;
    lower = [
        0.0 0.0
        0.0 0.0
        1.0 1.0
    ],
    upper = [
        0.0 0.0
        0.0 0.0
        1.0 1.0
    ]
)

transition_probs = [prob1, prob2, prob3]
initial_state = [1]
mdp = IntervalMarkovDecisionProcess(transition_probs, initial_state)

# output

FactoredRobustMarkovDecisionProcess
├─ 1 state variables with cardinality: (3,)
├─ 1 action variables with cardinality: (2,)
├─ Initial states: [1]
├─ Transition marginals:
│  └─ Marginal 1:
│     ├─ Conditional variables: states = (1,), actions = (1,)
│     └─ Ambiguity set type: Interval (dense, Matrix{Float64})
└─Inferred properties
   ├─Model type: Interval MDP
   ├─Number of states: 3
   ├─Number of actions: 2
   ├─Default model checking algorithm: Robust Value Iteration
   └─Default Bellman operator algorithm: O-Maximization
```

```jldoctest robust_vi
reach_states = [3]
time_horizon = 10
prop = FiniteTimeReachability(reach_states, time_horizon)
spec = Specification(prop, Pessimistic, Maximize)

# output

Specification
├─ Satisfaction mode: Pessimistic
├─ Strategy mode: Maximize
└─ Property: FiniteTimeReachability
   ├─ Time horizon: 10
   └─ Reach states: CartesianIndex{1}[CartesianIndex(3,)]
```


```jldoctest robust_vi
# Verification
problem = VerificationProblem(mdp, spec)
sol = solve(problem, RobustValueIteration(default_bellman_algorithm(mdp)); callback = (V, k) -> println("Iteration ", k))
V, k, res = sol  # or `value_function(sol), num_iterations(sol), residual(sol)`

# output

Iteration 1
Iteration 2
Iteration 3
Iteration 4
Iteration 5
Iteration 6
Iteration 7
Iteration 8
Iteration 9
Iteration 10
IntervalMDP.VerificationSolution{Float64, Vector{Float64}, Nothing}([0.9597716063999999, 0.9710050144, 1.0], [0.01593864639999998, 0.011487926399999848, -0.0], 10, nothing)

```

```jldoctest robust_vi
# Control synthesis
problem = ControlSynthesisProblem(mdp, spec)
sol = solve(problem, RobustValueIteration(default_bellman_algorithm(mdp)); callback = (V, k) -> println("Iteration ", k))
σ, V, k, res = sol # or `strategy(sol), value_function(sol), num_iterations(sol), residual(sol)`

# output

Iteration 1
Iteration 2
Iteration 3
Iteration 4
Iteration 5
Iteration 6
Iteration 7
Iteration 8
Iteration 9
Iteration 10
IntervalMDP.ControlSynthesisSolution{TimeVaryingStrategy{1, Vector{Tuple{Int32}}}, Float64, Vector{Float64}, Nothing}(TimeVaryingStrategy{1, Vector{Tuple{Int32}}}(Vector{Tuple{Int32}}[[(1,), (2,), (1,)], [(1,), (2,), (1,)], [(1,), (2,), (1,)], [(1,), (2,), (1,)], [(1,), (2,), (1,)], [(1,), (2,), (1,)], [(1,), (2,), (1,)], [(1,), (2,), (1,)], [(1,), (2,), (1,)], [(1,), (2,), (1,)]]), [0.9597716063999999, 0.9710050144, 1.0], [0.01593864639999998, 0.011487926399999848, -0.0], 10, nothing)
```
"""
function solve(problem::VerificationProblem, alg::RobustValueIteration; kwargs...)
    V, k, res, _ = _value_iteration!(problem, alg; kwargs...)
    return VerificationSolution(V, res, k)
end

function solve(problem::ControlSynthesisProblem, alg::RobustValueIteration; kwargs...)
    V, k, res, strategy_cache = _value_iteration!(problem, alg; kwargs...)
    strategy = cachetostrategy(strategy_cache)

    return ControlSynthesisSolution(strategy, V, res, k)
end

function _value_iteration!(problem::AbstractIntervalMDPProblem, alg; callback = nothing)
    mp = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(spec)

    # It is more efficient to use allocate first and reuse across iterations
    workspace = construct_workspace(mp, bellman_algorithm(alg))
    strategy_cache = construct_strategy_cache(problem)

    value_function = ValueFunction(problem)
    initialize!(value_function, spec)
    nextiteration!(value_function)

    step!(workspace, strategy_cache, value_function, 0, mp, spec)
    k = 1

    if !isnothing(callback)
        callback(value_function.current, k)
    end

    while !term_criteria(value_function.current, k, lastdiff!(value_function))
        nextiteration!(value_function)

        step!(workspace, strategy_cache, value_function, k, mp, spec)
        k += 1

        if !isnothing(callback)
            callback(value_function.current, k)
        end
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

function ValueFunction(problem::AbstractIntervalMDPProblem)
    mp = system(problem)
    previous = arrayfactory(mp, valuetype(mp), state_values(mp))
    previous .= zero(valuetype(mp))
    current = copy(previous)

    return ValueFunction(previous, current)
end

function lastdiff!(V::ValueFunction{R}) where {R}
    # Reuse prev to store the latest difference
    V.previous .-= V.current
    rmul!(V.previous, -one(R))

    return V.previous
end

function nextiteration!(V)
    copy!(V.previous, V.current)

    return V
end

function step!(workspace, strategy_cache, value_function, k, mp, spec)
    bellman!(
        workspace,
        select_strategy_cache(strategy_cache, k),
        value_function.current,
        value_function.previous,
        select_model(mp, k);  # For time-varying available and labelling functions
        upper_bound = isoptimistic(spec),
        maximize = ismaximize(spec),
        prop = system_property(spec),
    )
    step_postprocess_value_function!(value_function, spec)
    step_postprocess_strategy_cache!(strategy_cache)
end

select_strategy_cache(strategy_cache::OptimizingStrategyCache, k) = strategy_cache
select_strategy_cache(strategy_cache::NonOptimizingStrategyCache, k) =
    strategy_cache[time_length(strategy_cache) - k]

select_model(mp::IntervalMarkovProcess, k) = FactoredRMDP(
    state_values(mp),
    action_values(mp),
    source_shape(mp),
    marginals(mp),
    select_available_actions(available_actions(mp), k),
    initial_states(mp),
    Val(false),
)

select_available_actions(aa::SingleTimeStepAvailableActions, k) = aa
select_available_actions(aa::TimeVaryingAvailableActions, k) =
    aa.actions[time_length(aa) - k]

select_model(mp::ProductProcess, k) = ProductProcess(
    select_model(markov_process(mp), k),
    automaton(mp),
    select_labelling_function(labelling_function(mp), k),
)

select_labelling_function(lf::AbstractSingleStepLabelling, k) = lf
select_labelling_function(lf::TimeVaryingLabelling, k) =
    lf.labelling_functions[time_length(lf) - k]
