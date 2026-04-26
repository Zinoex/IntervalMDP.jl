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

function initialize!(value_function::ValueFunction, prop::AbstractReachability)
    initialize!(value_function, prop, Val(isupper(value_function)))
end

termination_criteria(::RobustValueIteration, spec) = termination_criteria(spec)
termination_criteria(::GeneralizedSamplingbasedRobustDynamicProgramming, spec) =
    termination_criteria(spec)

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

function solve(
    problem::VerificationProblem,
    alg::GeneralizedSamplingbasedRobustDynamicProgramming;
    kwargs...,
)
    V, k, res, _ = _value_iteration!(problem, alg; kwargs...)
    return VerificationSolution(V, res, k)
end

function solve(
    problem::ControlSynthesisProblem,
    alg::GeneralizedSamplingbasedRobustDynamicProgramming;
    kwargs...,
)
    V, k, res, strategy_cache = _value_iteration!(problem, alg; kwargs...)
    strategy = cachetostrategy(strategy_cache)

    return ControlSynthesisSolution(strategy, V, res, k)
end

function _value_iteration!(
    problem::AbstractIntervalMDPProblem,
    alg::ModelCheckingAlgorithm;
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

function bellman_update!(
    ::RobustValueIteration,
    workspace,
    strategy_cache,
    update_sequence,
    value_function::StateValueFunction,
    k,
    mp,
    spec,
)
    # `expectation_v!` writes V'[s] directly using per-state action scratch
    # in `workspace.actions` — no `(action × state)` Q-array is allocated
    # along the hot path. For RobustVI the update sequence is a
    # `StateUpdateSequence` (yields `s`), so this dispatches to the
    # state-outer + `extract_strategy!` path. The `StateValueArray`
    # wrappers tag the buffers as state-value-shape at the type level.
    expectation_v!(
        workspace,
        select_strategy_cache(strategy_cache, k),
        StateValueArray(value_function.current),
        StateValueArray(value_function.previous),
        select_model(mp, k), # For time-varying available and labelling functions
        update_sequence;
        upper_bound = isoptimistic(spec),
        maximize = ismaximize(spec),
        prop = system_property(spec),
    )

    # Post-process to compute V(s) = g(s, V'(s)) where the definition of g
    # depends on the objective (reachability / safety / reward / discount).
    step_postprocess_value_function!(value_function, spec)
    step_postprocess_strategy_cache!(strategy_cache)
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
    # Use the V-shape primitive — for the sampling-based path the update
    # sequence is a `StateActionUpdateSequence`, so this dispatches to
    # `sa_sweep!`, which relaxes V[s] and the strategy cache in place as
    # each (a, s) pair is visited. States not visited in this iteration
    # retain `V_prev`.
    expectation_v!(
        workspace,
        select_strategy_cache(strategy_cache, k),
        StateValueArray(value_function.current),
        StateValueArray(value_function.previous),
        select_model(mp, k),
        update_sequence;
        upper_bound = isoptimistic(spec),
        maximize = ismaximize(spec),
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
