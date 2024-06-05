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
    value_iteration(problem::Problem)

Solve minimizes/mazimizes optimistic/pessimistic specification problems using value iteration for interval Markov processes. 

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

transition_probs = [["a1", "a2"] => prob1, ["a1", "a2"] => prob2, ["sinking"] => prob3]
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
function value_iteration(
    problem::Problem,
)
    no_policy_cache = NoPolicyCache(system(problem))
    V, k, res, _ = _value_iteration!(no_policy_cache, problem)

    return V, k, res
end

function _value_iteration!(
    policy_cache::AbstractPolicyCache,
    problem::Problem,
)
    mp = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(spec)
    upper_bound = satisfaction_mode(spec) == Optimistic
    maximize = strategy_mode(spec) == Maximize

    # It is more efficient to use allocate first and reuse across iterations
    workspace = construct_workspace(mp, policy_cache)

    value_function = ValueFunction(problem)
    initialize!(value_function, spec)

    step!(workspace, value_function, 0, mp, spec; upper_bound = upper_bound, maximize = maximize)
    k = 1

    while !term_criteria(value_function.current, k, lastdiff!(value_function))
        nextiteration!(value_function)

        step!(workspace, value_function, k, mp, spec; upper_bound = upper_bound, maximize = maximize)
        k += 1
    end

    # lastdiff! uses previous to store the latest difference
    # and it is already computed from the condition in the loop
    return value_function.current, k, value_function.previous, policy_cache
end

mutable struct ValueFunction{R, A <: AbstractArray{R}}
    previous::A
    current::A
end

function ValueFunction(
    problem::Problem,
)
    mp = system(problem)

    previous = construct_value_function(gap(transition_prob(mp, 1)), num_states(mp))
    current = copy(previous)

    return ValueFunction(previous, current)
end

function construct_value_function(::MR, num_states) where {R, MR <: AbstractMatrix{R}}
    V = zeros(R, num_states)
    return V
end

function lastdiff!(V)
    # Reuse prev to store the latest difference
    V.previous .-= V.current
    rmul!(V.previous, -1.0)

    return V.previous
end

function nextiteration!(V)
    copyto!(V.previous, V.current)

    return V
end

function step!(workspace, value_function, k, mp::StationaryIntervalMarkovProcess, spec; upper_bound, maximize)
    prob = transition_prob(mp)
    bellman!(
        workspace,
        value_function.current,
        value_function.previous,
        prob;
        upper_bound = upper_bound,
        maximize = maximize,
    )
    postprocess_value_function!(value_function, spec)
    postprocess_policy_cache!(workspace.policy_cache)
end

function step!(workspace, value_function, k, mp::TimeVaryingIntervalMarkovProcess, spec; upper_bound, maximize)
    prob = transition_prob(mp, time_length(mp) - k)
    bellman!(
        workspace,
        value_function.current,
        value_function.previous,
        prob;
        upper_bound = upper_bound,
        maximize = maximize,
    )
    postprocess_value_function!(value_function, spec)
    postprocess_policy_cache!(workspace.policy_cache)
end