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
    value_iteration(problem::Problem{<:IntervalMarkovChain, <:Specification})

Solve optimistic/pessimistic specification problems using value iteration for interval Markov chain.

### Examples

```jldoctest
prob = IntervalProbabilities(;
    lower = [
        0.0 0.5 0.0
        0.1 0.3 0.0
        0.2 0.1 1.0
    ],
    upper = [
        0.5 0.7 0.0
        0.6 0.5 0.0
        0.7 0.3 1.0
    ],
)

mc = IntervalMarkovChain(prob, 1)

terminal_states = [3]
time_horizon = 10
prop = FiniteTimeReachability(terminal_states, time_horizon)
spec = Specification(prop, Pessimistic)
problem = Problem(mc, spec)
V, k, residual = value_iteration(problem)
```

"""
function value_iteration(
    problem::Problem{M, S},
) where {
    M <: Union{IntervalMarkovChain, TimeVaryingIntervalMarkovChain},
    S <: Specification,
}
    mc = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(spec)
    upper_bound = satisfaction_mode(spec) == Optimistic

    prob = transition_prob(mc, time_length(mc))

    # It is more efficient to use allocate first and reuse across iterations
    ordering = construct_ordering(gap(prob))

    value_function = IMCValueFunction(problem)
    initialize!(value_function, spec)

    step_imc!(value_function, ordering, prob; upper_bound = upper_bound)
    postprocess!(value_function, spec)
    k = 1

    while !term_criteria(value_function.cur, k, lastdiff!(value_function))
        nextiteration!(value_function)

        prob = transition_prob(mc, time_length(mc) - k)
        step_imc!(value_function, ordering, prob; upper_bound = upper_bound)
        postprocess!(value_function, spec)

        k += 1
    end

    # lastdiff! uses prev to store the latest difference
    # and it is already computed from the condition in the loop
    return value_function.cur, k, value_function.prev
end

function construct_value_function(::MR, num_states) where {R, MR <: AbstractMatrix{R}}
    V = zeros(R, num_states)
    return V
end

mutable struct IMCValueFunction{VR}
    prev::VR
    cur::VR
end

function IMCValueFunction(problem::Problem{M, S}) where {M <: IntervalMarkovChain, S}
    mc = system(problem)

    # Transition probabilities are always defined for time-step 1
    prob = transition_prob(mc, 1)
    prev = construct_value_function(gap(prob), num_states(mc))
    cur = copy(prev)

    return IMCValueFunction(prev, cur)
end

function lastdiff!(V)
    # Reuse prev to store the latest difference
    V.prev .-= V.cur
    rmul!(V.prev, -1.0)

    return V.prev
end

function nextiteration!(V)
    copyto!(V.prev, V.cur)

    return V
end

function step_imc!(
    value_function::IMCValueFunction,
    ordering,
    prob::IntervalProbabilities;
    upper_bound,
)
    bellman!(ordering, value_function.cur, value_function.prev, prob; max = upper_bound)

    return value_function
end

"""
    value_iteration(problem::Problem{<:IntervalMarkovDecisionProcess, <:Specification})

Solve minimizes/mazimizes optimistic/pessimistic specification problems using value iteration for interval Markov decision processes. 

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
    problem::Problem{M, S},
) where {M <: IntervalMarkovDecisionProcess, S <: Specification}
    no_policy_cache = NoPolicyCache()
    V, k, res, no_policy_cache = _value_iteration!(no_policy_cache, problem)

    return V, k, res
end

function _value_iteration!(
    policy_cache::AbstractPolicyCache,
    problem::Problem{M, S},
) where {M <: IntervalMarkovDecisionProcess, S <: Specification}
    mdp = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(spec)
    upper_bound = satisfaction_mode(spec) == Optimistic
    maximize = strategy_mode(spec) == Maximize

    prob = transition_prob(mdp)
    sptr = stateptr(mdp)

    # It is more efficient to use allocate first and reuse across iterations
    ordering = construct_ordering(gap(prob))

    value_function = IMDPValueFunction(problem)
    initialize!(value_function, spec)

    value_function, policy_cache = step_imdp!(
        value_function,
        policy_cache,
        ordering,
        prob,
        sptr;
        maximize = maximize,
        upper_bound = upper_bound,
    )
    postprocess!(value_function, spec)
    k = 1

    while !term_criteria(value_function.cur, k, lastdiff!(value_function))
        nextiteration!(value_function)
        value_function, policy_cache = step_imdp!(
            value_function,
            policy_cache,
            ordering,
            prob,
            sptr;
            maximize = maximize,
            upper_bound = upper_bound,
        )
        postprocess!(value_function, spec)

        k += 1
    end

    # lastdiff! uses prev to store the latest difference
    # and it is already computed from the condition in the loop
    return value_function.cur, k, value_function.prev, policy_cache
end

mutable struct IMDPValueFunction{VR}
    prev::VR
    cur::VR
    action_values::VR
end

function IMDPValueFunction(
    problem::Problem{M, S},
) where {M <: IntervalMarkovDecisionProcess, S}
    mdp = system(problem)

    prev = construct_value_function(gap(transition_prob(mdp)), num_states(mdp))
    cur = copy(prev)

    action_values = similar(prev, num_choices(mdp))

    return IMDPValueFunction(prev, cur, action_values)
end

function step_imdp!(
    value_function,
    policy_cache,
    ordering,
    prob::IntervalProbabilities,
    stateptr;
    maximize,
    upper_bound,
)
    bellman!(
        ordering,
        value_function.action_values,
        value_function.prev,
        prob;
        max = upper_bound,
    )

    return extract_policy!(value_function, policy_cache, stateptr, maximize)
end

function extract_policy!(
    value_function::IMDPValueFunction,
    policy_cache::NoPolicyCache,
    stateptr::VT,
    maximize,
) where {VT <: AbstractVector}
    gt = maximize ? (>) : (<)

    @inbounds for j in 1:(length(stateptr) - 1)
        s1 = stateptr[j]
        s2 = stateptr[j + 1]

        opt = value_function.action_values[s1]
        for i in (s1 + 1):(s2 - 1)
            if gt(value_function.action_values[i], opt)
                opt = value_function.action_values[i]
            end
        end

        value_function.cur[j] = opt
    end

    return value_function, policy_cache
end

function extract_policy!(
    value_function::IMDPValueFunction,
    policy_cache::TimeVaryingPolicyCache,
    stateptr::VT,
    maximize,
) where {VT <: AbstractVector}
    gt = maximize ? (>) : (<)

    @inbounds for j in 1:(length(stateptr) - 1)
        s1 = stateptr[j]
        s2 = stateptr[j + 1]

        opt_val = value_function.action_values[s1]
        opt_index = s1

        for i in (s1 + 1):(s2 - 1)
            if gt(value_function.action_values[i], opt_val)
                opt_val = value_function.action_values[i]
                opt_index = i
            end
        end

        policy_cache.cur_policy[j] = opt_index
        value_function.cur[j] = opt_val
    end

    push!(policy_cache.policy, copy(policy_cache.cur_policy))

    return value_function, policy_cache
end

function extract_policy!(
    value_function::IMDPValueFunction,
    policy_cache::StationaryPolicyCache,
    stateptr::VT,
    maximize,
) where {VT <: AbstractVector}
    gt = maximize ? (>) : (<)

    @inbounds for j in 1:(length(stateptr) - 1)
        s1 = stateptr[j]
        s2 = stateptr[j + 1]

        if iszero(policy_cache.policy[j])
            opt_val = value_function.action_values[s1]
            opt_index = s1
        else
            opt_val = value_function.prev[j]
            opt_index = policy_cache.policy[j]
        end

        for i in s1:(s2 - 1)
            if gt(value_function.action_values[i], opt_val)
                opt_val = value_function.action_values[i]
                opt_index = i
            end
        end

        policy_cache.policy[j] = opt_index
        value_function.cur[j] = opt_val
    end

    return value_function, policy_cache
end
