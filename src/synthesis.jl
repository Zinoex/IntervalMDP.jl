
function control_synthesis(problem::Problem{<:IntervalMarkovDecisionProcess}; maximize = true)
    spec = specification(problem)
    return control_synthesis(problem, Val(isfinitetime(spec)); maximize = maximize)
end

function control_synthesis(problem::Problem{<:IntervalMarkovDecisionProcess}, time_varying::Val{true}; maximize = true)
    upper_bound = satisfaction_mode(problem) == Optimistic
    policy_indices = extract_time_varying_policy(problem; maximize = maximize, upper_bound = upper_bound)

    act = actions(system(problem))

    return [act[indices] for indices in policy_indices]
end

function extract_time_varying_policy(problem::Problem{<:IntervalMarkovDecisionProcess, <:AbstractReachability}; maximize, upper_bound)
    mdp = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(problem)

    prob = transition_prob(mdp)
    maxactions = maximum(diff(stateptr(mdp)))
    target = reach(spec)

    # It is more efficient to use allocate first and reuse across iterations
    p = deepcopy(gap(prob))  # Deep copy as it may be a vector of vectors and we need sparse arrays to store the same indices
    ordering = construct_ordering(p)

    value_function = IMDPValueFunction(problem)
    initialize!(value_function, target, 1.0)

    policy = Vector{Vector{Int}}(undef, 1)

    step_policy = step_imdp_with_extract!(
        ordering,
        p,
        mdp,
        prob,
        maxactions,
        value_function;
        maximize = maximize,
        upper_bound = upper_bound,
    )
    policy[1] = step_policy
    k = 1

    while !term_criteria(value_function.cur, k, lastdiff!(value_function))
        nextiteration!(value_function)
        step_policy = step_imdp_with_extract!(
            ordering,
            p,
            mdp,
            prob,
            maxactions,
            value_function;
            maximize = maximize,
            upper_bound = upper_bound,
        )
        push!(policy, step_policy)
        k += 1
    end

    # The policy vector is built starting from the target state
    # at the final time step, so we need to reverse it.
    return reverse(policy)
end

function extract_time_varying_policy(problem::Problem{<:IntervalMarkovDecisionProcess, <:AbstractReward}; maximize, upper_bound)
    mdp = system(problem)
    spec = specification(problem)
    term_criteria = termination_criteria(problem)

    prob = transition_prob(mdp)
    maxactions = maximum(diff(stateptr(mdp)))

    # It is more efficient to use allocate first and reuse across iterations
    p = deepcopy(gap(prob))  # Deep copy as it may be a vector of vectors and we need sparse arrays to store the same indices
    ordering = construct_ordering(p)

    value_function = IMDPValueFunction(problem)
    initialize!(value_function, 1:num_states(mdp), reward(spec))

    policy = Vector{Vector{Int}}(undef, 1)

    step_policy = step_imdp_with_extract!(
        ordering,
        p,
        mdp,
        prob,
        maxactions,
        value_function;
        maximize = maximize,
        upper_bound = upper_bound,
        discount = discount(spec),
    )
    # Add immediate reward
    value_function.cur += reward(spec)
    policy[1] = step_policy
    k = 1

    while !term_criteria(value_function.cur, k, lastdiff!(value_function))
        nextiteration!(value_function)
        step_policy = step_imdp_with_extract!(
            ordering,
            p,
            mdp,
            prob,
            maxactions,
            value_function;
            maximize = maximize,
            upper_bound = upper_bound,
            discount = discount(spec),
        )
        # Add immediate reward
        value_function.cur += reward(spec)
        push!(policy, step_policy)
        k += 1
    end

    # The policy vector is built starting from the target state
    # at the final time step, so we need to reverse it.
    return reverse(policy)
end

function step_imdp_with_extract!(
    ordering,
    p,
    mdp,
    prob::IntervalProbabilities,
    maxactions,
    value_function;
    maximize,
    upper_bound,
    discount = 1.0,
)
    sptr = stateptr(mdp)

    partial_ominmax!(
        ordering,
        p,
        prob,
        value_function.prev,
        value_function.nonterminal_actions;
        max = upper_bound,
    )

    optfun = maximize ? argmax : argmin

    p = view(p, :, value_function.nonterminal_actions)
    mul!(value_function.nonterminal, value_function.prev_transpose, p)
    rmul!(value_function.nonterminal, discount)

    # Copy to ensure that terminal states are assigned their first (and only) action.
    indices = sptr[1:num_states(mdp)]

    @inbounds for j in value_function.nonterminal_states
        @inbounds s1 = sptr[j]
        @inbounds s2 = sptr[j + 1]

        @inbounds indices[j] = optfun(view(value_function.nonterminal, s1:(s2 - 1))) + s1 - 1
        @inbounds value_function.cur[j] = value_function.nonterminal[indices[j]]
    end

    return indices
end

function control_synthesis(problem::Problem{<:IntervalMarkovDecisionProcess}, time_varying::Val{false}; maximize = true)
    sys = system(problem)

    upper_bound = satisfaction_mode(problem) == Optimistic
    V, _, _ = value_iteration(problem; maximize = maximize, upper_bound = upper_bound)

    indices = extract_stationary_policy(sys, V; maximize = maximize, upper_bound = upper_bound)

    return actions(sys)[indices]
end

function extract_stationary_policy(system::IntervalMarkovDecisionProcess, V; maximize, upper_bound)
    sptr = stateptr(system)
    p = ominmax(transition_prob(system), V; max = upper_bound)

    optfun = maximize ? argmax : argmin

    V = transpose(transpose(V) * p)

    indices = Vector{Int}(undef, num_states(system))

    @inbounds for j in 1:num_states(system)
        @inbounds s1 = sptr[j]
        @inbounds s2 = sptr[j + 1]

        # Offset into action_vals array
        @inbounds indices[j] = optfun(view(V, s1:(s2 - 1))) + s1 - 1
    end

    return indices
end