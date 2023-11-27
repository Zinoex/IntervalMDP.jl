
function control_synthesis(problem::Problem{<:IntervalMarkovDecisionProcess}; maximize = true)
    spec = specification(problem)
    return control_synthesis(problem, Val(isfinitetime(spec)); maximize = maximize)
end

function control_synthesis(problem::Problem{<:IntervalMarkovDecisionProcess}, time_varying::Val{true}; maximize = true)
    spec = specification(problem)
end

function control_synthesis(problem::Problem{<:IntervalMarkovDecisionProcess}, time_varying::Val{false}; maximize = true)
    sys = system(problem)
    spec = specification(problem)

    upper_bound = satisfaction_mode(problem) == Optimistic
    V, _, _ = value_iteration(problem; maximize = maximize, upper_bound = upper_bound)

    indices = extract_stationary_policy(sys, V; maximize = maximize, upper_bound = upper_bound)

    return actions(sys)[indices]
end

function extract_stationary_policy(system::IntervalMarkovDecisionProcess, V; maximize, upper_bound)
    sp = stateptr(system)
    p = ominmax(transition_prob(system), V; max = upper_bound)

    optfun = maximize ? argmax : argmin

    V = transpose(transpose(V) * p)

    indices = Vector{Int}(undef, num_states(system))

    @inbounds for j in 1:num_states(system)
        @inbounds s1 = sp[j]
        @inbounds s2 = sp[j + 1]

        # Offset into action_vals array
        @inbounds indices[j] = optfun(view(V, s1:(s2 - 1))) + s1 - 1
    end

    return indices
end