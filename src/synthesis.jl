"""
    control_synthesis(problem::Problem{<:IntervalMarkovDecisionProcess})

Compute the optimal control policy for the given problem (system + specification). If the specification is finite time, then the policy is time-varying,
with the returned policy being in step order (i.e., the first element of the returned vector is the policy for the first time step).
If the specification is infinite time, then the policy is stationary and only a single vector of length `num_states(system)` is returned.
"""
function control_synthesis(problem::Problem{<:IntervalMarkovDecisionProcess})
    spec = specification(problem)
    prop = system_property(spec)

    policy_cache = create_policy_cache(problem, Val(isfinitetime(prop)))
    V, k, res = value_iteration!(policy_cache, problem)

    return policy_indices_to_actions(problem, policy_cache), V, k, res
end
