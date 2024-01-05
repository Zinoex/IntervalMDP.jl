
function satisfaction_probability(
    problem::Problem{S, <:Specification{<:LTLfFormula}},
) where {S <: IntervalMarkovProcess}
    spec = specification(problem)
    prod_system, terminal_states = product_system(problem)

    new_spec = FiniteTimeReachability(
        terminal_states,
        num_states(product_system),
        time_horizon(spec),
    )
    problem = Problem(prod_system, new_spec, satisfaction_mode(problem))
    return satisfaction_probability(problem)
end

"""
    satisfaction_probability(problem::Problem{<:IntervalMarkovProcess, <:AbstractReachability})

Compute the probability of satisfying the reachability-like specification from the initial state.
"""
function satisfaction_probability(
    problem::Problem{<:IntervalMarkovProcess, <:AbstractReachability},
)
    upper_bound = satisfaction_mode(problem) == Optimistic
    V, _, _ = value_iteration(problem; upper_bound = upper_bound)
    V = Vector(V)   # Convert to CPU vector if not already

    sys = system(problem)
    sat_prob = V[initial_states(sys)]

    return sat_prob
end
