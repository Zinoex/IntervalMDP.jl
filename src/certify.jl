
function satisfaction_probability(s::System, f::Specification, mode::SatisfactionMode = Pessimistic)
    return satisfaction_probability(Problem(s, f, mode))
end

function satisfaction_probability(problem::Problem{S, LTLfFormula}) where {S <: System}
    spec = specification(problem)
    prod_system, terminal_states = product_system(problem)

    new_spec = FiniteTimeReachability(terminal_states, num_states(product_system), time_horizon(spec))
    problem = Problem(prod_system, new_spec, satisfaction_mode(problem))
    return satisfaction_probability(problem)
end

function satisfaction_probability(problem::Problem{<:IntervalMarkovChain, <:AbstractReachability})
    upper_bound = satisfaction_mode(problem) == Optimistic
    V, _, _ = interval_value_iteration(problem; upper_bound = upper_bound)

    sat_prob = V[initial_state(system(problem))]

    return sat_prob
end