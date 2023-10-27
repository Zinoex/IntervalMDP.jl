
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

function satisfaction_probability(problem::Problem{<:IntervalMarkovChain, FiniteTimeReachability})
    k_max = time_horizon(specification(problem))
    criteria = FixedIterationsCriteria(k_max)

    maximize = satisfaction_mode(problem) == Pessimistic
    V, k, _ = interval_value_iteration(problem, criteria; max = maximize)

    @assert k == k_max
    sat_prob = V[initial_state(system(problem))]

    return sat_prob
end

function satisfaction_probability(problem::Problem{<:IntervalMarkovChain, InfiniteTimeReachability})
    convergence_limit = eps(specification(problem))
    criteria = ConvergenceCriteria(convergence_limit)

    maximize = satisfaction_mode(problem) == Pessimistic
    V, _, u = interval_value_iteration(problem, criteria; maximize)

    @assert maximum(u) < convergence_limit
    sat_prob = V[initial_state(system(problem))]

    return sat_prob
end