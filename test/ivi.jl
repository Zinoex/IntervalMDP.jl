# Vector of vectors
prob1 = StateIntervalProbabilities(; lower = [0.0, 0.1, 0.2], upper = [0.5, 0.6, 0.7])
prob2 = StateIntervalProbabilities(; lower = [0.5, 0.3, 0.1], upper = [0.7, 0.5, 0.3])
prob3 = StateIntervalProbabilities(; lower = [0.0, 0.0, 1.0], upper = [0.0, 0.0, 1.0])
prob = [prob1, prob2, prob3]

mc = IntervalMarkovChain(prob, 1)

problem = Problem(mc, FiniteTimeReachability([3], 10))
V_fixed_it, k, _ = interval_value_iteration(problem; upper_bound = false)
@test k == 10

problem = Problem(mc, InfiniteTimeReachability([3], 1e-6))
V_conv, _, u = interval_value_iteration(problem; upper_bound = false)
@test maximum(u) <= 1e-6

# Matrix
prob = MatrixIntervalProbabilities(;
    lower = [
        0.0 0.5 0.0;
        0.1 0.3 0.0;
        0.2 0.1 1.0
    ],
    upper = [
        0.5 0.7 0.0;
        0.6 0.5 0.0;
        0.7 0.3 1.0
    ],
)
mc = IntervalMarkovChain(prob, 1)

problem = Problem(mc, FiniteTimeReachability([3], 10))
V_fixed_it, k, _ = interval_value_iteration(problem; upper_bound = false)
@test k == 10

problem = Problem(mc, InfiniteTimeReachability([3], 1e-6))
V_conv, _, u = interval_value_iteration(problem; upper_bound = false)
@test maximum(u) <= 1e-6
