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

problem = Problem(mc, FiniteTimeReachability([3], 3, 10))
V_fixed_it, k, _ = interval_value_iteration(problem; upper_bound = false)
@test k == 10

problem = Problem(mc, InfiniteTimeReachability([3], 3, 1e-6))
V_conv, _, u = interval_value_iteration(problem; upper_bound = false)
@test maximum(u) <= 1e-6

problem = Problem(mc, FiniteTimeReachAvoid([3], [2], 3, 10))
V_fixed_it, k, _ = interval_value_iteration(problem; upper_bound = false)
@test k == 10

problem = Problem(mc, InfiniteTimeReachAvoid([3], [2], 3, 1e-6))
V_conv, _, u = interval_value_iteration(problem; upper_bound = false)
@test maximum(u) <= 1e-6
