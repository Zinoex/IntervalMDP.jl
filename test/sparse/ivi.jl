
# Vector of vectors
prob1 = StateIntervalProbabilities(;
    lower = SparseVector(3, [2, 3], [0.1, 0.2]),
    upper = SparseVector(3, [1, 2, 3], [0.5, 0.6, 0.7]),
)
prob2 = StateIntervalProbabilities(;
    lower = SparseVector(3, [1, 2, 3], [0.5, 0.3, 0.1]),
    upper = SparseVector(3, [1, 2, 3], [0.7, 0.5, 0.3]),
)
prob3 = StateIntervalProbabilities(;
    lower = SparseVector(3, [3], [1.0]),
    upper = SparseVector(3, [3], [1.0]),
)
prob = [prob1, prob2, prob3]

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

# Matrix
prob = MatrixIntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(3, [2, 3], [0.1, 0.2]),
        SparseVector(3, [1, 2, 3], [0.5, 0.3, 0.1]),
        SparseVector(3, [3], [1.0]),
    ),
    upper = sparse_hcat(
        SparseVector(3, [1, 2, 3], [0.5, 0.6, 0.7]),
        SparseVector(3, [1, 2, 3], [0.7, 0.5, 0.3]),
        SparseVector(3, [3], [1.0]),
    ),
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