using Revise, Test
using IntervalMDP, SparseArrays, CUDA

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

mc = IntervalMDP.cu(IntervalMarkovChain(prob, [1]))

prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic)
problem = Problem(mc, spec)
V_fixed_it, k, _ = value_iteration(problem)
@test k == 10

prop = FiniteTimeReachability([3], 11)
spec = Specification(prop, Pessimistic)
problem = Problem(mc, spec)
V_fixed_it2, k, _ = value_iteration(problem)
@test k == 11
@test all(V_fixed_it .<= V_fixed_it2)

prop = InfiniteTimeReachability([3], 1e-6)
spec = Specification(prop, Pessimistic)
problem = Problem(mc, spec)
V_conv, _, u = value_iteration(problem)
@test maximum(u) <= 1e-6

prop = FiniteTimeReachAvoid([3], [2], 10)
spec = Specification(prop, Pessimistic)
problem = Problem(mc, spec)
V_fixed_it, k, _ = value_iteration(problem)
@test k == 10

prop = InfiniteTimeReachAvoid([3], [2], 1e-6)
spec = Specification(prop, Pessimistic)
problem = Problem(mc, spec)
V_conv, _, u = value_iteration(problem)
@test maximum(u) <= 1e-6

prop = IntervalMDP.cu(FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10))
spec = Specification(prop, Pessimistic)
problem = Problem(mc, spec)
V_fixed_it, k, _ = value_iteration(problem)
@test k == 10

# problem = Problem(mc, InfiniteTimeReward([2.0, 1.0, 0.0], 0.9, 1e-6))
# V_conv, _, u = value_iteration(problem)
# @test maximum(u) <= 1e-6
