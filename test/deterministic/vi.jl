prob = transition_hcat(3, [2, 3], [1, 2], [3])

mc = DeterministicMarkovChain(prob, [1])

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

prop = FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10)
spec = Specification(prop, Pessimistic)
problem = Problem(mc, spec)
V_fixed_it, k, _ = value_iteration(problem)
@test k == 10
