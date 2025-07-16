using Revise, Test
using IntervalMDP

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

mc = IntervalMarkovChain(prob, [1])
@test initial_states(mc) == [1]

mc = IntervalMarkovChain(prob)

prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
sol = solve(problem)
V_fixed_it, k, res = sol
@test k == 10
@test all(V_fixed_it .>= 0.0)

@test value_function(sol) == V_fixed_it
@test num_iterations(sol) == k
@test residual(sol) == res

prop = FiniteTimeReachability([3], 11)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_fixed_it2, k, _ = solve(problem)
@test k == 11
@test all(V_fixed_it .<= V_fixed_it2)

prop = InfiniteTimeReachability([3], 1e-6)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_conv, _, u = solve(problem)
@test maximum(u) <= 1e-6
@test all(V_conv .>= 0.0)

prop = FiniteTimeReachAvoid([3], [2], 10)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_fixed_it, k, _ = solve(problem)
@test k == 10
@test all(V_fixed_it .>= 0.0)

prop = FiniteTimeReachAvoid([3], [2], 11)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_fixed_it2, k, _ = solve(problem)
@test k == 11
@test all(V_fixed_it2 .>= 0.0)
@test all(V_fixed_it .<= V_fixed_it2)

prop = InfiniteTimeReachAvoid([3], [2], 1e-6)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_conv, _, u = solve(problem)
@test maximum(u) <= 1e-6
@test all(V_conv .>= 0.0)

prop = FiniteTimeSafety([3], 10)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_fixed_it, k, _ = solve(problem)
@test k == 10
@test all(V_fixed_it .>= 0.0)

prop = FiniteTimeSafety([3], 11)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_fixed_it2, k, _ = solve(problem)
@test k == 11
@test all(V_fixed_it2 .>= 0.0)
@test all(V_fixed_it2 .<= V_fixed_it)

prop = InfiniteTimeSafety([3], 1e-6)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_conv, _, u = solve(problem)
@test maximum(u) <= 1e-6
@test all(V_conv .>= 0.0)

prop = FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_fixed_it, k, _ = solve(problem)
@test k == 10
@test all(V_fixed_it .>= 0.0)

prop = FiniteTimeReward([2.0, 1.0, -1.0], 0.9, 10)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_fixed_it2, k, _ = solve(problem)
@test k == 10
@test all(V_fixed_it2 .<= V_fixed_it)

prop = InfiniteTimeReward([2.0, 1.0, 0.0], 0.9, 1e-6)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_conv, _, u = solve(problem)
@test maximum(u) <= 1e-6
@test all(V_conv .>= 0.0)

prop = InfiniteTimeReward([2.0, 1.0, -1.0], 0.9, 1e-6)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_conv, _, u = solve(problem)
@test maximum(u) <= 1e-6
