using Revise, Test
using IntervalMDP, SparseArrays

prob = IntervalProbabilities(;
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

mc = IntervalMarkovChain(prob, [1])

prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic)
problem = VerificationProblem(mc, spec)
V_fixed_it, k, _ = solve(problem)
@test k == 10
@test all(V_fixed_it .>= 0.0)

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
