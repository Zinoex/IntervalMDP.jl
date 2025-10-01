using Revise, Test
using IntervalMDP

@testset for N in [Float32, Float64, Rational{BigInt}]
    prob = IntervalAmbiguitySets(;
        lower = N[
            0 1//2 0
            1//10 3//10 0
            1//5 1//10 1
        ],
        upper = N[
            1//2 7//10 0
            3//5 1//2 0
            7//10 3//10 1
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
    @test all(V_fixed_it .>= N(0))
    @test all(V_fixed_it .<= N(1))

    @test value_function(sol) == V_fixed_it
    @test num_iterations(sol) == k
    @test residual(sol) == res

    prop = FiniteTimeReachability([3], 11)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it2, k, _ = solve(problem)
    @test k == 11
    @test all(V_fixed_it .<= V_fixed_it2)

    prop = InfiniteTimeReachability([3], N(1//1_000_000))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_conv, _, u = solve(problem)
    @test maximum(u) <= N(1//1_000_000)
    @test all(V_conv .>= N(0))
    @test all(V_conv .<= N(1))

    prop = FiniteTimeReachAvoid([3], [2], 10)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it, k, _ = solve(problem)
    @test k == 10
    @test all(V_fixed_it .>= N(0))
    @test all(V_fixed_it .<= N(1))

    prop = FiniteTimeReachAvoid([3], [2], 11)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it2, k, _ = solve(problem)
    @test k == 11
    @test all(V_fixed_it2 .>= N(0))
    @test all(V_fixed_it2 .<= N(1))
    @test all(V_fixed_it .<= V_fixed_it2)

    prop = InfiniteTimeReachAvoid([3], [2], N(1//1_000_000))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_conv, _, u = solve(problem)
    @test maximum(u) <= N(1//1_000_000)
    @test all(V_conv .>= N(0))
    @test all(V_conv .<= N(1))

    prop = FiniteTimeSafety([3], 10)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it, k, _ = solve(problem)
    @test k == 10
    @test all(V_fixed_it .>= N(0))
    @test all(V_fixed_it .<= N(1))

    prop = FiniteTimeSafety([3], 11)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it2, k, _ = solve(problem)
    @test k == 11
    @test all(V_fixed_it2 .>= N(0))
    @test all(V_fixed_it2 .<= N(1))
    @test all(V_fixed_it2 .<= V_fixed_it)

    prop = InfiniteTimeSafety([3], N(1//1_000_000))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_conv, _, u = solve(problem)
    @test maximum(u) <= N(1//1_000_000)
    @test all(V_conv .>= N(0))
    @test all(V_conv .<= N(1))

    prop = FiniteTimeReward(N[2, 1, 0], N(9//10), 10)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it, k, _ = solve(problem)
    @test k == 10
    @test all(V_fixed_it .>= N(0))

    prop = FiniteTimeReward(N[2, 1, -1], N(9//10), 10)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it2, k, _ = solve(problem)
    @test k == 10
    @test all(V_fixed_it2 .<= V_fixed_it)

    prop = InfiniteTimeReward(N[2, 1, 0], N(9//10), N(1//1_000_000))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_conv, _, u = solve(problem)
    @test maximum(u) <= N(1//1_000_000)
    @test all(V_conv .>= N(0))

    prop = InfiniteTimeReward(N[2, 1, -1], N(9//10), N(1//1_000_000))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_conv, _, u = solve(problem)
    @test maximum(u) <= N(1//1_000_000)
end
