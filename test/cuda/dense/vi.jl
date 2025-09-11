using Revise, Test
using IntervalMDP, CUDA


@testset for N in [Float32, Float64]
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

    mc = IntervalMDP.cu(IntervalMarkovChain(prob, [1]))
    @test IntervalMDP.cpu(initial_states(mc)) == [1]

    mc = IntervalMDP.cu(IntervalMarkovChain(prob))
    mc_cpu = IntervalMarkovChain(prob)  # For comparison

    prop = FiniteTimeReachability([3], 10)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    sol = solve(problem)
    V_fixed_it, k, res = sol

    @test value_function(sol) == V_fixed_it
    @test num_iterations(sol) == k
    @test residual(sol) == res

    V_fixed_it = IntervalMDP.cpu(V_fixed_it)  # Convert to CPU for testing
    @test k == 10
    @test all(V_fixed_it .>= N(0))
    @test all(V_fixed_it .<= N(1))

    problem = VerificationProblem(mc_cpu, spec)
    V_fixed_it_cpu, k_cpu, res_cpu = solve(problem)
    @test k == k_cpu
    @test V_fixed_it ≈ V_fixed_it_cpu atol=1e-5
    @test IntervalMDP.cpu(res) ≈ res_cpu atol=1e-5

    prop = FiniteTimeReachability([3], 11)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it2, k, res = solve(problem)
    V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)  # Convert to CPU for testing
    @test k == 11
    @test all(V_fixed_it .<= V_fixed_it2)

    problem = VerificationProblem(mc_cpu, spec)
    V_fixed_it_cpu, k_cpu, res_cpu = solve(problem)
    @test k == k_cpu
    @test V_fixed_it2 ≈ V_fixed_it_cpu atol=1e-5
    @test IntervalMDP.cpu(res) ≈ res_cpu atol=1e-5

    prop = InfiniteTimeReachability([3], N(1//1_000_000))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_conv, _, u = solve(problem)
    V_conv = IntervalMDP.cpu(V_conv)  # Convert to CPU for testing
    @test maximum(u) <= N(1//1_000_000)
    @test all(V_conv .>= N(0))
    @test all(V_conv .<= N(1))

    prop = FiniteTimeReachAvoid([3], [2], 10)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it, k, res = solve(problem)
    V_fixed_it = IntervalMDP.cpu(V_fixed_it)  # Convert to CPU for testing
    @test k == 10
    @test all(V_fixed_it .>= N(0))
    @test all(V_fixed_it .<= N(1))

    problem = VerificationProblem(mc_cpu, spec)
    V_fixed_it_cpu, k_cpu, res_cpu = solve(problem)
    @test k == k_cpu
    @test V_fixed_it ≈ V_fixed_it_cpu atol=1e-5
    @test IntervalMDP.cpu(res) ≈ res_cpu atol=1e-5

    prop = FiniteTimeReachAvoid([3], [2], 11)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it2, k, res = solve(problem)
    V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)  # Convert to CPU for testing
    @test k == 11
    @test all(V_fixed_it2 .>= N(0))
    @test all(V_fixed_it2 .<= N(1))
    @test all(V_fixed_it .<= V_fixed_it2)

    problem = VerificationProblem(mc_cpu, spec)
    V_fixed_it_cpu, k_cpu, res_cpu = solve(problem)
    @test k == k_cpu
    @test V_fixed_it2 ≈ V_fixed_it_cpu atol=1e-5
    @test IntervalMDP.cpu(res) ≈ res_cpu atol=1e-5

    prop = InfiniteTimeReachAvoid([3], [2], N(1//1_000_000))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_conv, _, u = solve(problem)
    V_conv = IntervalMDP.cpu(V_conv)  # Convert to CPU for testing
    @test maximum(u) <= N(1//1_000_000)
    @test all(V_conv .>= N(0))
    @test all(V_conv .<= N(1))

    prop = FiniteTimeSafety([3], 10)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it, k, res = solve(problem)
    V_fixed_it = IntervalMDP.cpu(V_fixed_it)  # Convert to CPU for testing
    @test k == 10
    @test all(V_fixed_it .>= N(0))
    @test all(V_fixed_it .<= N(1))

    problem = VerificationProblem(mc_cpu, spec)
    V_fixed_it_cpu, k_cpu, res_cpu = solve(problem)
    @test k == k_cpu
    @test V_fixed_it ≈ V_fixed_it_cpu atol=1e-5
    @test IntervalMDP.cpu(res) ≈ res_cpu atol=1e-5

    prop = FiniteTimeSafety([3], 11)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it2, k, res = solve(problem)
    V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)  # Convert to CPU for testing
    @test k == 11
    @test all(V_fixed_it2 .>= N(0))
    @test all(V_fixed_it2 .<= N(1))
    @test all(V_fixed_it2 .<= V_fixed_it)

    problem = VerificationProblem(mc_cpu, spec)
    V_fixed_it_cpu, k_cpu, res_cpu = solve(problem)
    @test k == k_cpu
    @test V_fixed_it2 ≈ V_fixed_it_cpu atol=1e-5
    @test IntervalMDP.cpu(res) ≈ res_cpu atol=1e-5

    prop = InfiniteTimeSafety([3], N(1//1_000_000))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_conv, _, u = solve(problem)
    V_conv = IntervalMDP.cpu(V_conv)  # Convert to CPU for testing
    @test maximum(u) <= N(1//1_000_000)
    @test all(V_conv .>= N(0))
    @test all(V_conv .<= N(1))

    prop = IntervalMDP.cu(FiniteTimeReward(N[2, 1, 0], N(9//10), 10))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it, k, res = solve(problem)
    V_fixed_it = IntervalMDP.cpu(V_fixed_it)  # Convert to CPU for testing
    @test k == 10
    @test all(V_fixed_it .>= N(0))

    prop = FiniteTimeReward(N[2, 1, 0], N(9//10), 10)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc_cpu, spec)
    V_fixed_it_cpu, k_cpu, res_cpu = solve(problem)
    @test k == k_cpu
    @test V_fixed_it ≈ V_fixed_it_cpu atol=1e-5
    @test IntervalMDP.cpu(res) ≈ res_cpu atol=1e-5

    prop = IntervalMDP.cu(FiniteTimeReward(N[2, 1, -1], N(9//10), 10))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_fixed_it2, k, res = solve(problem)
    V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)  # Convert to CPU for testing
    @test k == 10
    @test all(V_fixed_it2 .<= V_fixed_it)

    prop = FiniteTimeReward(N[2, 1, -1], N(9//10), 10)
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc_cpu, spec)
    V_fixed_it_cpu, k_cpu, res_cpu = solve(problem)
    @test k == k_cpu
    @test V_fixed_it2 ≈ V_fixed_it_cpu atol=1e-5
    @test IntervalMDP.cpu(res) ≈ res_cpu atol=1e-5

    prop = IntervalMDP.cu(InfiniteTimeReward(N[2, 1, 0], N(9//10), N(1//1_000_000)))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_conv, _, u = solve(problem)
    V_conv = IntervalMDP.cpu(V_conv)  # Convert to CPU for testing
    @test maximum(u) <= N(1//1_000_000)
    @test all(V_conv .>= N(0))

    prop = IntervalMDP.cu(InfiniteTimeReward(N[2, 1, -1], N(9//10), N(1//1_000_000)))
    spec = Specification(prop, Pessimistic)
    problem = VerificationProblem(mc, spec)
    V_conv, _, u = solve(problem)
    V_conv = IntervalMDP.cpu(V_conv)  # Convert to CPU for testing
    @test maximum(u) <= N(1//1_000_000)
end