@testmodule ImdpModels begin
    using IntervalMDP

    function build(N)
        prob1 = IntervalAmbiguitySets(;
            lower = N[
                0 1//2
                1//10 3//10
                1//5 1//10
            ],
            upper = N[
                1//2 7//10
                3//5 1//2
                7//10 3//10
            ],
        )

        prob2 = IntervalAmbiguitySets(;
            lower = N[
                1//10 1//5
                1//5 3//10
                3//10 2//5
            ],
            upper = N[
                3//5 3//5
                1//2 1//2
                2//5 2//5
            ],
        )

        prob3 = IntervalAmbiguitySets(; lower = N[
            0 0
            0 0
            1 1
        ], upper = N[
            0 0
            0 0
            1 1
        ])

        transition_probs = [prob1, prob2, prob3]

        mdp = IntervalMarkovDecisionProcess(transition_probs)
        implicit_mdp = IntervalMarkovDecisionProcess([prob1, prob2])

        return (; prob1, prob2, prob3, transition_probs, mdp, implicit_mdp)
    end
end

@testitem "base/imdp: construction" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; transition_probs) = ImdpModels.build(N)
        istates = [1]

        mdp = IntervalMarkovDecisionProcess(transition_probs, istates)
        @test initial_states(mdp) == istates
    end
end

@testitem "base/imdp: bellman" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp) = ImdpModels.build(N)

        V = N[1, 2, 3]

        Vres = IntervalMDP.bellman(V, mdp; upper_bound = false, maximize = true)
        @test Vres ≈ N[
            (1 // 2) * 1 + (3 // 10) * 2 + (1 // 5) * 3,
            (3 // 10) * 1 + (3 // 10) * 2 + (2 // 5) * 3,
            1 * 3,
        ]

        Vres = similar(Vres)
        IntervalMDP.bellman!(Vres, V, mdp; upper_bound = false, maximize = true)
        @test Vres ≈ N[
            (1 // 2) * 1 + (3 // 10) * 2 + (1 // 5) * 3,
            (3 // 10) * 1 + (3 // 10) * 2 + (2 // 5) * 3,
            1 * 3,
        ]

        Vres = IntervalMDP.bellman(V, mdp; upper_bound = true, maximize = false)
        @test Vres ≈ N[
            (1 // 2) * 1 + (3 // 10) * 2 + (1 // 5) * 3,
            (1 // 5) * 1 + (2 // 5) * 2 + (2 // 5) * 3,
            1 * 3,
        ]

        Vres = similar(Vres)
        IntervalMDP.bellman!(Vres, V, mdp; upper_bound = true, maximize = false)
        @test Vres ≈ N[
            (1 // 2) * 1 + (3 // 10) * 2 + (1 // 5) * 3,
            (1 // 5) * 1 + (2 // 5) * 2 + (2 // 5) * 3,
            1 * 3,
        ]
    end
end

@testitem "base/imdp: explicit sink — IntervalMarkovChain dimension mismatch" setup =
    [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; transition_probs) = ImdpModels.build(N)

        transition_prob = IntervalMDP.interval_prob_hcat(transition_probs)
        @test_throws DimensionMismatch IntervalMarkovChain(transition_prob)
    end
end

@testitem "base/imdp: explicit sink — finite time reachability" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp) = ImdpModels.build(N)

        prop = FiniteTimeReachability([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= N(0))
        @test V_fixed_it1[3] == N(1)

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
        @test V_fixed_it2[3] == N(1)

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= N(0))
        @test V_fixed_it1[3] == N(1)

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
        @test V_fixed_it2[3] == N(1)
    end
end

@testitem "base/imdp: explicit sink — infinite time reachability" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp) = ImdpModels.build(N)

        prop = InfiniteTimeReachability([3], N(1//1_000_000))
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv, _, u = solve(problem)
        @test maximum(u) <= N(1//1_000_000)
        @test all(V_conv .>= N(0))
        @test V_conv[3] == N(1)
    end
end

@testitem "base/imdp: explicit sink — exact time reachability" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp) = ImdpModels.build(N)

        prop = ExactTimeReachability([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= N(0))
        @test all(V_fixed_it1 .<= N(1))

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= N(0))
        @test all(V_fixed_it1 .<= N(1))

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)

        # Compare exact time to finite time
        prop = ExactTimeReachability([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10

        prop = FiniteTimeReachability([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
    end
end

@testitem "base/imdp: explicit sink — finite time reach/avoid" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp) = ImdpModels.build(N)

        prop = FiniteTimeReachAvoid([3], [2], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= N(0))
        @test all(V_fixed_it1 .<= N(1))
        @test V_fixed_it1[3] == N(1)
        @test V_fixed_it1[2] == N(0)

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
        @test V_fixed_it2[3] == N(1)
        @test V_fixed_it2[2] == N(0)

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= N(0))
        @test all(V_fixed_it1 .<= N(1))
        @test V_fixed_it1[3] == N(1)
        @test V_fixed_it1[2] == N(0)

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
        @test V_fixed_it2[3] == N(1)
        @test V_fixed_it2[2] == N(0)
    end
end

@testitem "base/imdp: explicit sink — infinite time reach/avoid" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp) = ImdpModels.build(N)

        prop = InfiniteTimeReachAvoid([3], [2], N(1//1_000_000))
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv, _, u = solve(problem)
        @test maximum(u) <= N(1//1_000_000)
        @test all(V_conv .>= N(0))
        @test all(V_conv .<= N(1))
        @test V_conv[3] == N(1)
        @test V_conv[2] == N(0)
    end
end

@testitem "base/imdp: explicit sink — exact time reach/avoid" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp) = ImdpModels.build(N)

        prop = ExactTimeReachAvoid([3], [2], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= N(0))
        @test all(V_fixed_it1 .<= N(1))
        @test V_fixed_it1[2] == N(0)

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
        @test V_fixed_it2[2] == N(0)

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= N(0))
        @test V_fixed_it1[2] == N(0)

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
        @test V_fixed_it2[2] == N(0)

        # Compare exact time to finite time
        prop = ExactTimeReachAvoid([3], [2], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10

        prop = FiniteTimeReachAvoid([3], [2], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
    end
end

@testitem "base/imdp: explicit sink — finite time reward" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp) = ImdpModels.build(N)

        prop = FiniteTimeReward(N[2, 1, 0], N(9//10), 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= N(0))

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= N(0))

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
    end
end

@testitem "base/imdp: explicit sink — infinite time reward" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp) = ImdpModels.build(N)

        prop = InfiniteTimeReward(N[2, 1, 0], N(9//10), N(1//1_000_000))
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv, _, u = solve(problem)
        @test maximum(u) <= N(1//1_000_000)
        @test all(V_conv .>= N(0))
    end
end

@testitem "base/imdp: explicit sink — expected exit time" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp) = ImdpModels.build(N)

        prop = ExpectedExitTime([3], N(1//1_000_000))

        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv1, _, u = solve(problem)
        @test maximum(u) <= N(1//1_000_000)
        @test all(V_conv1 .>= N(0))
        @test V_conv1[3] == N(0)

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv2, _, u = solve(problem)
        @test maximum(u) <= N(1//1_000_000)
        @test all(V_conv1 .<= V_conv2)
        @test V_conv2[3] == N(0)

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_conv1, _, u = solve(problem)
        @test maximum(u) <= N(1//1_000_000)
        @test all(V_conv1 .>= N(0))
        @test V_conv1[3] == N(0)

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_conv2, _, u = solve(problem)
        @test maximum(u) <= N(1//1_000_000)
        @test all(V_conv1 .<= V_conv2)
        @test V_conv2[3] == N(0)
    end
end

@testitem "base/imdp: implicit sink — finite time reachability" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp, implicit_mdp) = ImdpModels.build(N)

        prop = FiniteTimeReachability([3], 10)

        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end
end

@testitem "base/imdp: implicit sink — infinite time reachability" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp, implicit_mdp) = ImdpModels.build(N)

        prop = InfiniteTimeReachability([3], N(1//1_000_000))
        spec = Specification(prop, Pessimistic, Maximize)

        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end
end

@testitem "base/imdp: implicit sink — exact time reachability" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp, implicit_mdp) = ImdpModels.build(N)

        prop = ExactTimeReachability([3], 10)

        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end
end

@testitem "base/imdp: implicit sink — finite time reach/avoid" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp, implicit_mdp) = ImdpModels.build(N)

        prop = FiniteTimeReachAvoid([3], [2], 10)

        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end
end

@testitem "base/imdp: implicit sink — infinite time reach/avoid" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp, implicit_mdp) = ImdpModels.build(N)

        prop = InfiniteTimeReachAvoid([3], [2], N(1//1_000_000))
        spec = Specification(prop, Pessimistic, Maximize)

        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end
end

@testitem "base/imdp: implicit sink — exact time reach/avoid" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp, implicit_mdp) = ImdpModels.build(N)

        prop = ExactTimeReachAvoid([3], [2], 10)

        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end
end

@testitem "base/imdp: implicit sink — finite time reward" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp, implicit_mdp) = ImdpModels.build(N)

        prop = FiniteTimeReward(N[2, 1, 0], N(9//10), 10)

        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end
end

@testitem "base/imdp: implicit sink — infinite time reward" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp, implicit_mdp) = ImdpModels.build(N)

        prop = InfiniteTimeReward(N[2, 1, 0], N(9//10), N(1//1_000_000))
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end
end

@testitem "base/imdp: implicit sink — expected exit time" setup = [ImdpModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; mdp, implicit_mdp) = ImdpModels.build(N)

        prop = ExpectedExitTime([3], N(1//1_000_000))
        spec = Specification(prop, Pessimistic, Maximize)

        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end
end
