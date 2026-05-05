@testitem "IMDP verification parity (Pessimistic, Maximize)" tags =
    [:base, :imdp_verification_parity_pessimistic_maximize] begin
    using IntervalMDP
    @testset "IMDP verification parity (Pessimistic, Maximize)" for N in
        [Float32, Float64, Rational{BigInt}]
        prob = IntervalAmbiguitySets(;
            lower = N[0 1 // 2 0; 1 // 10 3 // 10 0; 1 // 5 1 // 10 1],
            upper = N[1 // 2 7 // 10 0; 3 // 5 1 // 2 0; 7 // 10 3 // 10 1],
        )
        prob2 = IntervalAmbiguitySets(;
            lower = N[1 // 10 1 // 5 0; 1 // 5 1 // 5 0; 3 // 10 2 // 5 1],
            upper = N[1 // 2 1 // 2 0; 1 // 2 2 // 5 0; 2 // 5 2 // 5 1],
        )
        transition_probs = [prob, prob2, prob2]
        mdp = IntervalMarkovDecisionProcess(transition_probs, [1])
        rvi = RobustValueIteration(default_bellman_algorithm(mdp))
        gsdp =
            GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
        prop = FiniteTimeReachability([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        (V_rvi, k_rvi, _) = solve(problem, rvi)
        (V_gsdp, k_gsdp, _) = solve(problem, gsdp)
        @test k_rvi == k_gsdp
        @test V_rvi == V_gsdp
        prop = InfiniteTimeReachability([3], N(1 // 1000000))
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        (V_rvi, k_rvi, _) = solve(problem, rvi)
        (V_gsdp, k_gsdp, _) = solve(problem, gsdp)
        @test k_rvi == k_gsdp
        @test V_rvi == V_gsdp
        prop = FiniteTimeSafety([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        (V_rvi, k_rvi, _) = solve(problem, rvi)
        (V_gsdp, k_gsdp, _) = solve(problem, gsdp)
        @test k_rvi == k_gsdp
        @test V_rvi == V_gsdp
        prop = FiniteTimeReachAvoid([3], [2], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        (V_rvi, k_rvi, _) = solve(problem, rvi)
        (V_gsdp, k_gsdp, _) = solve(problem, gsdp)
        @test k_rvi == k_gsdp
        @test V_rvi == V_gsdp
        prop = FiniteTimeReward(N[2, 1, 0], N(9 // 10), 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        (V_rvi, k_rvi, _) = solve(problem, rvi)
        (V_gsdp, k_gsdp, _) = solve(problem, gsdp)
        @test k_rvi == k_gsdp
        @test V_rvi == V_gsdp
    end
end

@testitem "IMDP control synthesis parity (Pessimistic, Maximize)" tags =
    [:base, :imdp_control_synthesis_parity_pessimistic_maximize] begin
    using IntervalMDP
    @testset "IMDP control synthesis parity (Pessimistic, Maximize)" for N in
        [Float32, Float64]
        prob = IntervalAmbiguitySets(;
            lower = N[0 1 // 2 0; 1 // 10 3 // 10 0; 1 // 5 1 // 10 1],
            upper = N[1 // 2 7 // 10 0; 3 // 5 1 // 2 0; 7 // 10 3 // 10 1],
        )
        prob2 = IntervalAmbiguitySets(;
            lower = N[1 // 10 1 // 5 0; 1 // 5 1 // 5 0; 3 // 10 2 // 5 1],
            upper = N[1 // 2 1 // 2 0; 1 // 2 2 // 5 0; 2 // 5 2 // 5 1],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob2, prob2], [1])
        rvi = RobustValueIteration(default_bellman_algorithm(mdp))
        gsdp =
            GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
        prop = InfiniteTimeReachability([3], N(1 // 1000000))
        spec = Specification(prop, Pessimistic, Maximize)
        problem = ControlSynthesisProblem(mdp, spec)
        sol_rvi = solve(problem, rvi)
        sol_gsdp = solve(problem, gsdp)
        @test num_iterations(sol_rvi) == num_iterations(sol_gsdp)
        @test value_function(sol_rvi) == value_function(sol_gsdp)
    end
end

@testitem "RandomSubsetStateActions(0) leaves V at V_0" tags =
    [:base, :randomsubsetstateactions_0_leaves_v_at_v_0] begin
    using IntervalMDP
    @testset "RandomSubsetStateActions(0) leaves V at V_0" for N in [Float32, Float64]
        using Random
        Random.seed!(0)
        prob = IntervalAmbiguitySets(;
            lower = N[0 1 // 2 0; 1 // 10 3 // 10 0; 1 // 5 1 // 10 1],
            upper = N[1 // 2 7 // 10 0; 3 // 5 1 // 2 0; 7 // 10 3 // 10 1],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob], [1])
        prop = FiniteTimeReachability([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        alg = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(mdp),
            IntervalMDP.RandomSubsetStateActions(0),
        )
        (V, k, _) = solve(problem, alg)
        @test k == 10
        @test V == N[0, 0, 1]
    end
end

@testitem "RandomSubsetStateActions bounded + below full-sweep" tags =
    [:base, :randomsubsetstateactions_bounded_below_full_sweep] begin
    using IntervalMDP
    @testset "RandomSubsetStateActions bounded + below full-sweep" for N in
        [Float32, Float64]
        using Random
        Random.seed!(123)
        prob = IntervalAmbiguitySets(;
            lower = N[0 1 // 2 0; 1 // 10 3 // 10 0; 1 // 5 1 // 10 1],
            upper = N[1 // 2 7 // 10 0; 3 // 5 1 // 2 0; 7 // 10 3 // 10 1],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob], [1])
        prop = FiniteTimeReachability([3], 20)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        full = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(mdp),
            IntervalMDP.AllSampling(),
        )
        partial = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(mdp),
            IntervalMDP.RandomSubsetStateActions(50),
        )
        (V_full, _, _) = solve(problem, full)
        (V_partial, _, _) = solve(problem, partial)
        @test all(V_partial .>= N(0))
        @test all(V_partial .<= N(1))
        @test all(V_partial .<= V_full .+ 10 * eps(N))
    end
end
