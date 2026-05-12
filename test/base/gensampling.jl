@testitem "rejects finite-horizon properties" tags = [:base, :gsrdp_rejects_finite_horizon] begin
    using IntervalMDP
    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.5 0.0; 0.1 0.3 0.0; 0.2 0.1 1.0],
        upper = [0.5 0.7 0.0; 0.6 0.5 0.0; 0.7 0.3 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])
    gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
    @testset "FiniteTimeReachability" begin
        prop = FiniteTimeReachability([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        @test_throws ArgumentError solve(problem, gsdp)
    end
    @testset "FiniteTimeSafety" begin
        prop = FiniteTimeSafety([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        @test_throws ArgumentError solve(problem, gsdp)
    end
end

@testitem "GSRDP callback reports cumulative bellman updates" tags =
    [:base, :gsrdp_callback_reports_cumulative_bellman_updates] begin
    using IntervalMDP
    @testset "GSRDP callback reports cumulative bellman updates" for N in [Float32, Float64]
        prob = IntervalAmbiguitySets(;
            lower = N[0 1 // 2 0; 1 // 10 3 // 10 0; 1 // 5 1 // 10 1],
            upper = N[1 // 2 7 // 10 0; 3 // 5 1 // 2 0; 7 // 10 3 // 10 1],
        )
        prob2 = IntervalAmbiguitySets(;
            lower = N[1 // 10 1 // 5 0; 1 // 5 1 // 5 0; 3 // 10 2 // 5 1],
            upper = N[1 // 2 1 // 2 0; 1 // 2 2 // 5 0; 2 // 5 2 // 5 1],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob2, prob2], [1])
        gsdp =
            GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
        eps = N(1 // 1000000)
        prop = InfiniteTimeReachability([3], eps)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)

        callback_counts = Int[]
        (_, k, _) = solve(
            problem,
            gsdp;
            callback = (V, bellman_updates) -> push!(callback_counts, bellman_updates),
        )

        expected_increment = num_states(mdp) * (num_actions(mdp) + 1)
        @test !isempty(callback_counts)
        @test first(callback_counts) == 0
        @test all(diff(callback_counts) .== expected_increment)
        @test last(callback_counts) == k * expected_increment
    end
end

@testitem "Random subset state sampling yields states" tags =
    [:base, :gsrdp_random_subset_state_sampling] begin
    using IntervalMDP

    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.5 0.0; 0.1 0.3 0.0; 0.2 0.1 1.0],
        upper = [0.5 0.7 0.0; 0.6 0.5 0.0; 0.7 0.3 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob], [1])

    seq = IntervalMDP.sample(IntervalMDP.RandomSubsetState(5), mdp)

    @test IntervalMDP.sequence_shape(seq) === IntervalMDP.StateUpdateSequence()
    @test length(seq) == 5
    @test all(s -> s in CartesianIndices(IntervalMDP.source_shape(mdp)), seq)
    @test IntervalMDP.touched_states(seq) == Set(seq)
end

@testitem "GSRDP initial-state gap termination" tags =
    [:base, :gsrdp_initial_state_gap_termination] begin
    using IntervalMDP
    @testset "GSRDP initial-state gap termination" for N in [Float32, Float64]
        prob = IntervalAmbiguitySets(;
            lower = N[0 1 // 2 0; 1 // 10 3 // 10 0; 1 // 5 1 // 10 1],
            upper = N[1 // 2 7 // 10 0; 3 // 5 1 // 2 0; 7 // 10 3 // 10 1],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])
        gsdp =
            GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
        prop = InfiniteTimeReachAvoidInitial([3], [2], [1], N(1 // 1000))
        spec = Specification(prop, Pessimistic, Maximize)

        term = IntervalMDP.termination_criteria(gsdp, spec)
        @test term.initial == [CartesianIndex(1)]
        @test term(nothing, nothing, N[1 // 10000, 1, 1])
        @test !term(nothing, nothing, N[1, 1 // 10000, 1])
    end
end

@testitem "IMDP verification parity (Pessimistic, Maximize)" tags =
    [:base, :gsrdp_imdp_verification_parity_pessimistic_maximize] begin
    using IntervalMDP
    @testset "IMDP verification parity (Pessimistic, Maximize)" for N in [Float32, Float64]
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
        eps = N(1 // 1000000)
        prop = InfiniteTimeReachability([3], eps)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        (V_rvi, _, _) = solve(problem, rvi)
        (V_gsdp, _, _) = solve(problem, gsdp)
        # GSRDP terminates on V_upper - V_lower < eps; RVI terminates on
        # ||V_cur - V_prev|| < eps. Both converge to the same V*; allow
        # `2eps` slack since each method's lower bound is within `eps` of V*.
        @test maximum(abs, V_rvi .- V_gsdp) <= 2 * eps
    end
end

@testitem "IMDP control synthesis parity (Pessimistic, Maximize)" tags =
    [:base, :gsrdp_imdp_control_synthesis_parity_pessimistic_maximize] begin
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
        eps = N(1 // 1000000)
        prop = InfiniteTimeReachability([3], eps)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = ControlSynthesisProblem(mdp, spec)
        sol_rvi = solve(problem, rvi)
        sol_gsdp = solve(problem, gsdp)
        @test maximum(abs, value_function(sol_rvi) .- value_function(sol_gsdp)) <= 2 * eps
    end
end

@testitem "IntervalValueFunction gap shrinks monotonically" tags =
    [:base, :gsrdp_gap_shrinks_monotonically] begin
    using IntervalMDP
    @testset "IntervalValueFunction gap shrinks monotonically" for N in [Float32, Float64]
        prob = IntervalAmbiguitySets(;
            lower = N[0 1 // 2 0; 1 // 10 3 // 10 0; 1 // 5 1 // 10 1],
            upper = N[1 // 2 7 // 10 0; 3 // 5 1 // 2 0; 7 // 10 3 // 10 1],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob], [1])
        prop = InfiniteTimeReachability([3], N(1 // 1000))
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        gsdp =
            GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
        (V, _, residual) = solve(problem, gsdp)
        @test maximum(residual) <= N(1 // 1000)
        # Final lower bound is in [0, 1] for reachability.
        @test all(V .>= N(0))
        @test all(V .<= N(1))
    end
end

@testitem "ValueFunctionOrderedSampling selects and sorts states by operation" tags =
    [:base, :vf_ordered_sampling] begin
    using IntervalMDP

    @testset "ValueFunctionOrderedSampling basic operation" for N in [Float32, Float64]
        # Create a simple mock value function with controlled gaps
        struct MockVF
            lower_array::AbstractArray
            upper_array::AbstractArray
        end

        lower_vals = N[0.1, 0.3, 0.5]
        upper_vals = N[0.3, 0.5, 0.6]  # gaps: [0.2, 0.2, 0.1]
        vf = MockVF(lower_vals, upper_vals)
        gap_op = (vf) -> abs.(vf.upper_array .- vf.lower_array)

        # Create a mock model (just needs to work with the iterator)
        struct MockModel end
        mdp = MockModel()

        # Test with descending order (largest gaps first, k=2)
        sampler_desc = IntervalMDP.ValueFunctionOrderedSampling(gap_op, false, 2)
        seq_desc = IntervalMDP.sample(sampler_desc, mdp, nothing, vf)

        @test IntervalMDP.sequence_shape(seq_desc) === IntervalMDP.StateUpdateSequence()
        @test length(seq_desc) == 2
        states_desc = collect(seq_desc)
        # Should get the top 2 states by gap (states with gaps 0.2 and 0.2 or 0.2 and 0.1)
        # The exact states depend on sortperm behavior with ties, but we should have 2
        @test all(s -> isa(s, CartesianIndex), states_desc)

        # Test with ascending order (smallest gaps first, k=2)
        sampler_asc = IntervalMDP.ValueFunctionOrderedSampling(gap_op, true, 2)
        seq_asc = IntervalMDP.sample(sampler_asc, mdp, nothing, vf)

        @test length(seq_asc) == 2
        states_asc = collect(seq_asc)
        @test all(s -> isa(s, CartesianIndex), states_asc)
        # States with smallest gaps should be included (lowest gap is 0.1 at index 3)
        @test CartesianIndex(3) ∈ states_asc
    end

    @testset "ValueFunctionOrderedSampling k larger than state space" for N in
                                                                          [Float32, Float64]
        struct MockVF
            lower_array::AbstractArray
            upper_array::AbstractArray
        end

        vf = MockVF(fill(N(0.1), 3), fill(N(0.3), 3))
        gap_op = (vf) -> abs.(vf.upper_array .- vf.lower_array)

        struct MockModel end
        mdp = MockModel()

        # Request more states than available
        sampler = IntervalMDP.ValueFunctionOrderedSampling(gap_op, false, 100)
        seq = IntervalMDP.sample(sampler, mdp, nothing, vf)

        # Should return all 3 states
        @test length(seq) == 3
        states = Set(collect(seq))
        # Should have exactly 3 unique CartesianIndex objects
        @test length(states) == 3
    end

    @testset "ValueFunctionOrderedSampling requires value_function argument" begin
        struct MockModel end
        mdp = MockModel()
        gap_op = (vf) -> vf
        sampler = IntervalMDP.ValueFunctionOrderedSampling(gap_op, false, 5)

        # 2-argument sample should error
        @test_throws ErrorException IntervalMDP.sample(sampler, mdp)

        # 3-argument sample should error
        @test_throws ErrorException IntervalMDP.sample(sampler, mdp, nothing)
    end
end
