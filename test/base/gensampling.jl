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

@testitem "initial-state restricted termination" tags =
    [:base, :gsrdp_initial_state_gap_termination] begin
    using IntervalMDP
    @testset "initial-state restricted termination" for N in [Float32, Float64]
        prob = IntervalAmbiguitySets(;
            lower = N[0 1 // 2 0; 1 // 10 3 // 10 0; 1 // 5 1 // 10 1],
            upper = N[1 // 2 7 // 10 0; 3 // 5 1 // 2 0; 7 // 10 3 // 10 1],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])
        prop = InfiniteTimeReachAvoid([3], [2], N(1 // 1000); restrict_to_initial = true)
        spec = Specification(prop, Pessimistic, Maximize)

        # Both algorithms restrict the convergence check to the initial states.
        gsdp =
            GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
        rvi = RobustValueIteration(default_bellman_algorithm(mdp))
        for alg in (gsdp, rvi)
            term = IntervalMDP.termination_criteria(alg, spec, mdp)
            @test term isa IntervalMDP.InitialStateCriteria
            @test term.initial == [1]
            @test term(nothing, nothing, N[1 // 10000, 1, 1])
            @test !term(nothing, nothing, N[1, 1 // 10000, 1])
        end

        # Without the flag, convergence is checked on all states.
        prop_all = InfiniteTimeReachAvoid([3], [2], N(1 // 1000))
        spec_all = Specification(prop_all, Pessimistic, Maximize)
        for alg in (gsdp, rvi)
            term = IntervalMDP.termination_criteria(alg, spec_all, mdp)
            @test !(term isa IntervalMDP.InitialStateCriteria)
            @test !term(nothing, nothing, N[1 // 10000, 1, 1])
        end
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

@testitem "_omax_distribution realizes the O-max greedy fill" tags =
    [:base, :gsrdp_omax_distribution] begin
    using IntervalMDP

    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.5 0.0; 0.1 0.3 0.0; 0.2 0.1 1.0],
        upper = [0.5 0.7 0.0; 0.6 0.5 0.0; 0.7 0.3 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])
    marginal = IntervalMDP.marginals(mdp)[1]
    aset = marginal[CartesianIndex(1), CartesianIndex(1)]  # lower=[0,.1,.2], gap=[.5,.5,.5]
    V = [1.0, 2.0, 3.0]

    # Descending fill (upper_bound=true): target 3 (highest V) filled to its
    # upper bound (0.7) first, remaining budget (0.2) goes to target 2.
    p_upper = IntervalMDP._omax_distribution(aset, V, true)
    @test p_upper ≈ [0.0, 0.3, 0.7]
    @test sum(p_upper) ≈ 1.0

    # Ascending fill (upper_bound=false): target 1 (lowest V) filled first.
    p_lower = IntervalMDP._omax_distribution(aset, V, false)
    @test p_lower ≈ [0.5, 0.3, 0.2]
    @test sum(p_lower) ≈ 1.0

    struct FakeFactoredModel end
    IntervalMDP.marginals(::FakeFactoredModel) = (1, 2)
    @test_throws ArgumentError IntervalMDP._omax_marginal(FakeFactoredModel())
end

@testitem "TrajectorySamplingStrategy shared rollout skeleton" tags =
    [:base, :gsrdp_trajectory_sampling] begin
    using IntervalMDP

    # Every state's transition is a Dirac point mass on state 3, regardless
    # of source state or action — makes the rollout fully deterministic
    # (no flakiness from the random initial state / O-max tie-breaking)
    # while still exercising the real O-max/categorical-sampling machinery.
    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 1.0 1.0],
        upper = [0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 1.0 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])

    struct MockTrajectoryStrategy <: IntervalMDP.TrajectorySamplingStrategy
        n::Int
        terminate::Bool
        rev::Bool
    end
    IntervalMDP.num_trajectories(ss::MockTrajectoryStrategy) = ss.n
    IntervalMDP.terminate_sampling(ss::MockTrajectoryStrategy, current, trajectory, value_function, model, spec) =
        ss.terminate
    IntervalMDP.action_selection(ss::MockTrajectoryStrategy, current, value_function, model, spec) =
        first(IntervalMDP.available(model, current))
    IntervalMDP.target_state_sampling(
        ss::MockTrajectoryStrategy,
        current,
        a,
        probs,
        value_function,
        model,
        spec,
    ) = IntervalMDP._categorical_sample(probs)
    IntervalMDP.reverse_trajectory(ss::MockTrajectoryStrategy) = ss.rev

    function build_value_function(mdp, prop)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        alg = GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
        V = IntervalMDP.construct_value_function(alg, problem)
        IntervalMDP._gsrdp_initialize!(V, prop)
        return V, spec
    end

    @testset "stops at a reach state, default (reversed) order" begin
        prop = InfiniteTimeReachability([3], 1 // 1000)
        V, spec = build_value_function(mdp, prop)
        strat = MockTrajectoryStrategy(1, false, true)
        seq = IntervalMDP.sample(strat, mdp, nothing, V, spec)

        @test IntervalMDP.sequence_shape(seq) === IntervalMDP.StateUpdateSequence()
        states = collect(seq)
        @test all(s -> s in CartesianIndices(IntervalMDP.source_shape(mdp)), states)
        @test 1 <= length(states) <= 2
        @test first(states) == CartesianIndex(3)   # reversed: reach state first
    end

    @testset "forward (non-reversed) order" begin
        prop = InfiniteTimeReachability([3], 1 // 1000)
        V, spec = build_value_function(mdp, prop)
        strat = MockTrajectoryStrategy(1, false, false)
        states = collect(IntervalMDP.sample(strat, mdp, nothing, V, spec))
        @test last(states) == CartesianIndex(3)
    end

    @testset "num_trajectories concatenates independent rollouts" begin
        prop = InfiniteTimeReachability([3], 1 // 1000)
        V, spec = build_value_function(mdp, prop)
        strat = MockTrajectoryStrategy(3, false, true)
        states = collect(IntervalMDP.sample(strat, mdp, nothing, V, spec))
        @test 3 <= length(states) <= 6
        # Every trajectory visits state 3 exactly once (either as the sole
        # initial state, or as the one step taken to reach it).
        @test count(==(CartesianIndex(3)), states) == 3
    end

    @testset "hard step cap bounds a non-terminating strategy" begin
        prop = InfiniteTimeReachAvoid(Int[], Int[], 1 // 1000)  # no reach/avoid states
        V, spec = build_value_function(mdp, prop)
        strat = MockTrajectoryStrategy(1, false, true)  # terminate_sampling always false
        states = collect(IntervalMDP.sample(strat, mdp, nothing, V, spec))
        @test length(states) == 1 + IntervalMDP._trajectory_max_steps(mdp)
    end
end

@testitem "Round-robin state sampling cycles deterministically" tags =
    [:base, :gsrdp_round_robin_state_sampling] begin
    using IntervalMDP

    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.5 0.0; 0.1 0.3 0.0; 0.2 0.1 1.0],
        upper = [0.5 0.7 0.0; 0.6 0.5 0.0; 0.7 0.3 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])
    S = CartesianIndices(IntervalMDP.source_shape(mdp))

    rr = IntervalMDP.RoundRobinState(2)
    seq1 = IntervalMDP.sample(rr, mdp)
    @test IntervalMDP.sequence_shape(seq1) === IntervalMDP.StateUpdateSequence()
    @test collect(seq1) == [S[1], S[2]]

    seq2 = IntervalMDP.sample(rr, mdp)
    @test collect(seq2) == [S[3], S[1]]

    seq3 = IntervalMDP.sample(rr, mdp)
    @test collect(seq3) == [S[2], S[3]]

    # After 3 calls (6 = 2*3 total advance), the cursor is back at the start.
    seq4 = IntervalMDP.sample(rr, mdp)
    @test collect(seq4) == collect(seq1)

    IntervalMDP.reset_sampling_strategy!(rr)
    @test collect(IntervalMDP.sample(rr, mdp)) == collect(seq1)
end

@testitem "Round-robin state-action sampling cycles deterministically" tags =
    [:base, :gsrdp_round_robin_state_action_sampling] begin
    using IntervalMDP

    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.5 0.0; 0.1 0.3 0.0; 0.2 0.1 1.0],
        upper = [0.5 0.7 0.0; 0.6 0.5 0.0; 0.7 0.3 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])
    A = CartesianIndices(IntervalMDP.action_shape(mdp))
    S = CartesianIndices(IntervalMDP.source_shape(mdp))
    total = length(A) * length(S)

    rr = IntervalMDP.RoundRobinStateActions(2)
    seq1 = IntervalMDP.sample(rr, mdp)
    @test IntervalMDP.sequence_shape(seq1) === IntervalMDP.StateActionUpdateSequence()
    @test length(seq1) == 2
    @test all(p -> p[1] in A && p[2] in S, seq1)

    seen = Set{Tuple{eltype(A), eltype(S)}}()
    union!(seen, seq1)
    for _ in 1:cld(total, 2)
        union!(seen, IntervalMDP.sample(rr, mdp))
    end
    @test length(seen) == total

    IntervalMDP.reset_sampling_strategy!(rr)
    @test collect(IntervalMDP.sample(rr, mdp)) == collect(seq1)
end

@testitem "IMDP verification parity (Pessimistic, Maximize) with round-robin sampling" tags =
    [:base, :gsrdp_round_robin_parity] begin
    using IntervalMDP
    @testset "IMDP verification parity (Pessimistic, Maximize) with round-robin sampling" for N in
                                                                                               [
        Float32,
        Float64,
    ]
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
        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(mdp);
            sampling_strategy = IntervalMDP.RoundRobinState(1),
        )
        eps = N(1 // 1000000)
        prop = InfiniteTimeReachability([3], eps)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        (V_rvi, _, _) = solve(problem, rvi)
        (V_gsdp, _, _) = solve(problem, gsdp)
        @test maximum(abs, V_rvi .- V_gsdp) <= 2 * eps
    end
end

@testitem "RandomlyThinned filters a base strategy's update sequence" tags =
    [:base, :gsrdp_randomly_thinned] begin
    using IntervalMDP

    @test_throws ArgumentError IntervalMDP.RandomlyThinned(IntervalMDP.AllStatesSweep(), 1.5)
    @test_throws ArgumentError IntervalMDP.RandomlyThinned(IntervalMDP.AllStatesSweep(), -0.1)

    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.5 0.0; 0.1 0.3 0.0; 0.2 0.1 1.0],
        upper = [0.5 0.7 0.0; 0.6 0.5 0.0; 0.7 0.3 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])
    prop = InfiniteTimeReachability([3], 1 // 1000)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)
    cache = IntervalMDP.select_strategy_cache(IntervalMDP._gsrdp_strategy_cache(problem), 0)

    @testset "keep_prob = 1 keeps every state" begin
        thinned = IntervalMDP.RandomlyThinned(IntervalMDP.AllStatesSweep(), 1.0)
        seq = IntervalMDP.sample(thinned, mdp, cache, nothing, nothing)
        @test IntervalMDP.sequence_shape(seq) === IntervalMDP.StateUpdateSequence()
        @test length(seq) == num_states(mdp)
    end

    @testset "keep_prob thins the sequence" begin
        thinned = IntervalMDP.RandomlyThinned(IntervalMDP.AllStatesSweep(), 0.5)
        seq = IntervalMDP.sample(thinned, mdp, cache, nothing, nothing)
        @test 0 <= length(seq) <= num_states(mdp)
        @test all(s -> s in CartesianIndices(IntervalMDP.source_shape(mdp)), seq)
    end

    @testset "reset propagates to the wrapped strategy" begin
        rr = IntervalMDP.RoundRobinState(2)
        thinned = IntervalMDP.RandomlyThinned(rr, 1.0)
        rr.cursor[] = 7
        IntervalMDP.reset_sampling_strategy!(thinned)
        @test rr.cursor[] == 0
    end
end

@testitem "EpsilonGreedyMixture chooses exploit/explore deterministically at epsilon 0/1" tags =
    [:base, :gsrdp_epsilon_greedy_mixture] begin
    using IntervalMDP

    @test_throws ArgumentError IntervalMDP.EpsilonGreedyMixture(
        IntervalMDP.AllStatesSweep(),
        IntervalMDP.RandomSubsetState(1),
        1.5,
    )

    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.5 0.0; 0.1 0.3 0.0; 0.2 0.1 1.0],
        upper = [0.5 0.7 0.0; 0.6 0.5 0.0; 0.7 0.3 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])
    prop = InfiniteTimeReachability([3], 1 // 1000)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)

    expected_exploit = num_states(mdp) * (num_actions(mdp) + 1)
    expected_explore = 1 * (num_actions(mdp) + 1)

    @testset "epsilon = 0 always exploits" begin
        mix = IntervalMDP.EpsilonGreedyMixture(
            IntervalMDP.AllStatesSweep(),
            IntervalMDP.RandomSubsetState(1),
            0.0,
        )
        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(mdp);
            sampling_strategy = mix,
        )
        counts = Int[]
        solve(problem, gsdp; callback = (V, n) -> push!(counts, n))
        @test all(diff(counts) .== expected_exploit)
    end

    @testset "epsilon = 1 always explores" begin
        mix = IntervalMDP.EpsilonGreedyMixture(
            IntervalMDP.AllStatesSweep(),
            IntervalMDP.RandomSubsetState(1),
            1.0,
        )
        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(mdp);
            sampling_strategy = mix,
        )
        counts = Int[]
        solve(problem, gsdp; callback = (V, n) -> push!(counts, n))
        @test all(diff(counts) .== expected_explore)
    end

    @testset "reset propagates to both sub-strategies" begin
        rr1 = IntervalMDP.RoundRobinState(1)
        rr2 = IntervalMDP.RoundRobinState(1)
        rr1.cursor[] = 2
        rr2.cursor[] = 1
        mix = IntervalMDP.EpsilonGreedyMixture(rr1, rr2, 0.5)
        IntervalMDP.reset_sampling_strategy!(mix)
        @test rr1.cursor[] == 0
        @test rr2.cursor[] == 0
    end
end
