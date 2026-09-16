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

@testitem "_omax_best_action finds the argmax action, with exclude" tags =
    [:base, :gsrdp_omax_best_action] begin
    using IntervalMDP

    # 1 source state, 2 actions: action 1 -> Dirac target 1, action 2 -> Dirac target 2.
    prob = IntervalAmbiguitySets(; lower = [1.0 0.0; 0.0 1.0], upper = [1.0 0.0; 0.0 1.0])
    mdp = IntervalMarkovDecisionProcess([prob], [1])
    V = [10.0, 20.0]

    a, v = IntervalMDP._omax_best_action(mdp, CartesianIndex(1), V, true)
    @test a == CartesianIndex(2)
    @test v ≈ 20.0

    a2, v2 = IntervalMDP._omax_best_action(mdp, CartesianIndex(1), V, true; exclude = CartesianIndex(2))
    @test a2 == CartesianIndex(1)
    @test v2 ≈ 10.0

    # A single-action state, excluding its only action leaves no candidate.
    prob1a = IntervalAmbiguitySets(; lower = hcat([1.0, 0.0]), upper = hcat([1.0, 0.0]))
    mdp1a = IntervalMarkovDecisionProcess([prob1a], [1])
    a3, v3 = IntervalMDP._omax_best_action(mdp1a, CartesianIndex(1), V, true; exclude = CartesianIndex(1))
    @test a3 === nothing
    @test v3 === nothing
end

@testitem "TrajectorySampling: shared TrajectorySampling.TrajectorySampling behaviour" tags =
    [:base, :gsrdp_greedy_trajectory_shared] begin
    using IntervalMDP

    strategies = [
        IntervalMDP.TrajectorySampling.TransitionProbabilityTrajectorySampling(),
        IntervalMDP.TrajectorySampling.ExpectedGapTrajectorySampling(),
        IntervalMDP.TrajectorySampling.ReachProbabilityTrajectorySampling(),
        IntervalMDP.TrajectorySampling.ActionUncertaintyTrajectorySampling(),
    ]

    @testset "num_trajectories == 1, terminate_sampling is the false stub" for strat in strategies
        @test IntervalMDP.num_trajectories(strat) == 1
        @test IntervalMDP.terminate_sampling(strat, CartesianIndex(1), [CartesianIndex(1)], nothing, nothing, nothing) ==
              false
    end

    @testset "action_selection picks argmax upper-bound Q" begin
        # State 2's action 1 -> Dirac target 1 (upper Q = V_upper[1]); action 2 -> Dirac
        # target 2 (upper Q = V_upper[2]). V_upper[1] > V_upper[2], so action 1 wins.
        prob1 = IntervalAmbiguitySets(; lower = [1.0 1.0; 0.0 0.0], upper = [1.0 1.0; 0.0 0.0])
        prob2 = IntervalAmbiguitySets(; lower = [1.0 0.0; 0.0 1.0], upper = [1.0 0.0; 0.0 1.0])
        mdp = IntervalMarkovDecisionProcess([prob1, prob2], [1])
        vf = (upper = (current = [10.0, 5.0],), lower = (current = [0.0, 5.0],))

        for strat in strategies
            a = IntervalMDP.action_selection(strat, CartesianIndex(2), vf, mdp, nothing)
            @test a == CartesianIndex(1)
        end
    end
end

@testitem "TrajectorySampling: target_state_sampling weighting" tags =
    [:base, :gsrdp_greedy_trajectory_weighting] begin
    using IntervalMDP

    @testset "TransitionProbabilityTrajectorySampling samples directly from probs" begin
        strat = IntervalMDP.TrajectorySampling.TransitionProbabilityTrajectorySampling()
        probs = [0.0, 1.0]
        @test IntervalMDP.target_state_sampling(strat, nothing, nothing, probs, nothing, nothing, nothing) ==
              CartesianIndex(2)
    end

    @testset "ExpectedGapTrajectorySampling weights by U - L" begin
        strat = IntervalMDP.TrajectorySampling.ExpectedGapTrajectorySampling()
        probs = [0.5, 0.5]
        vf = (upper = (current = [1.0, 1.0],), lower = (current = [1.0, 0.0],))  # gap = [0, 1]
        @test IntervalMDP.target_state_sampling(strat, nothing, nothing, probs, vf, nothing, nothing) ==
              CartesianIndex(2)
    end

    @testset "ReachProbabilityTrajectorySampling weights by U" begin
        strat = IntervalMDP.TrajectorySampling.ReachProbabilityTrajectorySampling()
        probs = [0.5, 0.5]
        vf = (upper = (current = [0.0, 1.0],), lower = (current = [0.0, 0.0],))
        @test IntervalMDP.target_state_sampling(strat, nothing, nothing, probs, vf, nothing, nothing) ==
              CartesianIndex(2)
    end

    @testset "ActionUncertaintyTrajectorySampling weights by action uncertainty at s'" begin
        # 2 states, 2 actions: state 1's actions both Dirac -> state 1; state 2's
        # action 1 -> Dirac state 1, action 2 -> Dirac state 2.
        prob1 = IntervalAmbiguitySets(; lower = [1.0 1.0; 0.0 0.0], upper = [1.0 1.0; 0.0 0.0])
        prob2 = IntervalAmbiguitySets(; lower = [1.0 0.0; 0.0 1.0], upper = [1.0 0.0; 0.0 1.0])
        mdp = IntervalMarkovDecisionProcess([prob1, prob2], [1])
        vf = (upper = (current = [10.0, 5.0],), lower = (current = [0.0, 5.0],))

        # Direct check of the uncertainty primitive itself, hand-computed:
        #   state 1: a_L ties (both actions -> state 1), L(1) = V_lower[1] = 0;
        #            U^{-a_L}(1) = U(1, other action) = V_upper[1] = 10 => uncertainty 10.
        #   state 2: a_L = action 2 (L=5 > 0), L(2) = 5;
        #            U^{-a_L}(2) = U(2, action 1) = V_upper[1] = 10 => uncertainty 5.
        @test IntervalMDP.TrajectorySampling._action_uncertainty(mdp, CartesianIndex(1), vf) ≈ 10.0
        @test IntervalMDP.TrajectorySampling._action_uncertainty(mdp, CartesianIndex(2), vf) ≈ 5.0

        # With all transition mass on state 2, the (positive) uncertainty at state 1
        # is irrelevant — target_state_sampling must pick state 2 deterministically.
        strat = IntervalMDP.TrajectorySampling.ActionUncertaintyTrajectorySampling()
        probs = [0.0, 1.0]
        @test IntervalMDP.target_state_sampling(strat, CartesianIndex(1), CartesianIndex(1), probs, vf, mdp, nothing) ==
              CartesianIndex(2)
    end
end

@testitem "TrajectorySampling: end-to-end solve() with TransitionProbabilityTrajectorySampling" tags =
    [:base, :gsrdp_greedy_trajectory_solve] begin
    using IntervalMDP
    @testset "IMDP verification parity (Pessimistic, Maximize) with trajectory sampling" for N in
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
            sampling_strategy = IntervalMDP.TrajectorySampling.TransitionProbabilityTrajectorySampling(),
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

@testitem "_successor_states / _state_indices" tags = [:base, :gsrdp_priority_successor_states] begin
    using IntervalMDP, SparseArrays

    # 3-state chain, single action per state: 1 -> {1,2}; 2 -> {2,3}; 3 absorbing.
    # `support` returns the FULL column range for a dense IntervalAmbiguitySets
    # (see its docstring), so a sparse representation is needed here to actually
    # exercise successor filtering rather than trivially returning every state.
    prob1 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
    )
    prob2 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
    )
    prob3 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [3], [1.0])),
        upper = sparse_hcat(SparseVector(3, [3], [1.0])),
    )
    mdp = IntervalMarkovDecisionProcess([prob1, prob2, prob3], [1])

    @test IntervalMDP._successor_states(mdp, CartesianIndex(1)) ==
          Set([CartesianIndex(1), CartesianIndex(2)])
    @test IntervalMDP._successor_states(mdp, CartesianIndex(2)) ==
          Set([CartesianIndex(2), CartesianIndex(3)])
    @test IntervalMDP._successor_states(mdp, CartesianIndex(3)) == Set([CartesianIndex(3)])
    @test collect(IntervalMDP._state_indices(mdp)) ==
          [CartesianIndex(1), CartesianIndex(2), CartesianIndex(3)]
end

@testitem "PriorityQueueSampling: compute_priority formulas" tags =
    [:base, :gsrdp_priority_compute_priority] begin
    using IntervalMDP

    # Same 2-state/2-action model and hand-verified _action_uncertainty values as the
    # TrajectorySampling.ActionUncertaintyTrajectorySampling weighting test above.
    prob1 = IntervalAmbiguitySets(; lower = [1.0 1.0; 0.0 0.0], upper = [1.0 1.0; 0.0 0.0])
    prob2 = IntervalAmbiguitySets(; lower = [1.0 0.0; 0.0 1.0], upper = [1.0 0.0; 0.0 1.0])
    mdp = IntervalMarkovDecisionProcess([prob1, prob2], [1])
    vf = (upper = (current = [10.0, 5.0],), lower = (current = [0.0, 5.0],))

    gap_ss = IntervalMDP.PriorityQueueSampling.GapPriorityQueueSampling(1)
    @test IntervalMDP.compute_priority(gap_ss, CartesianIndex(1), vf, mdp, nothing) ≈ 10.0
    @test IntervalMDP.compute_priority(gap_ss, CartesianIndex(2), vf, mdp, nothing) ≈ 0.0

    upper_ss = IntervalMDP.PriorityQueueSampling.UpperBoundPriorityQueueSampling(1)
    @test IntervalMDP.compute_priority(upper_ss, CartesianIndex(1), vf, mdp, nothing) ≈ 10.0
    @test IntervalMDP.compute_priority(upper_ss, CartesianIndex(2), vf, mdp, nothing) ≈ 5.0

    au_ss = IntervalMDP.PriorityQueueSampling.ActionUncertaintyPriorityQueueSampling(1)
    @test IntervalMDP.compute_priority(au_ss, CartesianIndex(1), vf, mdp, nothing) ≈ 10.0
    @test IntervalMDP.compute_priority(au_ss, CartesianIndex(2), vf, mdp, nothing) ≈ 5.0
end

@testitem "PriorityQueueSampling: shared sample — init, incremental recompute, reset, fairness" tags =
    [:base, :gsrdp_priority_shared] begin
    using IntervalMDP, SparseArrays

    # 3-state chain, single action per state: 1 -> {1,2}; 2 -> {2,3}; 3 absorbing.
    # Sparse, so state 1's successor set is genuinely {1,2}, not every state (see
    # the note in the `_successor_states` test above).
    prob1 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [1, 2], [0.3, 0.7])),
    )
    prob2 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
        upper = sparse_hcat(SparseVector(3, [2, 3], [0.3, 0.7])),
    )
    prob3 = IntervalAmbiguitySets(;
        lower = sparse_hcat(SparseVector(3, [3], [1.0])),
        upper = sparse_hcat(SparseVector(3, [3], [1.0])),
    )
    mdp = IntervalMarkovDecisionProcess([prob1, prob2, prob3], [1])

    @testset "first call does a full sweep; later calls only touch the stale set" begin
        vf = (upper = (current = [1.0, 0.5, 1.0],), lower = (current = [0.0, 0.0, 1.0],))
        ss = IntervalMDP.PriorityQueueSampling.GapPriorityQueueSampling(1)

        # gap = [1.0, 0.5, 0.0] -> state 1 strictly highest, no tie to worry about.
        seq1 = collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        @test seq1 == [CartesianIndex(1)]
        @test ss.priorities[] ≈ [1.0, 0.5, 0.0]
        @test ss.initialized[]
        @test ss.previous_selected[] == [1]

        # Successor set of state 1 is {1, 2}: mutate state 2's value (a successor) and
        # state 3's value (not a successor of state 1) before the next call.
        vf.upper.current[2] = 0.9    # successor -> must be picked up
        vf.upper.current[3] = 0.99   # not a successor -> must stay stale (cached gap 0.0)

        seq2 = collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        @test seq2 == [CartesianIndex(1)]   # still the highest priority (1.0, recomputed, unchanged)
        @test ss.priorities[] ≈ [1.0, 0.9, 0.0]   # state 3 untouched despite the live value change
    end

    @testset "reset_sampling_strategy! clears cached state" begin
        ss = IntervalMDP.PriorityQueueSampling.UpperBoundPriorityQueueSampling(1)
        vf = (upper = (current = [1.0, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))
        collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))
        @test ss.initialized[]

        IntervalMDP.reset_sampling_strategy!(ss)
        @test !ss.initialized[]
        @test isempty(ss.previous_selected[])
        @test ss.clock[] == 0
    end

    @testset "ties break toward least-recently-selected, not lowest index" begin
        ss = IntervalMDP.PriorityQueueSampling.GapPriorityQueueSampling(1)
        vf = (upper = (current = [1.0, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))

        first = collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))[1]
        @test first in (CartesianIndex(1), CartesianIndex(2))
        other = first == CartesianIndex(1) ? CartesianIndex(2) : CartesianIndex(1)

        # Both states 1 and 2 still tie at gap 1.0 (vf untouched), so whichever wasn't
        # just selected must win now — starvation would instead reselect `first` again.
        second = collect(IntervalMDP.sample(ss, mdp, nothing, vf, nothing))[1]
        @test second == other
    end
end

@testitem "PriorityQueueSampling: end-to-end solve() parity with all three concrete strategies" tags =
    [:base, :gsrdp_priority_solve] begin
    using IntervalMDP

    @testset "$(nameof(typeof(ss)))" for ss in [
        IntervalMDP.PriorityQueueSampling.GapPriorityQueueSampling(2),
        IntervalMDP.PriorityQueueSampling.UpperBoundPriorityQueueSampling(2),
        IntervalMDP.PriorityQueueSampling.ActionUncertaintyPriorityQueueSampling(2),
    ]
        prob = IntervalAmbiguitySets(;
            lower = [0.0 0.5 0.0; 0.1 0.3 0.0; 0.2 0.1 1.0],
            upper = [0.5 0.7 0.0; 0.6 0.5 0.0; 0.7 0.3 1.0],
        )
        prob2 = IntervalAmbiguitySets(;
            lower = [0.1 0.2 0.0; 0.2 0.2 0.0; 0.3 0.4 1.0],
            upper = [0.5 0.5 0.0; 0.5 0.4 0.0; 0.4 0.4 1.0],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob2, prob2], [1])
        rvi = RobustValueIteration(default_bellman_algorithm(mdp))
        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(mdp);
            sampling_strategy = ss,
        )
        eps = 1e-6
        prop = InfiniteTimeReachability([3], eps)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        (V_rvi, _, _) = solve(problem, rvi)
        (V_gsdp, _, _) = solve(problem, gsdp)
        @test maximum(abs, V_rvi .- V_gsdp) <= 1000 * eps
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
