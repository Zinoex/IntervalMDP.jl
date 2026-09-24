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

@testitem "GSRDP callback reports the states each iteration relaxed" tags =
    [:base, :gsrdp_callback_reports_state_sequence] begin
    using IntervalMDP, Random
    # The three-argument callback form sees the update sequence; the two-argument form
    # must keep working untouched, since every existing caller uses it.
    @testset "GSRDP callback reports the states each iteration relaxed" for sampling in [
        IntervalMDP.ExhaustiveState(),
        IntervalMDP.RandomSubsetState(2),
    ]
        prob = IntervalAmbiguitySets(;
            lower = Float64[0 1//2 0; 1//10 3//10 0; 1//5 1//10 1],
            upper = Float64[1//2 7//10 0; 3//5 1//2 0; 7//10 3//10 1],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])
        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(mdp);
            sampling_strategy = sampling,
        )
        prop = InfiniteTimeReachability([3], 1e-6)
        problem = VerificationProblem(mdp, Specification(prop, Pessimistic, Maximize))

        # Three-argument form: record both the count and the reported sequence.
        counts = Int[]
        seqs = Any[]
        Random.seed!(1234)          # so the sampled variant replays identically below
        solve(problem, gsdp; callback = (V, n, seq) -> (push!(counts, n); push!(seqs, seq)))

        # The pre-update fire has nothing sampled yet.
        @test first(counts) == 0
        @test isnothing(first(seqs))
        @test all(!isnothing, seqs[2:end])

        # `bellman_updates` advances by exactly `|state_seq| * (num_actions + 1)`, which
        # is what ties the reported sequence to the work the solver actually did.
        per_state = num_actions(mdp) + 1
        @test diff(counts) == [length(seq) * per_state for seq in seqs[2:end]]

        # Every reported state is a real state of the model.
        S = CartesianIndices(IntervalMDP.source_shape(mdp))
        @test all(all(s -> s in S, seq) for seq in seqs[2:end])

        # A full sweep relaxes every state exactly once per iteration.
        if sampling isa IntervalMDP.ExhaustiveState
            @test all(length(seq) == num_states(mdp) for seq in seqs[2:end])
        end

        # Two-argument form still fires, with the identical update counts.
        legacy = Int[]
        Random.seed!(1234)
        solve(problem, gsdp; callback = (V, n) -> push!(legacy, n))
        @test legacy == counts
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

@testitem "GSRDP honours the satisfaction mode (parity with RVI, both modes)" tags =
    [:base, :gsrdp_satisfaction_mode_parity] begin
    using IntervalMDP
    # `upper_bound` selects the adversary's direction inside the ambiguity set, so
    # getting it wrong makes GSRDP silently return the *opposite* mode's value. The
    # other parity fixtures in this file cannot catch that: every state there has an
    # action reaching the target with probability exactly 1, so V* = 1 under both
    # modes. This fixture separates the modes by 0.48.
    #
    # Reach-avoid, not plain reachability: separating the modes needs an absorbing
    # non-target state, and GSRDP's upper bracket (initialised to 1 everywhere) never
    # contracts on one. Reach-avoid pins the avoid state in `step_postprocess`, so the
    # gap closes.
    #
    #   s1 -> s2 in [2/5, 4/5],  s1 -> s4 in [1/5, 3/5]
    #   s2 -> s3 in [2/5, 4/5],  s2 -> s4 in [1/5, 3/5]
    #   s3 (reach) and s4 (avoid) absorbing
    #
    # Pessimistic V = [0.16, 0.4, 1, 0];  Optimistic V = [0.64, 0.8, 1, 0].
    @testset "GSRDP honours the satisfaction mode" for N in [Float32, Float64]
        prob = IntervalAmbiguitySets(;
            lower = N[0 0 0 0; 2//5 0 0 0; 0 2//5 1 0; 1//5 1//5 0 1],
            upper = N[0 0 0 0; 4//5 0 0 0; 0 4//5 1 0; 3//5 3//5 0 1],
        )
        eps = N(1 // 1000000)
        prop = InfiniteTimeReachAvoid([3], [4], eps)

        model = IntervalMarkovChain(prob, [1])
        rvi = RobustValueIteration(default_bellman_algorithm(model))
        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(model),
        )

        V_rvi = Dict{SatisfactionMode, Any}()
        for mode in (Pessimistic, Optimistic)
            problem = VerificationProblem(model, Specification(prop, mode, Maximize))
            (V_rvi[mode], _, _) = solve(problem, rvi)
            (V_gsdp, _, _) = solve(problem, gsdp)
            @test maximum(abs, V_rvi[mode] .- V_gsdp) <= 2 * eps
        end

        # Guard against the fixture degenerating: if the two modes ever collapse
        # onto the same values, the assertions above stop testing the sign.
        @test maximum(abs, V_rvi[Pessimistic] .- V_rvi[Optimistic]) > N(1 // 10)
    end
end

@testitem "Reach-avoid keeps a valid bracket, and gap reports an inverted one" tags =
    [:base, :gsrdp_reach_avoid_bracket] begin
    using IntervalMDP
    # `initialize!(..., AbstractReachAvoid, Val(true))` used to seed avoid states at
    # `-1.0` — the `AbstractSafety` encoding, which only works because safety shifts
    # everything back by `+1.0` in `postprocess_value_function!`. Reach-avoid has no
    # such shift, so the upper bound started below the lower bound on avoid states.
    # `gap` hid it: it took `abs`, so `GapTerminationCriteria` read an inverted
    # bracket as a small gap.
    @testset "Reach-avoid bracket" for N in [Float32, Float64]
        prob = IntervalAmbiguitySets(;
            lower = N[0 0 0 0; 2//5 0 0 0; 0 2//5 1 0; 1//5 1//5 0 1],
            upper = N[0 0 0 0; 4//5 0 0 0; 0 4//5 1 0; 3//5 3//5 0 1],
        )
        model = IntervalMarkovChain(prob, [1])
        prop = InfiniteTimeReachAvoid([3], [4], N(1 // 1000000))
        problem = VerificationProblem(model, Specification(prop, Pessimistic, Maximize))

        V = IntervalMDP.IntervalValueFunction(
            IntervalMDP.StateValueFunction(problem, IntervalMDP.Lower),
            IntervalMDP.StateValueFunction(problem, IntervalMDP.Upper),
        )
        IntervalMDP.initialize!(V, prop)

        # The avoid state is 0, not -1, and the bracket is valid everywhere.
        @test V.upper.current[4] == N(0)
        @test all(V.upper.current .>= V.lower.current)

        # `gap` is the signed width `upper - lower`, not `abs`.
        @test IntervalMDP.gap(V) == V.upper.current .- V.lower.current

        # An inverted bracket is reported rather than folded away.
        V.upper.current[1] = V.lower.current[1] - N(1 // 2)
        @test_throws IntervalMDP.InvertedBracketError IntervalMDP.gap(V)
    end
end

@testitem "GSRDP reach-avoid parity with RVI over multiple actions" tags =
    [:base, :gsrdp_reach_avoid_multiaction_parity] begin
    using IntervalMDP
    # The multi-action counterpart of the satisfaction-mode fixture. It is the case
    # the `-1.0` avoid initialisation broke: `bellman_update!` picks the optimal
    # action from the *upper* bracket, so a negative upper bound on the avoid state
    # made the maximiser prefer the action that jumps straight into it, dragging the
    # lower bracket onto the wrong action (`V = [0.0, 0.4, 1.0, 0.0]` under
    # `Pessimistic`).
    #
    # One `IntervalAmbiguitySets` per state; its columns are that state's actions.
    # Action 1 is the reach-avoid chain; action 2 jumps straight to the avoid state,
    # so a `Maximize` strategy must never pick it.
    #
    # Pessimistic V = [0.16, 0.4, 1, 0];  Optimistic V = [0.64, 0.8, 1, 0].
    @testset "GSRDP reach-avoid multi-action parity" for N in [Float32, Float64]
        s1 = IntervalAmbiguitySets(;
            lower = N[0 0; 2//5 0; 0 0; 1//5 1],
            upper = N[0 0; 4//5 0; 0 0; 3//5 1],
        )
        s2 = IntervalAmbiguitySets(;
            lower = N[0 0; 0 0; 2//5 0; 1//5 1],
            upper = N[0 0; 0 0; 4//5 0; 3//5 1],
        )
        s3 = IntervalAmbiguitySets(;
            lower = N[0 0; 0 0; 1 1; 0 0],
            upper = N[0 0; 0 0; 1 1; 0 0],
        )
        s4 = IntervalAmbiguitySets(;
            lower = N[0 0; 0 0; 0 0; 1 1],
            upper = N[0 0; 0 0; 0 0; 1 1],
        )
        model = IntervalMarkovDecisionProcess([s1, s2, s3, s4], [1])

        eps = N(1 // 1000000)
        prop = InfiniteTimeReachAvoid([3], [4], eps)
        rvi = RobustValueIteration(default_bellman_algorithm(model))
        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(model),
        )

        expected = Dict(
            Pessimistic => N[4 // 25, 2 // 5, 1, 0],
            Optimistic => N[16 // 25, 4 // 5, 1, 0],
        )
        for mode in (Pessimistic, Optimistic)
            problem = VerificationProblem(model, Specification(prop, mode, Maximize))
            (V_rvi, _, _) = solve(problem, rvi)
            (V_gsdp, _, _) = solve(problem, gsdp)
            @test maximum(abs, V_rvi .- V_gsdp) <= 2 * eps
            @test maximum(abs, vec(V_gsdp) .- expected[mode]) <= 2 * eps
        end
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

@testitem "_categorical_sample rejects a degenerate weight vector" tags =
    [:base, :gsrdp_categorical_sample] begin
    using IntervalMDP, Random

    # `_categorical_sample` used to answer a weight vector with no positive
    # mass with `firstindex`: `u = rand() * sum(w) == 0`, so `acc >= u` passes
    # on the first index. Every weighted trajectory sampler can hand it such a
    # vector (a successor set that scores zero throughout), and the rollout then
    # moved to target state 1 regardless of its transition probability.
    @testset "no positive mass => nothing" begin
        @test IntervalMDP._categorical_sample(zeros(4)) === nothing
        @test IntervalMDP._categorical_sample(zeros(Float32, 3)) === nothing
        @test IntervalMDP._categorical_sample(Float64[]) === nothing
        @test IntervalMDP._categorical_sample([-1.0, -1.0]) === nothing
    end

    @testset "negative entries are ignored, not cancelled against" begin
        # Reach-avoid seeds `U` at -1.0 on avoid states, so a tilted weight
        # vector really can carry negatives. They must not shrink the sampling
        # range: a large negative alongside a positive weight still leaves the
        # positive one drawable.
        @test all(IntervalMDP._categorical_sample([-5.0, 1.0]) == 2 for _ in 1:100)
    end

    @testset "zero and negative entries are never selected" begin
        # Deterministic, not statistical: only one entry is positive, and the
        # `rand() === 0.0` draw must not divert to index 1 either.
        @test all(IntervalMDP._categorical_sample([0.0, 1.0, 0.0]) == 2 for _ in 1:100)
        @test all(IntervalMDP._categorical_sample([-1.0, 0.0, 2.0]) == 3 for _ in 1:100)
    end

    @testset "samples proportionally to the positive weights" begin
        Random.seed!(1234)
        counts = zeros(Int, 2)
        for _ in 1:40_000
            counts[IntervalMDP._categorical_sample([0.25, 0.75])] += 1
        end
        @test counts[1] / 40_000 ≈ 0.25 atol = 0.02
        @test counts[2] / 40_000 ≈ 0.75 atol = 0.02
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

    a2, v2 = IntervalMDP._omax_best_action(
        mdp,
        CartesianIndex(1),
        V,
        true;
        exclude = CartesianIndex(2),
    )
    @test a2 == CartesianIndex(1)
    @test v2 ≈ 10.0

    # A single-action state, excluding its only action leaves no candidate.
    prob1a = IntervalAmbiguitySets(; lower = hcat([1.0, 0.0]), upper = hcat([1.0, 0.0]))
    mdp1a = IntervalMarkovDecisionProcess([prob1a], [1])
    a3, v3 = IntervalMDP._omax_best_action(
        mdp1a,
        CartesianIndex(1),
        V,
        true;
        exclude = CartesianIndex(1),
    )
    @test a3 === nothing
    @test v3 === nothing
end

@testitem "_predecessor_states / _predecessor_index / _state_indices" tags =
    [:base, :gsrdp_priority_predecessor_states] begin
    using IntervalMDP, SparseArrays

    # 3-state chain, single action per state: 1 -> {1,2}; 2 -> {2,3}; 3 absorbing,
    # so the predecessor relation is 1 <- {1}; 2 <- {1,2}; 3 <- {2,3}.
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

    @test IntervalMDP._predecessor_states(mdp, CartesianIndex(1)) ==
          Set([CartesianIndex(1)])
    @test IntervalMDP._predecessor_states(mdp, CartesianIndex(2)) ==
          Set([CartesianIndex(1), CartesianIndex(2)])
    @test IntervalMDP._predecessor_states(mdp, CartesianIndex(3)) ==
          Set([CartesianIndex(2), CartesianIndex(3)])
    @test collect(IntervalMDP._state_indices(mdp)) ==
          [CartesianIndex(1), CartesianIndex(2), CartesianIndex(3)]

    @testset "_predecessor_index agrees with the per-state query" begin
        index = IntervalMDP._predecessor_index(mdp)
        for i in 1:3
            @test Set(CartesianIndex(src) for (src, _) in index[i]) ==
                  IntervalMDP._predecessor_states(mdp, CartesianIndex(i))
        end

        # ... and carries maxₐ p̄(target | source, a) per edge.
        @test sort(index[1]) == [(1, 0.3)]
        @test sort(index[2]) == [(1, 0.7), (2, 0.3)]
        @test sort(index[3]) == [(2, 0.7), (3, 1.0)]
    end
end

@testitem "Sampling handles implicit sink states (source_dims < state_vars)" tags =
    [:base, :gsrdp_implicit_sink] begin
    using IntervalMDP

    # A model may declare fewer source states than target states, leaving the
    # trailing targets implicit and absorbing ("implicit sink states", see
    # `FactoredRobustMarkovDecisionProcess`). Those have no ambiguity-set column
    # and no actions, so sampling must never treat one as a state to act from,
    # nor emit one in an update sequence — the strategy cache is sized by
    # `source_shape`, and `available` on `AllAvailableActions` ignores the state
    # and offers every action regardless.
    #
    # 2 source states, 3 targets, 2 actions; target 3 is the sink.
    #   state 1: a1 -> state 2,  a2 -> sink
    #   state 2: a1 -> state 1,  a2 -> sink
    dirac(rows) = IntervalAmbiguitySets(; lower = rows, upper = rows)
    prob1 = dirac([0.0 0.0; 1.0 0.0; 0.0 1.0])
    prob2 = dirac([1.0 0.0; 0.0 0.0; 0.0 1.0])
    mdp = IntervalMarkovDecisionProcess([prob1, prob2], [1])

    @test IntervalMDP.source_shape(mdp) == (2,)
    @test IntervalMDP.state_values(mdp) == (3,)

    vf = (upper = (current = [1.0, 1.0, 0.0],), lower = (current = [0.0, 0.5, 0.0],))

    # Trajectory sampling's own handling of implicit sinks is covered in
    # `trajectorysampling.jl`; what's left here is the shared machinery the
    # priority-queue strategies also rely on.

    @testset "_is_source_state / _target_indices" begin
        @test IntervalMDP._is_source_state(mdp, CartesianIndex(1))
        @test IntervalMDP._is_source_state(mdp, CartesianIndex(2))
        @test !IntervalMDP._is_source_state(mdp, CartesianIndex(3))   # the sink
        @test collect(IntervalMDP._target_indices(mdp)) ==
              [CartesianIndex(1), CartesianIndex(2), CartesianIndex(3)]
    end

    @testset "_action_uncertainty at the sink is zero, not a BoundsError" begin
        # Regression: this used to call `available(mdp, sink)` and then index
        # the marginal past its last (source, action) column.
        @test IntervalMDP._action_uncertainty(mdp, CartesianIndex(3), vf, nothing) == 0.0
    end

    @testset "bellman_residual_delta at the sink is zero, not a BoundsError" begin
        @test IntervalMDP.PriorityQueueSampling.bellman_residual_delta(
            CartesianIndex(3),
            vf,
            mdp,
            nothing,
        ) == 0.0
    end

    @testset "the sink is never a predecessor, but may be a target" begin
        # `support` on a dense IntervalAmbiguitySets returns the full target
        # range, so membership has to be decided on a nonzero upper transition
        # probability — going by support alone would make every state a
        # predecessor of every other here.
        @test IntervalMDP._predecessor_states(mdp, CartesianIndex(1)) ==
              Set([CartesianIndex(2)])
        @test IntervalMDP._predecessor_states(mdp, CartesianIndex(2)) ==
              Set([CartesianIndex(1)])

        # The sink owns no ambiguity-set column, so it can never appear in a
        # predecessor set — it does, however, have predecessors of its own.
        @test IntervalMDP._predecessor_states(mdp, CartesianIndex(3)) ==
              Set([CartesianIndex(1), CartesianIndex(2)])

        index = IntervalMDP._predecessor_index(mdp)
        @test length(index) == 3          # indexed by target, so the sink has a slot
        @test sort(index[3]) == [(1, 1.0), (2, 1.0)]
    end
end

@testitem "GSRDP parity: implicit vs explicit sink state, all sampling strategies" tags =
    [:base, :gsrdp_implicit_sink_solve] begin
    using IntervalMDP

    # The same system written two ways: `explicit_mdp` spells out state 3's
    # absorbing self-loop as a third source column; `implicit_mdp` omits it and
    # lets state 3 be an implicit sink. Both must give the same values.
    prob1 = IntervalAmbiguitySets(;
        lower = Float64[0 1//2; 1//10 3//10; 1//5 1//10],
        upper = Float64[1//2 7//10; 3//5 1//2; 7//10 3//10],
    )
    prob2 = IntervalAmbiguitySets(;
        lower = Float64[1//10 1//5; 1//5 1//5; 3//10 2//5],
        upper = Float64[1//2 1//2; 1//2 2//5; 2//5 2//5],
    )
    sink = IntervalAmbiguitySets(;
        lower = Float64[0 0; 0 0; 1 1],
        upper = Float64[0 0; 0 0; 1 1],
    )

    explicit_mdp = IntervalMarkovDecisionProcess([prob1, prob2, sink], [1])
    implicit_mdp = IntervalMarkovDecisionProcess([prob1, prob2], [1])
    @test IntervalMDP.source_shape(implicit_mdp) == (2,)
    @test IntervalMDP.state_values(implicit_mdp) == (3,)

    eps = 1e-6
    prop = InfiniteTimeReachability([3], eps)
    spec = Specification(prop, Pessimistic, Maximize)

    (V_ref, _, _) = solve(
        VerificationProblem(explicit_mdp, spec),
        RobustValueIteration(default_bellman_algorithm(explicit_mdp)),
    )

    # Under RVI the two encodings agree exactly — the implicit sink is handled
    # identically to a spelled-out absorbing state.
    (V_ref_implicit, _, _) = solve(
        VerificationProblem(implicit_mdp, spec),
        RobustValueIteration(default_bellman_algorithm(implicit_mdp)),
    )
    @test V_ref_implicit == V_ref

    # Trajectory sampling's parity on this same fixture is in
    # `trajectorysampling.jl`, alongside the rest of its suite.
    strategies = [
        IntervalMDP.PriorityQueueSampling.GapPriorityQueueSampling(2),
        IntervalMDP.PriorityQueueSampling.UpperBoundPriorityQueueSampling(2),
        IntervalMDP.PriorityQueueSampling.ActionUncertaintyPriorityQueueSampling(2),
    ]

    # GSRDP is compared against RVI with a looser tolerance than the usual
    # `2 * eps`: the two terminate independently, each within `eps` of the fixed
    # point under its own residual rule, so their difference is not strictly
    # bounded by `2 * eps`. Values here are all ~1.0, so `1e-4` still catches any
    # real mishandling of the sink (a grossly wrong value, or a BoundsError).
    @testset "verification parity: $(typeof(ss).name.name)" for ss in strategies
        alg = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(implicit_mdp);
            sampling_strategy = ss,
        )
        (V, _, _) = solve(VerificationProblem(implicit_mdp, spec), alg)
        @test maximum(abs, V_ref .- V) <= 1e-4
    end

    @testset "control synthesis yields a valid, source-shaped strategy" begin
        alg = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(implicit_mdp);
            sampling_strategy = IntervalMDP.PriorityQueueSampling.ActionUncertaintyPriorityQueueSampling(
                2,
            ),
        )
        sol = solve(ControlSynthesisProblem(implicit_mdp, spec), alg)
        # `checkstrategy` asserts the shape matches `source_shape` and every
        # action is in range — i.e. nothing wrote through the sink.
        @test IntervalMDP.checkstrategy(strategy(sol), implicit_mdp) === nothing
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

    @test_throws ArgumentError IntervalMDP.RandomlyThinned(
        IntervalMDP.ExhaustiveState(),
        1.5,
    )
    @test_throws ArgumentError IntervalMDP.RandomlyThinned(
        IntervalMDP.ExhaustiveState(),
        -0.1,
    )

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
        thinned = IntervalMDP.RandomlyThinned(IntervalMDP.ExhaustiveState(), 1.0)
        seq = IntervalMDP.sample(thinned, mdp, cache, nothing, nothing)
        @test IntervalMDP.sequence_shape(seq) === IntervalMDP.StateUpdateSequence()
        @test length(seq) == num_states(mdp)
    end

    @testset "keep_prob thins the sequence" begin
        thinned = IntervalMDP.RandomlyThinned(IntervalMDP.ExhaustiveState(), 0.5)
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
        IntervalMDP.ExhaustiveState(),
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
            IntervalMDP.ExhaustiveState(),
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
            IntervalMDP.ExhaustiveState(),
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
