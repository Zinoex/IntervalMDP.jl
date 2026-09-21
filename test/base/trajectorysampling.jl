# Tests for the configurable trajectory sampler (`src/trajectorysampling.jl`).
#
# The sampler is a product of orthogonal choices, so these are organized the
# same way: the selection policies, then each score family, then the concrete
# transition, then the rollout, the termination rules, the Gauss-Seidel
# batching, and finally end-to-end parity against robust value iteration.

@testitem "TrajectorySampling: selection policies" tags =
    [:base, :trajectory_sampling, :trajectory_policies] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    @testset "constructor validation" begin
        @test_throws ArgumentError TS.EpsilonGreedy(-0.1)
        @test_throws ArgumentError TS.EpsilonGreedy(1.5)
        @test_throws ArgumentError TS.Boltzmann(0.0)
        @test_throws ArgumentError TS.Boltzmann(-1.0)
        @test TS.EpsilonGreedy(0.0).p == 0.0
        @test TS.Boltzmann(2.0).T == 2.0
    end

    cands = [:a, :b, :c]
    scores = [1.0, 5.0, 2.0]

    @testset "EpsilonGreedy(0) is deterministic argmax" begin
        @test all(_ -> TS._select(TS.EpsilonGreedy(0.0), cands, scores) === :b, 1:50)
    end

    @testset "EpsilonGreedy(1) is uniform" begin
        draws = Set(TS._select(TS.EpsilonGreedy(1.0), cands, scores) for _ in 1:400)
        @test draws == Set(cands)
    end

    @testset "Boltzmann concentrates as T -> 0 and flattens as T -> inf" begin
        # Score gaps are 3 and 4; at T = 0.01 that is exp(-300), so the argmax
        # is drawn with overwhelming probability.
        @test all(_ -> TS._select(TS.Boltzmann(0.01), cands, scores) === :b, 1:50)
        draws = Set(TS._select(TS.Boltzmann(1000.0), cands, scores) for _ in 1:400)
        @test draws == Set(cands)
    end

    @testset "-Inf is never selected unless every score is -Inf" begin
        # `LogScore` produces -Inf for a non-positive inner score.
        partial = [-Inf, 1.0, -Inf]
        @test all(_ -> TS._select(TS.EpsilonGreedy(0.0), cands, partial) === :b, 1:50)
        @test all(_ -> TS._select(TS.Boltzmann(1.0), cands, partial) === :b, 1:100)

        # All -Inf: no signal to act on, so both policies fall back to uniform
        # rather than erroring or always answering the first index.
        allneg = [-Inf, -Inf, -Inf]
        @test Set(TS._select(TS.EpsilonGreedy(0.0), cands, allneg) for _ in 1:400) ==
              Set(cands)
        @test Set(TS._select(TS.Boltzmann(1.0), cands, allneg) for _ in 1:400) == Set(cands)
    end
end

@testitem "TrajectorySampling: action scores f_A" tags =
    [:base, :trajectory_sampling, :trajectory_action_scores] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    # One ambiguity set, two targets: lower = [0.2, 0.3], upper = [0.6, 0.8],
    # so the probability budget above the lower bounds is 1 - 0.5 = 0.5.
    as_matrix_l = reshape([0.2, 0.3], 2, 1)
    as_matrix_u = reshape([0.6, 0.8], 2, 1)
    amb = IntervalAmbiguitySets(; lower = as_matrix_l, upper = as_matrix_u)
    mdp = IntervalMarkovDecisionProcess([amb, amb], [1])
    as = IntervalMDP._omax_marginal(mdp)[CartesianIndex(1), CartesianIndex(1)]

    U = [1.0, 0.0]
    L = [0.5, 0.0]

    # dir = true (O-maximization): fill the higher-valued target first.
    #   target 1 takes min(0.5, 0.6-0.2) = 0.4 -> p = [0.6, 0.4]
    #   U-expectation = 0.6*1.0 + 0.4*0.0 = 0.6
    #   L-expectation = 0.6*0.5 + 0.4*0.0 = 0.3
    @testset "upper / lower / weighted-average, O-maximization" begin
        @test TS._action_score(TS.UpperBoundScore(), as, U, L, true) ≈ 0.6
        @test TS._action_score(TS.LowerBoundScore(), as, U, L, true) ≈ 0.3
        # L + beta*(U - L)
        @test TS._action_score(TS.WeightedAverageScore(0.0), as, U, L, true) ≈ 0.3
        @test TS._action_score(TS.WeightedAverageScore(1.0), as, U, L, true) ≈ 0.6
        @test TS._action_score(TS.WeightedAverageScore(0.5), as, U, L, true) ≈ 0.45
    end

    # dir = false (O-minimization): fill the lower-valued target first.
    #   target 2 takes min(0.5, 0.8-0.3) = 0.5 -> p = [0.2, 0.8]
    #   U-expectation = 0.2
    @testset "O-minimization flips the fill order" begin
        @test TS._action_score(TS.UpperBoundScore(), as, U, L, false) ≈ 0.2
        @test TS._action_score(TS.LowerBoundScore(), as, U, L, false) ≈ 0.1
    end

    @testset "WeightedAverageScore validates beta in [0, 1]" begin
        @test_throws ArgumentError TS.WeightedAverageScore(-0.1)
        @test_throws ArgumentError TS.WeightedAverageScore(1.1)
    end

    @testset "LogScore wraps any inner score" begin
        @test TS._action_score(TS.LogScore(TS.UpperBoundScore()), as, U, L, true) ≈ log(0.6)
        @test TS._action_score(TS.LogScore(TS.LowerBoundScore()), as, U, L, true) ≈ log(0.3)
        @test TS._action_score(TS.LogScore(TS.WeightedAverageScore(0.5)), as, U, L, true) ≈
              log(0.45)
    end

    @testset "_safe_log maps non-positive to -Inf, never NaN" begin
        @test TS._safe_log(1.0) == 0.0
        @test TS._safe_log(0.5) ≈ log(0.5)
        @test TS._safe_log(0.0) == -Inf
        # A reach-avoid value function carries negative values on avoid states
        # until `postprocess_value_function!` runs; `log` would give NaN, which
        # would then poison `maximum`/`argmax` in `_select`.
        @test TS._safe_log(-0.25) == -Inf
        @test !isnan(TS._safe_log(-0.25))
    end

    @testset "LogScore is monotone in its inner score" begin
        zero_U = [0.0, 0.0]
        @test TS._action_score(TS.LogScore(TS.UpperBoundScore()), as, zero_U, L, true) ==
              -Inf
    end
end

@testitem "TrajectorySampling: concrete transition" tags =
    [:base, :trajectory_sampling, :trajectory_transition] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    amb = IntervalAmbiguitySets(;
        lower = reshape([0.2, 0.3], 2, 1),
        upper = reshape([0.6, 0.8], 2, 1),
    )
    mdp = IntervalMarkovDecisionProcess([amb, amb], [1])
    as = IntervalMDP._omax_marginal(mdp)[CartesianIndex(1), CartesianIndex(1)]

    vf = (upper = (current = [1.0, 0.0],), lower = (current = [0.0, 1.0],))

    @testset "bound selects which value function is optimized against" begin
        @test TS._bound_values(IntervalMDP.Upper, vf) == [1.0, 0.0]
        @test TS._bound_values(IntervalMDP.Lower, vf) == [0.0, 1.0]
        @test TS.ConcreteTransition().bound === IntervalMDP.Upper
        @test TS.ConcreteTransition(; bound = IntervalMDP.Lower).bound === IntervalMDP.Lower
    end

    @testset "adversary defaults to the spec's direction" begin
        opt = Specification(InfiniteTimeReachability([2], 1e-6), Optimistic, Maximize)
        pes = Specification(InfiniteTimeReachability([2], 1e-6), Pessimistic, Maximize)

        @test TS._adversary_direction(nothing, opt) == true
        @test TS._adversary_direction(nothing, pes) == false
        # Unit tests that drive the helpers without a full Specification.
        @test TS._adversary_direction(nothing, nothing) == true
        # An explicit adversary overrides the spec in both directions.
        @test TS._adversary_direction(Optimistic, pes) == true
        @test TS._adversary_direction(Pessimistic, opt) == false
        @test TS.ConcreteTransition().adversary === nothing
    end

    @testset "all four (bound, adversary) combinations realize the right p" begin
        U, L = [1.0, 0.0], [0.0, 1.0]
        # Against U with argmax: fill target 1 (the higher-valued one).
        @test IntervalMDP._omax_distribution(as, U, true) ≈ [0.6, 0.4]
        # Against U with argmin: fill target 2 first.
        @test IntervalMDP._omax_distribution(as, U, false) ≈ [0.2, 0.8]
        # Against L the value order is reversed, so the two swap.
        @test IntervalMDP._omax_distribution(as, L, true) ≈ [0.2, 0.8]
        @test IntervalMDP._omax_distribution(as, L, false) ≈ [0.6, 0.4]
    end
end

@testitem "TrajectorySampling: successor scores f_S" tags =
    [:base, :trajectory_sampling, :trajectory_state_scores] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    probs = [0.6, 0.4]
    supp = [1, 2]
    vf = (upper = (current = [1.0, 0.5],), lower = (current = [0.2, 0.1],))
    # f_greedy (Upper) = [1.0, 0.5];  f_explore = U - L = [0.8, 0.4]
    # p * f_greedy = [0.6, 0.2]

    @testset "GreedyScore" begin
        @test TS._state_scores(TS.GreedyScore(), probs, supp, vf) ≈ [0.6, 0.2]
        # `bound` picks which value function f_greedy reads.
        @test TS._state_scores(
            TS.GreedyScore(; bound = IntervalMDP.Lower),
            probs,
            supp,
            vf,
        ) ≈ [0.6 * 0.2, 0.4 * 0.1]
    end

    @testset "ExplorationScore adds beta * p * (U - L)" begin
        # [0.6*(1.0 + 1*0.8), 0.4*(0.5 + 1*0.4)] = [1.08, 0.36]
        @test TS._state_scores(TS.ExplorationScore(1.0), probs, supp, vf) ≈ [1.08, 0.36]
        # beta = 0 reduces exactly to GreedyScore.
        @test TS._state_scores(TS.ExplorationScore(0.0), probs, supp, vf) ≈
              TS._state_scores(TS.GreedyScore(), probs, supp, vf)
        @test_throws ArgumentError TS.ExplorationScore(-1.0)
    end

    @testset "GapWeightedExplorationScore discounts by f_gap(delta)" begin
        # greedy = [0.6, 0.2], best = 0.6, so delta = [0.0, 0.4].
        exp_expected = [
            0.6 + 1.0 * 0.6 * 0.8 * exp(-0.0 / 1.0),
            0.2 + 1.0 * 0.4 * 0.4 * exp(-0.4 / 1.0),
        ]
        @test TS._state_scores(
            TS.GapWeightedExplorationScore(1.0, TS.ExponentialGap(1.0)),
            probs,
            supp,
            vf,
        ) ≈ exp_expected

        poly_expected = [0.6 + 1.0 * 0.6 * 0.8 * 1.0^2, 0.2 + 1.0 * 0.4 * 0.4 * (1 - 0.4)^2]
        @test TS._state_scores(
            TS.GapWeightedExplorationScore(1.0, TS.PolynomialGap(2.0)),
            probs,
            supp,
            vf,
        ) ≈ poly_expected
    end

    @testset "delta is non-negative and zero at the best successor" begin
        # delta = max_x p(x)V(x) - p(s')V(s'), both terms weighted, so the
        # best successor scores delta = 0 and every other one delta > 0.
        for pr in ([0.6, 0.4], [0.1, 0.9], [0.5, 0.5])
            greedy = [pr[i] * vf.upper.current[i] for i in supp]
            deltas = maximum(greedy) .- greedy
            @test all(>=(0.0), deltas)
            @test minimum(deltas) == 0.0
        end
    end

    @testset "gap functions" begin
        @test TS._gap_weight(TS.ExponentialGap(2.0), 0.0) == 1.0
        @test TS._gap_weight(TS.ExponentialGap(2.0), 1.0) ≈ exp(-0.5)
        @test TS._gap_weight(TS.PolynomialGap(3.0), 0.0) == 1.0
        @test TS._gap_weight(TS.PolynomialGap(3.0), 0.25) ≈ 0.75^3
        # delta > 1 would make (1-delta) negative; clamped so a fractional tau
        # cannot produce a complex result.
        @test TS._gap_weight(TS.PolynomialGap(0.5), 2.0) == 0.0
        @test_throws ArgumentError TS.ExponentialGap(0.0)
        @test_throws ArgumentError TS.PolynomialGap(-1.0)
    end
end

@testitem "TrajectorySampling: epsilon-greedy ignores the exploration terms" tags =
    [:base, :trajectory_sampling, :trajectory_policy_asymmetry] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    # This pins the one place the implementation is deliberately NOT orthogonal,
    # following the writeup: SampleState's epsilon-greedy branch scores by
    # `p(x) * f_greedy(x)` alone, whatever `state_score` says, while Boltzmann
    # scores by the full f_S. The two therefore disagree whenever the
    # exploration term changes the ranking.
    amb = IntervalAmbiguitySets(;
        lower = reshape([0.5, 0.5], 2, 1),
        upper = reshape([0.5, 0.5], 2, 1),
    )
    mdp = IntervalMarkovDecisionProcess([amb, amb], [1])

    probs = [0.5, 0.5]
    # p*f_greedy = [0.25, 0.225]              -> argmax is state 1
    # f_S (beta = 10) = [0.5, 2.475]          -> argmax is state 2
    vf = (upper = (current = [0.5, 0.45],), lower = (current = [0.45, 0.0],))
    score = TS.ExplorationScore(10.0)
    s, a = CartesianIndex(1), CartesianIndex(1)

    greedy_scores = [probs[i] * vf.upper.current[i] for i in 1:2]
    full_scores = TS._state_scores(score, probs, [1, 2], vf)
    @test argmax(greedy_scores) == 1
    @test argmax(full_scores) == 2   # the two really do disagree

    @testset "EpsilonGreedy exploits on p * f_greedy" begin
        ss = TS.TrajectorySampling(;
            state_policy = TS.EpsilonGreedy(0.0),
            state_score = score,
        )
        @test all(
            _ -> TS._sample_state(ss, s, a, probs, vf, mdp, nothing) == CartesianIndex(1),
            1:50,
        )
    end

    @testset "Boltzmann honours the full f_S" begin
        ss = TS.TrajectorySampling(; state_policy = TS.Boltzmann(0.01), state_score = score)
        @test all(
            _ -> TS._sample_state(ss, s, a, probs, vf, mdp, nothing) == CartesianIndex(2),
            1:50,
        )
    end

    @testset "no successor carries mass -> nothing" begin
        ss = TS.TrajectorySampling()
        @test TS._sample_state(ss, s, a, [0.0, 0.0], vf, mdp, nothing) === nothing
    end
end

@testitem "TrajectorySampling: rollout" tags =
    [:base, :trajectory_sampling, :trajectory_rollout] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    # Every transition is a Dirac point mass on state 3, whatever the source
    # state or action — so the rollout is fully deterministic and the
    # assertions below are exact rather than probabilistic.
    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 1.0 1.0],
        upper = [0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 1.0 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])

    function build_value_function(mdp, prop)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        alg =
            GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
        V = IntervalMDP.construct_value_function(alg, problem)
        IntervalMDP._gsrdp_initialize!(V, prop)
        return V, spec
    end

    @testset "the terminal goal state is NOT part of the trajectory" begin
        # Initial state is 1; one step lands on the goal state 3, which ends the
        # rollout WITHOUT being appended — a state is pushed at the top of a
        # step, so the state that stops the loop is never relaxed.
        prop = InfiniteTimeReachability([3], 1 // 1000)
        V, spec = build_value_function(mdp, prop)
        ss = TS.TrajectorySampling()
        rho = TS._sample_trajectory(ss, mdp, V, spec)
        @test rho == [CartesianIndex(1)]
        @test !(CartesianIndex(3) in rho)
    end

    @testset "an obstacle state is excluded the same way" begin
        prop = InfiniteTimeReachAvoid([2], [3], 1 // 1000)
        V, spec = build_value_function(mdp, prop)
        ss = TS.TrajectorySampling()
        rho = TS._sample_trajectory(ss, mdp, V, spec)
        @test rho == [CartesianIndex(1)]
    end

    @testset "sequence shape and source-state membership" begin
        prop = InfiniteTimeReachability([3], 1 // 1000)
        V, spec = build_value_function(mdp, prop)
        seq = IntervalMDP.sample(TS.TrajectorySampling(), mdp, nothing, V, spec)
        @test IntervalMDP.sequence_shape(seq) === IntervalMDP.StateUpdateSequence()
        states = collect(seq)
        @test all(s -> s in CartesianIndices(IntervalMDP.source_shape(mdp)), states)
    end

    @testset "runs to the step cap when nothing stops it" begin
        # Goal is state 2, which is never reached: 1 -> 3 -> 3 -> ... The
        # default MaxSteps() cap is `num_states(mdp)` = 3 transitions, so three
        # states are appended.
        prop = InfiniteTimeReachability([2], 1 // 1000)
        V, spec = build_value_function(mdp, prop)
        rho = TS._sample_trajectory(TS.TrajectorySampling(), mdp, V, spec)
        @test rho == [CartesianIndex(1), CartesianIndex(3), CartesianIndex(3)]
        @test length(rho) == IntervalMDP.num_states(mdp)
    end

    @testset "reverse puts the goal-adjacent end first" begin
        prop = InfiniteTimeReachability([2], 1 // 1000)
        V, spec = build_value_function(mdp, prop)
        fwd = collect(
            IntervalMDP.sample(
                TS.TrajectorySampling(; reverse = false),
                mdp,
                nothing,
                V,
                spec,
            ),
        )
        rev = collect(
            IntervalMDP.sample(
                TS.TrajectorySampling(; reverse = true),
                mdp,
                nothing,
                V,
                spec,
            ),
        )
        @test fwd == [CartesianIndex(1), CartesianIndex(3), CartesianIndex(3)]
        @test rev == reverse(fwd)
    end

    @testset "implicit sink states are never emitted" begin
        # 2 source states, 3 targets; target 3 is an implicit sink with no
        # ambiguity-set column and no actions. It is absorbing, so the rollout
        # ends on reaching it, and it is excluded since it cannot be relaxed.
        dirac(rows) = IntervalAmbiguitySets(; lower = rows, upper = rows)
        sink_mdp = IntervalMarkovDecisionProcess(
            [dirac([0.0 0.0; 0.0 0.0; 1.0 1.0]), dirac([0.0 0.0; 0.0 0.0; 1.0 1.0])],
            [1],
        )
        @test IntervalMDP.source_shape(sink_mdp) == (2,)
        @test IntervalMDP.state_values(sink_mdp) == (3,)

        prop = InfiniteTimeReachability([3], 1 // 1000)
        V, spec = build_value_function(sink_mdp, prop)
        for _ in 1:20
            rho = TS._sample_trajectory(TS.TrajectorySampling(), sink_mdp, V, spec)
            @test all(s -> IntervalMDP._is_source_state(sink_mdp, s), rho)
            @test !(CartesianIndex(3) in rho)
        end
    end
end

@testitem "TrajectorySampling: termination rules" tags =
    [:base, :trajectory_sampling, :trajectory_terminate] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 1.0 1.0],
        upper = [0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 1.0 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])

    function build_value_function(mdp, prop)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        alg =
            GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
        V = IntervalMDP.construct_value_function(alg, problem)
        IntervalMDP._gsrdp_initialize!(V, prop)
        return V, spec
    end

    # Goal is state 2, never reached, so only the termination rules stop this.
    prop = InfiniteTimeReachability([2], 1 // 1000)
    V, spec = build_value_function(mdp, prop)

    @testset "MaxSteps truncates at n" begin
        for n in 0:4
            rho = TS._sample_trajectory(
                TS.TrajectorySampling(; terminate = [TS.MaxSteps(n)]),
                mdp,
                V,
                spec,
            )
            @test length(rho) == n
        end
        # An explicit MaxSteps larger than the state count is honoured, not
        # silently clamped by the backstop.
        @test length(
            TS._sample_trajectory(
                TS.TrajectorySampling(; terminate = [TS.MaxSteps(7)]),
                mdp,
                V,
                spec,
            ),
        ) == 7
        @test_throws ArgumentError TS.MaxSteps(-1)
        @test TS.MaxSteps().n === nothing
    end

    @testset "the unconditional backstop applies even with no MaxSteps rule" begin
        rho = TS._sample_trajectory(
            TS.TrajectorySampling(; terminate = TS.TerminationRule[]),
            mdp,
            V,
            spec,
        )
        @test length(rho) == IntervalMDP.num_states(mdp)
    end

    @testset "PredicateStop fires" begin
        stop_at_3 = (s, traj, i, vf, model, sp) -> s == CartesianIndex(3)
        rho = TS._sample_trajectory(
            TS.TrajectorySampling(; terminate = [TS.PredicateStop(stop_at_3)]),
            mdp,
            V,
            spec,
        )
        @test rho == [CartesianIndex(1)]
    end

    @testset "rules OR together" begin
        never = (s, traj, i, vf, model, sp) -> false
        rho = TS._sample_trajectory(
            TS.TrajectorySampling(; terminate = [TS.PredicateStop(never), TS.MaxSteps(1)]),
            mdp,
            V,
            spec,
        )
        @test length(rho) == 1
    end

    @testset "ExpectedGapStop: the BRTDP arithmetic" begin
        @test_throws ArgumentError TS.ExpectedGapStop(0.0)
        @test_throws ArgumentError TS.ExpectedGapStop(-1.0)
        @test TS.ExpectedGapStop().tau == 10.0

        s, a = CartesianIndex(1), CartesianIndex(1)
        # gap = U - L = [0.5, 0.2, 0.0]; at s = 1 the open gap is 0.5.
        vf = (upper = (current = [1.0, 0.4, 0.0],), lower = (current = [0.5, 0.2, 0.0],))
        # B = dot(probs, gap). With probs = [0, 1, 0], B = 0.2.
        probs = [0.0, 1.0, 0.0]
        stop(tau) = TS.terminate_post(
            TS.ExpectedGapStop(tau),
            s,
            a,
            probs,
            CartesianIndex[],
            0,
            vf,
            mdp,
            nothing,
        )
        # Stop when B < diff/tau, i.e. 0.2 < 0.5/tau, i.e. tau < 2.5.
        @test stop(2.0)            # 0.5/2.0 = 0.25 > 0.2 -> stop
        @test !stop(3.0)           # 0.5/3.0 ≈ 0.167 < 0.2 -> continue
        # Exactly at the threshold: 0.5/2.5 = 0.2, and the test is strict `<`.
        @test !stop(2.5)
    end

    @testset "ExpectedGapStop: a converged state ends the rollout" begin
        s, a = CartesianIndex(1), CartesianIndex(1)
        # U(s) == L(s), so the threshold is 0 and B >= 0 can never fall below
        # it — without this rule the trajectory would run to the step cap
        # through an already-tight region.
        vf = (upper = (current = [1.0, 1.0, 1.0],), lower = (current = [1.0, 1.0, 1.0],))
        @test TS.terminate_post(
            TS.ExpectedGapStop(10.0),
            s,
            a,
            [0.0, 1.0, 0.0],
            CartesianIndex[],
            0,
            vf,
            mdp,
            nothing,
        )
    end

    @testset "default hooks are false" begin
        # A rule implements only the hook it needs; the other defaults to false.
        @test !TS.terminate_post(
            TS.MaxSteps(1),
            CartesianIndex(1),
            CartesianIndex(1),
            [1.0, 0.0, 0.0],
            CartesianIndex[],
            0,
            V,
            mdp,
            spec,
        )
        @test !TS.terminate_pre(
            TS.ExpectedGapStop(10.0),
            CartesianIndex(1),
            CartesianIndex[],
            0,
            V,
            mdp,
            spec,
        )
    end
end

@testitem "TrajectorySampling: Gauss-Seidel batching" tags =
    [:base, :trajectory_sampling, :trajectory_gauss_seidel] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    prob = IntervalAmbiguitySets(;
        lower = [0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 1.0 1.0],
        upper = [0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 1.0 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob, prob], [1])

    # Goal is state 2 (never reached), so the deterministic rollout is exactly
    # [1, 3, 3] forward / [3, 3, 1] reversed, every time.
    prop = InfiniteTimeReachability([2], 1 // 1000)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)
    alg = GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
    V = IntervalMDP.construct_value_function(alg, problem)
    IntervalMDP._gsrdp_initialize!(V, prop)

    batch(ss) = collect(IntervalMDP.sample(ss, mdp, nothing, V, spec))
    full_reversed = [CartesianIndex(3), CartesianIndex(3), CartesianIndex(1)]

    @testset "gauss_seidel = false returns the whole trajectory every call" begin
        ss = TS.TrajectorySampling(; gauss_seidel = false)
        @test batch(ss) == full_reversed
        @test batch(ss) == full_reversed
    end

    @testset "k-sized batches partition the trajectory in order" begin
        ss = TS.TrajectorySampling(; gauss_seidel = true, k = 2)
        b1, b2 = batch(ss), batch(ss)
        @test b1 == full_reversed[1:2]
        @test b2 == full_reversed[3:3]
        @test vcat(b1, b2) == full_reversed
        # Exhausted, so the next call samples a fresh trajectory.
        @test batch(ss) == full_reversed[1:2]
    end

    @testset "k = 1, forward order" begin
        ss = TS.TrajectorySampling(; gauss_seidel = true, k = 1, reverse = false)
        forward = [CartesianIndex(1), CartesianIndex(3), CartesianIndex(3)]
        @test [only(batch(ss)) for _ in 1:3] == forward
    end

    @testset "k larger than the trajectory yields it in one batch" begin
        ss = TS.TrajectorySampling(; gauss_seidel = true, k = 99)
        @test batch(ss) == full_reversed
        @test batch(ss) == full_reversed
    end

    @testset "reset_sampling_strategy! discards a partial trajectory" begin
        ss = TS.TrajectorySampling(; gauss_seidel = true, k = 2)
        @test batch(ss) == full_reversed[1:2]
        IntervalMDP.reset_sampling_strategy!(ss)
        @test ss.buffer[] === nothing
        @test ss.cursor[] == 0
        # A fresh trajectory, not the tail of the abandoned one.
        @test batch(ss) == full_reversed[1:2]
    end

    @testset "k must be positive when gauss_seidel = true" begin
        @test_throws ArgumentError TS.TrajectorySampling(; gauss_seidel = true, k = 0)
        # Irrelevant, and hence unvalidated, when batching is off.
        @test TS.TrajectorySampling(; gauss_seidel = false, k = 0) isa TS.TrajectorySampling
    end

    @testset "an empty trajectory returns an empty sequence, not a hang" begin
        # MaxSteps(0) makes the rollout stop before appending anything.
        for gs in (false, true)
            ss = TS.TrajectorySampling(;
                gauss_seidel = gs,
                k = 2,
                terminate = [TS.MaxSteps(0)],
            )
            @test isempty(batch(ss))
            @test isempty(batch(ss))
        end
    end
end

@testitem "TrajectorySampling: end-to-end solve() parity with RVI" tags =
    [:base, :trajectory_sampling, :trajectory_solve] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    @testset "IMDP verification parity (Pessimistic, Maximize), $N" for N in
                                                                        [Float32, Float64]
        prob = IntervalAmbiguitySets(;
            lower = N[0 1//2 0; 1//10 3//10 0; 1//5 1//10 1],
            upper = N[1//2 7//10 0; 3//5 1//2 0; 7//10 3//10 1],
        )
        prob2 = IntervalAmbiguitySets(;
            lower = N[1//10 1//5 0; 1//5 1//5 0; 3//10 2//5 1],
            upper = N[1//2 1//2 0; 1//2 2//5 0; 2//5 2//5 1],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob2, prob2], [1])
        eps = N(1 // 1000000)
        prop = InfiniteTimeReachability([3], eps)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)

        (V_rvi, _, _) = solve(problem, RobustValueIteration(default_bellman_algorithm(mdp)))

        strategies = [
            "default" => TS.TrajectorySampling(),
            "boltzmann + exploration" => TS.TrajectorySampling(;
                action_policy = TS.Boltzmann(0.5),
                state_policy = TS.Boltzmann(0.5),
                state_score = TS.ExplorationScore(1.0),
            ),
            "gap-weighted (exponential)" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(0.5),
                state_score = TS.GapWeightedExplorationScore(1.0, TS.ExponentialGap(1.0)),
            ),
            "gap-weighted (polynomial)" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(0.5),
                state_score = TS.GapWeightedExplorationScore(1.0, TS.PolynomialGap(2.0)),
            ),
            "expected-gap stop" => TS.TrajectorySampling(;
                terminate = [TS.MaxSteps(), TS.ExpectedGapStop(10.0)],
            ),
            "gauss-seidel k = 2" => TS.TrajectorySampling(; gauss_seidel = true, k = 2),
            "lower-bound transition" => TS.TrajectorySampling(;
                transition = TS.ConcreteTransition(; bound = IntervalMDP.Lower),
            ),
        ]

        @testset "$name" for (name, ss) in strategies
            gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
                default_bellman_algorithm(mdp);
                sampling_strategy = ss,
            )
            (V, _, _) = solve(problem, gsdp)
            @test maximum(abs, V_rvi .- V) <= 2 * eps
        end
    end
end

@testitem "TrajectorySampling: parity on implicit vs explicit sink, and synthesis" tags =
    [:base, :trajectory_sampling, :trajectory_sink_solve] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    # The same system written two ways: `explicit_mdp` spells out state 3's
    # absorbing self-loop as a third source column; `implicit_mdp` omits it and
    # lets state 3 be an implicit sink. Both must give the same values. Unlike
    # the three-state fixture above, this one has non-trivial values, so it
    # actually discriminates between a correct and an incorrect sampler.
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

    eps = 1e-6
    prop = InfiniteTimeReachability([3], eps)
    spec = Specification(prop, Pessimistic, Maximize)

    (V_ref, _, _) = solve(
        VerificationProblem(explicit_mdp, spec),
        RobustValueIteration(default_bellman_algorithm(explicit_mdp)),
    )

    strategies = [
        "default" => TS.TrajectorySampling(),
        "boltzmann + gap-weighted" => TS.TrajectorySampling(;
            state_policy = TS.Boltzmann(0.5),
            state_score = TS.GapWeightedExplorationScore(1.0, TS.ExponentialGap(1.0)),
        ),
        "gauss-seidel k = 2" => TS.TrajectorySampling(; gauss_seidel = true, k = 2),
    ]

    @testset "$name on $label" for (name, ss) in strategies,
        (label, m) in ("explicit" => explicit_mdp, "implicit" => implicit_mdp)

        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(m);
            sampling_strategy = ss,
        )
        (V, _, _) = solve(VerificationProblem(m, spec), gsdp)
        @test maximum(abs, V_ref .- V) <= 1e-4
    end

    @testset "control synthesis runs and agrees with verification" begin
        gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(implicit_mdp);
            sampling_strategy = TS.TrajectorySampling(),
        )
        sol = solve(ControlSynthesisProblem(implicit_mdp, spec), gsdp)
        # `checkstrategy` asserts the shape matches `source_shape` and every
        # action is in range — i.e. nothing wrote through the sink.
        @test IntervalMDP.checkstrategy(strategy(sol), implicit_mdp) === nothing
    end
end
