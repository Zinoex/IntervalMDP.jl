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
        @test TS.Boltzmann(2.0).T == TS.FixedTemperature(2.0)
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

    # A minimizing controller is scored on the complemented bracket U' = 1 - L,
    # L' = 1 - U, so "upper" stays controller-optimistic (lowest plausible value).
    @testset "Minimize scores the complemented bracket" begin
        @test TS._action_score(TS.UpperBoundScore(), as, U, L, true; maximize = false) ≈ 0.7
        @test TS._action_score(TS.LowerBoundScore(), as, U, L, true; maximize = false) ≈ 0.4
        @test TS._action_score(TS.WeightedAverageScore(0.5), as, U, L, true; maximize = false) ≈
              0.55
        @test TS._action_score(TS.LogScore(TS.UpperBoundScore()), as, U, L, true; maximize = false) ≈
              log(0.7)
        # The keyword defaults to Maximize.
        @test TS._action_score(TS.UpperBoundScore(), as, U, L, true; maximize = true) ==
              TS._action_score(TS.UpperBoundScore(), as, U, L, true)
    end
end

@testitem "TrajectorySampling: rollout action follows the strategy mode" tags =
    [:base, :trajectory_sampling, :trajectory_action_direction] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    # State 1 has two deterministic actions: a1 -> state 2, a2 -> state 3; states 2-4 are
    # absorbing. Q-values are then just U and L at states 2 and 3. A greedy rollout must
    # pick argmax U under Maximize and argmin L (the minimizer's optimistic action) under
    # Minimize — not argmax U (the old direction-blind rule) and not argmin U.
    det(t) = Float64.(1:4 .== t)
    s1 = IntervalAmbiguitySets(; lower = hcat(det(2), det(3)), upper = hcat(det(2), det(3)))
    absorb(t) = IntervalAmbiguitySets(; lower = hcat(det(t), det(t)), upper = hcat(det(t), det(t)))
    mdp = IntervalMarkovDecisionProcess([s1, absorb(2), absorb(3), absorb(4)], [1])
    marginal = IntervalMDP._omax_marginal(mdp)
    ss = TS.TrajectorySampling(; action_policy = TS.EpsilonGreedy(0.0))
    pick(vf, strat) = Tuple(TS._sample_action(
        ss, CartesianIndex(1), vf, mdp,
        Specification(InfiniteTimeReachability([4], 1e-6), Pessimistic, strat),
        marginal, false,
    ))[1]
    bracket(u2, l2, u3, l3) = (upper = (current = [1.0, u2, u3, 1.0],),
                               lower = (current = [0.0, l2, l3, 1.0],))

    # a1: [.25, .9], a2: [.2, .3]. argmax U = a1, argmin L = a2.
    vf = bracket(0.9, 0.25, 0.3, 0.2)
    @test pick(vf, Maximize) == 1
    @test pick(vf, Minimize) == 2      # argmax U would give a1

    # a1: [.05, .2], a2: [.08, .1]. argmin U = a2, argmin L = a1.
    vf = bracket(0.2, 0.05, 0.1, 0.08)
    @test pick(vf, Minimize) == 1      # argmin U would give a2
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

@testitem "TrajectorySampling: exploration terms f_explore" tags =
    [:base, :trajectory_sampling, :trajectory_explore_terms] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    probs = [0.6, 0.4]
    supp = [1, 2]
    # `_state_scores` without the step context, for terms that read only `vf`.
    scores(score, probs, supp, vf) =
        TS._state_scores(score, probs, supp, vf, nothing, nothing, nothing, nothing, nothing)
    # Fresh vectors per call: the Bellman-error tests mutate them.
    fixture() = (upper = (current = [1.0, 0.5],), lower = (current = [0.2, 0.1],))
    vf = fixture()

    @testset "GapExplore is the default" begin
        @test TS.MixtureStateScore().explore isa TS.GapExplore
        @test TS._explore_values(TS.GapExplore(), supp, vf, nothing, nothing) ≈ [0.8, 0.4]
    end

    @testset "UpperBoundExplore reads U" begin
        @test TS._explore_values(TS.UpperBoundExplore(), supp, vf, nothing, nothing) ≈
              [1.0, 0.5]
    end

    @testset "Bellman-error terms track the last backup" begin
        vf = fixture()
        up = TS.BellmanErrorExplore()
        lo = TS.BellmanErrorExplore(; bound = IntervalMDP.Lower)
        mx = TS.MaxBellmanErrorExplore()
        @test up.bound === IntervalMDP.Upper

        # Never backed up: fall back to the gap.
        for term in (up, lo, mx)
            @test TS._explore_values(term, supp, vf, nothing, nothing) ≈ [0.8, 0.4]
        end

        # Hand state 1 to GSRDP, "back it up", and measure on the next call.
        for term in (up, lo, mx)
            TS._after_sample!(term, [CartesianIndex(1)], vf)
        end
        vf.upper.current[1] = 0.9
        vf.lower.current[1] = 0.25
        for term in (up, lo, mx)
            TS._before_sample!(term, vf)
        end
        # State 2 was never backed up, so it still reads its gap (0.4).
        @test TS._explore_values(up, supp, vf, nothing, nothing) ≈ [0.1, 0.4]
        @test TS._explore_values(lo, supp, vf, nothing, nothing) ≈ [0.05, 0.4]
        @test TS._explore_values(mx, supp, vf, nothing, nothing) ≈ [0.1, 0.4]

        # A second measurement with nothing pending leaves the readings alone.
        TS._before_sample!(up, vf)
        @test TS._explore_values(up, supp, vf, nothing, nothing) ≈ [0.1, 0.4]

        # A backup that did not move the value reads 0, not the gap.
        TS._after_sample!(up, [CartesianIndex(1)], vf)
        TS._before_sample!(up, vf)
        @test TS._explore_values(up, supp, vf, nothing, nothing) ≈ [0.0, 0.4]

        TS._reset_score!(up)
        @test TS._explore_values(up, supp, vf, nothing, nothing) ≈ [0.65, 0.4]
    end

    @testset "hooks are forwarded through the successor scores" begin
        vf = fixture()
        term = TS.BellmanErrorExplore()
        score = TS.LogStateScore(TS.MixtureStateScore(; explore = term, alpha_explore = 1.0))
        TS._after_sample!(score, [CartesianIndex(2)], vf)
        vf.upper.current[2] = 0.3
        TS._before_sample!(score, vf)
        # dU = [NaN -> gap 0.8, 0.2], so p·f_explore = [0.48, 0.08].
        @test scores(score, probs, supp, vf) ≈ log.([0.48, 0.08] ./ 0.56)
        TS._reset_score!(score)
        @test all(isnan, term.tracker.dU[]) || isempty(term.tracker.dU[])

        # A term without a tracker accepts the hooks and does nothing.
        @test TS._before_sample!(TS.MixtureStateScore(), vf) === nothing
        @test TS._after_sample!(TS.MixtureStateScore(), [CartesianIndex(1)], vf) === nothing
        @test TS._reset_score!(TS.LogStateScore(TS.MixtureStateScore())) === nothing
    end

    @testset "ActionUncertaintyExplore" begin
        # Every state has two precise actions over {1, 2, 3}; 2 and 3 are
        # absorbing under both, so there U^{-a_L}(t) − L(t) = U(t) − L(t).
        precise(p) = IntervalAmbiguitySets(; lower = p, upper = p)
        mdp = IntervalMarkovDecisionProcess(
            [
                precise([0.0 0.0; 0.5 0.8; 0.5 0.2]),
                precise([0.0 0.0; 1.0 1.0; 0.0 0.0]),
                precise([0.0 0.0; 0.0 0.0; 1.0 1.0]),
            ],
            [1],
        )
        vf = (upper = (current = [1.0, 0.8, 0.6],), lower = (current = [0.0, 0.3, 0.1],))
        term = TS.ActionUncertaintyExplore()

        # At state 1: L(1, a1) = 0.5·0.3 + 0.5·0.1 = 0.2 and
        # L(1, a2) = 0.8·0.3 + 0.2·0.1 = 0.26, so a_L = a2 and
        # U^{-a_L}(1) = U(1, a1) = 0.5·0.8 + 0.5·0.6 = 0.7.
        x = TS._explore_values(term, [1, 2, 3], vf, mdp, nothing)
        @test x ≈ [0.7 - 0.26, 0.5, 0.5]
        @test x ≈ [
            IntervalMDP._action_uncertainty(mdp, CartesianIndex(t), vf, nothing) for
            t in 1:3
        ]

        tr = TS.ConcreteTransition()
        score = TS.MixtureStateScore(; explore = term)
        p1 = [0.0, 0.5, 0.5]
        # p·U = [0.4, 0.3] and p·f_explore = [0.25, 0.25], each normalized.
        @test TS._state_scores(
            score,
            p1,
            [2, 3],
            vf,
            CartesianIndex(1),
            CartesianIndex(1),
            mdp,
            nothing,
            tr,
        ) ≈ 0.5 .* [4 / 7, 3 / 7] .+ 0.5 .* [0.5, 0.5]

        # Without the model there is nothing to compute it from.
        @test_throws ArgumentError scores(score, p1, [2, 3], vf)
    end

    @testset "_uniform_excess is E[max(0, q - c)] for uniform q" begin
        n = 200_000
        for (l, u, c) in ((0.2, 0.7, 0.51), (0.2, 0.7, 0.1), (0.2, 0.7, 0.9), (0.0, 1.0, 0.5))
            qs = range(l, u; length = n)
            @test TS._uniform_excess(l, u, c) ≈ sum(q -> max(0.0, q - c), qs) / n atol = 1e-5
        end
        # A point interval (a converged Q-value) never divides by zero.
        @test TS._uniform_excess(0.4, 0.4, 0.4) == 0.0
        @test TS._uniform_excess(0.4, 0.4, 0.1) ≈ 0.3
    end

    @testset "VPIExplore" begin
        precise(p) = IntervalAmbiguitySets(; lower = p, upper = p)
        mdp = IntervalMarkovDecisionProcess(
            [
                precise([0.0 0.0; 0.5 0.8; 0.5 0.2]),
                precise([0.0 0.0; 1.0 1.0; 0.0 0.0]),
                precise([0.0 0.0; 0.0 0.0; 1.0 1.0]),
            ],
            [1],
        )
        vf = (upper = (current = [1.0, 0.8, 0.6],), lower = (current = [0.0, 0.3, 0.1],))
        term = TS.VPIExplore()

        # State 1: Q(1, a1) ∈ [0.2, 0.7] (m = 0.45), Q(1, a2) ∈ [0.26, 0.76]
        # (m = 0.51), so a₁ = a2 with runner-up 0.45:
        #   VPI(a2) = E[max(0, 0.45 − q)], q ~ U[0.26, 0.76] = 0.19²/1
        #   VPI(a1) = E[max(0, q − 0.51)], q ~ U[0.2, 0.7]   = 0.19²/1
        # States 2 and 3: both actions share one interval of width 0.5, so
        # each side of the tie reads 0.25²/1.
        x = TS._explore_values(term, [1, 2, 3], vf, mdp, nothing)
        @test x ≈ [0.19^2, 0.25^2, 0.25^2]

        # Asymmetric intervals, where only one side can gain. At state 1, a1
        # moves to 2 and a2 to 3 deterministically, so Q(1, a1) = [L(2), U(2)]
        # = [0.2, 0.7] (m = 0.45) and Q(1, a2) = [L(3), U(3)] = [0.5, 0.6]
        # (m = 0.55). a₁ = a2, runner-up 0.45:
        #   VPI(a2) = E[max(0, 0.45 − q)], q ~ U[0.5, 0.6] = 0
        #   VPI(a1) = E[max(0, q − 0.55)], q ~ U[0.2, 0.7] = 0.15²/1
        split = IntervalMarkovDecisionProcess(
            [
                precise([0.0 0.0; 1.0 0.0; 0.0 1.0]),
                precise([0.0 0.0; 1.0 1.0; 0.0 0.0]),
                precise([0.0 0.0; 0.0 0.0; 1.0 1.0]),
            ],
            [1],
        )
        vf2 = (upper = (current = [1.0, 0.7, 0.6],), lower = (current = [0.0, 0.2, 0.5],))
        @test TS._state_vpi(split, CartesianIndex(1), vf2, nothing) ≈ 0.15^2

        # Minimize: a1 is now best (m = 0.45 < 0.55) with runner-up 0.55, so
        # the gain is a1 turning out *above* 0.55, again 0.15²/1, and a2 (all of
        # it above 0.45) cannot gain.
        spec_min = Specification(InfiniteTimeReachability([3], 1e-6), Optimistic, Minimize)
        @test TS._state_vpi(split, CartesianIndex(1), vf2, spec_min) ≈ 0.15^2
        # Narrowing a1 to [0.4, 0.5] leaves no chance of crossing either way.
        vf3 = (upper = (current = [1.0, 0.5, 0.6],), lower = (current = [0.0, 0.4, 0.5],))
        @test TS._state_vpi(split, CartesianIndex(1), vf3, nothing) == 0.0
        @test TS._state_vpi(split, CartesianIndex(1), vf3, spec_min) == 0.0

        # A single-action state has no decision to be uncertain about.
        single = IntervalMarkovDecisionProcess(
            [
                precise(reshape([0.0, 0.5, 0.5], 3, 1)),
                precise(reshape([0.0, 1.0, 0.0], 3, 1)),
                precise(reshape([0.0, 0.0, 1.0], 3, 1)),
            ],
            [1],
        )
        @test TS._explore_values(term, [1, 2, 3], vf, single, nothing) == [0.0, 0.0, 0.0]

        score = TS.MixtureStateScore(; explore = term)
        @test_throws ArgumentError scores(score, [0.0, 0.5, 0.5], [2, 3], vf)
    end
end

@testitem "TrajectorySampling: epsilon-greedy ignores the exploration terms" tags =
    [:base, :trajectory_sampling, :trajectory_policy_asymmetry] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    # This pins the one place the implementation is deliberately NOT orthogonal,
    # following the writeup: SampleState's epsilon-greedy branch scores by
    # `p(x) * f_exploit(x)` alone, whatever `state_score` says, while Boltzmann
    # scores by the full f_S. The two therefore disagree whenever the
    # exploration term changes the ranking.
    amb = IntervalAmbiguitySets(;
        lower = reshape([0.5, 0.5], 2, 1),
        upper = reshape([0.5, 0.5], 2, 1),
    )
    mdp = IntervalMarkovDecisionProcess([amb, amb], [1])

    probs = [0.5, 0.5]
    # p*f_exploit = [0.25, 0.225]             -> argmax is state 1
    # f_S (explore only) = [0.1, 0.9]         -> argmax is state 2
    vf = (upper = (current = [0.5, 0.45],), lower = (current = [0.45, 0.0],))
    score = TS.MixtureStateScore(; alpha_explore = 1.0)
    s, a = CartesianIndex(1), CartesianIndex(1)

    exploit_scores = [probs[i] * vf.upper.current[i] for i in 1:2]
    full_scores = TS._state_scores(score, probs, [1, 2], vf, nothing, nothing, nothing, nothing, nothing)
    @test argmax(exploit_scores) == 1
    @test argmax(full_scores) == 2   # the two really do disagree

    @testset "EpsilonGreedy exploits on p * f_exploit" begin
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

@testitem "TrajectorySampling: exploit/explore mixture score" tags =
    [:base, :trajectory_sampling, :trajectory_state_scores] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    probs = [0.5, 0.5]
    supp = [1, 2]
    vf = (upper = (current = [1.0, 0.5],), lower = (current = [0.9, 0.0],))
    # p·f_exploit (U)  = [0.5, 0.25] -> f'_exploit = [2/3, 1/3]
    # p·f_explore (gap) = [0.05, 0.25] -> f'_explore = [1/6, 5/6]
    exploit = [2 / 3, 1 / 3]
    explore = [1 / 6, 5 / 6]
    mix(ae, ap = 0.0) = TS.MixtureStateScore(; alpha_explore = ae, alpha_prob = ap)
    scores(score, probs, supp, vf) =
        TS._state_scores(score, probs, supp, vf, nothing, nothing, nothing, nothing, nothing)

    @testset "alpha schedules" begin
        @test_throws ArgumentError TS.FixedAlpha(-0.1)
        @test_throws ArgumentError TS.FixedAlpha(1.1)
        @test_throws ArgumentError TS.PowerAlpha(0.0)
        @test_throws ArgumentError TS.ClippedAlpha(TS.PowerAlpha(1.0), 0.6, 0.4)
        @test_throws ArgumentError TS.ClippedAlpha(TS.PowerAlpha(1.0), -0.1, 0.4)
        @test TS._alpha(TS.FixedAlpha(0.3), 0.9) == 0.3
        @test TS._alpha(TS.PowerAlpha(2.0), 0.5) ≈ 0.25
        @test TS._alpha(TS.PowerAlpha(1.0), 0.0) == 0.0
        @test TS._alpha(TS.CustomAlpha(d -> 2d), 0.25) ≈ 0.5
        @test TS._alpha(TS.CustomAlpha(d -> 2d), 0.75) == 1.0   # clamped
        clipped = TS.ClippedAlpha(TS.PowerAlpha(1.0), 0.1, 0.9)
        @test TS._alpha(clipped, 1.0) == 0.9
        @test TS._alpha(clipped, 0.0) == 0.1
        @test TS._alpha(clipped, 0.5) == 0.5
    end

    @testset "defaults: exploit Upper, gap explore, alpha_e = 0.5, alpha_p = 0" begin
        score = TS.MixtureStateScore()
        @test score.exploit === IntervalMDP.Upper
        @test score.explore isa TS.GapExplore
        @test score.alpha_explore == TS.FixedAlpha(0.5)
        @test score.alpha_prob == TS.FixedAlpha(0.0)
        @test TS._exploit_bound(score) === IntervalMDP.Upper
        @test TS._exploit_bound(TS.MixtureStateScore(; exploit = IntervalMDP.Lower)) ===
              IntervalMDP.Lower
    end

    @testset "alpha_e mixes the two normalized terms" begin
        @test scores(mix(0.0), probs, supp, vf) ≈ exploit
        @test scores(mix(1.0), probs, supp, vf) ≈ explore
        @test scores(mix(0.5), probs, supp, vf) ≈ (exploit .+ explore) ./ 2
        @test sum(scores(mix(0.3), probs, supp, vf)) ≈ 1
    end

    @testset "alpha_p flattens p towards uniform" begin
        skewed = [0.8, 0.2]
        # alpha_p = 1 -> p' uniform, so the skew is gone.
        @test scores(mix(0.0, 1.0), skewed, supp, vf) ≈ exploit
        pp = 0.5 .* skewed .+ 0.25
        w = pp .* vf.upper.current
        @test scores(mix(0.0, 0.5), skewed, supp, vf) ≈ w ./ sum(w)
    end

    @testset "exploit reads the chosen bound" begin
        score = TS.MixtureStateScore(; exploit = IntervalMDP.Lower, alpha_explore = 0.0)
        @test scores(score, probs, supp, vf) ≈ [1.0, 0.0]
    end

    @testset "a term that vanishes is uniform; negatives are clamped" begin
        zero_vf = (upper = (current = [0.0, -0.5],), lower = (current = [0.0, -0.5],))
        @test scores(mix(0.0), probs, supp, zero_vf) ≈ [0.5, 0.5]
        @test scores(mix(1.0), probs, supp, zero_vf) ≈ [0.5, 0.5]
    end

    @testset "explore term is pluggable" begin
        score = TS.MixtureStateScore(;
            explore = TS.UpperBoundExplore(),
            alpha_explore = 1.0,
        )
        @test scores(score, probs, supp, vf) ≈ exploit
    end

    @testset "LogStateScore wraps the mixture" begin
        @test scores(TS.LogStateScore(mix(0.5)), probs, supp, vf) ≈
              log.((exploit .+ explore) ./ 2)
        # A successor the mixture gives no weight maps to -Inf, never NaN.
        lopsided = (upper = (current = [1.0, 0.0],), lower = (current = [0.0, 0.0],))
        @test scores(TS.LogStateScore(mix(0.0)), probs, supp, lopsided) == [0.0, -Inf]
        @test TS._exploit_bound(TS.LogStateScore(TS.MixtureStateScore(; exploit = IntervalMDP.Lower))) ===
              IntervalMDP.Lower
    end

    @testset "a non-fixed schedule must be resolved first" begin
        score = TS.MixtureStateScore(; alpha_explore = TS.PowerAlpha(1.0))
        @test_throws ArgumentError scores(score, probs, supp, vf)
    end

    amb = IntervalAmbiguitySets(;
        lower = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0],
        upper = [1.0 1.0 0.0; 1.0 1.0 0.0; 1.0 1.0 1.0],
    )
    gap_vf = (
        upper = (current = [0.9, 0.7, 1.0],),
        lower = (current = [0.4, 0.0, 1.0],),
    )

    @testset "initial gap is the max over S0, or over all states" begin
        @test TS._initial_gap(IntervalMarkovDecisionProcess([amb, amb, amb], [1]), gap_vf) ≈
              0.5
        @test TS._initial_gap(IntervalMarkovDecisionProcess([amb, amb, amb], [1, 2]), gap_vf) ≈
              0.7
        @test TS._initial_gap(IntervalMarkovDecisionProcess([amb, amb, amb]), gap_vf) ≈ 0.7
    end

    @testset "the rollout context reads the max gap over S0, not Diff(s0)" begin
        # Whichever of states 1 (Diff 0.5) and 2 (Diff 0.7) the rollout starts
        # from, every schedule sees delta = 0.7.
        ss = TS.TrajectorySampling()
        mdp = IntervalMarkovDecisionProcess([amb, amb, amb], [1, 2])
        @test TS._rollout_context(ss, mdp, gap_vf).diff ≈ 0.7
        sched = TS.GapBasedTemperature(1.0, 0.0)
        @test TS._temperature(sched, TS._rollout_context(ss, mdp, gap_vf)) ≈ 0.7 / 0.3
    end

    @testset "initial_state = :max starts at the widest gap in S0, ties at random" begin
        @test TS.TrajectorySampling().initial_state === :random
        @test TS.TrajectorySampling(; initial_state = :max).initial_state === :max
        @test_throws ArgumentError TS.TrajectorySampling(; initial_state = :min)

        # Diff = (0.5, 0.7, 0.0): state 2 over S0 = {1, 2} and over all states.
        @test TS._max_gap_initial_state(
            IntervalMarkovDecisionProcess([amb, amb, amb], [1, 2]),
            gap_vf,
        ) == CartesianIndex(2)
        @test TS._max_gap_initial_state(IntervalMarkovDecisionProcess([amb, amb, amb]), gap_vf) ==
              CartesianIndex(2)
        @test TS._max_gap_initial_state(
            IntervalMarkovDecisionProcess([amb, amb, amb], [1, 3]),
            gap_vf,
        ) == CartesianIndex(1)

        # Diff = (0.5, 0.5, 0.0): states 1 and 2 tie and are both drawn.
        tie_vf = (upper = (current = [0.9, 0.5, 1.0],), lower = (current = [0.4, 0.0, 1.0],))
        tie_mdp = IntervalMarkovDecisionProcess([amb, amb, amb])
        picks = Set(TS._max_gap_initial_state(tie_mdp, tie_vf) for _ in 1:200)
        @test picks == Set([CartesianIndex(1), CartesianIndex(2)])
    end

    @testset "_resolve_score evaluates the schedules at the initial gap" begin
        fixed = TS.MixtureStateScore()
        @test TS._resolve_score(fixed, 0.5) === fixed
        @test TS._resolve_score(TS.LogStateScore(fixed), 0.5).inner === fixed

        term = TS.BellmanErrorExplore()
        score = TS.MixtureStateScore(;
            exploit = IntervalMDP.Lower,
            explore = term,
            alpha_explore = TS.PowerAlpha(2.0),
            alpha_prob = TS.ClippedAlpha(TS.PowerAlpha(1.0), 0.0, 0.3),
        )
        resolved = TS._resolve_score(score, 0.5)
        @test resolved.alpha_explore == TS.FixedAlpha(0.25)
        @test resolved.alpha_prob == TS.FixedAlpha(0.3)
        @test resolved.exploit === IntervalMDP.Lower
        # Shares the term, so the lifecycle hooks on the strategy's score reach it.
        @test resolved.explore === term

        wrapped = TS._resolve_score(TS.LogStateScore(score), 0.5)
        @test wrapped isa TS.LogStateScore
        @test wrapped.inner.alpha_explore == TS.FixedAlpha(0.25)
    end

    @testset "_sample_state: Boltzmann follows f_S, epsilon-greedy p·f_exploit" begin
        mdp = IntervalMarkovDecisionProcess([amb, amb, amb], [1])
        s, a = CartesianIndex(1), CartesianIndex(1)
        p = [0.5, 0.5, 0.0]
        vf3 = (upper = (current = [1.0, 0.5, 1.0],), lower = (current = [0.9, 0.0, 1.0],))
        score = mix(1.0)   # explore only: state 2 has the larger gap

        ss = TS.TrajectorySampling(; state_policy = TS.Boltzmann(0.001), state_score = score)
        @test all(_ -> TS._sample_state(ss, s, a, p, vf3, mdp, nothing) == CartesianIndex(2), 1:50)

        ss = TS.TrajectorySampling(; state_policy = TS.EpsilonGreedy(0.0), state_score = score)
        @test all(_ -> TS._sample_state(ss, s, a, p, vf3, mdp, nothing) == CartesianIndex(1), 1:50)
    end
end

@testitem "TrajectorySampling: successor filters" tags =
    [:base, :trajectory_sampling, :trajectory_filters] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    # States 1 and 2 loop among themselves; 3 is the goal.
    amb1 = IntervalAmbiguitySets(;
        lower = [0.2 0.2; 0.2 0.2; 0.1 0.1],
        upper = [0.6 0.6; 0.6 0.6; 0.3 0.3],
    )
    absorbing = IntervalAmbiguitySets(;
        lower = [0.0 0.0; 0.0 0.0; 1.0 1.0],
        upper = [0.0 0.0; 0.0 0.0; 1.0 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([amb1, amb1, absorbing], [1])
    spec = Specification(InfiniteTimeReachability([3], 1e-3), Pessimistic, Maximize)
    s, a = CartesianIndex(1), CartesianIndex(1)

    @testset "constructor: off by default, eps only with the flag" begin
        ss = TS.TrajectorySampling()
        @test !ss.exclude_visited && !ss.exclude_converged && ss.converged_eps === nothing
        @test TS.TrajectorySampling(; exclude_converged = true, converged_eps = 0.01).converged_eps ==
              0.01
        @test_throws ArgumentError TS.TrajectorySampling(; converged_eps = 0.01)
        @test_throws ArgumentError TS.TrajectorySampling(;
            exclude_converged = true,
            converged_eps = -1.0,
        )
    end

    vf = (upper = (current = [0.9, 0.5, 1.0],), lower = (current = [0.1, 0.4995, 1.0],))
    # Diff = [0.8, 0.0005, 0.0]

    @testset "_filter_successors!" begin
        @test TS._filter_successors!([1, 2, 3], mdp, vf, nothing, nothing) == [1, 2, 3]
        @test TS._filter_successors!([1, 2, 3], mdp, vf, nothing, 1e-3) == [1]
        @test TS._filter_successors!([1, 2, 3], mdp, vf, nothing, 1e-4) == [1, 2]
        @test TS._filter_successors!([1, 2, 3], mdp, vf, nothing, 0.0) == [1, 2, 3]
        @test TS._filter_successors!([1, 2, 3], mdp, vf, Set([CartesianIndex(1)]), nothing) ==
              [2, 3]
        @test TS._filter_successors!(
            [1, 2, 3],
            mdp,
            vf,
            Set([CartesianIndex(1)]),
            1e-3,
        ) == Int[]
        # Terminal states are exempt from the convergence filter (not from `visited`).
        goal = Set([CartesianIndex(3)])
        @test TS._filter_successors!([1, 2, 3], mdp, vf, nothing, 1e-3, goal) == [1, 3]
        @test TS._filter_successors!([1, 2, 3], mdp, vf, Set([CartesianIndex(1)]), 1e-3, goal) ==
              [3]
        # So are implicit sinks: target 3 below has no source column.
        dirac(rows) = IntervalAmbiguitySets(; lower = rows, upper = rows)
        sink_mdp = IntervalMarkovDecisionProcess(
            [dirac([0.0 0.0; 0.0 0.0; 1.0 1.0]), dirac([0.0 0.0; 0.0 0.0; 1.0 1.0])],
            [1],
        )
        @test TS._filter_successors!([1, 2, 3], sink_mdp, vf, nothing, 1e-3) == [1, 3]
    end

    @testset "_sample_state honours the filters under every policy" begin
        probs = [0.4, 0.4, 0.2]
        for (policy, score) in (
            (TS.EpsilonGreedy(1.0), TS.MixtureStateScore()), # always the uniform branch
            (TS.EpsilonGreedy(0.0), TS.MixtureStateScore()), # always the argmax (state 1)
            (TS.Boltzmann(1.0), TS.MixtureStateScore()),
            (TS.Boltzmann(1.0), TS.LogStateScore(TS.MixtureStateScore())),
        )
            ss = TS.TrajectorySampling(; state_policy = policy, state_score = score)
            draw(; kw...) = TS._sample_state(ss, s, a, probs, vf, mdp, nothing, policy, score; kw...)
            @test all(_ -> draw(; visited = Set([CartesianIndex(1)])) != CartesianIndex(1), 1:100)
            @test all(_ -> draw(; converged_eps = 1e-4) in (CartesianIndex(1), CartesianIndex(2)), 1:100)
            @test all(_ -> draw(; converged_eps = 1e-3) == CartesianIndex(1), 1:20)
            @test draw(; visited = Set([CartesianIndex(1)]), converged_eps = 1e-3) === nothing
        end
    end

    open_vf = (upper = (current = [1.0, 1.0, 1.0],), lower = (current = [0.0, 0.0, 1.0],))

    @testset "exclude_visited: a rollout never revisits a state" begin
        # Uniform successor draws on a looping model: without the filter, rollouts revisit.
        loose = TS.TrajectorySampling(;
            state_policy = TS.EpsilonGreedy(1.0),
            terminate = [TS.MaxSteps(50)],
        )
        @test any(_ -> !allunique(TS._sample_trajectory(loose, mdp, open_vf, spec)), 1:200)

        strict = TS.TrajectorySampling(;
            state_policy = TS.EpsilonGreedy(1.0),
            terminate = [TS.MaxSteps(50)],
            exclude_visited = true,
        )
        @test all(_ -> allunique(TS._sample_trajectory(strict, mdp, open_vf, spec)), 1:200)
    end

    @testset "exclude_converged: a rollout never enters a converged state" begin
        # State 2 is converged; with the filter, rollouts from state 1 never reach it.
        conv_vf = (upper = (current = [1.0, 0.5, 1.0],), lower = (current = [0.0, 0.5, 1.0],))
        ss = TS.TrajectorySampling(;
            state_policy = TS.EpsilonGreedy(1.0),
            terminate = [TS.MaxSteps(50)],
            exclude_converged = true,
        )
        @test all(
            _ -> !(CartesianIndex(2) in TS._sample_trajectory(ss, mdp, conv_vf, spec)),
            1:200,
        )
        # The default threshold is the property's convergence_eps (1e-3 here): a gap
        # of 5e-4 counts as converged, one of 2e-3 does not.
        near = (upper = (current = [1.0, 0.5005, 1.0],), lower = (current = [0.0, 0.5, 1.0],))
        @test all(_ -> !(CartesianIndex(2) in TS._sample_trajectory(ss, mdp, near, spec)), 1:200)
        far = (upper = (current = [1.0, 0.502, 1.0],), lower = (current = [0.0, 0.5, 1.0],))
        @test any(_ -> CartesianIndex(2) in TS._sample_trajectory(ss, mdp, far, spec), 1:200)
    end

    @testset "exclude_converged: rollouts still end at the goal" begin
        # The goal (state 3) has a zero gap but stays drawable, so rollouts end by
        # reaching it rather than all running to the 50-step cap.
        conv_vf = (upper = (current = [1.0, 0.5, 1.0],), lower = (current = [0.0, 0.5, 1.0],))
        ss = TS.TrajectorySampling(;
            state_policy = TS.EpsilonGreedy(1.0),
            terminate = [TS.MaxSteps(50)],
            exclude_converged = true,
        )
        lens = [length(TS._sample_trajectory(ss, mdp, conv_vf, spec)) for _ in 1:200]
        @test count(<(50), lens) > 150
    end
end

@testitem "TrajectorySampling: temperature schedules" tags =
    [:base, :trajectory_sampling, :trajectory_temperature] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    struct OtherSchedule <: TS.TemperatureSchedule end

    @testset "constructor validation" begin
        @test_throws ArgumentError TS.FixedTemperature(0.0)
    end

    @testset "a bare number means a fixed temperature" begin
        @test TS.Boltzmann(0.5).T === TS.FixedTemperature(0.5)
        @test_throws ArgumentError TS.Boltzmann(0.0)
        @test_throws ArgumentError TS.Boltzmann(-1.0)
    end

    @testset "each policy keyword takes only its own schedules" begin
        gap = TS.Boltzmann(TS.GapBasedTemperature(1.0))
        @test TS.TrajectorySampling(; action_policy = gap).action_policy === gap
        @test TS.TrajectorySampling(; action_policy = TS.Boltzmann(0.5)) isa TS.TrajectorySampling
        @test_throws ArgumentError TS.TrajectorySampling(;
            action_policy = TS.Boltzmann(OtherSchedule()),
        )
        @test TS.TrajectorySampling(; state_policy = TS.Boltzmann(0.5)) isa TS.TrajectorySampling
        @test_throws ArgumentError TS.TrajectorySampling(; state_policy = gap)
        # Only a Boltzmann policy carries a schedule.
        @test TS.TrajectorySampling(; state_policy = TS.EpsilonGreedy(0.2)) isa
              TS.TrajectorySampling
    end

    @testset "TemperatureContext clamps the gap to [0, 1]" begin
        # Round-off can push U − L below zero, and a caller-supplied value
        # function can carry a gap above one; either would put the odds
        # δ / (1 − δ) outside [0, ∞].
        @test TS.TemperatureContext(-0.5).diff == 0.0
        @test TS.TemperatureContext(1.5).diff == 1.0
    end

    @testset "gap based is k times the odds of the initial gap, floored at t_min" begin
        @test_throws ArgumentError TS.GapBasedTemperature(0.0)
        @test_throws ArgumentError TS.GapBasedTemperature(-1.0)
        @test_throws ArgumentError TS.GapBasedTemperature(1.0, -0.1)
        @test TS.GapBasedTemperature(1.0).t_min == 0.1

        sched = TS.GapBasedTemperature(2.0, 0.0)
        @test TS._temperature(sched, TS.TemperatureContext(0.5)) == 2.0
        @test TS._temperature(sched, TS.TemperatureContext(0.2)) ≈ 2.0 * 0.2 / 0.8
        @test TS._temperature(sched, TS.TemperatureContext(0.9)) ≈ 2.0 * 0.9 / 0.1

        # The floor holds the temperature up once the odds drop below it.
        floored = TS.GapBasedTemperature(2.0, 0.25)
        @test TS._temperature(floored, TS.TemperatureContext(0.5)) == 2.0
        @test TS._temperature(floored, TS.TemperatureContext(0.01)) == 0.25
        @test TS._temperature(floored, TS.TemperatureContext(0.0)) == 0.25

        # Without a floor the endpoints (T = 0, T = ∞) are clamped to a valid
        # temperature, and resolve to argmax and uniform respectively.
        cold = TS._resolve_policy(TS.Boltzmann(sched), TS.TemperatureContext(0.0))
        hot = TS._resolve_policy(TS.Boltzmann(sched), TS.TemperatureContext(1.0))
        @test cold.T.t == floatmin(Float64)
        @test hot.T.t == floatmax(Float64)

        cands, scores = [:a, :b, :c], [0.1, 0.9, -Inf]
        @test all(TS._select(cold, cands, scores) == :b for _ in 1:200)
        draws = [TS._select(hot, cands, scores) for _ in 1:2000]
        @test :c ∉ draws                                    # -Inf is never drawn
        @test count(==(:a), draws) / 2000 ≈ 0.5 atol = 0.05
    end

    @testset "resolution against a rollout context" begin
        ctx = TS.TemperatureContext(0.5)

        eg = TS.EpsilonGreedy(0.3)
        @test TS._resolve_policy(eg, ctx) === eg

        varying = TS.Boltzmann(TS.GapBasedTemperature(2.0, 0.01))
        @test TS._resolve_policy(varying, ctx) == TS.Boltzmann(2.0)

        # An unresolved schedule has no temperature to draw with; that is an
        # error rather than a silent default.
        @test_throws ArgumentError TS._select(varying, [:a, :b], [1.0, 2.0])
    end
end

@testitem "TrajectorySampling: temperature follows the initial gap in the rollout" tags =
    [:base, :trajectory_sampling, :trajectory_temperature_rollout] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    # State 1 has two deterministic actions: a1 steps onto the goal (state 2),
    # a2 loops back to state 1. Under UpperBoundScore the two score U(2) = 1.0
    # and U(1) = 0.9, so a cold action policy always takes a1 and ends the
    # rollout at once, while a hot one also takes a2 and wanders.
    det(t) = Float64.(1:2 .== t)
    s1 = IntervalAmbiguitySets(; lower = hcat(det(2), det(1)), upper = hcat(det(2), det(1)))
    goal = IntervalAmbiguitySets(; lower = hcat(det(2), det(2)), upper = hcat(det(2), det(2)))
    mdp = IntervalMarkovDecisionProcess([s1, goal], [1])
    prop = InfiniteTimeReachability([2], 1 // 1000)
    spec = Specification(prop, Pessimistic, Maximize)

    # Holding U fixed and moving only L varies the initial gap — and therefore
    # the temperature — while leaving the action scores untouched.
    function value_function(lower)
        alg =
            GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))
        V = IntervalMDP.construct_value_function(alg, VerificationProblem(mdp, spec))
        IntervalMDP._gsrdp_initialize!(V, prop)
        V.upper.current .= [0.9, 1.0]
        V.lower.current .= lower
        return V
    end

    # δ = 0.9 resolves to T = 100·0.9/0.1 = 900: exp(0.1/900) ≈ 1, i.e. an even
    # draw. δ = 0 resolves to the floor T = 0.001: exp(0.1/0.001) = e^100, i.e.
    # the argmax every time.
    sched = TS.GapBasedTemperature(100.0, 0.001)
    ss = TS.TrajectorySampling(; action_policy = TS.Boltzmann(sched))

    @testset "a converged S0 draws at t_min: straight to the goal" begin
        V = value_function([0.9, 1.0])   # δ = 0
        rhos = [TS._sample_trajectory(ss, mdp, V, spec) for _ in 1:200]
        @test all(rho -> rho == [CartesianIndex(1)], rhos)
    end

    @testset "a wide-open S0 draws hot: it wanders" begin
        V = value_function([0.0, 1.0])   # δ = 0.9
        rhos = [TS._sample_trajectory(ss, mdp, V, spec) for _ in 1:200]
        # The default MaxSteps() cap is num_states(mdp) = 2 transitions, so a
        # rollout that does not step onto the goal immediately has length 2.
        @test any(rho -> length(rho) > 1, rhos)
        @test any(rho -> length(rho) == 1, rhos)
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

    @testset "ExpectedGapStop: no allocation that grows with |S|" begin
        # It runs at every step of every rollout, so it must not build `U .- L`.
        rule = TS.ExpectedGapStop(10.0)
        s, a = CartesianIndex(1), CartesianIndex(1)
        function allocs(n)
            vf = (upper = (current = fill(1.0, n),), lower = (current = fill(0.5, n),))
            probs = zeros(n)
            probs[2] = 1.0
            check() = TS.terminate_post(rule, s, a, probs, CartesianIndex[], 0, vf, mdp, nothing)
            check()   # compile
            return @allocated check()
        end
        @test allocs(10_000) <= allocs(10)
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

    @testset "dedup drops repeats within a batch, keeping the first" begin
        @test !TS.TrajectorySampling().dedup
        ss = TS.TrajectorySampling(; gauss_seidel = false, dedup = true)
        @test batch(ss) == [CartesianIndex(3), CartesianIndex(1)]

        # Per batch, not per trajectory: the repeat split across batches stays.
        ss = TS.TrajectorySampling(; gauss_seidel = true, k = 2, reverse = false, dedup = true)
        @test batch(ss) == [CartesianIndex(1), CartesianIndex(3)]
        @test batch(ss) == [CartesianIndex(3)]

        ss = TS.TrajectorySampling(; gauss_seidel = true, k = 99, dedup = true)
        @test batch(ss) == [CartesianIndex(3), CartesianIndex(1)]
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
            "boltzmann actions + mixture" => TS.TrajectorySampling(;
                action_policy = TS.Boltzmann(0.5),
                state_policy = TS.Boltzmann(0.5),
                state_score = TS.MixtureStateScore(),
            ),
            "mixture: upper bound explore" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(0.5),
                state_score = TS.MixtureStateScore(; explore = TS.UpperBoundExplore()),
            ),
            "mixture: action uncertainty explore" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(0.5),
                state_score = TS.MixtureStateScore(; explore = TS.ActionUncertaintyExplore()),
            ),
            "mixture: state-level VPI explore" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(0.5),
                state_score = TS.MixtureStateScore(; explore = TS.VPIExplore()),
            ),
            "mixture: Bellman error (U) explore" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(0.5),
                state_score = TS.MixtureStateScore(; explore = TS.BellmanErrorExplore()),
            ),
            "mixture: Bellman error (L) explore" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(0.5),
                state_score = TS.MixtureStateScore(;
                    explore = TS.BellmanErrorExplore(; bound = IntervalMDP.Lower),
                ),
            ),
            "mixture: max Bellman error explore, gauss-seidel" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(0.5),
                state_score = TS.MixtureStateScore(; explore = TS.MaxBellmanErrorExplore()),
                gauss_seidel = true,
                k = 2,
            ),
            "mixture (fixed alphas)" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(0.1),
                state_score = TS.MixtureStateScore(),
            ),
            "mixture (power alphas, clipped)" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(0.1),
                state_score = TS.MixtureStateScore(;
                    alpha_explore = TS.ClippedAlpha(TS.PowerAlpha(1.0), 0.1, 0.9),
                    alpha_prob = TS.PowerAlpha(2.0),
                ),
            ),
            "log mixture, Bellman error, gauss-seidel" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(1.0),
                state_score = TS.LogStateScore(
                    TS.MixtureStateScore(;
                        explore = TS.BellmanErrorExplore(),
                        alpha_explore = TS.PowerAlpha(1.0),
                    ),
                ),
                gauss_seidel = true,
                k = 2,
            ),
            "exclude visited" => TS.TrajectorySampling(; exclude_visited = true),
            "expected-gap stop" => TS.TrajectorySampling(;
                terminate = [TS.MaxSteps(), TS.ExpectedGapStop(10.0)],
            ),
            # `t_min` is 0.5, not something near zero: a temperature that cools
            # all the way down is pure exploitation, which can lock the rollout
            # onto one state and stall the gap criterion — the same hazard
            # documented under "Why the default e is not zero".
            "gap-based action temperature" => TS.TrajectorySampling(;
                action_policy = TS.Boltzmann(TS.GapBasedTemperature(1.0, 0.5)),
            ),
            "gauss-seidel k = 2" => TS.TrajectorySampling(; gauss_seidel = true, k = 2),
            "max-gap initial state" => TS.TrajectorySampling(; initial_state = :max),
            "lower-bound transition" => TS.TrajectorySampling(;
                transition = TS.ConcreteTransition(; bound = IntervalMDP.Lower),
            ),
            "power rule + expected-gap stop" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(1.0),
                state_score = TS.LogStateScore(TS.MixtureStateScore()),
                terminate = [TS.MaxSteps(), TS.ExpectedGapStop(10.0)],
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

        # exclude_converged is meant for convergence on the initial states, so its
        # parity is checked there, at the initial state, and without the warning.
        init_prop = InfiniteTimeReachability([3], eps; restrict_to_initial = true)
        init_problem = VerificationProblem(mdp, Specification(init_prop, Pessimistic, Maximize))
        @testset "$name (restrict_to_initial)" for (name, ss) in [
            "exclude converged" => TS.TrajectorySampling(; exclude_converged = true),
            "both filters, power rule, gauss-seidel" => TS.TrajectorySampling(;
                state_policy = TS.Boltzmann(1.0),
                state_score = TS.LogStateScore(TS.MixtureStateScore()),
                exclude_visited = true,
                exclude_converged = true,
                gauss_seidel = true,
                k = 2,
            ),
        ]
            gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
                default_bellman_algorithm(mdp);
                sampling_strategy = ss,
            )
            (V, _, _) = @test_logs solve(init_problem, gsdp)
            @test abs(V_rvi[1] - V[1]) <= 2 * eps
        end

        @testset "exclude_converged warns once per solve without restrict_to_initial" begin
            ss = TS.TrajectorySampling(; exclude_converged = true)
            gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(
                default_bellman_algorithm(mdp);
                sampling_strategy = ss,
            )
            # Exactly one warning over the whole solve, and again on the next solve.
            @test_logs (:warn, r"restrict_to_initial") solve(problem, gsdp)
            @test_logs (:warn, r"restrict_to_initial") solve(problem, gsdp)
            # Not without the filter.
            plain = GeneralizedSamplingbasedRobustDynamicProgramming(
                default_bellman_algorithm(mdp);
                sampling_strategy = TS.TrajectorySampling(),
            )
            @test_logs solve(problem, plain)
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
        "boltzmann + mixture" => TS.TrajectorySampling(;
            state_policy = TS.Boltzmann(0.5),
            state_score = TS.MixtureStateScore(),
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
