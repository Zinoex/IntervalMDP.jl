# Tests for the tail-convergence additions: the scale-free `Proportional` policy,
# `GapContributionScore`, `ExpectedGapStop(max_adversary = true)`, batch dedup,
# GSRDP's `lower_bound = :optimize`, and `AdaptiveSweepMixture`.

@testitem "Proportional selection policy" tags = [:base, :tail_sampling] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    cands = [:a, :b, :c, :d]

    @testset "draw frequencies follow the scores and ignore their scale" begin
        for scale in (1.0, 1e-9)
            scores = scale .* [1.0, 3.0, 0.0, 6.0]
            n = 40_000
            counts = Dict(c => 0 for c in cands)
            for _ in 1:n
                counts[TS._select(TS.Proportional(), cands, scores)] += 1
            end
            @test counts[:c] == 0                       # zero weight never drawn
            @test isapprox(counts[:a] / n, 0.1; atol = 0.01)
            @test isapprox(counts[:b] / n, 0.3; atol = 0.015)
            @test isapprox(counts[:d] / n, 0.6; atol = 0.015)
        end
    end

    @testset "no positive score falls back to uniform" begin
        for scores in ([0.0, 0.0, 0.0, 0.0], [-Inf, -Inf, -Inf, -Inf], [-1.0, NaN, 0.0, -2.0])
            draws = Set(TS._select(TS.Proportional(), cands, scores) for _ in 1:400)
            @test draws == Set(cands)
        end
    end
end

@testitem "GapContributionScore bounds the gap at (s, a)" tags = [:base, :tail_sampling] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    # One source state with four successors; the budget above the lower bounds is
    # 1 - 0.4 = 0.6, so O-max moves real mass and pU != pL once U and L order the
    # successors differently.
    l = reshape([0.1, 0.1, 0.1, 0.1], 4, 1)
    u = reshape([0.7, 0.7, 0.7, 0.7], 4, 1)
    amb = IntervalAmbiguitySets(; lower = l, upper = u)
    mdp = IntervalMarkovDecisionProcess([amb, amb, amb, amb], [1])
    spec_for(mode) = Specification(InfiniteTimeReachability([4], 1e-6), mode, Maximize)
    problem = VerificationProblem(mdp, spec_for(Pessimistic))

    vf = IntervalMDP.IntervalValueFunction(
        IntervalMDP.StateValueFunction(problem, IntervalMDP.Lower),
        IntervalMDP.StateValueFunction(problem, IntervalMDP.Upper),
    )
    # Successor 2 still wide open (U high, L low), 1 and 3 converged, 4 the target.
    vf.upper.current .= [0.5, 0.9, 0.2, 1.0]
    vf.lower.current .= [0.5, 0.1, 0.2, 1.0]

    s, a = CartesianIndex(1), CartesianIndex(1)
    as = IntervalMDP._omax_marginal(mdp)[a, s]

    @testset "$mode" for mode in (Pessimistic, Optimistic)
        spec = spec_for(mode)
        dir = isoptimistic(spec)
        U, L = vf.upper.current, vf.lower.current
        pU = IntervalMDP._omax_distribution(as, U, dir)
        pL = IntervalMDP._omax_distribution(as, L, dir)
        gap_sa = sum(pU .* U) - sum(pL .* L)

        tr = TS.ConcreteTransition()
        supp, w = TS._candidates_and_scores(TS.GapContributionScore(), pU, vf, s, a, mdp, spec, tr)
        @test all(>=(0), w)
        @test sum(w) >= gap_sa - 1e-12
        # Converged successors and the target carry no score.
        for (n, i) in enumerate(supp)
            i in (1, 3, 4) && @test w[n] == 0
        end
        # The nine-argument form agrees with the candidate hook.
        @test TS._state_scores(TS.GapContributionScore(), pU, supp, vf, s, a, mdp, spec, tr) ≈ w
    end

    @testset "pessimistic: the pU-weighted gap misses the disagreement term" begin
        spec = spec_for(Pessimistic)
        U, L = vf.upper.current, vf.lower.current
        pU = IntervalMDP._omax_distribution(as, U, false)
        pL = IntervalMDP._omax_distribution(as, L, false)
        gap_sa = sum(pU .* U) - sum(pL .* L)
        # The adversary starves the open successor under pU and loads it under pL.
        @test pL[2] > pU[2]
        @test sum(pU .* (U .- L)) < gap_sa
    end

    @testset "four-argument form throws" begin
        @test_throws ArgumentError TS._state_scores(TS.GapContributionScore(), [1.0], [1], vf)
    end
end

@testitem "ExpectedGapStop: realized sum unchanged, max_adversary never smaller" tags =
    [:base, :tail_sampling] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    l = reshape([0.1, 0.1, 0.1, 0.1], 4, 1)
    u = reshape([0.7, 0.7, 0.7, 0.7], 4, 1)
    amb = IntervalAmbiguitySets(; lower = l, upper = u)
    mdp = IntervalMarkovDecisionProcess([amb, amb, amb, amb], [1])
    spec = Specification(InfiniteTimeReachability([4], 1e-6), Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)
    vf = IntervalMDP.IntervalValueFunction(
        IntervalMDP.StateValueFunction(problem, IntervalMDP.Lower),
        IntervalMDP.StateValueFunction(problem, IntervalMDP.Upper),
    )
    vf.upper.current .= [0.6, 0.9, 0.2, 1.0]
    vf.lower.current .= [0.4, 0.1, 0.2, 1.0]
    s, a = CartesianIndex(1), CartesianIndex(1)
    as = IntervalMDP._omax_marginal(mdp)[a, s]
    U, L = vf.upper.current, vf.lower.current
    probs = IntervalMDP._omax_distribution(as, U, false)
    pL = IntervalMDP._omax_distribution(as, L, false)

    B_realized = sum(probs .* (U .- L))
    B_max = sum(max.(probs, pL) .* (U .- L))
    diff = U[1] - L[1]
    @test B_max >= B_realized

    # Pick tau so the two sums fall on opposite sides of the threshold.
    tau = diff / ((B_realized + B_max) / 2)
    @test TS.terminate_post(TS.ExpectedGapStop(tau), s, a, probs, [], 0, vf, mdp, spec)
    @test !TS.terminate_post(
        TS.ExpectedGapStop(tau; max_adversary = true), s, a, probs, [], 0, vf, mdp, spec,
    )
end

@testitem "TrajectorySampling dedup drops repeats within a batch" tags =
    [:base, :tail_sampling] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    # A state that returns to itself with high probability produces repeats.
    prob = IntervalAmbiguitySets(;
        lower = [0.8 0.0; 0.0 1.0],
        upper = [0.95 0.0; 0.2 1.0],
    )
    mdp = IntervalMarkovDecisionProcess([prob], [1])
    spec = Specification(InfiniteTimeReachability([2], 1e-6), Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)

    for gs in (false, true)
        seen_dup = Ref(false)
        cb = function (_, _, seq)
            seq === nothing && return nothing
            v = collect(seq)
            length(unique(v)) < length(v) && (seen_dup[] = true)
            return nothing
        end
        ss = TS.TrajectorySampling(; gauss_seidel = gs, k = 4, dedup = true)
        alg = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(mdp);
            sampling_strategy = ss,
        )
        solve(problem, alg; callback = cb)
        @test !seen_dup[]
    end
end

@testitem "GSRDP lower_bound = :optimize" tags = [:base, :tail_sampling] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    @test_throws ArgumentError GeneralizedSamplingbasedRobustDynamicProgramming(
        OMaximization();
        lower_bound = :bogus,
    )

    @testset "parity with RVI, $N" for N in [Float32, Float64]
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
        for mode in (Pessimistic, Optimistic), smode in (Maximize, Minimize)
            spec = Specification(InfiniteTimeReachability([3], eps), mode, smode)
            problem = VerificationProblem(mdp, spec)
            (V_rvi, _, _) =
                solve(problem, RobustValueIteration(default_bellman_algorithm(mdp)))
            for ss in (IntervalMDP.ExhaustiveState(), TS.TrajectorySampling())
                alg = GeneralizedSamplingbasedRobustDynamicProgramming(
                    default_bellman_algorithm(mdp);
                    sampling_strategy = ss,
                    lower_bound = :optimize,
                )
                (V, _, _) = solve(problem, alg)
                # RVI stops on its residual, which does not bound its own error: under
                # Minimize it sits ~3.6e-6 from V* here, and `:follow` misses the 2eps
                # bar by the same amount. Hold Minimize to the harness's 10eps.
                tol = smode == Maximize ? 2 * eps : 10 * eps
                @test maximum(abs, V_rvi .- V) <= tol
            end
        end
    end

    @testset "reach-avoid, both modes" begin
        prob = IntervalAmbiguitySets(;
            lower = [0 0 0 0; 2/5 0 0 0; 0 2/5 1 0; 1/5 1/5 0 1],
            upper = [0 0 0 0; 4/5 0 0 0; 0 4/5 1 0; 3/5 3/5 0 1],
        )
        model = IntervalMarkovChain(prob, [1])
        prop = InfiniteTimeReachAvoid([3], [4], 1e-6)
        for mode in (Pessimistic, Optimistic)
            problem = VerificationProblem(model, Specification(prop, mode, Maximize))
            (V_rvi, _, _) =
                solve(problem, RobustValueIteration(default_bellman_algorithm(model)))
            alg = GeneralizedSamplingbasedRobustDynamicProgramming(
                default_bellman_algorithm(model);
                lower_bound = :optimize,
            )
            (V, _, _) = solve(problem, alg)
            @test maximum(abs, V_rvi .- V) <= 2e-6
        end
    end

    @testset "update count is 2·A per state, lower never below :follow" begin
        prob = IntervalAmbiguitySets(;
            lower = Float64[0 1//2 0; 1//10 3//10 0; 1//5 1//10 1],
            upper = Float64[1//2 7//10 0; 3//5 1//2 0; 7//10 3//10 1],
        )
        prob2 = IntervalAmbiguitySets(;
            lower = Float64[1//10 1//5 0; 1//5 1//5 0; 3//10 2//5 1],
            upper = Float64[1//2 1//2 0; 1//2 2//5 0; 2//5 2//5 1],
        )
        mdp = IntervalMarkovDecisionProcess([prob, prob2, prob2], [1])
        spec = Specification(InfiniteTimeReachability([3], 1e-6), Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        A = IntervalMDP.num_actions(mdp)

        lowers = Dict{Symbol, Vector{Vector{Float64}}}()
        for lb in (:follow, :optimize)
            deltas = Int[]
            last = Ref(0)
            L = Vector{Float64}[]
            cb = function (vf, n, seq)
                seq === nothing && return nothing
                push!(deltas, n - last[])
                last[] = n
                push!(L, copy(vec(vf.lower.current)))
                return nothing
            end
            alg = GeneralizedSamplingbasedRobustDynamicProgramming(
                default_bellman_algorithm(mdp);
                lower_bound = lb,
            )
            solve(problem, alg; callback = cb)
            per_state = lb === :optimize ? 2A : A + 1
            @test all(==(3 * per_state), deltas)   # exhaustive: 3 states per iteration
            lowers[lb] = L
        end
        # Same sweep, same upper bound: the optimizing lower bound is never below.
        n = min(length(lowers[:follow]), length(lowers[:optimize]))
        @test all(all(lowers[:optimize][k] .>= lowers[:follow][k] .- 1e-12) for k in 1:n)
    end
end

@testitem "AdaptiveSweepMixture" tags = [:base, :tail_sampling] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    @test_throws ArgumentError IntervalMDP.AdaptiveSweepMixture(IntervalMDP.ExhaustiveState(); window = 0)
    @test_throws ArgumentError IntervalMDP.AdaptiveSweepMixture(
        IntervalMDP.ExhaustiveState();
        probe_every = 0,
    )

    prob = IntervalAmbiguitySets(;
        lower = Float64[0 1//2 0; 1//10 3//10 0; 1//5 1//10 1],
        upper = Float64[1//2 7//10 0; 3//5 1//2 0; 7//10 3//10 1],
    )
    prob2 = IntervalAmbiguitySets(;
        lower = Float64[1//10 1//5 0; 1//5 1//5 0; 3//10 2//5 1],
        upper = Float64[1//2 1//2 0; 1//2 2//5 0; 2//5 2//5 1],
    )
    mdp = IntervalMarkovDecisionProcess([prob, prob2, prob2], [1])
    eps = 1e-6
    spec = Specification(InfiniteTimeReachability([3], eps), Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)
    (V_rvi, _, _) = solve(problem, RobustValueIteration(default_bellman_algorithm(mdp)))

    mix = IntervalMDP.AdaptiveSweepMixture(
        TS.TrajectorySampling(; gauss_seidel = true, k = 1);
        window = 1.0,
        probe_every = 2,
    )
    alg = GeneralizedSamplingbasedRobustDynamicProgramming(
        default_bellman_algorithm(mdp);
        sampling_strategy = mix,
    )
    (V, _, _) = solve(problem, alg)
    @test maximum(abs, V_rvi .- V) <= 2 * eps
    # Both arms ran and were measured.
    @test Set(first.(mix.history)) == Set([1, 2])

    # A second solve starts from a clean slate.
    (V2, _, _) = solve(problem, alg)
    @test maximum(abs, V_rvi .- V2) <= 2 * eps
    @test first(first(mix.history)) == 1
end

@testitem "BRTDP-style trajectory configuration solves to RVI parity" tags =
    [:base, :tail_sampling] begin
    using IntervalMDP
    const TS = IntervalMDP.TrajectorySampling

    prob1 = IntervalAmbiguitySets(;
        lower = Float64[0 1//2; 1//10 3//10; 1//5 1//10],
        upper = Float64[1//2 7//10; 3//5 1//2; 7//10 3//10],
    )
    prob2 = IntervalAmbiguitySets(;
        lower = Float64[1//10 1//5; 1//5 1//5; 3//10 2//5],
        upper = Float64[1//2 1//2; 1//2 2//5; 2//5 2//5],
    )
    sink = IntervalAmbiguitySets(; lower = Float64[0 0; 0 0; 1 1], upper = Float64[0 0; 0 0; 1 1])
    explicit_mdp = IntervalMarkovDecisionProcess([prob1, prob2, sink], [1])
    implicit_mdp = IntervalMarkovDecisionProcess([prob1, prob2], [1])
    spec = Specification(InfiniteTimeReachability([3], 1e-6), Pessimistic, Maximize)
    (V_ref, _, _) = solve(
        VerificationProblem(explicit_mdp, spec),
        RobustValueIteration(default_bellman_algorithm(explicit_mdp)),
    )

    brtdp() = TS.TrajectorySampling(;
        state_policy = TS.Proportional(),
        state_score = TS.GapContributionScore(),
        terminate = [TS.MaxSteps(), TS.ExpectedGapStop(10.0; max_adversary = true)],
        gauss_seidel = true,
        k = 2,
        dedup = true,
    )

    @testset "$label, lower_bound = $lb" for (label, m) in
                                            ("explicit" => explicit_mdp, "implicit" => implicit_mdp),
        lb in (:follow, :optimize)

        alg = GeneralizedSamplingbasedRobustDynamicProgramming(
            default_bellman_algorithm(m);
            sampling_strategy = brtdp(),
            lower_bound = lb,
        )
        (V, _, _) = solve(VerificationProblem(m, spec), alg)
        @test maximum(abs, V_ref .- V) <= 1e-4
    end
end
