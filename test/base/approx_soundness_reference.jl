@testmodule ApproxSoundnessReference begin
    # Cross-check for the Lean theorems `IntervalMDP.Factored.mcCormick_sound` (A3) and
    # `IntervalMDP.Factored.recursiveOMax_sound` (A2) against the exact value
    # `IntervalMDP.Factored.vertexValue_eq_opt` (lean/IntervalMDPProofs/Factored.lean): on factored
    # IMDPs with two or more state variables, the Bellman update computed with
    # `LPMcCormickRelaxation()` (HiGHS, an external, unverified LP solver) and with
    # `OMaximization()` lies on the conservative side of the one computed with
    # `VertexEnumeration()`, which is the exact optimum over the (non-convex) product set:
    #     upper_bound = false (pessimistic):  approx ≤ exact,
    #     upper_bound = true  (optimistic):   approx ≥ exact,
    # for both strategy modes (`maximize = true/false`), up to `TOL`.
    using IntervalMDP
    using Random

    const TOL = 1e-9
    const APPROXIMATIONS = (LPMcCormickRelaxation(), OMaximization())

    # One interval column on `1:d` around a random distribution: `0 ≤ l ≤ u ≤ 1`, `∑ l ≤ 1 ≤ ∑ u`.
    function random_column(rng, d)
        p = rand(rng, d)
        p ./= sum(p)
        l = max.(0.0, p .- 0.3 .* rand(rng, d))
        u = min.(1.0, p .+ 0.3 .* rand(rng, d))
        return l, u
    end

    # Marginal for state variable `i` conditioned on `state_indices` and on action variable 1.
    function random_marginal(rng, state_vars, action_vars, i, state_indices)
        source_dims = state_vars[collect(state_indices)]
        ncols = action_vars[1] * prod(source_dims)
        lower = zeros(state_vars[i], ncols)
        upper = zeros(state_vars[i], ncols)
        for j in 1:ncols
            lower[:, j], upper[:, j] = random_column(rng, state_vars[i])
        end
        return Marginal(
            IntervalAmbiguitySets(; lower = lower, upper = upper),
            state_indices,
            (1,),
            Tuple(source_dims),
            action_vars,
        )
    end

    # Random fIMDP; each marginal depends on a random nonempty, increasing subset of the state
    # variables and on the (single) action variable.
    function random_fimdp(rng, state_vars, action_vars)
        N = length(state_vars)
        marginals = ntuple(N) do i
            deps = Tuple(sort(randperm(rng, N)[1:rand(rng, 1:N)]))
            random_marginal(rng, state_vars, action_vars, i, deps)
        end
        return FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)
    end

    # The same interval set for every marginal and every column (binary state variables).
    function uniform_fimdp(N, lower, upper; num_actions = 1)
        state_vars = ntuple(_ -> 2, N)
        ncols = num_actions * prod(state_vars)
        marginals = ntuple(N) do _
            Marginal(
                IntervalAmbiguitySets(;
                    lower = repeat(lower, 1, ncols),
                    upper = repeat(upper, 1, ncols),
                ),
                Tuple(1:N),
                (1,),
                state_vars,
                (num_actions,),
            )
        end
        return FactoredRobustMarkovDecisionProcess(state_vars, (num_actions,), marginals)
    end

    exact(V, mdp; upper_bound, maximize) = IntervalMDP.bellman(
        V,
        mdp,
        VertexEnumeration();
        upper_bound = upper_bound,
        maximize = maximize,
    )

    # `true` iff every approximation is conservative w.r.t. vertex enumeration, in all four
    # (upper_bound, maximize) combinations.
    function conservative(V, mdp)
        ok = true
        for upper_bound in (false, true), maximize in (false, true)
            Vexact = exact(V, mdp; upper_bound = upper_bound, maximize = maximize)
            for alg in APPROXIMATIONS
                Vapprox = IntervalMDP.bellman(
                    V,
                    mdp,
                    alg;
                    upper_bound = upper_bound,
                    maximize = maximize,
                )
                ok &= if upper_bound
                    all(Vapprox .>= Vexact .- TOL)
                else
                    all(Vapprox .<= Vexact .+ TOL)
                end
            end
        end
        return ok
    end
end

@testitem "base/approx_soundness_reference: two binary variables, O-max not exact" setup =
    [ApproxSoundnessReference] begin
    # Inventory Observation O17: O-max 0.4 / 0.6, exact 0.48 / 0.52.
    mdp = ApproxSoundnessReference.uniform_fimdp(2, [0.4, 0.4], [0.6, 0.6])
    V = [1.0 0.0; 0.0 1.0]
    @test ApproxSoundnessReference.conservative(V, mdp)

    Vlow = ApproxSoundnessReference.exact(V, mdp; upper_bound = false, maximize = true)
    Vup = ApproxSoundnessReference.exact(V, mdp; upper_bound = true, maximize = true)
    @test all(isapprox.(Vlow, 0.48; atol = 1e-12))
    @test all(isapprox.(Vup, 0.52; atol = 1e-12))
    @test all(isapprox.(IntervalMDP.bellman(V, mdp, OMaximization()), 0.4; atol = 1e-12))
end

@testitem "base/approx_soundness_reference: degenerate intervals l = u" setup =
    [ApproxSoundnessReference] begin
    # Every marginal set is a single distribution, so all three algorithms are exact.
    mdp = ApproxSoundnessReference.uniform_fimdp(3, [0.3, 0.7], [0.3, 0.7])
    V = reshape(Float64.(1:8), 2, 2, 2)
    @test ApproxSoundnessReference.conservative(V, mdp)
    for upper_bound in (false, true)
        Vexact = ApproxSoundnessReference.exact(
            V,
            mdp;
            upper_bound = upper_bound,
            maximize = true,
        )
        for alg in ApproxSoundnessReference.APPROXIMATIONS
            Vapprox = IntervalMDP.bellman(V, mdp, alg; upper_bound = upper_bound)
            @test all(isapprox.(Vapprox, Vexact; atol = 1e-8))
        end
    end
end

@testitem "base/approx_soundness_reference: five binary variables (McCormick tree ≠ chain)" setup =
    [ApproxSoundnessReference] begin
    # Inventory Finding F6: for N = 5 Julia's McCormick tree is (((1, 2), 3), (4, 5)); still sound.
    mdp = ApproxSoundnessReference.uniform_fimdp(5, [0.2, 0.3], [0.7, 0.8]; num_actions = 2)
    V = [Float64(isodd(sum(Tuple(I)))) for I in CartesianIndices(ntuple(_ -> 2, 5))]
    @test ApproxSoundnessReference.conservative(V, mdp)
end

@testitem "base/approx_soundness_reference: ties and a constant value function" setup =
    [ApproxSoundnessReference] begin
    using Random

    rng = MersenneTwister(5)
    mdp = ApproxSoundnessReference.random_fimdp(rng, (2, 3), (2,))
    @test ApproxSoundnessReference.conservative(ones(2, 3), mdp)
    @test ApproxSoundnessReference.conservative(Float64.(rand(rng, 0:1, 2, 3)), mdp)
end

@testitem "base/approx_soundness_reference: random small fIMDPs" setup =
    [ApproxSoundnessReference] begin
    using Random

    rng = MersenneTwister(20261016)
    for _ in 1:25
        N = rand(rng, 2:3)
        state_vars = Tuple(rand(rng, 2:3, N))
        action_vars = (rand(rng, 1:2),)
        mdp = ApproxSoundnessReference.random_fimdp(rng, state_vars, action_vars)
        V = rand(rng, state_vars...)
        @test ApproxSoundnessReference.conservative(V, mdp)
    end
end
