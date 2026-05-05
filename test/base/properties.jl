# Property-based tests via Supposition.jl. Each property runs many
# randomly-generated valid IMDPs through the solver and checks an
# invariant (boundedness, monotonicity, soundness vs. vertex enumeration,
# GenSamplingDP parity). Generators are kept small (≤ 4 states, ≤ 3
# actions) so a single @check completes well under a second.

@testitem "properties — setup smoke" tags = [:base, :property] begin
    using IntervalMDP, Random

    function _valid_bounds(rng, n_target, n_columns)
        lower = zeros(n_target, n_columns)
        upper = zeros(n_target, n_columns)
        for j in 1:n_columns
            p_raw = rand(rng, n_target)
            p = p_raw ./ sum(p_raw)
            r = rand(rng, n_target) .* min.(p, 1 .- p) .* 0.6
            lower[:, j] .= max.(0.0, p .- r)
            upper[:, j] .= min.(1.0, p .+ r)
        end
        return lower, upper
    end

    rng = MersenneTwister(0)
    lower, upper = _valid_bounds(rng, 4, 6)
    @test all(0 .<= lower .<= upper .<= 1)
    @test all(sum(lower; dims = 1) .<= 1)
    @test all(sum(upper; dims = 1) .>= 1)
end

# Property 1 — Boundedness for finite-time reachability.
@testitem "property: V is bounded in [0,1] for finite reachability" tags =
    [:base, :property] begin
    using IntervalMDP, Random, Supposition

    function _valid_bounds(rng, n_target, n_columns)
        lower = zeros(n_target, n_columns)
        upper = zeros(n_target, n_columns)
        for j in 1:n_columns
            p_raw = rand(rng, n_target)
            p = p_raw ./ sum(p_raw)
            r = rand(rng, n_target) .* min.(p, 1 .- p) .* 0.6
            lower[:, j] .= max.(0.0, p .- r)
            upper[:, j] .= min.(1.0, p .+ r)
        end
        return lower, upper
    end

    @check db = false function v_bounded(
        seed = Data.Integers{UInt32}(),
        n_states = Data.Integers(2, 4),
        n_actions = Data.Integers(1, 3),
        horizon = Data.Integers(1, 5),
    )
        rng = MersenneTwister(seed)
        lower, upper = _valid_bounds(rng, n_states, n_states * n_actions)
        transition_probs = [
            IntervalAmbiguitySets(;
                lower = lower[:, ((s - 1) * n_actions + 1):(s * n_actions)],
                upper = upper[:, ((s - 1) * n_actions + 1):(s * n_actions)],
            ) for s in 1:n_states
        ]
        mdp = IntervalMarkovDecisionProcess(transition_probs, [1])

        prop = FiniteTimeReachability([n_states], horizon)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, _, _ = solve(problem)

        all(0 <= v <= 1 for v in V)
    end
end

# Property 2 — Monotonicity in horizon.
@testitem "property: V_T ≤ V_{T+1} for finite reachability" tags = [:base, :property] begin
    using IntervalMDP, Random, Supposition

    function _valid_bounds(rng, n_target, n_columns)
        lower = zeros(n_target, n_columns)
        upper = zeros(n_target, n_columns)
        for j in 1:n_columns
            p_raw = rand(rng, n_target)
            p = p_raw ./ sum(p_raw)
            r = rand(rng, n_target) .* min.(p, 1 .- p) .* 0.6
            lower[:, j] .= max.(0.0, p .- r)
            upper[:, j] .= min.(1.0, p .+ r)
        end
        return lower, upper
    end

    @check db = false function v_monotone(
        seed = Data.Integers{UInt32}(),
        n_states = Data.Integers(2, 4),
        n_actions = Data.Integers(1, 3),
        horizon = Data.Integers(1, 4),
    )
        rng = MersenneTwister(seed)
        lower, upper = _valid_bounds(rng, n_states, n_states * n_actions)
        transition_probs = [
            IntervalAmbiguitySets(;
                lower = lower[:, ((s - 1) * n_actions + 1):(s * n_actions)],
                upper = upper[:, ((s - 1) * n_actions + 1):(s * n_actions)],
            ) for s in 1:n_states
        ]
        mdp = IntervalMarkovDecisionProcess(transition_probs, [1])

        prop_T = FiniteTimeReachability([n_states], horizon)
        spec_T = Specification(prop_T, Pessimistic, Maximize)
        V_T, _, _ = solve(VerificationProblem(mdp, spec_T))

        prop_T1 = FiniteTimeReachability([n_states], horizon + 1)
        spec_T1 = Specification(prop_T1, Pessimistic, Maximize)
        V_T1, _, _ = solve(VerificationProblem(mdp, spec_T1))

        all(V_T .<= V_T1 .+ 10 * eps(eltype(V_T)))
    end
end

# Property 3 — GenSamplingDP(AllSampling()) parity with RobustVI.
@testitem "property: GenSamplingDP(AllSampling()) ≡ RobustVI on V" tags = [:base, :property] begin
    using IntervalMDP, Random, Supposition

    function _valid_bounds(rng, n_target, n_columns)
        lower = zeros(n_target, n_columns)
        upper = zeros(n_target, n_columns)
        for j in 1:n_columns
            p_raw = rand(rng, n_target)
            p = p_raw ./ sum(p_raw)
            r = rand(rng, n_target) .* min.(p, 1 .- p) .* 0.6
            lower[:, j] .= max.(0.0, p .- r)
            upper[:, j] .= min.(1.0, p .+ r)
        end
        return lower, upper
    end

    @check db = false function gensampling_parity(
        seed = Data.Integers{UInt32}(),
        n_states = Data.Integers(2, 4),
        n_actions = Data.Integers(1, 3),
        horizon = Data.Integers(1, 5),
    )
        rng = MersenneTwister(seed)
        lower, upper = _valid_bounds(rng, n_states, n_states * n_actions)
        transition_probs = [
            IntervalAmbiguitySets(;
                lower = lower[:, ((s - 1) * n_actions + 1):(s * n_actions)],
                upper = upper[:, ((s - 1) * n_actions + 1):(s * n_actions)],
            ) for s in 1:n_states
        ]
        mdp = IntervalMarkovDecisionProcess(transition_probs, [1])

        prop = FiniteTimeReachability([n_states], horizon)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)

        rvi = RobustValueIteration(default_bellman_algorithm(mdp))
        gsdp =
            GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))

        V_rvi, _, _ = solve(problem, rvi)
        V_gsdp, _, _ = solve(problem, gsdp)

        V_rvi == V_gsdp
    end
end

# Property 4 — Soundness of O-Maximization vs. vertex enumeration.
@testitem "property: O-Max ≤ Vertex enumeration (lower bound)" tags = [:base, :property] begin
    using IntervalMDP, Random, Supposition

    function _valid_bounds(rng, n_target, n_columns)
        lower = zeros(n_target, n_columns)
        upper = zeros(n_target, n_columns)
        for j in 1:n_columns
            p_raw = rand(rng, n_target)
            p = p_raw ./ sum(p_raw)
            r = rand(rng, n_target) .* min.(p, 1 .- p) .* 0.6
            lower[:, j] .= max.(0.0, p .- r)
            upper[:, j] .= min.(1.0, p .+ r)
        end
        return lower, upper
    end

    @check db = false function omax_le_vertex(
        seed = Data.Integers{UInt32}(),
        n_states = Data.Integers(2, 4),
        n_actions = Data.Integers(1, 2),
    )
        rng = MersenneTwister(seed)
        lower, upper = _valid_bounds(rng, n_states, n_states * n_actions)
        transition_probs = [
            IntervalAmbiguitySets(;
                lower = lower[:, ((s - 1) * n_actions + 1):(s * n_actions)],
                upper = upper[:, ((s - 1) * n_actions + 1):(s * n_actions)],
            ) for s in 1:n_states
        ]
        mdp = IntervalMarkovDecisionProcess(transition_probs, [1])

        V = rand(rng, n_states)
        upd = IntervalMDP.sample(IntervalMDP.default_sampling_strategy(), mdp)
        Q_omax = let
    _Qres = Array{eltype(V)}(undef, (IntervalMDP.action_values(mdp)..., size(V)...))
    _ws = IntervalMDP.construct_workspace(mdp)
    _sc = IntervalMDP.construct_strategy_cache(mdp)
    IntervalMDP.bellman_q!(_ws, _sc, IntervalMDP.StateActionValueArray(_Qres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false)
    _Qres
end
        Q_vertex =
            let
    _Qres = Array{eltype(V)}(undef, (IntervalMDP.action_values(mdp)..., size(V)...))
    _ws = IntervalMDP.construct_workspace(mdp)
    _sc = IntervalMDP.construct_strategy_cache(mdp)
    IntervalMDP.bellman_q!(_ws, _sc, IntervalMDP.StateActionValueArray(_Qres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false)
    _Qres
end

        tol = 1e-9
        all(Q_omax .<= Q_vertex .+ tol)
    end
end
