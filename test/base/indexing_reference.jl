@testitem "base/indexing_reference: Marginal sub2ind is column-major (actions first)" begin
    # Cross-check for the Lean theorem `IntervalMDP.Index.marginalSub2ind_eq_linear`
    # (lean/IntervalMDPProofs/Index/Marginal.lean): `sub2ind(marginal, a, s)` equals
    # `LinearIndices((action_vars[action_indices]..., source_dims[state_indices]...))[a[action_indices]..., s[state_indices]...]`.
    using Random
    using IntervalMDP: sub2ind

    rng = MersenneTwister(20261008)

    # Strictly increasing random subset of 1:n (possibly empty if `allow_empty`).
    function random_indices(rng, n, allow_empty)
        while true
            inds = Tuple(i for i in 1:n if rand(rng, Bool))
            if allow_empty || !isempty(inds)
                return inds
            end
        end
    end

    for _ in 1:200
        num_state_vars = rand(rng, 1:3)
        num_action_vars = rand(rng, 1:3)
        state_vars = Tuple(rand(rng, 1:3) for _ in 1:num_state_vars)
        action_vars = Tuple(rand(rng, 1:3) for _ in 1:num_action_vars)

        state_indices = random_indices(rng, num_state_vars, false)
        action_indices = random_indices(rng, num_action_vars, true)
        marginal_source_dims = getindex.((state_vars,), state_indices)
        marginal_action_vars = getindex.((action_vars,), action_indices)

        num_target = rand(rng, 1:3)
        num_sets = prod(marginal_source_dims) * prod(marginal_action_vars; init = 1)
        ambiguity_sets = IntervalAmbiguitySets(;
            lower = zeros(num_target, num_sets),
            upper = ones(num_target, num_sets),
        )
        marginal = Marginal(
            ambiguity_sets,
            state_indices,
            action_indices,
            marginal_source_dims,
            marginal_action_vars,
        )

        layout = LinearIndices((marginal_action_vars..., marginal_source_dims...))
        seen = Set{Int}()
        for jₛ in CartesianIndices(state_vars), jₐ in CartesianIndices(action_vars)
            s = Tuple(jₛ)
            a = Tuple(jₐ)
            expected = layout[
                getindex.((a,), action_indices)...,
                getindex.((s,), state_indices)...,
            ]
            @test sub2ind(marginal, a, s) == expected
            @test sub2ind(marginal, jₐ, jₛ) == expected
            push!(seen, sub2ind(marginal, a, s))
        end
        # Bijective onto 1:num_sets (`marginalSub2ind_bijective`).
        @test seen == Set(1:num_sets)
    end
end

@testitem "base/indexing_reference: IntervalAmbiguitySets sub2ind on single-action layouts" begin
    # Cross-check for `IntervalMDP.Index.intervalSub2ind_correct`: with one action and the
    # marginal conditioning on state variable 1 only (the layout built by
    # `IntervalMarkovDecisionProcess`), `sub2ind(::IntervalAmbiguitySets, jₐ, jₛ) = jₛ[1]`
    # equals `sub2ind(::Marginal, jₐ, jₛ)`.
    using IntervalMDP: sub2ind

    for num_states in 1:4
        ambiguity_sets = IntervalAmbiguitySets(;
            lower = zeros(num_states, num_states),
            upper = ones(num_states, num_states),
        )
        marginal = Marginal(ambiguity_sets, (num_states,), (1,))
        for s in 1:num_states
            @test sub2ind(ambiguity_sets, (1,), (s,)) == sub2ind(marginal, (1,), (s,)) == s
        end
    end
end
