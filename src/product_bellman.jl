

function bellman!(
    workspace::ProductWorkspace,
    strategy_cache,
    Vres::AbstractArray,
    V::AbstractArray,
    model::ProductProcess,
    update_sequence = sample(default_sampling_strategy(), model, strategy_cache);
    upper_bound = false,
    maximize = true,
    prop = nothing,
)
    mp = markov_process(model)
    lf = labelling_function(model)
    dfa = automaton(model)

    return _bellman_helper!(
        workspace,
        strategy_cache,
        Vres,
        V,
        dfa,
        lf,
        mp,
        update_sequence;
        upper_bound = upper_bound,
        maximize = maximize,
        prop = prop,
    )
end

function _bellman_helper!(
    workspace::ProductWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    dfa::DFA,
    lf::DeterministicLabelling,
    mp::IntervalMarkovProcess,
    update_sequence;
    upper_bound = false,
    maximize = true,
    prop = nothing,
)
    W = workspace.intermediate_values

    @inbounds for state in dfa
        # If a DFA property is given, skip terminal states
        if !isnothing(prop) && state ∈ terminal(prop)
            continue
        end

        local_strategy_cache = localize_strategy_cache(strategy_cache, state)

        # Select the value function for the current DFA state
        # according to the appropriate DFA transition function
        map!(W, CartesianIndices(state_values(mp))) do idx
            return V[idx, dfa[state, lf[idx]]]
        end

        # For each state in the product process, compute the Bellman operator
        # for the corresponding Markov process
        bellman!(
            workspace.underlying_workspace,
            local_strategy_cache,
            selectdim(Vres, ndims(Vres), state),
            W,
            mp,
            update_sequence; #TODO: need to separate automata states
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::ProductWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V::AbstractArray{R},
    dfa::DFA,
    lf::ProbabilisticLabelling,
    mp::IntervalMarkovProcess,
    update_sequence;
    upper_bound = false,
    maximize = true,
    prop = nothing,
) where {R}
    W = workspace.intermediate_values

    @inbounds for state in dfa
        # If a DFA property is given, skip terminal states
        if !isnothing(prop) && state ∈ terminal(prop)
            continue
        end

        local_strategy_cache = localize_strategy_cache(strategy_cache, state)

        # Select the value function for the current DFA state
        # according to the appropriate DFA transition function
        map!(W, CartesianIndices(state_values(mp))) do idx
            v = zero(R)

            for (label, prob) in enumerate(lf[idx])
                new_dfa_state = dfa[state, label]
                v += prob * V[idx, new_dfa_state]
            end

            return v
        end

        # For each state in the product process, compute the Bellman operator
        # for the corresponding Markov process
        bellman!(
            workspace.underlying_workspace,
            local_strategy_cache,
            selectdim(Vres, ndims(Vres), state),
            W,
            mp,
            update_sequence; #TODO: need to separate automata states
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function localize_strategy_cache(strategy_cache::NoStrategyCache, dfa_state)
    return strategy_cache
end

function localize_strategy_cache(strategy_cache::TimeVaryingStrategyCache, dfa_state)
    return TimeVaryingStrategyCache(
        selectdim(
            strategy_cache.cur_strategy,
            ndims(strategy_cache.cur_strategy),
            dfa_state,
        ),
    )
end

function localize_strategy_cache(strategy_cache::StationaryStrategyCache, dfa_state)
    return StationaryStrategyCache(
        selectdim(strategy_cache.strategy, ndims(strategy_cache.strategy), dfa_state),
    )
end

function localize_strategy_cache(strategy_cache::ActiveGivenStrategyCache, dfa_state)
    return ActiveGivenStrategyCache(
        selectdim(strategy_cache.strategy, ndims(strategy_cache.strategy), dfa_state),
    )
end