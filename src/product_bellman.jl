# Bellman primitives for `ProductProcess` (an `IntervalMarkovProcess`
# composed with a DFA via a labelling function). The product expectation
# splits along the DFA axis: for each DFA state, we project the value
# function via the labelling function and recurse into the underlying
# Markov process's Bellman primitive on a slice of the result buffer.
#
# Both `bellman_v!` (V-shape) and `bellman_q!` (Q-shape) are supported.
# In each case the DFA axis is the last axis of the corresponding result
# buffer (so a V-shape Vres has shape `(state_shape..., dfa_states)`,
# and a Q-shape Qres has shape
# `(action_shape..., state_shape..., dfa_states)`).

###############################################################################
# Q-shape: bellman_q!                                                         #
###############################################################################

function bellman_q!(
    workspace::ProductWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Qres::StateActionValueArray,
    V::StateValueArray,
    model::ProductProcess,
    update_sequence = sample(default_sampling_strategy(), model, strategy_cache);
    upper_bound = false,
    maximize = true,
    prop = nothing,
)
    return _bellman_q_product!(
        workspace,
        strategy_cache,
        parent(Qres),
        parent(V),
        automaton(model),
        labelling_function(model),
        markov_process(model),
        update_sequence;
        upper_bound = upper_bound,
        maximize = maximize,
        prop = prop,
    )
end

function _bellman_q_product!(
    workspace::ProductWorkspace,
    strategy_cache,
    Qres,
    V,
    dfa::DFA,
    lf::DeterministicLabelling,
    mp::IntervalMarkovProcess,
    update_sequence;
    upper_bound,
    maximize,
    prop,
)
    W = workspace.intermediate_values

    @inbounds for state in dfa
        if !isnothing(prop) && state ∈ terminal(prop)
            continue
        end
        local_sc = localize_strategy_cache(strategy_cache, state)
        # Project V to the underlying-MDP V[s] for this DFA state.
        map!(W, CartesianIndices(state_values(mp))) do idx
            return V[idx, dfa[state, lf[idx]]]
        end
        bellman_q!(
            workspace.underlying_workspace,
            local_sc,
            StateActionValueArray(selectdim(Qres, ndims(Qres), state)),
            StateValueArray(W),
            mp,
            update_sequence;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end
    return Qres
end

function _bellman_q_product!(
    workspace::ProductWorkspace,
    strategy_cache,
    Qres,
    V::AbstractArray{R},
    dfa::DFA,
    lf::ProbabilisticLabelling,
    mp::IntervalMarkovProcess,
    update_sequence;
    upper_bound,
    maximize,
    prop,
) where {R}
    W = workspace.intermediate_values

    @inbounds for state in dfa
        if !isnothing(prop) && state ∈ terminal(prop)
            continue
        end
        local_sc = localize_strategy_cache(strategy_cache, state)
        map!(W, CartesianIndices(state_values(mp))) do idx
            v = zero(R)
            for (label, prob) in enumerate(lf[idx])
                v += prob * V[idx, dfa[state, label]]
            end
            return v
        end
        bellman_q!(
            workspace.underlying_workspace,
            local_sc,
            StateActionValueArray(selectdim(Qres, ndims(Qres), state)),
            StateValueArray(W),
            mp,
            update_sequence;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end
    return Qres
end

###############################################################################
# V-shape: bellman_v!                                                         #
###############################################################################

function bellman_v!(
    workspace::ProductWorkspace,
    strategy_cache,
    Vres::StateValueArray,
    V::StateValueArray,
    model::ProductProcess,
    update_sequence = sample(default_sampling_strategy(), model, strategy_cache);
    upper_bound = false,
    maximize = true,
    prop = nothing,
)
    return _bellman_v_product!(
        workspace,
        strategy_cache,
        parent(Vres),
        parent(V),
        automaton(model),
        labelling_function(model),
        markov_process(model),
        update_sequence;
        upper_bound = upper_bound,
        maximize = maximize,
        prop = prop,
    )
end

function _bellman_v_product!(
    workspace::ProductWorkspace,
    strategy_cache,
    Vres,
    V,
    dfa::DFA,
    lf::DeterministicLabelling,
    mp::IntervalMarkovProcess,
    update_sequence;
    upper_bound,
    maximize,
    prop,
)
    W = workspace.intermediate_values

    @inbounds for state in dfa
        if !isnothing(prop) && state ∈ terminal(prop)
            continue
        end
        local_sc = localize_strategy_cache(strategy_cache, state)
        map!(W, CartesianIndices(state_values(mp))) do idx
            return V[idx, dfa[state, lf[idx]]]
        end
        bellman_v!(
            workspace.underlying_workspace,
            local_sc,
            StateValueArray(selectdim(Vres, ndims(Vres), state)),
            StateValueArray(W),
            mp,
            update_sequence;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end
    return Vres
end

function _bellman_v_product!(
    workspace::ProductWorkspace,
    strategy_cache,
    Vres,
    V::AbstractArray{R},
    dfa::DFA,
    lf::ProbabilisticLabelling,
    mp::IntervalMarkovProcess,
    update_sequence;
    upper_bound,
    maximize,
    prop,
) where {R}
    W = workspace.intermediate_values

    @inbounds for state in dfa
        if !isnothing(prop) && state ∈ terminal(prop)
            continue
        end
        local_sc = localize_strategy_cache(strategy_cache, state)
        map!(W, CartesianIndices(state_values(mp))) do idx
            v = zero(R)
            for (label, prob) in enumerate(lf[idx])
                v += prob * V[idx, dfa[state, label]]
            end
            return v
        end
        bellman_v!(
            workspace.underlying_workspace,
            local_sc,
            StateValueArray(selectdim(Vres, ndims(Vres), state)),
            StateValueArray(W),
            mp,
            update_sequence;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end
    return Vres
end

###############################################################################
# Strategy cache localisation (per DFA state)                                 #
###############################################################################

localize_strategy_cache(strategy_cache::NoStrategyCache, dfa_state) = strategy_cache

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
