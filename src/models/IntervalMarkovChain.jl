function IntervalMarkovChain(ambiguity_set::IntervalAmbiguitySets{R, MR}, initial_states=AllStates()) where {R, MR <: AbstractMatrix{R}}
    state_vars = (num_target(ambiguity_set),)

    state_indices = (1,)
    action_indices = (1,)
    source_dims = (num_sets(ambiguity_set),)
    action_vars = (1,)
    marginal = Marginal(ambiguity_set, state_indices, action_indices, source_dims, action_vars)

    return FactoredRMDP( # wrap in a FactoredRMDP for consistency
        state_vars,
        action_vars,
        source_dims,
        (marginal,),
        initial_states,
    )
end