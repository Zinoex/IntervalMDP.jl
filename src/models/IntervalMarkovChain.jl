function IntervalMarkovChain(marginal::Marginal{<:IntervalAmbiguitySets}, initial_states=AllStates())
    state_vars = (Int32(num_target(marginal)),)
    source_dims = source_shape(marginal)

    if action_shape(marginal) != (1,)
        throw(DimensionMismatch("The action shape of the marginal must be (1,) for an IntervalMarkovChain. Got $(action_shape(marginal))."))
    end

    action_vars = (Int32(1),)

    return FactoredRMDP( # wrap in a FactoredRMDP for consistency
        state_vars,
        action_vars,
        source_dims,
        (marginal,),
        initial_states,
    )
end


function IntervalMarkovChain(ambiguity_set::IntervalAmbiguitySets, initial_states=AllStates())
    state_indices = (1,)
    action_indices = (1,)
    source_dims = (num_sets(ambiguity_set),)
    action_vars = (1,)
    marginal = Marginal(ambiguity_set, state_indices, action_indices, source_dims, action_vars)

    return IntervalMarkovChain(marginal, initial_states)
end