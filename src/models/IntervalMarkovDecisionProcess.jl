function IntervalMarkovDecisionProcess(marginal::Marginal{<:IntervalAmbiguitySets}, initial_states::InitialStates = AllStates())
    state_vars = (Int32(num_target(marginal)),)
    action_vars = action_shape(marginal)
    source_dims = source_shape(marginal)
    transition = (marginal,)

    return FactoredRMDP(
        state_vars,
        action_vars,
        source_dims,
        transition,
        initial_states
    )
end

function IntervalMarkovDecisionProcess(ambiguity_set::IntervalAmbiguitySets, num_actions::Integer, initial_states::InitialStates = AllStates())
    if num_sets(ambiguity_set) % num_actions != 0
        throw(ArgumentError("The number of sets in the ambiguity set must be a multiple of the number of actions."))
    end

    source_dims = (num_sets(ambiguity_set) ÷ num_actions,)
    action_vars = (num_actions,)
    marginal = Marginal(ambiguity_set, source_dims, action_vars)

    return IntervalMarkovDecisionProcess(marginal, initial_states)
end

function IntervalMarkovDecisionProcess(
    ps::Vector{<:IntervalAmbiguitySets},
    initial_states::InitialStates = AllStates(),
)
    marginal = interval_prob_hcat(ps)
    return IntervalMarkovDecisionProcess(marginal, initial_states)
end

function interval_prob_hcat(
    ps::Vector{<:IntervalAmbiguitySets{R, MR}},
) where {R, MR <: AbstractMatrix{R}}
    if length(ps) == 0
        throw(ArgumentError("Cannot concatenate an empty vector of IntervalAmbiguitySets."))
    end

    num_actions = num_sets(ps[1])
    for (i, p) in enumerate(ps)
        if num_sets(p) != num_actions
            throw(DimensionMismatch("All IntervalAmbiguitySets must have the same number of sets (actions). Expected $num_actions, was $(num_sets(p)) at index $i."))
        end
    end

    l = mapreduce(p -> p.lower, hcat, ps)
    g = mapreduce(p -> p.gap, hcat, ps)

    ambiguity_set = IntervalAmbiguitySets(l, g)

    source_dims = (length(ps),)
    action_vars = (num_actions,)
    marginal = Marginal(ambiguity_set, source_dims, action_vars)

    return marginal
end