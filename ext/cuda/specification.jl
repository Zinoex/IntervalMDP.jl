
Adapt.@adapt_structure FiniteTimeReward
Adapt.@adapt_structure InfiniteTimeReward
Adapt.@adapt_structure Problem

function IntervalMDP.checkterminalprob!(
    terminal_states,
    system::IntervalMarkovChain{<:IntervalProbabilities{R, VR, MR}},
) where {R, VR, MR <: CuSparseMatrixCSC}
    prob = transition_prob(system)
    lower_prob = lower(prob)
    gap_prob = gap(prob)

    for j in terminal_states
        target = CuVector([Float64(i == j) for i in 1:num_states(system)])

        @assert all(iszero, @view(gap_prob[:, j])) &&
                all(iszero, @view(lower_prob[:, j]) - target) "The terminal state $j must have a transition probability of 1 to itself and 0 to all other states"
    end
end

function checkterminalprob!(
    terminal_states,
    system::IntervalMarkovDecisionProcess{<:IntervalProbabilities{R, VR, MR}},
) where {R, VR, MR <: AbstractMatrix}
    prob = transition_prob(system)
    lower_prob = lower(prob)
    gap_prob = gap(prob)
    sptr = stateptr(system)

    for j in terminal_states
        target = CuVector([Float64(i == j) for i in 1:num_states(system)])

        for s in sptr[j]:(sptr[j + 1] - 1)
            @assert all(iszero, @view(gap_prob[:, s])) &&
                    all(iszero, @view(lower_prob[:, s]) - target) "All actions for the terminal state $j must have a transition probability of 1 to itself and 0 to all other states"
        end
    end
end
