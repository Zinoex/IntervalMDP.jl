
function IntervalMarkovDecisionProcess(states::Vector{S}, actions::Vector{A}, transition_intervals::Dict{Tuple{S,A,S}, Tuple{Float64, Float64}}, rewards::Dict{Tuple{S,A}, Float64}) where {S,A}
    # Validate inputs
    for ((s, a, s_next), (p_min, p_max)) in transition_intervals
        @assert 0.0 <= p_min <= p_max <= 1.0 "Transition probabilities must be in [0, 1] and p_min <= p_max"
    end
    for s in states
        for a in actions
            total_min = sum(p_min for ((s2, a2, s3), (p_min, p_max)) in transition_intervals if s2 == s && a2 == a)
            total_max = sum(p_max for ((s2, a2, s3), (p_min, p_max)) in transition_intervals if s2 == s && a2 == a)
            @assert total_min <= 1.0 "Total minimum transition probability from state $s with action $a exceeds 1"
            @assert total_max <= 1.0 "Total maximum transition probability from state $s with action $a exceeds 1"
        end
    end

    return IntervalMarkovDecisionProcess{S,A}(states, actions, transition_intervals, rewards)
end