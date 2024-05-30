"""
    IntervalMarkovProcess{P <: IntervalProbabilities}

An abstract type for interval Markov processes including [`IntervalMarkovChain`](@ref) and [`IntervalMarkovDecisionProcess`](@ref).
"""
abstract type IntervalMarkovProcess{P <: IntervalProbabilities} end

function all_initial_states(num_states)
    if num_states <= typemax(Int32)
        return Base.OneTo(Int32(num_states))
    else
        return Base.OneTo(Int64(num_states))
    end
end
