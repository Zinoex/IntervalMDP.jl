abstract type System end


struct IntervalMarkovChain{P <: IntervalProbabilities} <: System
    transition_prob::P
    initial_state::Int32
    num_states::Int32

    function IntervalMarkovChain(transition_prob::P, initial_state::Int32) where {P <: IntervalProbabilities}
        num_states = checksize!(transition_prob)
    
        return new{P}(transition_prob, initial_state, num_states)
    end
end

function checksize!(p::AbstractVector{<:AbstractVector})
    num_states = length(p)
    for j in eachindex(p)
        if length(p[j]) != num_states
            throw(DimensionMismatch("The number of transition probabilities in the vector at index $j is not equal to the number of states in the problem"))
        end
    end

    return num_states
end

function checksize!(p::AbstractMatrix)
    num_states = size(p, 1)
    if size(p, 2) != num_states
        throw(DimensionMismatch("The number of transition probabilities in the matrix is not equal to the number of states in the problem"))
    end

    return num_states
end

transition_prob(s::IntervalMarkovChain) = s.transition_prob
initial_state(s::IntervalMarkovChain) = s.initial_state
num_states(s::IntervalMarkovChain) = s.num_states