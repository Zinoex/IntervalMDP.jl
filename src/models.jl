abstract type System end


struct IntervalMarkovChain{P <: IntervalProbabilities, T <: Integer} <: System
    transition_prob::P
    initial_state::T
    num_states::T

    function IntervalMarkovChain(transition_prob::P, initial_state::T) where {P <: IntervalProbabilities, T <: Integer}
        num_states = checksize!(transition_prob)
    
        return new{P, T}(transition_prob, initial_state, num_states)
    end
end

function checksize!(p::AbstractVector{<:StateIntervalProbabilities})
    g = gap(p)
    num_states = length(g)
    for j in eachindex(g)
        if length(g[j]) != num_states
            throw(DimensionMismatch("The number of transition probabilities in the vector at index $j is not equal to the number of states in the problem"))
        end
    end

    return num_states
end

function checksize!(p::MatrixIntervalProbabilities)
    g = gap(p)
    num_states = size(g, 1)
    if size(g, 2) != num_states
        throw(DimensionMismatch("The number of transition probabilities in the matrix is not equal to the number of states in the problem"))
    end

    return num_states
end

transition_prob(s::IntervalMarkovChain) = s.transition_prob
initial_state(s::IntervalMarkovChain) = s.initial_state
num_states(s::IntervalMarkovChain) = s.num_states