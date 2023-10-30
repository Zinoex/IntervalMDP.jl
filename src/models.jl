abstract type System end

### Interval Markov Chain
struct IntervalMarkovChain{P <: IntervalProbabilities, T <: Integer} <: System
    transition_prob::P
    initial_state::T
    num_states::T

    function IntervalMarkovChain(
        transition_prob::P,
        initial_state::T,
    ) where {P <: IntervalProbabilities, T <: Integer}
        num_states = checksize_imc!(transition_prob)

        return new{P, T}(transition_prob, initial_state, num_states)
    end
end

function checksize_imc!(p::AbstractVector{<:StateIntervalProbabilities})
    g = gap(p)
    num_states = length(g)
    for j in eachindex(g)
        if length(g[j]) != num_states
            throw(
                DimensionMismatch(
                    "The number of transition probabilities in the vector at index $j is not equal to the number of states in the problem",
                ),
            )
        end
    end

    return num_states
end

function checksize_imc!(p::MatrixIntervalProbabilities)
    g = gap(p)
    num_states = size(g, 1)
    if size(g, 2) != num_states
        throw(
            DimensionMismatch(
                "The number of transition probabilities in the matrix is not equal to the number of states in the problem",
            ),
        )
    end

    return num_states
end

transition_prob(s::IntervalMarkovChain) = s.transition_prob
initial_state(s::IntervalMarkovChain) = s.initial_state
num_states(s::IntervalMarkovChain) = s.num_states

### Interval Markov Decision Process
struct IntervalMarkovDecisionProcess{P <: IntervalProbabilities, T <: Integer, VT <: AbstractVector{T}, VA <: AbstractVector} <: System
    transition_prob::P
    stateptr::VT
    action_vals::VA
    initial_state::T
    num_states::T

    function IntervalMarkovDecisionProcess(
        transition_prob::P,
        stateptr::VT,
        action_vals::VA,
        initial_state::T,
    ) where {P <: IntervalProbabilities, T <: Integer, VT <: AbstractVector{T}, VA <: AbstractVector}
        num_states = checksize_imdp!(p, stateptr)

        return new{P, T, VT, VA}(transition_prob, stateptr, action_vals, initial_state, num_states)
    end
end

function IntervalMarkovDecisionProcess(transition_probs::Vector{P}, action_vals::VA, initial_state::VT)  where {P <: IntervalProbabilities, VT <: AbstractVector{<: Integer}, VA <: AbstractVector}
    transition_prob = type_specific_hcat(transition_probs)
    lengths = map(length, transition_probs)
    stateptr = T[1; cumsum(lengths) .+ 1]

    return IntervalMarkovDecisionProcess(transition_prob, stateptr, action_vals, initial_state)
end

function checksize_imdp!(p::AbstractVector{<:StateIntervalProbabilities}, stateptr)
    g = gap(p)
    num_states = length(stateptr) - 1

    num_actions_per_state = diff(stateptr)
    @assert all(num_actions_per_state .> 0) "The number of actions per state must be positive"

    for j in eachindex(g)
        if length(g[j]) != num_states
            throw(
                DimensionMismatch(
                    "The number of transition probabilities in the vector at index $j is not equal to the number of states in the problem",
                ),
            )
        end
    end

    return num_states
end

function checksize_imdp!(p::MatrixIntervalProbabilities, stateptr)
    g = gap(p)
    num_states = length(stateptr) - 1
    
    num_actions_per_state = diff(stateptr)
    @assert all(num_actions_per_state .> 0) "The number of actions per state must be positive"

    if size(g, 2) != num_states
        throw(
            DimensionMismatch(
                "The number of transition probabilities in the matrix is not equal to the number of states in the problem",
            ),
        )
    end

    return num_states
end

transition_prob(s::IntervalMarkovDecisionProcess) = s.transition_prob
initial_state(s::IntervalMarkovDecisionProcess) = s.initial_state
num_states(s::IntervalMarkovDecisionProcess) = s.num_states
stateptr(s::IntervalMarkovDecisionProcess) = s.stateptr
