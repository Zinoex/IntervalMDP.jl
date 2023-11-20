abstract type System end

### Interval Markov Chain
struct IntervalMarkovChain{P <: MatrixIntervalProbabilities, T <: Integer} <: System
    transition_prob::P
    initial_state::T
    num_states::T
end

function IntervalMarkovChain(
    transition_prob::P,
    initial_state::T,
) where {P <: MatrixIntervalProbabilities, T <: Integer}
    num_states = checksize_imc!(transition_prob)
    num_states = T(num_states)

    return IntervalMarkovChain(transition_prob, initial_state, num_states)
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
struct IntervalMarkovDecisionProcess{
    P <: MatrixIntervalProbabilities,
    T <: Integer,
    VT <: AbstractVector{T},
    VA <: AbstractVector,
} <: System
    transition_prob::P
    stateptr::VT
    action_vals::VA
    initial_state::T
    num_states::T
end

function IntervalMarkovDecisionProcess(
    transition_prob::P,
    stateptr::VT,
    action_vals::VA,
    initial_state::T,
) where {
    P <: MatrixIntervalProbabilities,
    T <: Integer,
    VT <: AbstractVector{T},
    VA <: AbstractVector,
}
    num_states = checksize_imdp!(transition_prob, stateptr)
    num_states = T(num_states)

    return IntervalMarkovDecisionProcess(
        transition_prob,
        stateptr,
        action_vals,
        initial_state,
        num_states,
    )
end

function IntervalMarkovDecisionProcess(
    transition_probs::Vector{P},
    action_vals::VA,
    initial_state::T,
) where {P <: MatrixIntervalProbabilities, T <: Integer, VA <: AbstractVector}
    transition_prob, stateptr = interval_prob_hcat(T, transition_probs)

    return IntervalMarkovDecisionProcess(
        transition_prob,
        stateptr,
        action_vals,
        initial_state,
    )
end

function IntervalMarkovDecisionProcess(
    transition_probs::Vector{Pair{VA, P}},
    initial_state::T,
) where {P <: MatrixIntervalProbabilities, T <: Integer, VA <: AbstractVector}
    action_vals = mapreduce(first, vcat, transition_probs)
    transition_probs = map(x -> x[2], transition_probs)

    return IntervalMarkovDecisionProcess(transition_probs, action_vals, initial_state)
end

function checksize_imdp!(p::MatrixIntervalProbabilities, stateptr)
    g = gap(p)
    num_states = length(stateptr) - 1

    num_actions_per_state = diff(stateptr)
    @assert all(num_actions_per_state .> 0) "The number of actions per state must be positive"

    if size(g, 1) != num_states
        throw(
            DimensionMismatch(
                "The number of transition probabilities in the matrix is not equal to the number of states in the problem",
            ),
        )
    end

    return num_states
end

transition_prob(s::IntervalMarkovDecisionProcess) = s.transition_prob
actions(s::IntervalMarkovDecisionProcess) = s.action_vals
initial_state(s::IntervalMarkovDecisionProcess) = s.initial_state
num_states(s::IntervalMarkovDecisionProcess) = s.num_states
stateptr(s::IntervalMarkovDecisionProcess) = s.stateptr
