"""
    IntervalMarkovProcess

An abstract type for interval Markov processes including [`IntervalMarkovChain`](@ref) and [`IntervalMarkovDecisionProcess`](@ref).
"""
abstract type IntervalMarkovProcess <: StochasticProcess end

"""
    initial_states(mp::IntervalMarkovProcess)

Return the initial states. If the initial states are not specified, return `nothing`.
"""
initial_states(mp::IntervalMarkovProcess) = mp.initial_states

"""
    AllStates

A type to represent all states in a Markov process. This type is used to specify all states as the initial states.
"""
struct AllStates end
const InitialStates = Union{AllStates, AbstractVector}

"""
    num_states(mp::IntervalMarkovProcess)

Return the number of states.
"""
num_states(mp::IntervalMarkovProcess) = mp.num_states

"""
    transition_prob(mp::IntervalMarkovProcess)

Return the interval on transition probabilities.
"""
transition_prob(mp::IntervalMarkovProcess) = mp.transition_prob
