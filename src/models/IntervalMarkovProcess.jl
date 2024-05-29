"""
    IntervalMarkovProcess{P <: IntervalProbabilities}

An abstract type for interval Markov processes including [`IntervalMarkovChain`](@ref) and [`IntervalMarkovDecisionProcess`](@ref).
"""
abstract type IntervalMarkovProcess{P <: IntervalProbabilities} end

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

##############
# Stationary #
# ############
"""
    StationaryIntervalMarkovProcess{P <: IntervalProbabilities}

An abstract type for stationary interval Markov processes including [`IntervalMarkovChain`](@ref) and [`IntervalMarkovDecisionProcess`](@ref).
"""
abstract type StationaryIntervalMarkovProcess{P <: IntervalProbabilities} <:
              IntervalMarkovProcess{P} end
transition_prob(mp::StationaryIntervalMarkovProcess, t) = transition_prob(mp)
time_length(mp::StationaryIntervalMarkovProcess) = typemax(Int64)

"""
    transition_prob(mp::StationaryIntervalMarkovProcess)

Return the interval on transition probabilities.
"""
transition_prob(mp::StationaryIntervalMarkovProcess) = mp.transition_prob

################
# Time-varying #
# ##############
"""
    TimeVaryingIntervalMarkovProcess{P <: IntervalProbabilities}

An abstract type for time-varying interval Markov processes including [`TimeVaryingIntervalMarkovChain`](@ref).
"""
abstract type TimeVaryingIntervalMarkovProcess{P <: IntervalProbabilities} <:
              IntervalMarkovProcess{P} end
transition_probs(s::TimeVaryingIntervalMarkovProcess) = s.transition_probs

"""
    time_length(mp::TimeVaryingIntervalMarkovProcess)

Return the time length of the time-varying interval Markov process. Model checking for this type of process
must be done over for a finite time property of equal length.
"""
time_length(mp::TimeVaryingIntervalMarkovProcess) = length(transition_probs(mp))

"""
    transition_prob(s::TimeVaryingIntervalMarkovProcess, t)

Return the interval on transition probabilities at time step ``t``.
"""
function transition_prob(mp::TimeVaryingIntervalMarkovProcess, t)
    if t < 1 || t > time_length(mp)
        throw(DomainError("Time step must be between 1 and $(time_length(mp))"))
    end

    return transition_probs(mp)[t]
end