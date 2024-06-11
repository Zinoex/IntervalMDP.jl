"""
    IntervalMarkovProcess

An abstract type for interval Markov processes including [`IntervalMarkovChain`](@ref) and [`IntervalMarkovDecisionProcess`](@ref).
"""
abstract type IntervalMarkovProcess end

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

abstract type SimpleIntervalMarkovProcess <: IntervalMarkovProcess end
dims(::SimpleIntervalMarkovProcess) = one(Int32)
product_num_states(mp::SimpleIntervalMarkovProcess) = [num_states(mp)]

##############
# Stationary #
##############
"""
    StationaryIntervalMarkovProcess

An abstract type for stationary interval Markov processes including [`IntervalMarkovDecisionProcess`](@ref).
"""
abstract type StationaryIntervalMarkovProcess <: SimpleIntervalMarkovProcess end
transition_prob(mp::StationaryIntervalMarkovProcess, t) = transition_prob(mp)
time_length(mp::StationaryIntervalMarkovProcess) = typemax(Int64)

"""
    transition_prob(mp::StationaryIntervalMarkovProcess)

Return the interval on transition probabilities.
"""
transition_prob(mp::StationaryIntervalMarkovProcess) = mp.transition_prob

################
# Time-varying #
################
"""
    TimeVaryingIntervalMarkovProcess

An abstract type for time-varying interval Markov processes including [`TimeVaryingIntervalMarkovChain`](@ref).
"""
abstract type TimeVaryingIntervalMarkovProcess <:
              SimpleIntervalMarkovProcess end
transition_probs(s::TimeVaryingIntervalMarkovProcess) = s.transition_probs

"""
    time_length(mp::TimeVaryingIntervalMarkovProcess)

Return the time length of the time-varying interval Markov process. Model checking for this type of process
must be done over for a finite time property of equal length.
"""
time_length(mp::TimeVaryingIntervalMarkovProcess) = length(transition_probs(mp))

"""
    transition_prob(mp::TimeVaryingIntervalMarkovProcess, t)

Return the interval on transition probabilities at time step ``t`` in the range `1:time_length(mp)`.
"""
function transition_prob(mp::TimeVaryingIntervalMarkovProcess, t)
    if t < 1 || t > time_length(mp)
        throw(DomainError("Time step must be between 1 and $(time_length(mp))"))
    end

    return transition_probs(mp)[t]
end

##############################
# Composite Markov processes #
##############################
"""
ProductIntervalMarkovProcess

An abstract type for composite interval Markov processes including [`SequentialIntervalMarkovProcess`](@ref) and [`ProductIntervalMarkovProcess`](@ref).
"""
abstract type CompositeIntervalMarkovProcess <: IntervalMarkovProcess end

"""
    SequentialIntervalMarkovProcess

An abstract type for sequential interval Markov processes.
"""
abstract type SequentialIntervalMarkovProcess <: CompositeIntervalMarkovProcess end

"""
    ProductIntervalMarkovProcess

An abstract type for product interval Markov processes including [`ParallelProduct`](@ref).
"""
abstract type ProductIntervalMarkovProcess <: CompositeIntervalMarkovProcess end
