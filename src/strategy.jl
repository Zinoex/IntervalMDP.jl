abstract type AbstractStrategy end

struct NoStrategy <: AbstractStrategy end
checkstrategy(::NoStrategy, system) = nothing

"""
    StationaryStrategy

A stationary strategy is a strategy that is the same for all time steps.
"""
struct StationaryStrategy{N, A <: AbstractArray{NTuple{N, Int32}}} <: AbstractStrategy
    strategy::A
end
Base.getindex(strategy::StationaryStrategy, k) = strategy.strategy
time_length(::StationaryStrategy) = typemax(Int64)

function checkstrategy(strategy::StationaryStrategy, system)
    checkstrategy(strategy.strategy, system)
end

function checkstrategy(strategy::AbstractArray, system::ProductProcess)
    mp = markov_process(system)
    dfa = automaton(system)

    for state in dfa
        checkstrategy(selectdim(strategy, ndims(strategy), state), mp)
    end
end

function checkstrategy(strategy::AbstractArray, system::FactoredRMDP)
    if size(strategy) != source_shape(system)
        throw(
            DimensionMismatch(
                "The strategy shape $(size(strategy)) does not match the source shape of the system $(source_shape(system)).",
            ),
        )
    end

    for jₛ in CartesianIndices(source_shape(system))
        if !all(1 .<= strategy[jₛ] .<= action_shape(system))
            throw(
                DomainError(
                    "The strategy includes at least one invalid action (less than 1 or greater than num_actions for the state).",
                ),
            )
        end
    end
end

"""
    TimeVaryingStrategy

A time-varying strategy is a strategy that _may_ vary over time. Since we need to store the strategy for each time step, 
the strategy is finite, and thus only applies to finite time specifications, of the same length as the strategy.
"""
struct TimeVaryingStrategy{N, A <: AbstractArray{NTuple{N, Int32}}} <: AbstractStrategy
    strategy::Vector{A}
end
Base.getindex(strategy::TimeVaryingStrategy, k) = strategy.strategy[k]
time_length(strategy::TimeVaryingStrategy) = length(strategy.strategy)

function checkstrategy(strategy::TimeVaryingStrategy, system)
    for strategy_step in strategy.strategy
        checkstrategy(strategy_step, system)
    end
end
