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

    as = action_shape(system)
    invalid = any(strategy) do s
        for i in eachindex(s)
            if s[i] < 1
                return true
            end

            if s[i] > as[i]
                return true
            end
        end

        return false
    end

    if invalid
        throw(
            DomainError(
                "The strategy includes at least one invalid action (less than 1 or greater than num_actions for some action variable).",
            ),
        )
    end
end

function showstrategy(io::IO, first_prefix, prefix, strategy::StationaryStrategy)
    println(io, first_prefix, styled"{code:StationaryStrategy}")
    println(io, prefix, styled"└─ Strategy shape: {magenta:$(size(strategy.strategy))}")
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

function showstrategy(io::IO, first_prefix, prefix, strategy::TimeVaryingStrategy)
    println(io, first_prefix, styled"{code:TimeVaryingStrategy}")
    println(io, prefix, styled"├─ Time length: {magenta:$(length(strategy.strategy))}")
    println(io, prefix, styled"└─ Strategy shape: {magenta:$(size(strategy.strategy[1]))}")
end
