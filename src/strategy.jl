# The difference between these three type hierachies probably needs an explanation!
# 1. (Abstract)Strategy is the hierarchy of return types. For `value_iteration` no strategy is returned,
#    but `control_synthesis` returns a strategy. The strategy is either stationary or time-varying, depending
#    on the specification (finite horizon vs until convergence).
# 2. (Abstract)StrategyCache is the hierarchy of caches, which is used during the value iteration algorithm to
#    store and build the strategies.
# 3. (Abstract)StrategyConfig is the hierarchy of configurations for the strategy cache. When constructing the
#    cache, the configuration is used to determine the type of cache to construct (`construct_strategy_cache`).
#    It is primarily used internally, where `value_iteration` creates a `NoStrategyConfig`, and `control_synthesis`
#    creates a `StationaryStrategyConfig` or `TimeVaryingStrategyConfig` depending on the specification.
#    Then `construct_strategy_cache` creates the appropriate cache type based on the configuration, including
#    the size of and the device to store the cache depending on the model type.
#
# Part of the purpose of the split between these three hierarchies is: (i) keep the cache internal and allow
# dispatching on the returned strategy type and (ii) to allow to specifcying the cache type without 
# specifying the size of and the device to store the cache manually.

############
# Strategy #
############
abstract type AbstractStrategy end

struct NoStrategy <: AbstractStrategy end
checkstrategy!(::NoStrategy, system) = nothing

struct StationaryStrategy{A <: AbstractArray{Int32}} <: AbstractStrategy
    strategy::A
end
Base.getindex(strategy::StationaryStrategy, k) = strategy.strategy

function checkstrategy!(strategy::StationaryStrategy, system)
    checkstrategy!(strategy.strategy, system)
end

function checkstrategy!(strategy::AbstractArray, system)
    num_actions = stateptr(system)[2:end] .- stateptr(system)[1:(end - 1)]
    if !all(1 .<= vec(strategy) .<= num_actions)
        throw(
            DomainError(
                "The strategy includes at least one invalid action (less than 1 or greater than num_actions for the state).",
            ),
        )
    end
end

struct TimeVaryingStrategy{A <: AbstractArray{Int32}} <: AbstractStrategy
    strategy::Vector{A}
end
Base.getindex(strategy::TimeVaryingStrategy, k) = strategy.strategy[k]
time_length(strategy::TimeVaryingStrategy) = length(strategy.strategy)

function checkstrategy!(strategy::TimeVaryingStrategy, system)
    for strategy_step in strategy.strategy
        checkstrategy!(strategy_step, system)
    end
end

###################
# Strategy config #
###################
abstract type AbstractStrategyConfig end

"""
    NoStrategyConfig

A configuration for a strategy cache that does not store policies.
See [`construct_strategy_cache`](@ref) for more details on how to construct the cache from the configuration.
"""
struct NoStrategyConfig <: AbstractStrategyConfig end

"""
    GivenStrategyConfig

A configuration for a strategy cache where a given strategy is applied.
"""
struct GivenStrategyConfig <: AbstractStrategyConfig end

"""
    StationaryStrategyConfig

A configuration for a strategy cache that stores stationary policies.
Note that the strategy is updated at each iteration of the value iteration algorithm,
if a new choice is strictly better than the previous one. See [1, Section 4.3] for more details why this is necessary.
See [`construct_strategy_cache`](@ref) for more details on how to construct the cache from the configuration.

[1] Forejt, Vojtěch, et al. "Automated verification techniques for probabilistic systems." Formal Methods for Eternal Networked Software Systems: 11th International School on Formal Methods for the Design of Computer, Communication and Software Systems, SFM 2011, Bertinoro, Italy, June 13-18, 2011. Advanced Lectures 11 (2011): 53-113.
"""
struct StationaryStrategyConfig <: AbstractStrategyConfig end

"""
    TimeVaryingStrategyConfig

A configuration for a strategy cache that stores time-varying policies.
See [`construct_strategy_cache`](@ref) for more details on how to construct the cache from the configuration.
"""
struct TimeVaryingStrategyConfig <: AbstractStrategyConfig end

##################
# Strategy cache #
##################
abstract type AbstractStrategyCache end
abstract type OptimizingStrategyCache <: AbstractStrategyCache end
abstract type NonOptimizingStrategyCache <: AbstractStrategyCache end

"""
    construct_strategy_cache(mp::Union{IntervalProbabilities, IntervalMarkovProcess}, config::AbstractStrategyConfig)

Construct a strategy cache from a configuration for a given interval Markov process. The resuling cache type
depends on the configuration and the device to store the strategy depends on the device of the Markov process.
"""
function construct_strategy_cache end

construct_strategy_cache(mp::IntervalMarkovProcess, config, strategy = NoStrategy()) =
    construct_strategy_cache(mp, config, strategy, product_num_states(mp))

# Strategy cache for applying given policies - useful for dispatching
struct GivenStrategyCache{S <: AbstractStrategy} <: NonOptimizingStrategyCache
    strategy::S
end

construct_strategy_cache(::IntervalMarkovProcess, ::GivenStrategyConfig, strategy, dims) =
    GivenStrategyCache(strategy)

struct ActiveGivenStrategyCache{A <: AbstractArray{Int32}} <: NonOptimizingStrategyCache
    strategy::A
end
Base.getindex(cache::GivenStrategyCache, k) = ActiveGivenStrategyCache(cache.strategy[k])
Base.getindex(cache::ActiveGivenStrategyCache, j) = cache.strategy[j]

function extract_strategy!(::GivenStrategyCache, values, V, j, maximize)
    throw(ArgumentError("The strategy is given and not supposed to optimize over actions."))
end
function extract_strategy!(::ActiveGivenStrategyCache, values, V, j, maximize)
    throw(ArgumentError("The strategy is given and not supposed to optimize over actions."))
end
postprocess_strategy_cache!(::GivenStrategyCache) = nothing
postprocess_strategy_cache!(::ActiveGivenStrategyCache) = nothing

# Strategy cache for not storing policies - useful for dispatching
struct NoStrategyCache <: OptimizingStrategyCache end

function construct_strategy_cache(
    ::Union{IntervalProbabilities, OrthogonalIntervalProbabilities},
    ::NoStrategyConfig,
)
    return NoStrategyCache()
end

construct_strategy_cache(::IntervalMarkovProcess, ::NoStrategyConfig, strategy, dims) =
    NoStrategyCache()

function extract_strategy!(::NoStrategyCache, values, V, j, maximize)
    return maximize ? maximum(values) : minimum(values)
end
postprocess_strategy_cache!(::NoStrategyCache) = nothing

# Strategy cache for storing time-varying policies
struct TimeVaryingStrategyCache{A <: AbstractArray{Int32}} <: OptimizingStrategyCache
    cur_strategy::A
    strategy::Vector{A}
end

function TimeVaryingStrategyCache(cur_strategy::A) where {A}
    return TimeVaryingStrategyCache(cur_strategy, Vector{A}())
end

function construct_strategy_cache(
    mp::IntervalMarkovProcess,
    ::TimeVaryingStrategyConfig,
    strategy,
    dims,
)
    cur_strategy = construct_action_cache(transition_prob(mp), dims)
    return TimeVaryingStrategyCache(cur_strategy)
end

function construct_action_cache(
    ::IntervalProbabilities{R, VR},
    dims,
) where {R <: Real, VR <: AbstractVector{R}}
    return zeros(Int32, dims)
end

function construct_action_cache(
    ::OrthogonalIntervalProbabilities{N, <:IntervalProbabilities{R, VR}},
    dims,
) where {N, R <: Real, VR <: AbstractVector{R}}
    return zeros(Int32, dims)
end

cachetostrategy(strategy_cache::TimeVaryingStrategyCache) =
    TimeVaryingStrategy([indices for indices in reverse(strategy_cache.strategy)])

function extract_strategy!(
    strategy_cache::TimeVaryingStrategyCache,
    values::AbstractArray{R},
    V,
    j,
    maximize,
) where {R <: Real}
    opt_val = maximize ? typemin(R) : typemax(R)
    opt_index = 1
    neutral = (opt_val, opt_index)

    return _extract_strategy!(strategy_cache.cur_strategy, values, neutral, j, maximize)
end
function postprocess_strategy_cache!(strategy_cache::TimeVaryingStrategyCache)
    push!(strategy_cache.strategy, copy(strategy_cache.cur_strategy))
end

# Strategy cache for storing stationary policies
struct StationaryStrategyCache{A <: AbstractArray{Int32}} <: OptimizingStrategyCache
    strategy::A
end

function construct_strategy_cache(
    mp::IntervalMarkovProcess,
    ::StationaryStrategyConfig,
    strategy,
    dims,
)
    strategy = construct_action_cache(transition_prob(mp), dims)
    return StationaryStrategyCache(strategy)
end

cachetostrategy(strategy_cache::StationaryStrategyCache) =
    StationaryStrategy(strategy_cache.strategy)

function extract_strategy!(
    strategy_cache::StationaryStrategyCache,
    values::AbstractArray{R},
    V,
    j,
    maximize,
) where {R <: Real}
    neutral = if iszero(strategy_cache.strategy[j])
        maximize ? typemin(R) : typemax(R), 1
    else
        V[j], strategy_cache.strategy[j]
    end

    return _extract_strategy!(strategy_cache.strategy, values, neutral, j, maximize)
end
postprocess_strategy_cache!(::StationaryStrategyCache) = nothing

# Shared between stationary and time-varying strategies
function _extract_strategy!(cur_strategy, values, neutral, j, maximize)
    gt = maximize ? (>) : (<)

    opt_val, opt_index = neutral

    for (i, v) in enumerate(values)
        if gt(v, opt_val)
            opt_val = v
            opt_index = i
        end
    end

    @inbounds cur_strategy[j] = opt_index
    return opt_val
end
