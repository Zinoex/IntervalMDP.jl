abstract type AbstractStrategyConfig end

"""
    NoStrategyConfig

A configuration for a strategy cache that does not store policies.
See [`construct_strategy_cache`](@ref) for more details on how to construct the cache from the configuration.
"""
struct NoStrategyConfig <: AbstractStrategyConfig end

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

# Abstract type
abstract type AbstractStrategyCache end

"""
    construct_strategy_cache(mp::Union{IntervalProbabilities, IntervalMarkovProcess}, config::AbstractStrategyConfig)

Construct a strategy cache from a configuration for a given interval Markov process. The resuling cache type
depends on the configuration and the device to store the strategy depends on the device of the Markov process.
"""
function construct_strategy_cache end

# Strategy cache for not storing policies - useful for dispatching
struct NoStrategyCache <: AbstractStrategyCache end

function construct_strategy_cache(::IntervalProbabilities, ::NoStrategyConfig)
    return NoStrategyCache()
end

function construct_strategy_cache(::SimpleIntervalMarkovProcess, ::NoStrategyConfig)
    return NoStrategyCache()
end

function construct_strategy_cache(::SimpleIntervalMarkovProcess, ::NoStrategyConfig, dims)
    return NoStrategyCache()
end

function extract_strategy!(::NoStrategyCache, values, V, j, s₁, maximize)
    return maximize ? maximum(values) : minimum(values)
end
postprocess_strategy_cache!(::NoStrategyCache) = nothing

# Strategy cache for storing time-varying policies
struct TimeVaryingStrategyCache{A <: AbstractArray{Int32}} <: AbstractStrategyCache
    cur_strategy::A
    strategy::Vector{A}
end

function TimeVaryingStrategyCache(cur_strategy::A) where {A}
    return TimeVaryingStrategyCache(cur_strategy, Vector{A}())
end

function construct_strategy_cache(
    mp::M,
    ::TimeVaryingStrategyConfig,
) where {M <: SimpleIntervalMarkovProcess}
    cur_strategy = construct_action_cache(transition_prob(mp, 1), num_states(mp))
    return TimeVaryingStrategyCache(cur_strategy)
end

function construct_strategy_cache(
    mp::M,
    ::TimeVaryingStrategyConfig,
    dims,
) where {M <: SimpleIntervalMarkovProcess}
    cur_strategy = construct_action_cache(transition_prob(mp, 1), dims)
    return TimeVaryingStrategyCache(cur_strategy)
end

function construct_action_cache(
    ::IntervalProbabilities{R, VR},
    dims,
) where {R <: Real, VR <: AbstractVector{R}}
    return zeros(Int32, dims)
end

function cachetostrategy(strategy_cache::TimeVaryingStrategyCache)
    return [Array(indices) for indices in reverse(strategy_cache.strategy)]
end

function extract_strategy!(
    strategy_cache::TimeVaryingStrategyCache,
    values::AbstractArray{R},
    V,
    j,
    s₁,
    maximize,
) where {R <: Real}
    opt_val = maximize ? typemin(R) : typemax(R)
    opt_index = s₁
    neutral = (opt_val, opt_index)

    return _extract_strategy!(strategy_cache.cur_strategy, values, neutral, j, s₁, maximize)
end
function postprocess_strategy_cache!(strategy_cache::TimeVaryingStrategyCache)
    push!(strategy_cache.strategy, copy(strategy_cache.cur_strategy))
end

# Strategy cache for storing stationary policies
struct StationaryStrategyCache{A <: AbstractArray{Int32}} <: AbstractStrategyCache
    strategy::A
end

function construct_strategy_cache(
    mp::M,
    ::StationaryStrategyConfig,
) where {M <: SimpleIntervalMarkovProcess}
    strategy = construct_action_cache(transition_prob(mp), num_states(mp))
    return StationaryStrategyCache(strategy)
end

function construct_strategy_cache(
    mp::M,
    ::StationaryStrategyConfig,
    dims,
) where {M <: SimpleIntervalMarkovProcess}
    strategy = construct_action_cache(transition_prob(mp), dims)
    return StationaryStrategyCache(strategy)
end

function cachetostrategy(strategy_cache::StationaryStrategyCache)
    return Array(strategy_cache.strategy)
end

function extract_strategy!(
    strategy_cache::StationaryStrategyCache,
    values::AbstractArray{R},
    V,
    j,
    s₁,
    maximize,
) where {R <: Real}
    neutral = if iszero(strategy_cache.strategy[j])
        maximize ? typemin(R) : typemax(R), s₁
    else
        V[j], strategy_cache.strategy[j]
    end

    return _extract_strategy!(strategy_cache.strategy, values, neutral, j, s₁, maximize)
end
postprocess_strategy_cache!(::StationaryStrategyCache) = nothing

# Shared between stationary and time-varying strategies
function _extract_strategy!(cur_strategy, values, neutral, j, s₁, maximize)
    gt = maximize ? (>) : (<)

    opt_val, opt_index = neutral

    for (i, v) in enumerate(values)
        if gt(v, opt_val)
            opt_val = v
            opt_index = s₁ + i - 1
        end
    end

    @inbounds cur_strategy[j] = opt_index
    return opt_val
end

# Composite types
abstract type ProductStrategyCache <: AbstractStrategyCache end

struct ParallelProductStrategyCache <: ProductStrategyCache
    orthogonal_caches::Vector{AbstractStrategyCache}
end
orthogonal_caches(cache::ParallelProductStrategyCache) = cache.orthogonal_caches

function construct_strategy_cache(mp::ParallelProduct, config::AbstractStrategyConfig)
    dims = Tuple(product_num_states(mp) |> recursiveflatten |> collect)

    return ParallelProductStrategyCache([
        construct_strategy_cache(orthogonal_process, config, dims) for
        orthogonal_process in orthogonal_processes(mp)
    ])
end

function construct_strategy_cache(mp::ParallelProduct, config::AbstractStrategyConfig, dims)
    return ParallelProductStrategyCache([
        construct_strategy_cache(orthogonal_process, config, dims) for
        orthogonal_process in orthogonal_processes(mp)
    ])
end

function postprocess_strategy_cache!(strategy_cache::ParallelProductStrategyCache)
    for orthogonal_cache in orthogonal_caches(strategy_cache)
        postprocess_strategy_cache!(orthogonal_cache)
    end
end

function cachetostrategy(strategy_cache::ParallelProductStrategyCache)
    return [
        cachetostrategy(orthogonal_cache) for
        orthogonal_cache in orthogonal_caches(strategy_cache)
    ]
end
