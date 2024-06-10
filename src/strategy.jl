abstract type AbstractStrategyConfig end

struct NoStrategyConfig <: AbstractStrategyConfig end
struct StationaryStrategyConfig <: AbstractStrategyConfig end
struct TimeVaryingStrategyConfig <: AbstractStrategyConfig end

# Abstract type
abstract type AbstractStrategyCache end

# Strategy cache for not storing policies - useful for dispatching
struct NoStrategyCache <: AbstractStrategyCache end

function construct_strategy_cache(mp, ::NoStrategyConfig)
    return NoStrategyCache()
end

function extract_strategy!(
    ::NoStrategyCache,
    values,
    V,
    j,
    s₁,
    maximize,
)
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

function construct_strategy_cache(mp::M, ::TimeVaryingStrategyConfig) where {R, P <: IntervalProbabilities{R, <:AbstractVector{R}}, M <: SimpleIntervalMarkovProcess{P}}
    cur_strategy = zeros(Int32, num_states(mp))
    return TimeVaryingStrategyCache(cur_strategy)
end

function cachetostrategy(
    strategy_cache::TimeVaryingStrategyCache,
)
    return [Vector(indices) for indices in reverse(strategy_cache.strategy)]
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

function construct_strategy_cache(mp::M, ::StationaryStrategyConfig) where {R, P <: IntervalProbabilities{R, <:AbstractVector{R}}, M <: IntervalMarkovDecisionProcess{P}}
    cur_strategy = zeros(Int32, num_states(mp))
    return StationaryStrategyCache(cur_strategy)
end

function cachetostrategy(
    strategy_cache::StationaryStrategyCache,
)
    return Vector(strategy_cache.strategy)
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
function _extract_strategy!(
    cur_strategy,
    values,
    neutral,
    j,
    s₁,
    maximize,
)
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