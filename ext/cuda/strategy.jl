function IntervalMDP.construct_action_cache(
    ::IntervalProbabilities{R, VR},
    dims,
) where {R <: Real, VR <: AbstractGPUVector{R}}
    return CUDA.zeros(Int32, dims)
end

abstract type ActiveCache end

struct NoStrategyActiveCache <: ActiveCache end
Adapt.@adapt_structure NoStrategyActiveCache
@inline function active_cache(::IntervalMDP.NoStrategyCache)
    return NoStrategyActiveCache()
end

struct TimeVaryingStrategyActiveCache{V <: AbstractVector{Int32}} <: ActiveCache
    cur_strategy::V
end
Adapt.@adapt_structure TimeVaryingStrategyActiveCache
@inline function active_cache(strategy_cache::IntervalMDP.TimeVaryingStrategyCache)
    return TimeVaryingStrategyActiveCache(strategy_cache.cur_strategy)
end

struct StationaryStrategyActiveCache{V <: AbstractVector{Int32}} <: ActiveCache
    strategy::V
end
Adapt.@adapt_structure StationaryStrategyActiveCache
@inline function active_cache(strategy_cache::IntervalMDP.StationaryStrategyCache)
    return StationaryStrategyActiveCache(strategy_cache.strategy)
end

@inline function extract_strategy_warp!(
    ::NoStrategyActiveCache,
    values::AbstractVector{Tv},
    V,
    j,
    s₁,
    action_reduce,
    lane,
) where {Tv}
    assume(warpsize() == 32)
    action_min, action_neutral = action_reduce[1], action_reduce[3]

    warp_aligned_length = kernel_nextwarp(length(values))
    @inbounds opt_val = action_neutral

    s = lane
    @inbounds while s <= warp_aligned_length
        new_val = if s <= length(values)
            values[s]
        else
            action_neutral
        end
        opt_val = action_min(new_val, opt_val)

        s += warpsize()
    end

    opt_val = CUDA.reduce_warp(action_min, opt_val)
    return opt_val
end

@inline function extract_strategy_warp!(
    cache::TimeVaryingStrategyActiveCache,
    values::AbstractVector{Tv},
    V,
    j,
    s₁,
    action_reduce,
    lane,
) where {Tv}
    assume(warpsize() == 32)
    action_lt, action_neutral = action_reduce[2], action_reduce[3]

    warp_aligned_length = kernel_nextwarp(length(values))
    opt_val, opt_idx = action_neutral, s₁

    s = lane
    @inbounds while s <= warp_aligned_length
        new_val, new_idx = if s <= length(values)
            values[s], s₁ + s - 1
        else
            action_neutral, s₁
        end
        opt_val, opt_idx = argop(action_lt, opt_val, opt_idx, new_val, new_idx)

        s += warpsize()
    end

    opt_val, opt_idx = argmin_warp(action_lt, opt_val, opt_idx)

    if lane == 1
        @inbounds cache.cur_strategy[j] = opt_idx
    end

    return opt_val
end

@inline function extract_strategy_warp!(
    cache::StationaryStrategyActiveCache,
    values::AbstractVector{Tv},
    V,
    j,
    s₁,
    action_reduce,
    lane,
) where {Tv}
    assume(warpsize() == 32)
    action_lt, action_neutral = action_reduce[2], action_reduce[3]

    warp_aligned_length = kernel_nextwarp(length(values))
    opt_val, opt_idx = if iszero(cache.strategy[j])
        action_neutral, s₁
    else
        V[j], cache.strategy[j]
    end

    s = lane
    @inbounds while s <= warp_aligned_length
        new_val, new_idx = if s <= length(values)
            values[s], s₁ + s - 1
        else
            action_neutral, s₁
        end
        opt_val, opt_idx = argop(action_lt, opt_val, opt_idx, new_val, new_idx)

        s += warpsize()
    end

    opt_val, opt_idx = argmin_warp(action_lt, opt_val, opt_idx)

    if lane == 1
        @inbounds cache.strategy[j] = opt_idx
    end

    return opt_val
end
