abstract type ActiveCache end
abstract type OptimizingActiveCache <: ActiveCache end

struct NoStrategyActiveCache <: OptimizingActiveCache end
Adapt.@adapt_structure NoStrategyActiveCache
@inline function active_cache(::IntervalMDP.NoStrategyCache)
    return NoStrategyActiveCache()
end

struct TimeVaryingStrategyActiveCache{N, V <: AbstractVector{NTuple{N, Int32}}} <:
       OptimizingActiveCache
    cur_strategy::V
end
Adapt.@adapt_structure TimeVaryingStrategyActiveCache
@inline function active_cache(strategy_cache::IntervalMDP.TimeVaryingStrategyCache)
    return TimeVaryingStrategyActiveCache(strategy_cache.cur_strategy)
end

struct StationaryStrategyActiveCache{N, V <: AbstractVector{NTuple{N, Int32}}} <:
       OptimizingActiveCache
    strategy::V
end
Adapt.@adapt_structure StationaryStrategyActiveCache
@inline function active_cache(strategy_cache::IntervalMDP.StationaryStrategyCache)
    return StationaryStrategyActiveCache(strategy_cache.strategy)
end

abstract type NonOptimizingActiveCache <: ActiveCache end

struct GivenStrategyActiveCache{N, V <: AbstractVector{NTuple{N, Int32}}} <:
       NonOptimizingActiveCache
    strategy::V
end
Adapt.@adapt_structure GivenStrategyActiveCache
@inline function active_cache(strategy_cache::IntervalMDP.ActiveGivenStrategyCache)
    return GivenStrategyActiveCache(strategy_cache.strategy)
end
Base.@propagate_inbounds Base.getindex(cache::GivenStrategyActiveCache, j) =
    cache.strategy[j]

Base.@propagate_inbounds function extract_strategy_warp!(
    ::NoStrategyActiveCache,
    values::AbstractVector{Tv},
    jₛ,
    action_reduce,
) where {Tv}
    assume(warpsize() == 32)
    action_min, action_neutral = action_reduce[1], action_reduce[3]

    warp_aligned_length = kernel_nextwarp(length(values))
    opt_val = action_neutral

    s = laneid()
    while s <= warp_aligned_length
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

Base.@propagate_inbounds function extract_strategy_warp!(
    cache::TimeVaryingStrategyActiveCache{1, <:AbstractVector{Tuple{Int32}}},
    values::AbstractVector{Tv},
    jₛ,
    action_reduce,
) where {Tv}
    assume(warpsize() == 32)
    _, action_lt, action_neutral = action_reduce

    warp_aligned_length = kernel_nextwarp(length(values))
    opt_val, opt_idx = action_neutral, one(Int32)

    s = laneid()
    while s <= warp_aligned_length
        new_val, new_idx = if s <= length(values)
            values[s], s
        else
            action_neutral, one(Int32)
        end
        opt_val, opt_idx = argop(action_lt, opt_val, opt_idx, new_val, new_idx)

        s += warpsize()
    end

    opt_val, opt_idx = argmin_warp(action_lt, opt_val, opt_idx)

    if laneid() == one(Int32)
        @inbounds cache.cur_strategy[jₛ] = (opt_idx,)
    end

    return opt_val
end

Base.@propagate_inbounds function extract_strategy_warp!(
    cache::StationaryStrategyActiveCache{1, <:AbstractVector{Tuple{Int32}}},
    values::AbstractVector{Tv},
    jₛ,
    action_reduce,
) where {Tv}
    assume(warpsize() == 32)
    _, action_lt, action_neutral = action_reduce

    warp_aligned_length = kernel_nextwarp(length(values))
    opt_val, opt_idx = if iszero(cache.strategy[jₛ][one(Int32)])
        action_neutral, one(Int32)
    else
        s = cache.strategy[jₛ][1]
        values[s], s
    end

    s = laneid()
    while s <= warp_aligned_length
        new_val, new_idx = if s <= length(values)
            values[s], s
        else
            action_neutral, one(Int32)
        end
        opt_val, opt_idx = argop(action_lt, opt_val, opt_idx, new_val, new_idx)

        s += warpsize()
    end

    opt_val, opt_idx = argmin_warp(action_lt, opt_val, opt_idx)

    if laneid() == 1
        @inbounds cache.strategy[jₛ] = (opt_idx,)
    end

    return opt_val
end

@inline function extract_strategy_block!(
    ::NoStrategyActiveCache,
    values::AbstractArray{Tv},
    jₛ,
    action_reduce
) where {Tv}
    action_min, action_neutral = action_reduce[1], action_reduce[3]

    loop_length = nextmult(blockDim().x, length(values))
    opt_val = action_neutral

    s = threadIdx().x
    @inbounds while s <= loop_length
        new_val = if s <= length(values)
            values[s]
        else
            action_neutral
        end
        opt_val = action_min(new_val, opt_val)

        s += blockDim().x
    end

    opt_val = CUDA.reduce_block(action_min, opt_val, action_neutral, Val(true))
    return opt_val
end

@inline function extract_strategy_block!(
    cache::TimeVaryingStrategyActiveCache{1, <:AbstractArray},
    values::AbstractArray{Tv},
    jₛ,
    action_reduce
) where {Tv}
    action_lt, action_neutral = action_reduce[2], action_reduce[3]

    loop_length = nextmult(blockDim().x, length(values))
    opt_val, opt_idx = action_neutral, one(Int32)

    s = threadIdx().x
    @inbounds while s <= loop_length
        new_val, new_idx = if s <= length(values)
            values[s], s
        else
            action_neutral, one(Int32)
        end
        opt_val, opt_idx = argop(action_lt, opt_val, opt_idx, new_val, new_idx)

        s += blockDim().x
    end

    opt_val, opt_idx = argmin_block(action_lt, opt_val, opt_idx, action_neutral, one(Int32), Val(true))

    if lane == 1
        action_shape = CartesianIndices(size(values))
        @inbounds cache.cur_strategy[jₛ] = Tuple(action_shape[opt_idx])
    end

    return opt_val
end

@inline function extract_strategy_block!(
    cache::StationaryStrategyActiveCache{1, <:AbstractArray},
    values::AbstractArray{Tv},
    jₛ,
    action_reduce
) where {Tv}
    action_lt, action_neutral = action_reduce[2], action_reduce[3]

    loop_length = nextmult(blockDim().x, length(values))
    opt_val, opt_idx = if iszero(cache.strategy[jₛ][1])
        action_neutral, one(Int32)
    else
        s = cache.strategy[jₛ]
        values[s...], s
    end

    s = threadIdx().x
    @inbounds while s <= loop_length
        new_val, new_idx = if s <= length(values)
            values[s], s
        else
            action_neutral, one(Int32)
        end
        opt_val, opt_idx = argop(action_lt, opt_val, opt_idx, new_val, new_idx)

        s += blockDim().x
    end

    opt_val, opt_idx = argmin_block(action_lt, opt_val, opt_idx, action_neutral, one(Int32), Val(true))

    if lane == 1
        action_shape = CartesianIndices(size(values))
        @inbounds cache.cur_strategy[jₛ] = Tuple(action_shape[opt_idx])
    end

    return opt_val
end
