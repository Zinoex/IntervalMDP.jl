# function IntervalMDP.construct_strategy_cache(mp::M, ::TimeVaryingStrategyConfig) where {M <: SimpleIntervalMarkovProcess}
#     cur_strategy = CUDA.zeros(Int32, num_states(mp))
#     return IntervalMDP.TimeVaryingStrategyCache(stateptr(mp), cur_strategy)
# end

# function IntervalMDP.construct_strategy_cache(mp::M, ::StationaryStrategyConfig) where {M <: SimpleIntervalMarkovProcess}
#     cur_strategy = CUDA.zeros(Int32, num_states(mp))
#     return IntervalMDP.StationaryStrategyCache(stateptr(mp), cur_strategy)
# end

@inline function extract_strategy_warp!(
    ::IntervalMDP.NoStrategyCache,
    values::AbstractVector{Tv},
    V,
    j,
    s₁,
    action_min,
    lane
) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(values))
    @inbounds opt_val = values[1]

    # Add the lower bound multiplied by the value
    s = lane
    @inbounds while s <= warp_aligned_length
        if s <= length(values)
            opt_val = action_min(opt_val, values[s])
        end

        s += warpsize()
    end

    opt_val = CUDA.reduce_warp(action_min, opt_val)
    return opt_val
end