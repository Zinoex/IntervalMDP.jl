
function IntervalMDP.construct_value_function(
    ::MR,
    num_states,
) where {R, MR <: CuSparseMatrixCSC{R}}
    V = CUDA.zeros(R, num_states)
    return V
end

function IntervalMDP.extract_policy!(
    value_function::IntervalMDP.IMDPValueFunction,
    policy_cache::IntervalMDP.NoPolicyCache,
    stateptr::VT,
    maximize,
) where {VT <: CuVector}
    R = eltype(value_function.cur)
    V_per_state = CuVectorOfVector(stateptr, value_function.action_values, maximum(diff(stateptr)))

    kernel = @cuda launch=false reduce_vov_kernel!(
        maximize ? max : min,
        maximize ? typemin(R) : typemax(R),
        value_function.cur,
        V_per_state,
    )

    config = launch_configuration(kernel.fun)

    threads = prevwarp(device(), config.threads)

    states_per_block = threads ÷ 32
    blocks = min(65535, ceil(Int64, length(V_per_state) / states_per_block))

    kernel(
        maximize ? max : min,
        maximize ? typemin(R) : typemax(R),
        value_function.cur,
        V_per_state;
        threads=threads,
        blocks=blocks
    )

    return value_function, policy_cache
end

function reduce_vov_kernel!(
    op,
    neutral,
    res::CuDeviceVector{Tv, A},
    vov::CuDeviceVectorOfVector{Tv, Ti, A},
) where {Tv, Ti, A}
    assume(warpsize() == 32)

    thread_id = (blockIdx().x - one(Ti)) * blockDim().x + threadIdx().x
    wid, lane = fldmod1(thread_id, warpsize())

    while wid <= length(vov)
        # Tree reduce
        @inbounds subset = vov[wid]
        bound = kernel_nextwarp(length(subset))

        val = neutral

        # Reduce within each warp
        i = lane
        while i <= bound
            val = op(val, if i <= length(subset)
                @inbounds subset[i]
            else
                neutral
            end)

            val = CUDA.reduce_warp(op, val)

            i += blockDim().x
        end

        if lane == 1
            @inbounds res[wid] = val
        end

        thread_id += gridDim().x * blockDim().x
        wid, lane = fldmod1(thread_id, warpsize())
    end
end

function IntervalMDP.extract_policy!(
    value_function::IntervalMDP.IMDPValueFunction,
    policy_cache::IntervalMDP.TimeVaryingPolicyCache,
    stateptr::VT,
    maximize,
) where {T, VT <: CuVector{T}}
    R = eltype(value_function.cur)
    V_per_state = CuVectorOfVector(stateptr, value_function.action_values, maximum(diff(stateptr)))

    # Transfer to GPU if not already
    policy_cache = IntervalMDP.cu(policy_cache)
    
    argop = time_varying_argop(maximize)

    kernel = @cuda launch=false argreduce_vov_kernel!(
        argop,
        maximize ? typemin(R) : typemax(R),
        zero(T),
        value_function.cur,
        policy_cache.cur_policy,
        V_per_state,
    )

    config = launch_configuration(kernel.fun)

    threads = prevwarp(device(), config.threads)

    states_per_block = threads ÷ 32
    blocks = min(65535, ceil(Int64, length(V_per_state) / states_per_block))

    kernel(
        argop,
        maximize ? typemin(R) : typemax(R),
        zero(T),
        value_function.cur,
        policy_cache.cur_policy,
        V_per_state;
        threads=threads,
        blocks=blocks
    )

    push!(policy_cache.policy, copy(policy_cache.cur_policy))

    return value_function, policy_cache
end

function time_varying_argop(maximize)
    gt = maximize ? (>) : (<)

    @inline function argop(val::Tv, idx::Ti, other_val::Tv, other_idx::Ti) where {Ti, Tv}
        if !iszero(other_idx) && gt(other_val, val)
            return other_val, other_idx
        else
            return val, idx
        end
    end

    return argop
end

function IntervalMDP.extract_policy!(
    value_function::IntervalMDP.IMDPValueFunction,
    policy_cache::IntervalMDP.StationaryPolicyCache,
    stateptr::VT,
    maximize,
) where {T, VT <: CuVector{T}}
    V_per_state = CuVectorOfVector(stateptr, value_function.action_values, maximum(diff(stateptr)))

    # Transfer to GPU if not already
    policy_cache = IntervalMDP.cu(policy_cache)
    
    argop = time_varying_argop(maximize)

    kernel = @cuda launch=false argreduce_vov_kernel!(
        argop,
        value_function.prev,
        policy_cache.cur_policy,
        value_function.cur,
        policy_cache.cur_policy,
        V_per_state,
    )

    config = launch_configuration(kernel.fun)

    threads = prevwarp(device(), config.threads)

    states_per_block = threads ÷ 32
    blocks = min(65535, ceil(Int64, length(V_per_state) / states_per_block))

    kernel(
        argop,
        value_function.prev,
        policy_cache.cur_policy,
        value_function.cur,
        V_per_state;
        threads=threads,
        blocks=blocks
    )

    push!(policy_cache.policy, copy(policy_cache.cur_policy))

    return value_function, policy_cache
end

function argreduce_vov_kernel!(
    argop,
    neutral_val,
    neural_idx,
    res_val::CuDeviceVector{Tv, A},
    res_idx::CuDeviceVector{Ti, A},
    vov::CuDeviceVectorOfVector{Tv, Ti, A},
) where {Tv, Ti, A}
    assume(warpsize() == 32)

    thread_id = (blockIdx().x - one(Ti)) * blockDim().x + threadIdx().x
    wid, lane = fldmod1(thread_id, warpsize())

    while wid <= length(vov)
        # Tree reduce
        @inbounds subset = vov[wid]
        bound = kernel_nextwarp(length(subset))

        neutral = get_neutral(neutral_val, neural_idx, wid)
        val, idx = neutral

        # Reduce within each warp
        i = lane
        while i <= bound
            new_val, new_idx = if i <= length(subset)
                @inbounds subset[i], Ti(i)
            else
                neutral
            end
            val, idx = argop(val, idx, new_val, new_idx)

            val, idx = argreduce_warp(argop, val, idx)

            i += blockDim().x
        end

        if lane == 1
            @inbounds res_val[wid] = val
            @inbounds res_idx[wid] = idx + subset.offset - 1
        end

        thread_id += gridDim().x * blockDim().x
        wid, lane = fldmod1(thread_id, warpsize())
    end
end

@inline get_neutral(neutral_val::Tv, neural_idx::Ti, wid) where {Tv <: Number, Ti <: Integer} = neutral_val, neural_idx
@inline get_neutral(neutral_val::VTv, neural_idx::VTi, wid) where {VTv <: AbstractArray, VTi <: AbstractArray} = neutral_val[wid], neural_idx[wid]

@inline function argreduce_warp(argop, val, idx)
    assume(warpsize() == 32)
    offset = 0x00000001
    while offset < warpsize()
        new_val, new_idx = shfl_down_sync(0xffffffff, val, offset), shfl_down_sync(0xffffffff, idx, offset)
        val, idx = argop(val, idx, new_val, new_idx)
        offset <<= 1
    end

    return val, idx
end