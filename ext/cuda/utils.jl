@inline function kernel_nextwarp(threads)
    assume(warpsize() == 32)

    ws = warpsize()
    return threads + (ws - threads % ws) % ws
end

@inline function nextmult(mult, value)
    return value + (mult - value % mult) % mult
end

@inline function cumsum_warp(val)
    assume(warpsize() == 32)
    offset = 0x00000001
    while offset < warpsize()
        up_val = shfl_up_sync(0xffffffff, val, offset)
        if laneid() > offset
            val += up_val
        end
        offset <<= one(Int32)
    end

    return val
end

Base.@propagate_inbounds function cumsum_block(val, workspace, wid)
    # Warp-reduction
    val = cumsum_warp(val)

    # Block-reduction
    if laneid() == Int32(32)
        workspace[wid] = val
    end
    sync_threads()

    if wid == one(Int32)
        wid_val = workspace[laneid()]
        wid_val = cumsum_warp(wid_val)
        workspace[laneid()] = wid_val
    end
    sync_threads()

    # Warp-correction
    if wid > one(Int32)
        val += workspace[wid - one(Int32)]
    end

    return val
end

@inline function argmin_warp(lt, val, idx)
    assume(warpsize() == 32)
    offset = Int32(16)
    while offset > 0
        up_val = shfl_down_sync(0xffffffff, val, offset)
        up_idx = shfl_down_sync(0xffffffff, idx, offset)
        val, idx = argop(lt, val, idx, up_val, up_idx)

        offset >>= one(Int32)
    end

    return val, idx
end

@inline function argmin_block(lt, val::T, idx, neutral_val, neutral_idx, shuffle::Val{true}) where {T}
    # shared mem for partial sums
    assume(warpsize() == 32)
    shared_val = CuStaticSharedArray(T, Int32(32))
    shared_idx = CuStaticSharedArray(Int32, Int32(32))

    wid, lane = fldmod1(threadIdx().x, warpsize())

    # each warp performs partial reduction
    val, idx = argmin_warp(lt, val, idx)

    # write reduced value to shared memory
    if lane == one(Int32)
        @inbounds shared_val[wid] = val
        @inbounds shared_idx[wid] = idx
    end

    # wait for all partial reductions
    sync_threads()

    # read from shared memory only if that warp existed
    val, idx = if threadIdx().x <= fld1(blockDim().x, warpsize())
         @inbounds shared_val[lane], @inbounds shared_idx[lane]
    else
        neutral_val, neutral_idx
    end

    # final reduce within first warp
    if wid == one(Int32)
        val, idx = argmin_warp(lt, val, idx)
    end

    return val, idx
end

@inline function argop(lt, val, idx, other_val, other_idx)
    if iszero(idx) || (!iszero(other_idx) && lt(other_val, val))
        return other_val, other_idx
    else
        return val, idx
    end
end

Base.@propagate_inbounds function swapelem(A::AbstractArray, i, j)
    @inbounds A[i], A[j] = A[j], A[i]
end

Base.@propagate_inbounds function swapelem(A::Nothing, i, j)
    # Do nothing
end
