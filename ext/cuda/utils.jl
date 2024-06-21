@inline function kernel_nextwarp(threads)
    assume(warpsize() == 32)

    ws = warpsize()
    return threads + (ws - threads % ws) % ws
end

@inline function cumsum_warp(val, lane)
    assume(warpsize() == 32)
    offset = 0x00000001
    while offset < warpsize()
        up_val = shfl_up_sync(0xffffffff, val, offset)
        if lane > offset
            val += up_val
        end
        offset <<= 1
    end

    return val
end

@inline function cumsum_block(val, workspace, wid, lane)
    # Warp-reduction
    val = cumsum_warp(val, lane)

    # Block-reduction
    if lane == 32
        workspace[wid] = val
    end
    sync_threads()

    if wid == 1
        wid_val = workspace[lane]
        wid_val = cumsum_warp(wid_val, lane)
        workspace[lane] = wid_val
    end
    sync_threads()

    # Warp-correction
    if wid > 1
        val += workspace[wid - 1]
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

        offset >>= 1
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

@inline function selectotherdims(A::AbstractArray, dim, idxs)
    head, tail = idxs[1:(dim - 1)], idxs[dim:end]

    return view(A, head..., Colon(), tail...)
end
