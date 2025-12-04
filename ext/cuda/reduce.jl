@inline function reduce_warp(op, val)
    assume(warpsize() == 32)
    offset = one(Int32)
    while offset < warpsize()
        val_up = shfl_down_sync(0xffffffff, val, offset)
        val = op(val, val_up)
        offset <<= one(Int32)
    end

    return val
end

@inline function reduce_block(op, val::T, neutral, shuffle::Val{true}) where {T}
    # shared mem for partial sums
    shared = CuStaticSharedArray(T, 32)
    return reduce_block(shared, op, val, neutral, shuffle)
end

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(shared, op, val::T, neutral, shuffle::Val{true}) where {T}
    assume(warpsize() == 32)
    assume(threadIdx().x >= 1)

    wid = fld1(threadIdx().x, warpsize())

    # each warp performs partial reduction
    val = reduce_warp(op, val)

    # write reduced value to shared memory
    if laneid() == one(Int32)
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    sync_threads()

    # read from shared memory only if that warp existed
    val = if threadIdx().x <= fld1(blockDim().x, warpsize())
        @inbounds shared[laneid()]
    else
        neutral
    end

    # final reduce within first warp
    if wid == one(Int32)
        val = reduce_warp(op, val)
    end

    return val
end
