
# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op, val::T, neutral, shuffle::Val{true}) where T
    # shared mem for partial sums
    assume(warpsize() == 32)
    assume(threadIdx().x >= 1)
    shared = CuStaticSharedArray(T, 32)

    wid = fld1(threadIdx().x, warpsize())
    lane = mod1(threadIdx().x, warpsize())

    # each warp performs partial reduction
    val = CUDA.reduce_warp(op, val)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    sync_threads()

    # read from shared memory only if that warp existed
    val = if threadIdx().x <= fld1(blockDim().x, warpsize())
         @inbounds shared[lane]
    else
        neutral
    end

    # final reduce within first warp
    if wid == 1
        val = CUDA.reduce_warp(op, val)
    end

    return val
end