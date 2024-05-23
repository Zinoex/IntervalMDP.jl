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
