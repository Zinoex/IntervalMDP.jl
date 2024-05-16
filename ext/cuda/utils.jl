@inline function kernel_nextwarp(threads)
    assume(warpsize() == 32)

    ws = warpsize()
    return threads + (ws - threads % ws) % ws
end
