
Base.@propagate_inbounds function block_bitonic_sort!(value, aux, lt)
    #### Sort the shared memory with bitonic sort
    nextpow2_length = Base._nextpow2(length(value))
    log_nextpow2_length = unsafe_trunc(Int32, trailing_zeros(nextpow2_length))

    logk = one(Int32)
    while logk <= log_nextpow2_length
        block_bitonic_sort_major_step!(value, aux, lt, logk)

        logk += one(Int32)
    end
end

Base.@propagate_inbounds function block_bitonic_sort_major_step!(value, aux, lt, logk)
    logj = logk - one(Int32)
    block_bitonic_sort_minor_step_merge!(value, aux, lt, logj)

    logj -= one(Int32)
    while logj >= zero(Int32)
        block_bitonic_sort_minor_step_cas!(value, aux, lt, logj)
        logj -= one(Int32)
    end
end

Base.@propagate_inbounds function block_bitonic_sort_minor_step_merge!(value, aux, lt, logj)
    j = one(Int32) << logj

    block_size = j * Int32(2)
    mask = block_size - one(Int32)
    
    lane = threadIdx().x - one(Int32)
    i = (lane & (j - one(Int32))) + ((lane >> logj) << (logj + one(Int32)))
    while i < length(value)
        l = i ⊻ mask
        if l < length(value) && !lt(value[i + one(Int32)], value[l + one(Int32)])
            swapelem(value, i + one(Int32), l + one(Int32))
            swapelem(aux, i + one(Int32), l + one(Int32))
        end

        lane += blockDim().x
        i = (lane & (j - one(Int32))) + ((lane >> logj) << (logj + one(Int32)))
    end

    sync_threads()
end

Base.@propagate_inbounds function block_bitonic_sort_minor_step_cas!(value, aux, lt, logj::Int32)
    j = one(Int32) << logj

    lane = threadIdx().x - one(Int32)
    i = (lane & (j - one(Int32))) + ((lane >> logj) << (logj + one(Int32))) + one(Int32)
    while i <= length(value)
        l = i + j
        if l <= length(value) && !lt(value[i], value[l])
            swapelem(value, i, l)
            swapelem(aux, i, l)
        end

        lane += blockDim().x
        i = (lane & (j - one(Int32))) + ((lane >> logj) << (logj + one(Int32))) + one(Int32)
    end

    sync_threads()
end

Base.@propagate_inbounds function block_bitonic_sortperm!(value, perm, aux, lt)
    #### Sort the shared memory with bitonic sort
    nextpow2_length = Base._nextpow2(length(value))

    k = Int32(2)
    while k <= nextpow2_length
        block_bitonic_sortperm_major_step!(value, perm, aux, lt, k)

        k *= Int32(2)
    end
end

Base.@propagate_inbounds function block_bitonic_sortperm_major_step!(value, perm, aux, lt, k)
    j = k ÷ Int32(2)
    block_bitonic_sortperm_minor_step!(value, perm, aux, lt, merge_other_lane, j)

    j ÷= Int32(2)
    while j >= one(Int32)
        block_bitonic_sortperm_minor_step!(
            value,
            perm,
            aux,
            lt,
            compare_and_swap_other_lane,
            j,
        )
        j ÷= Int32(2)
    end
end

Base.@propagate_inbounds function block_bitonic_sortperm_minor_step!(value, perm, aux, lt, other_lane, j)
    assume(j >= one(Int32))

    thread = threadIdx().x
    block = fld1(thread, j)
    lane = mod1(thread, j)
    i = (block - one(Int32)) * j * Int32(2) + lane
    l = (block - one(Int32)) * j * Int32(2) + other_lane(j, lane)

    while i <= length(perm)
        if l <= length(perm) && !lt(value[perm[i]], value[perm[l]])
            swapelem(perm, i, l)
            swapelem(aux, i, l)
        end

        thread += blockDim().x
        block = fld1(thread, j)
        lane = mod1(thread, j)
        i = (block - one(Int32)) * j * Int32(2) + lane
        l = (block - one(Int32)) * j * Int32(2) + other_lane(j, lane)
    end

    sync_threads()
end

@inline function merge_other_lane(j, lane)
    mask = Int32(2) * j - one(Int32)

    return (lane - one(Int32)) ⊻ mask + one(Int32)
end

@inline function compare_and_swap_other_lane(j, lane)
    return lane + j
end

Base.@propagate_inbounds function warp_bitonic_sort!(value, aux, lt)
    #### Sort the shared memory with bitonic sort
    nextpow2_length = Base._nextpow2(length(value))
    log_nextpow2_length = unsafe_trunc(Int32, trailing_zeros(nextpow2_length))

    logk = one(Int32)
    while logk <= log_nextpow2_length
        warp_bitonic_sort_major_step!(value, aux, lt, logk)

        logk += one(Int32)
    end
end

Base.@propagate_inbounds function warp_bitonic_sort_major_step!(value, aux, lt, logk)
    logj = logk - one(Int32)
    warp_bitonic_sort_minor_step_merge!(value, aux, lt, logj)

    logj -= one(Int32)
    while logj >= zero(Int32)
        warp_bitonic_sort_minor_step_cas!(value, aux, lt, logj)
        logj -= one(Int32)
    end
end

Base.@propagate_inbounds function warp_bitonic_sort_minor_step_merge!(value, aux, lt, logj::Int32)
    assume(warpsize() == Int32(32))

    j = one(Int32) << logj
    block_size = j * Int32(2)
    mask = block_size - one(Int32)

    lane = laneid() - one(Int32)
    i = (lane & (j - one(Int32))) + ((lane >> logj) << (logj + one(Int32)))
    while i < length(value)
        l = i ⊻ mask
        if l < length(value) && !lt(value[i + one(Int32)], value[l + one(Int32)])
            swapelem(value, i + one(Int32), l + one(Int32))
            swapelem(aux, i + one(Int32), l + one(Int32))
        end

        lane += warpsize()
        i = (lane & (j - one(Int32))) + ((lane >> logj) << (logj + one(Int32))) + one(Int32)
    end

    sync_warp()
end

Base.@propagate_inbounds function warp_bitonic_sort_minor_step_cas!(value, aux, lt, logj::Int32)
    assume(warpsize() == Int32(32))

    j = one(Int32) << logj

    lane = laneid() - one(Int32)
    i = (lane & (j - one(Int32))) + ((lane >> logj) << (logj + one(Int32))) + one(Int32)
    while i <= length(value)
        l = i + j
        if l <= length(value) && !lt(value[i], value[l])
            swapelem(value, i, l)
            swapelem(aux, i, l)
        end

        lane += warpsize()
        i = (lane & (j - one(Int32))) + ((lane >> logj) << (logj + one(Int32))) + one(Int32)
    end

    sync_warp()
end
