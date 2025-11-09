
Base.@propagate_inbounds function block_bitonic_sort!(value, aux, lt)
    #### Sort the shared memory with bitonic sort
    nextpow2_length = Base._nextpow2(length(value))

    k = Int32(2)
    while k <= nextpow2_length
        block_bitonic_sort_major_step!(value, aux, lt, k)

        k *= Int32(2)
    end
end

Base.@propagate_inbounds function block_bitonic_sort_major_step!(value, aux, lt, k)
    j = k ÷ Int32(2)
    block_bitonic_sort_minor_step!(value, aux, lt, merge_other_lane, j)

    j ÷= Int32(2)
    while j >= Int32(1)
        block_bitonic_sort_minor_step!(value, aux, lt, compare_and_swap_other_lane, j)
        j ÷= Int32(2)
    end
end

Base.@propagate_inbounds function block_bitonic_sort_minor_step!(value, aux, lt, other_lane, j)
    assume(j >= Int32(1))

    thread = threadIdx().x
    block = fld1(thread, j)
    lane = mod1(thread, j)
    i = (block - one(Int32)) * j * Int32(2) + lane
    l = (block - one(Int32)) * j * Int32(2) + other_lane(j, lane)

    while i <= length(value)
        if l <= length(value) && !lt(value[i], value[l])
            swapelem(value, i, l)
            swapelem(aux, i, l)
        end

        thread += blockDim().x
        block = fld1(thread, j)
        lane = mod1(thread, j)
        i = block * j * Int32(2) + lane
        l = block * j * Int32(2) + other_lane(j, lane)
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
    while j >= Int32(1)
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
    assume(j >= Int32(1))

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

Base.@propagate_inbounds function warp_bitonic_sort!(value, aux, lt)
    #### Sort the shared memory with bitonic sort
    nextpow2_length = Base._nextpow2(length(value))

    k = Int32(2)
    while k <= nextpow2_length
        warp_bitonic_sort_major_step!(value, aux, lt, k)

        k *= Int32(2)
    end
end

Base.@propagate_inbounds function warp_bitonic_sort_major_step!(value, aux, lt, k)
    j = k ÷ Int32(2)
    warp_bitonic_sort_minor_step!(value, aux, lt, merge_other_lane, j)

    j ÷= Int32(2)
    while j >= Int32(1)
        warp_bitonic_sort_minor_step!(value, aux, lt, compare_and_swap_other_lane, j)
        j ÷= Int32(2)
    end
end

Base.@propagate_inbounds function warp_bitonic_sort_minor_step!(value, aux, lt, other_lane, j::Int32)
    assume(warpsize() == Int32(32))
    assume(j >= Int32(1))

    thread = mod1(threadIdx().x, warpsize())
    block = fld1(thread, j)
    lane = mod1(thread, j)
    i = (block - one(Int32)) * j * Int32(2) + lane
    l = (block - one(Int32)) * j * Int32(2) + other_lane(j, lane)

    while i <= length(value)
        if l <= length(value) && !lt(value[i], value[l])
            swapelem(value, i, l)
            swapelem(aux, i, l)
        end

        thread += warpsize()
        block = fld1(thread, j)
        lane = mod1(thread, j)
        i = (block - one(Int32)) * j * Int32(2) + lane
        l = (block - one(Int32)) * j * Int32(2) + other_lane(j, lane)
    end

    sync_warp()
end

@inline function merge_other_lane(j, lane)
    mask = Int32(2) * j - one(Int32)

    return (lane - one(Int32)) ⊻ mask + one(Int32)
end

@inline function compare_and_swap_other_lane(j, lane)
    return lane + j
end
