
@inline function block_bitonic_sort!(value, aux, lt)
    #### Sort the shared memory with bitonic sort
    nextpow2_length = nextpow(Int32(2), length(value))

    k = Int32(2)
    while k <= nextpow2_length
        block_bitonic_sort_major_step!(value, aux, lt, k)

        k *= Int32(2)
    end
end

@inline function block_bitonic_sort_major_step!(value, aux, lt, k)
    j = k ÷ Int32(2)
    block_bitonic_sort_minor_step!(value, aux, lt, merge_other_lane, j)

    j ÷= Int32(2)
    while j >= Int32(1)
        block_bitonic_sort_minor_step!(value, aux, lt, compare_and_swap_other_lane, j)
        j ÷= Int32(2)
    end
end

@inline function block_bitonic_sort_minor_step!(value, aux, lt, other_lane, j)
    thread = threadIdx().x
    block, lane = fldmod1(thread, j)
    i = (block - one(Int32)) * j * Int32(2) + lane
    l = (block - one(Int32)) * j * Int32(2) + other_lane(j, lane)

    @inbounds while i <= length(value)
        if l <= length(value) && !lt(value[i], value[l])
            value[i], value[l] = value[l], value[i]
            aux[i], aux[l] = aux[l], aux[i]
        end

        thread += blockDim().x
        block, lane = fldmod1(thread, j)
        i = (block - one(Int32)) * j * Int32(2) + lane
        l = (block - one(Int32)) * j * Int32(2) + other_lane(j, lane)
    end

    sync_threads()
end

@inline function block_bitonic_sortperm!(value, perm, aux, lt)
    #### Sort the shared memory with bitonic sort
    nextpow2_length = nextpow(Int32(2), length(value))

    k = Int32(2)
    while k <= nextpow2_length
        block_bitonic_sortperm_major_step!(value, perm, aux, lt, k)

        k *= Int32(2)
    end
end

@inline function block_bitonic_sortperm_major_step!(value, perm, aux, lt, k)
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

@inline function block_bitonic_sortperm_minor_step!(value, perm, aux, lt, other_lane, j)
    thread = threadIdx().x
    block, lane = fldmod1(thread, j)
    i = (block - one(Int32)) * j * Int32(2) + lane
    l = (block - one(Int32)) * j * Int32(2) + other_lane(j, lane)

    @inbounds while i <= length(perm)
        if l <= length(perm) && !lt(value[perm[i]], value[perm[l]])
            perm[i], perm[l] = perm[l], perm[i]
            aux[i], aux[l] = aux[l], aux[i]
        end

        thread += blockDim().x
        block, lane = fldmod1(thread, j)
        i = (block - one(Int32)) * j * Int32(2) + lane
        l = (block - one(Int32)) * j * Int32(2) + other_lane(j, lane)
    end

    sync_threads()
end

@inline function warp_bitonic_sort!(value, aux, lt)
    #### Sort the shared memory with bitonic sort
    nextpow2_length = nextpow(Int32(2), length(value))

    k = Int32(2)
    while k <= nextpow2_length
        warp_bitonic_sort_major_step!(value, aux, lt, k)

        k *= Int32(2)
    end
end

@inline function warp_bitonic_sort_major_step!(value, aux, lt, k)
    j = k ÷ Int32(2)
    warp_bitonic_sort_minor_step!(value, aux, lt, merge_other_lane, j)

    j ÷= Int32(2)
    while j >= Int32(1)
        warp_bitonic_sort_minor_step!(value, aux, lt, compare_and_swap_other_lane, j)
        j ÷= Int32(2)
    end
end

@inline function warp_bitonic_sort_minor_step!(value, aux, lt, other_lane, j)
    assume(warpsize() == Int32(32))

    thread = mod1(threadIdx().x, warpsize())
    block, lane = fldmod1(thread, j)
    i = (block - one(Int32)) * j * Int32(2) + lane
    l = (block - one(Int32)) * j * Int32(2) + other_lane(j, lane)

    @inbounds while i <= length(value)
        if l <= length(value) && !lt(value[i], value[l])
            value[i], value[l] = value[l], value[i]
            aux[i], aux[l] = aux[l], aux[i]
        end

        thread += warpsize()
        block, lane = fldmod1(thread, j)
        i = (block - one(Int32)) * j * Int32(2) + lane
        l = (block - one(Int32)) * j * Int32(2) + other_lane(j, lane)
    end

    sync_warp()
end

@inline function merge_other_lane(j, lane)
    mask = create_mask(j)

    return (lane - one(Int32)) ⊻ mask + one(Int32)
end

@inline function create_mask(j)
    mask = Int32(2) * j - one(Int32)

    return mask
end

@inline function compare_and_swap_other_lane(j, lane)
    return lane + j
end
