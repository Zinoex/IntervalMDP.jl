
function IntervalMDP.bellman!(workspace::AbstractCuWorkspace, Vres, V, prob::IntervalProbabilities; max = true)
    # Test if custom kernel is beneficial
    Vres .= Transpose(Transpose(V) * lower(prob))

    gap_assignment!(workspace, Vres, V, prob; max = max)

    return Vres
end

function gap_assignment!(::CuDenseWorkspace, Vres, V, prob::IntervalProbabilities{Tv}; max = true) where {Tv}
    shmem = length(V) * (sizeof(Int32) + sizeof(Tv))

    kernel = @cuda launch = false dense_gap_assignment_kernel!(Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=))

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    # Execution plan:
    # - value assignment: 1 warp per state
    # - squeeze as many states as possible in a block
    # - use shared memory to store the values and permutation
    # - use bitonic sort to sort the values for all states in a block
    wanted_threads = min(1024, 32 * length(Vres))
    
    threads = min(max_threads, wanted_threads)
    warps = div(threads, 32)
    blocks = min(2^16 - 1, ceil(Int32, length(Vres) / warps))

    kernel(Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=); blocks = blocks, threads = threads, shmem = shmem)
end

function dense_gap_assignment_kernel!(
    Vres,
    V,
    gap::CuDeviceMatrix{Tv},
    sum_lower::CuDeviceVector{Tv},
    lt,
) where {Tv}
    value = CuDynamicSharedArray(Tv, length(V))
    perm = CuDynamicSharedArray(Int32, length(V), length(V) * sizeof(Tv))

    dense_initialize_sorting_shared_memory!(V, value, perm)
    block_bitonic_sort!(value, perm, lt)
    add_gap_mul_V_dense!(Vres, value, perm, gap, sum_lower)
end


@inline function dense_initialize_sorting_shared_memory!(
    V,
    value,
    perm,
)
    # Copy into shared memory
    i = threadIdx().x
    @inbounds while i <= length(V)
        value[i] = V[i]
        perm[i] = i
        i += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

@inline function add_gap_mul_V_dense!(
    Vres,
    value,
    perm,
    gap::CuDeviceMatrix{Tv},
    sum_lower::CuDeviceVector{Tv},
) where {Tv}
    assume(warpsize() == 32)

    thread_id = (blockIdx().x - one(Int32)) * blockDim().x + threadIdx().x
    wid, lane = fldmod1(thread_id, warpsize())

    @inbounds while wid <= length(Vres)
        j = Int32(wid)

        gapⱼ = @view gap[:, j]
        warp_aligned_length = kernel_nextwarp(length(gapⱼ))
        remaining = one(Tv) - sum_lower[j]
        gap_value = zero(Tv)

        s = lane
        while s <= warp_aligned_length
            # Find index of the permutation, and lookup the corresponding gap
            g = if s <= length(gapⱼ)
                gapⱼ[perm[s]]
            else
                # 0 gap is a neural element
                zero(Tv)
            end

            # Cummulatively sum the gap with a tree reduction
            cum_gap = cumsum_warp(g, lane)

            # Update the remaining probability
            remaining -= cum_gap
            remaining += g

            # Update the probability
            if s <= length(gapⱼ)
                sub = clamp(remaining, zero(Tv), g)
                gap_value += sub * value[s]
                remaining -= sub
            end

            # Update the remaining probability from the last thread in the warp
            remaining = shfl_sync(0xffffffff, remaining, warpsize())

            # Early exit if the remaining probability is zero
            if remaining <= zero(Tv)
                break
            end

            s += warpsize()
        end

        gap_value = CUDA.reduce_warp(+, gap_value)

        if lane == 1
            Vres[j] += gap_value
        end
        sync_warp()

        thread_id += gridDim().x * blockDim().x
        wid, lane = fldmod1(thread_id, warpsize())
    end

    return nothing
end



function gap_assignment!(workspace::CuSparseWorkspace, Vres, V, prob::IntervalProbabilities{Tv}; max = true) where {Tv}
    shmem = workspace.max_nonzeros * 2 * sizeof(Tv)

    kernel = @cuda launch = false sparse_gap_assignment_kernel!(workspace, Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=))

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    # Execution plan:
    # - value assignment: 1 warp per state
    # - squeeze as many states as possible in a block
    # - use shared memory to store the values and permutation
    # - use bitonic sort to sort the values for all states in a block
    wanted_threads = min(1024, nextwarp(device(), ceil(Int32, workspace.max_nonzeros / 2)))
    
    threads = min(max_threads, wanted_threads)
    blocks = min(2^16 - 1, ceil(Int32, length(Vres)))

    if threads < 32
        throw("The shared memory available is too little. For now until I have a special case implementation, either try the CPU implementation or use a larger GPU.")

        # TODO: A solution could be to use global memory as intermediate storage.
        # It will be slower, but it will work.
        # I think that it will require that we allocate a smaller amount of global memory 
        # and launch kernels in a loop to process all the states.
        # Alternatively, we will store the permutation only in shared memory.
    end

    kernel(workspace, Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=); blocks = blocks, threads = threads, shmem = shmem)
end

function sparse_gap_assignment_kernel!(
    workspace,
    Vres,
    V,
    gap::CuSparseDeviceMatrixCSC{Tv},
    sum_lower::CuDeviceVector{Tv},
    lt,
) where {Tv}
    assume(warpsize() == 32)

    value = CuDynamicSharedArray(Tv, workspace.max_nonzeros)
    prob = CuDynamicSharedArray(Tv, workspace.max_nonzeros, workspace.max_nonzeros * sizeof(Tv))

    # Grid-stride loop
    j = blockIdx().x

    @inbounds while j <= length(Vres)
        r = gap.colPtr[j]:(gap.colPtr[j + one(Int32)] - one(Int32))
        gindsⱼ = @view gap.rowVal[r]
        gvalsⱼ = @view gap.nzVal[r]
        sparse_initialize_sorting_shared_memory!(V, gindsⱼ, gvalsⱼ, value, prob)

        valueⱼ = @view value[1:length(gindsⱼ)]
        probⱼ = @view prob[1:length(gindsⱼ)]
        block_bitonic_sort!(valueⱼ, probⱼ, lt)
        if threadIdx().x <= warpsize()  # Only the first warp is used to reduce and update the value
            add_gap_mul_V_sparse!(j, Vres, valueⱼ, probⱼ, sum_lower)
        end

        j += gridDim().x
    end
end


@inline function sparse_initialize_sorting_shared_memory!(
    V,
    gapinds,
    gapvals,
    value,
    prob,
)
    # Copy into shared memory
    i = threadIdx().x
    @inbounds while i <= length(gapinds)
        value[i] = V[gapinds[i]]
        prob[i] = gapvals[i]
        i += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

@inline function add_gap_mul_V_sparse!(
    j,
    Vres,
    value,
    prob,
    sum_lower::CuDeviceVector{Tv},
) where {Tv}
    warp_aligned_length = kernel_nextwarp(length(prob))
    @inbounds remaining = one(Tv) - sum_lower[j]
    gap_value = zero(Tv)

    lane = mod1(threadIdx().x, warpsize())

    s = lane
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= length(prob)
            prob[s]
        else
            # 0 gap is a neural element
            zero(Tv)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_warp(g, lane)

        # Update the remaining probability
        remaining -= cum_gap
        remaining += g

        # Update the probability
        if s <= length(prob)
            sub = clamp(remaining, zero(Tv), g)
            gap_value += sub * value[s]
            remaining -= sub
        end

        # Update the remaining probability from the last thread in the warp
        remaining = shfl_sync(0xffffffff, remaining, warpsize())

        # Early exit if the remaining probability is zero
        if remaining <= zero(Tv)
            break
        end

        s += warpsize()
    end

    gap_value = CUDA.reduce_warp(+, gap_value)

    if lane == 1
        @inbounds Vres[j] += gap_value
    end
    sync_warp()

    return nothing
end