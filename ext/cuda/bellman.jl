
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
    blocks = min(2^16 - 1, cld(length(Vres), warps))

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

    warps = div(blockDim().x, warpsize())
    wid, lane = fldmod1(threadIdx().x, warpsize())

    j = wid + (blockIdx().x - one(Int32)) * warps
    @inbounds while j <= length(Vres)
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

        j += gridDim().x * warps
    end

    return nothing
end

function gap_assignment!(workspace::CuSparseWorkspace, Vres, V, prob::IntervalProbabilities{Tv}; max = true) where {Tv}
    # Try to find the best kernel for the problem.

    # Small amounts of shared memory per state, then use multiple states per block
    if try_small_sparse_gap_assignment!(workspace, Vres, V, prob; max = max)
        return nothing
    end

    # Try if we can fit all values and gaps into shared memory
    if try_large_ff_sparse_gap_assignment!(workspace, Vres, V, prob; max = max)
        return nothing
    end

    # Try if we can fit all values and permutation indices into shared memory (25% less memory relative to ff) 
    if try_large_fi_sparse_gap_assignment!(workspace, Vres, V, prob; max = max)
        return nothing
    end

    # Try if we can fit permutation indices into shared memory (50% less memory relative to ff) 
    if try_large_ii_sparse_gap_assignment!(workspace, Vres, V, prob; max = max)
        return nothing
    end

    throw(IntervalMDP.OutOfSharedMemory(workspace.max_nonzeros * 2 * sizeof(Int32)))
end

function try_small_sparse_gap_assignment!(workspace::CuSparseWorkspace, Vres, V, prob::IntervalProbabilities{Tv}; max = true) where {Tv}
    # Execution plan:
    # - at least 8 states per block
    # - one warp per state
    # - use shared memory to store the values and gap probability
    # - use bitonic sort in a warp to sort values

    desired_warps = 8
    shmem = workspace.max_nonzeros * 2 * sizeof(Tv) * desired_warps

    kernel = @cuda launch = false small_sparse_gap_assignment_kernel!(workspace, Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=))

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    if max_threads < desired_warps * 32
        return false
    end

    # TODO: Dynamically maximize the number of states per block

    threads = desired_warps * 32
    blocks = min(2^16 - 1, cld(length(Vres), desired_warps))

    kernel(workspace, Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=); blocks = blocks, threads = threads, shmem = shmem)

    return true
end

function small_sparse_gap_assignment_kernel!(
    workspace,
    Vres,
    V,
    gap::CuSparseDeviceMatrixCSC{Tv},
    sum_lower::CuDeviceVector{Tv},
    lt,
) where {Tv}
    assume(warpsize() == 32)
    warps = div(blockDim().x, warpsize())

    value = CuDynamicSharedArray(Tv, (workspace.max_nonzeros, warps))
    prob = CuDynamicSharedArray(Tv, (workspace.max_nonzeros, warps), workspace.max_nonzeros * warps * sizeof(Tv))

    wid = fld1(threadIdx().x, warpsize())

    value = @view value[:, wid]
    prob = @view prob[:, wid]

    # Grid-stride loop
    j = wid + (blockIdx().x - one(Int32)) * warps
    @inbounds while j <= length(Vres)
        r = gap.colPtr[j]:(gap.colPtr[j + one(Int32)] - one(Int32))
        gindsⱼ = @view gap.rowVal[r]
        gvalsⱼ = @view gap.nzVal[r]
        small_sparse_initialize_sorting_shared_memory!(V, gindsⱼ, gvalsⱼ, value, prob)

        valueⱼ = @view value[1:length(gindsⱼ)]
        probⱼ = @view prob[1:length(gindsⱼ)]
        warp_bitonic_sort!(valueⱼ, probⱼ, lt)
        small_add_gap_mul_V_sparse!(j, Vres, valueⱼ, probⱼ, sum_lower)

        j += gridDim().x * warps
    end
end


@inline function small_sparse_initialize_sorting_shared_memory!(
    V,
    gapinds,
    gapvals,
    value,
    prob,
)
    assume(warpsize() == 32)

    # Copy into shared memory
    i = mod1(threadIdx().x, warpsize())
    @inbounds while i <= length(gapinds)
        value[i] = V[gapinds[i]]
        prob[i] = gapvals[i]
        i += warpsize()
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_warp()
end

@inline function small_add_gap_mul_V_sparse!(
    j,
    Vres,
    value,
    prob,
    sum_lower::CuDeviceVector{Tv},
) where {Tv}
    assume(warpsize() == 32)

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

function try_large_ff_sparse_gap_assignment!(workspace::CuSparseWorkspace, Vres, V, prob::IntervalProbabilities{Tv}; max = true) where {Tv}
    # Execution plan:
    # - one state per block
    # - use shared memory to store the values and gap probability
    # - use bitonic sort in a block to sort the values 

    shmem = workspace.max_nonzeros * 2 * sizeof(Tv)

    kernel = @cuda launch = false ff_sparse_gap_assignment_kernel!(workspace, Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=))

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    if max_threads < 32
        return false
    end

    wanted_threads = min(1024, nextwarp(device(), cld(workspace.max_nonzeros, 2)))
    
    threads = min(max_threads, wanted_threads)
    blocks = min(2^16 - 1, length(Vres))

    kernel(workspace, Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=); blocks = blocks, threads = threads, shmem = shmem)

    return true
end

function ff_sparse_gap_assignment_kernel!(
    workspace,
    Vres,
    V,
    gap::CuSparseDeviceMatrixCSC{Tv},
    sum_lower::CuDeviceVector{Tv},
    lt,
) where {Tv}
    reduction_ws = CuStaticSharedArray(Tv, 32)
    value = CuDynamicSharedArray(Tv, workspace.max_nonzeros)
    prob = CuDynamicSharedArray(Tv, workspace.max_nonzeros, workspace.max_nonzeros * sizeof(Tv))

    # Grid-stride loop
    j = blockIdx().x
    @inbounds while j <= length(Vres)
        r = gap.colPtr[j]:(gap.colPtr[j + one(Int32)] - one(Int32))
        gindsⱼ = @view gap.rowVal[r]
        gvalsⱼ = @view gap.nzVal[r]
        ff_sparse_initialize_sorting_shared_memory!(V, gindsⱼ, gvalsⱼ, value, prob)

        valueⱼ = @view value[1:length(gindsⱼ)]
        probⱼ = @view prob[1:length(gindsⱼ)]
        block_bitonic_sort!(valueⱼ, probⱼ, lt)
        ff_add_gap_mul_V_sparse!(reduction_ws, j, Vres, valueⱼ, probⱼ, sum_lower)

        j += gridDim().x
    end

    return nothing
end


@inline function ff_sparse_initialize_sorting_shared_memory!(
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

@inline function ff_add_gap_mul_V_sparse!(
    reduction_ws,
    j,
    Vres,
    value,
    prob,
    sum_lower::CuDeviceVector{Tv},
) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(prob))
    @inbounds remaining = one(Tv) - sum_lower[j]
    gap_value = zero(Tv)

    wid, lane = fldmod1(threadIdx().x, warpsize())

    # Block-strided loop and save into register `gap_value`
    s = threadIdx().x
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= length(prob)
            prob[s]
        else
            # 0 gap is a neural element
            zero(Tv)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_block(g, reduction_ws, wid, lane)

        # Update the remaining probability
        remaining -= cum_gap
        remaining += g

        # Update the probability
        if s <= length(prob)
            sub = clamp(remaining, zero(Tv), g)
            gap_value += sub * value[s]
            remaining -= sub
        end

        # Update the remaining probability from the last thread in the block
        if threadIdx().x == blockDim().x
            reduction_ws[1] = remaining
        end
        sync_threads()

        remaining = reduction_ws[1]
        sync_threads()

        # Early exit if the remaining probability is zero
        if remaining <= zero(Tv)
            break
        end

        s += blockDim().x
    end

    # Warp-reduction
    gap_value = CUDA.reduce_warp(+, gap_value)
    sync_threads()

    # Block-reduction
    if wid == 1
        reduction_ws[lane] = zero(Tv)
    end
    sync_threads()

    if lane == 1
        reduction_ws[wid] = gap_value
    end
    sync_threads()

    if wid == 1
        gap_value = reduction_ws[lane]
        gap_value = CUDA.reduce_warp(+, gap_value)

        if lane == 1
            @inbounds Vres[j] += gap_value
        end
    end
    sync_threads()

    return nothing
end

function try_large_fi_sparse_gap_assignment!(workspace::CuSparseWorkspace, Vres, V, prob::IntervalProbabilities{Tv}; max = true) where {Tv}
    # Execution plan:
    # - one state per block
    # - use shared memory to store the values and permutation indices
    # - use bitonic sort in a block to sort the values 

    shmem = workspace.max_nonzeros * (sizeof(Tv) + sizeof(Int32))

    kernel = @cuda launch = false fi_sparse_gap_assignment_kernel!(workspace, Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=))

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    if max_threads < 32
        return false
    end

    wanted_threads = min(1024, nextwarp(device(), cld(workspace.max_nonzeros, 2)))
    
    threads = min(max_threads, wanted_threads)
    blocks = min(2^16 - 1, length(Vres))

    kernel(workspace, Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=); blocks = blocks, threads = threads, shmem = shmem)

    return true
end

function fi_sparse_gap_assignment_kernel!(
    workspace,
    Vres,
    V,
    gap::CuSparseDeviceMatrixCSC{Tv},
    sum_lower::CuDeviceVector{Tv},
    lt,
) where {Tv}
    reduction_ws = CuStaticSharedArray(Tv, 32)
    value = CuDynamicSharedArray(Tv, workspace.max_nonzeros)
    perm = CuDynamicSharedArray(Int32, workspace.max_nonzeros, workspace.max_nonzeros * sizeof(Tv))

    # Grid-stride loop
    j = blockIdx().x
    @inbounds while j <= length(Vres)
        r = gap.colPtr[j]:(gap.colPtr[j + one(Int32)] - one(Int32))
        gindsⱼ = @view gap.rowVal[r]
        gvalsⱼ = @view gap.nzVal[r]
        fi_sparse_initialize_sorting_shared_memory!(V, gindsⱼ, value, perm)

        valueⱼ = @view value[1:length(gindsⱼ)]
        permⱼ = @view perm[1:length(gindsⱼ)]
        block_bitonic_sort!(valueⱼ, permⱼ, lt)
        fi_add_gap_mul_V_sparse!(reduction_ws, j, Vres, valueⱼ, permⱼ, gvalsⱼ, sum_lower)

        sync_threads()
        j += gridDim().x
    end
end


@inline function fi_sparse_initialize_sorting_shared_memory!(
    V,
    gapinds,
    value,
    perm,
)
    # Copy into shared memory
    i = threadIdx().x
    @inbounds while i <= length(gapinds)
        value[i] = V[gapinds[i]]
        perm[i] = i
        i += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

@inline function fi_add_gap_mul_V_sparse!(
    reduction_ws,
    j,
    Vres,
    value,
    perm,
    gapvals,
    sum_lower::CuDeviceVector{Tv},
) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(gapvals))
    @inbounds remaining = one(Tv) - sum_lower[j]
    gap_value = zero(Tv)

    wid, lane = fldmod1(threadIdx().x, warpsize())

    # Block-strided loop and save into register `gap_value`
    s = threadIdx().x
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= length(gapvals)
            t = perm[s]
            gapvals[t]
        else
            # 0 gap is a neural element
            zero(Tv)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_block(g, reduction_ws, wid, lane)

        # Update the remaining probability
        remaining -= cum_gap
        remaining += g

        # Update the probability
        if s <= length(gapvals)
            sub = clamp(remaining, zero(Tv), g)
            gap_value += sub * value[s]
            remaining -= sub
        end

        # Update the remaining probability from the last thread in the block
        if threadIdx().x == blockDim().x
            reduction_ws[1] = remaining
        end
        sync_threads()

        remaining = reduction_ws[1]
        sync_threads()

        # Early exit if the remaining probability is zero
        if remaining <= zero(Tv)
            break
        end

        s += blockDim().x
    end

    # Warp-reduction
    gap_value = CUDA.reduce_warp(+, gap_value)
    sync_threads()

    # Block-reduction
    if wid == 1
        reduction_ws[lane] = zero(Tv)
    end
    sync_threads()

    if lane == 1
        reduction_ws[wid] = gap_value
    end
    sync_threads()

    if wid == 1
        gap_value = reduction_ws[lane]
        gap_value = CUDA.reduce_warp(+, gap_value)

        if lane == 1
            @inbounds Vres[j] += gap_value
        end
    end
    sync_threads()

    return nothing
end

function try_large_ii_sparse_gap_assignment!(workspace::CuSparseWorkspace, Vres, V, prob::IntervalProbabilities{Tv}; max = true) where {Tv}
    # Execution plan:
    # - one state per block
    # - use shared memory to store only permutation indices
    # - use bitonic sort in a block to sort the permutation indices by the values 

    shmem = workspace.max_nonzeros * 2 * sizeof(Int32)

    kernel = @cuda launch = false ii_sparse_gap_assignment_kernel!(workspace, Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=))

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    if max_threads < 32
        return false
    end

    wanted_threads = min(1024, nextwarp(device(), cld(workspace.max_nonzeros, 2)))
    
    threads = min(max_threads, wanted_threads)
    blocks = min(2^16 - 1, length(Vres))

    kernel(workspace, Vres, V, gap(prob), sum_lower(prob), max ? (>=) : (<=); blocks = blocks, threads = threads, shmem = shmem)

    return true
end

function ii_sparse_gap_assignment_kernel!(
    workspace,
    Vres,
    V,
    gap::CuSparseDeviceMatrixCSC{Tv},
    sum_lower::CuDeviceVector{Tv},
    lt,
) where {Tv}
    reduction_ws = CuStaticSharedArray(Tv, 32)
    Vperm = CuDynamicSharedArray(Int32, workspace.max_nonzeros)
    Pperm = CuDynamicSharedArray(Int32, workspace.max_nonzeros, workspace.max_nonzeros * sizeof(Int32))

    # Grid-stride loop
    j = blockIdx().x
    @inbounds while j <= length(Vres)
        r = gap.colPtr[j]:(gap.colPtr[j + one(Int32)] - one(Int32))
        gindsⱼ = @view gap.rowVal[r]
        gvalsⱼ = @view gap.nzVal[r]
        ii_sparse_initialize_sorting_shared_memory!(gindsⱼ, Vperm, Pperm)

        Vpermⱼ = @view Vperm[1:length(gindsⱼ)]
        Ppermⱼ = @view Pperm[1:length(gindsⱼ)]

        block_bitonic_sortperm!(V, Vpermⱼ, Ppermⱼ, lt)
        ii_add_gap_mul_V_sparse!(reduction_ws, j, Vres, V, Vpermⱼ, Ppermⱼ, gvalsⱼ, sum_lower)

        sync_threads()
        j += gridDim().x
    end
end


@inline function ii_sparse_initialize_sorting_shared_memory!(
    gapinds,
    Vperm,
    Pperm
)
    # Copy into shared memory
    i = threadIdx().x
    @inbounds while i <= length(gapinds)
        Vperm[i] = gapinds[i]
        Pperm[i] = i
        i += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

@inline function ii_add_gap_mul_V_sparse!(
    reduction_ws,
    j,
    Vres,
    value,
    Vperm,
    Pperm,
    gapvals,
    sum_lower::CuDeviceVector{Tv},
) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(gapvals))
    @inbounds remaining = one(Tv) - sum_lower[j]
    gap_value = zero(Tv)

    wid, lane = fldmod1(threadIdx().x, warpsize())

    # Block-strided loop and save into register `gap_value`
    s = threadIdx().x
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= length(gapvals)
            t = Pperm[s]
            gapvals[t]
        else
            # 0 gap is a neural element
            zero(Tv)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_block(g, reduction_ws, wid, lane)

        # Update the remaining probability
        remaining -= cum_gap
        remaining += g

        # Update the probability
        if s <= length(gapvals)
            sub = clamp(remaining, zero(Tv), g)
            gap_value += sub * value[Vperm[s]]
            remaining -= sub
        end

        # Update the remaining probability from the last thread in the block
        if threadIdx().x == blockDim().x
            reduction_ws[1] = remaining
        end
        sync_threads()

        remaining = reduction_ws[1]
        sync_threads()

        # Early exit if the remaining probability is zero
        if remaining <= zero(Tv)
            break
        end

        s += blockDim().x
    end

    # Warp-reduction
    gap_value = CUDA.reduce_warp(+, gap_value)
    sync_threads()

    # Block-reduction
    if wid == 1
        reduction_ws[lane] = zero(Tv)
    end
    sync_threads()

    if lane == 1
        reduction_ws[wid] = gap_value
    end
    sync_threads()

    if wid == 1
        gap_value = reduction_ws[lane]
        gap_value = CUDA.reduce_warp(+, gap_value)

        if lane == 1
            @inbounds Vres[j] += gap_value
        end
    end
    sync_threads()

    return nothing
end