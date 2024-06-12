function IntervalMDP.bellman!(
    workspace::CuDenseWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    stateptr;
    upper_bound = false,
    maximize = true,
) where {Tv}
    max_states_per_block = 32
    shmem = length(V) * (sizeof(Int32) + sizeof(Tv)) + max_states_per_block * workspace.max_actions * sizeof(Tv)

    kernel = @cuda launch = false dense_bellman_kernel!(
        workspace,
        strategy_cache,
        Vres,
        V,
        prob,
        stateptr,
        upper_bound ? (>=) : (<=),
        maximize ? max : min
    )

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    # Execution plan:
    # - value assignment: 1 warp per state
    # - squeeze as many states as possible in a block
    # - use shared memory to store the values and permutation
    # - use bitonic sort to sort the values for all states in a block
    wanted_threads = min(1024, max_states_per_block * length(Vres))

    threads = min(max_threads, wanted_threads)
    warps = div(threads, 32)
    blocks = min(2^16 - 1, cld(length(Vres), warps))
    shmem = length(V) * (sizeof(Int32) + sizeof(Tv)) + warps * workspace.max_actions * sizeof(Tv)

    CUDA.@sync kernel(
        workspace,
        strategy_cache,
        Vres,
        V,
        prob,
        stateptr,
        upper_bound ? (>=) : (<=),
        maximize ? max : min;
        blocks = blocks,
        threads = threads,
        shmem = shmem,
    )

    return Vres
end

function dense_bellman_kernel!(
    workspace,
    strategy_cache,
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    stateptr,
    value_lt,
    action_min,
) where {Tv}
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    # Prepare action workspace shared memory
    action_workspace = CuDynamicSharedArray(Tv, (nwarps, workspace.max_actions))
    @inbounds action_workspace = @view action_workspace[wid, :]

    # Prepare sorting shared memory
    value = CuDynamicSharedArray(Tv, length(V), nwarps * workspace.max_actions * sizeof(Tv))
    perm = CuDynamicSharedArray(Int32, length(V), (nwarps * workspace.max_actions + length(V)) * sizeof(Tv))

    # Perform sorting
    dense_initialize_sorting_shared_memory!(V, value, perm)
    block_bitonic_sort!(value, perm, value_lt)

    # O-maxmization
    dense_omaximization!(action_workspace, strategy_cache, Vres, value, perm, prob, stateptr, action_min)

    return nothing
end

@inline function dense_initialize_sorting_shared_memory!(V, value, perm)
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

@inline function dense_omaximization!(
    action_workspace,
    strategy_cache,
    Vres,
    value,
    perm,
    prob,
    stateptr,
    action_min,
)
    assume(warpsize() == 32)

    warps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    j = wid + (blockIdx().x - one(Int32)) * warps
    @inbounds while j <= length(Vres)
        state_dense_omaximization!(action_workspace, strategy_cache, Vres, value, perm, prob, stateptr, action_min, j)
        j += gridDim().x * warps
    end

    return nothing
end

@inline function state_dense_omaximization!(
    action_workspace,
    strategy_cache,
    Vres,
    value,
    perm,
    prob::IntervalProbabilities{Tv},
    stateptr,
    action_min,
    jₛ
) where {Tv}
    lane = mod1(threadIdx().x, warpsize())

    s₁, s₂ = stateptr[jₛ], stateptr[jₛ + one(Int32)]
    nactions = s₂ - s₁
    @inbounds action_values = @view action_workspace[1:nactions]

    k = one(Int32)
    @inbounds while k <= nactions
        jₐ = s₁ + k - one(Int32)
        lowerⱼ = @view lower(prob)[:, jₐ]
        gapⱼ = @view gap(prob)[:, jₐ]
        sum_lowerⱼ = sum_lower(prob)[jₐ]

        # Use O-maxmization to find the value for the action
        v = state_action_dense_omaximization!(
            value,
            perm,
            lowerⱼ,
            gapⱼ,
            sum_lowerⱼ,
            lane
        )

        if lane == one(Int32)
            action_values[k] = v
        end
        sync_warp()

        k += one(Int32)
    end

    # Find the best action
    v = extract_strategy_warp!(
        strategy_cache,
        action_values,
        Vres,
        jₛ,
        s₁,
        action_min,
        lane
    )

    if lane == one(Int32)
        Vres[jₛ] = v
    end
    sync_warp()
end

@inline function state_action_dense_omaximization!(
    value,
    perm,
    lower,
    gap,
    sum_lower::Tv,
    lane
) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(lower))
    remaining = one(Tv) - sum_lower
    gap_value = zero(Tv)

    # Add the lower bound multiplied by the value
    s = lane
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        if s <= length(lower)
            p = perm[s]

            gap_value += lower[p] * value[s]
        end

        s += warpsize()
    end

    # Add the gap multiplied by the value
    s = lane
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= length(gap)
            gap[perm[s]]
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
        if s <= length(gap)
            g = clamp(remaining, zero(Tv), g)
            gap_value += g * value[s]
            remaining -= g
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
    return gap_value
end


function IntervalMDP.bellman!(
    workspace::CuSparseWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    stateptr;
    max = true
) where {Tv}
    Vres .= Transpose(Transpose(V) * lower(prob))

    # Try to find the best kernel for the problem.

    # Small amounts of shared memory per state, then use multiple states per block
    if try_small_sparse_gap_assignment!(workspace, Vres, V, prob; max = max)
        return Vres
    end

    # Try if we can fit all values and gaps into shared memory
    if try_large_ff_sparse_gap_assignment!(workspace, Vres, V, prob; max = max)
        return Vres
    end

    # Try if we can fit all values and permutation indices into shared memory (25% less memory relative to ff) 
    if try_large_fi_sparse_gap_assignment!(workspace, Vres, V, prob; max = max)
        return Vres
    end

    # Try if we can fit permutation indices into shared memory (50% less memory relative to ff) 
    if try_large_ii_sparse_gap_assignment!(workspace, Vres, V, prob; max = max)
        return Vres
    end

    throw(IntervalMDP.OutOfSharedMemory(workspace.max_nonzeros * 2 * sizeof(Int32)))
end

function try_small_sparse_gap_assignment!(
    workspace::CuSparseWorkspace,
    Vres,
    V,
    prob::IntervalProbabilities{Tv};
    max = true,
) where {Tv}
    # Execution plan:
    # - at least 8 states per block
    # - one warp per state
    # - use shared memory to store the values and gap probability
    # - use bitonic sort in a warp to sort values

    desired_warps = 8
    shmem = workspace.max_nonzeros * 2 * sizeof(Tv) * desired_warps

    kernel = @cuda launch = false small_sparse_gap_assignment_kernel!(
        workspace,
        Vres,
        V,
        gap(prob),
        sum_lower(prob),
        max ? (>=) : (<=),
    )

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    if max_threads < desired_warps * 32
        return false
    end

    # TODO: Dynamically maximize the number of states per block

    threads = desired_warps * 32
    blocks = min(2^16 - 1, cld(length(Vres), desired_warps))

    kernel(
        workspace,
        Vres,
        V,
        gap(prob),
        sum_lower(prob),
        max ? (>=) : (<=);
        blocks = blocks,
        threads = threads,
        shmem = shmem,
    )

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
    prob = CuDynamicSharedArray(
        Tv,
        (workspace.max_nonzeros, warps),
        workspace.max_nonzeros * warps * sizeof(Tv),
    )

    wid = fld1(threadIdx().x, warpsize())

    value = @inbounds @view value[:, wid]
    prob = @inbounds @view prob[:, wid]

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

function try_large_ff_sparse_gap_assignment!(
    workspace::CuSparseWorkspace,
    Vres,
    V,
    prob::IntervalProbabilities{Tv};
    max = true,
) where {Tv}
    # Execution plan:
    # - one state per block
    # - use shared memory to store the values and gap probability
    # - use bitonic sort in a block to sort the values 

    shmem = workspace.max_nonzeros * 2 * sizeof(Tv)

    kernel = @cuda launch = false ff_sparse_gap_assignment_kernel!(
        workspace,
        Vres,
        V,
        gap(prob),
        sum_lower(prob),
        max ? (>=) : (<=),
    )

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    if max_threads < 32
        return false
    end

    wanted_threads = min(1024, nextwarp(device(), cld(workspace.max_nonzeros, 2)))

    threads = min(max_threads, wanted_threads)
    blocks = min(2^16 - 1, length(Vres))

    kernel(
        workspace,
        Vres,
        V,
        gap(prob),
        sum_lower(prob),
        max ? (>=) : (<=);
        blocks = blocks,
        threads = threads,
        shmem = shmem,
    )

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
    prob = CuDynamicSharedArray(
        Tv,
        workspace.max_nonzeros,
        workspace.max_nonzeros * sizeof(Tv),
    )

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

function try_large_fi_sparse_gap_assignment!(
    workspace::CuSparseWorkspace,
    Vres,
    V,
    prob::IntervalProbabilities{Tv};
    max = true,
) where {Tv}
    # Execution plan:
    # - one state per block
    # - use shared memory to store the values and permutation indices
    # - use bitonic sort in a block to sort the values 

    shmem = workspace.max_nonzeros * (sizeof(Tv) + sizeof(Int32))

    kernel = @cuda launch = false fi_sparse_gap_assignment_kernel!(
        workspace,
        Vres,
        V,
        gap(prob),
        sum_lower(prob),
        max ? (>=) : (<=),
    )

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    if max_threads < 32
        return false
    end

    wanted_threads = min(1024, nextwarp(device(), cld(workspace.max_nonzeros, 2)))

    threads = min(max_threads, wanted_threads)
    blocks = min(2^16 - 1, length(Vres))

    kernel(
        workspace,
        Vres,
        V,
        gap(prob),
        sum_lower(prob),
        max ? (>=) : (<=);
        blocks = blocks,
        threads = threads,
        shmem = shmem,
    )

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
    perm = CuDynamicSharedArray(
        Int32,
        workspace.max_nonzeros,
        workspace.max_nonzeros * sizeof(Tv),
    )

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

@inline function fi_sparse_initialize_sorting_shared_memory!(V, gapinds, value, perm)
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

function try_large_ii_sparse_gap_assignment!(
    workspace::CuSparseWorkspace,
    Vres,
    V,
    prob::IntervalProbabilities{Tv};
    max = true,
) where {Tv}
    # Execution plan:
    # - one state per block
    # - use shared memory to store only permutation indices
    # - use bitonic sort in a block to sort the permutation indices by the values 

    shmem = workspace.max_nonzeros * 2 * sizeof(Int32)

    kernel = @cuda launch = false ii_sparse_gap_assignment_kernel!(
        workspace,
        Vres,
        V,
        gap(prob),
        sum_lower(prob),
        max ? (>=) : (<=),
    )

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    if max_threads < 32
        return false
    end

    wanted_threads = min(1024, nextwarp(device(), cld(workspace.max_nonzeros, 2)))

    threads = min(max_threads, wanted_threads)
    blocks = min(2^16 - 1, length(Vres))

    kernel(
        workspace,
        Vres,
        V,
        gap(prob),
        sum_lower(prob),
        max ? (>=) : (<=);
        blocks = blocks,
        threads = threads,
        shmem = shmem,
    )

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
    Pperm = CuDynamicSharedArray(
        Int32,
        workspace.max_nonzeros,
        workspace.max_nonzeros * sizeof(Int32),
    )

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
        ii_add_gap_mul_V_sparse!(
            reduction_ws,
            j,
            Vres,
            V,
            Vpermⱼ,
            Ppermⱼ,
            gvalsⱼ,
            sum_lower,
        )

        sync_threads()
        j += gridDim().x
    end
end

@inline function ii_sparse_initialize_sorting_shared_memory!(gapinds, Vperm, Pperm)
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
