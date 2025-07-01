function IntervalMDP._bellman_helper!(
    workspace::CuSparseWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    stateptr;
    upper_bound = false,
    maximize = true,
) where {Tv}

    # Try to find the best kernel for the problem.

    # Small amounts of shared memory per state, then use multiple states per block
    if try_small_sparse_bellman!(
        workspace,
        strategy_cache,
        Vres,
        V,
        prob,
        stateptr;
        upper_bound = upper_bound,
        maximize = maximize,
    )
        return Vres
    end

    # Try if we can fit all values and gaps into shared memory
    if try_large_sparse_bellman!(
        Tv,
        Tv,
        workspace,
        strategy_cache,
        Vres,
        V,
        prob,
        stateptr;
        upper_bound = upper_bound,
        maximize = maximize,
    )
        return Vres
    end

    # Try if we can fit all values and permutation indices into shared memory (25% less memory relative to (Tv, Tv)) 
    if try_large_sparse_bellman!(
        Tv,
        Int32,
        workspace,
        strategy_cache,
        Vres,
        V,
        prob,
        stateptr;
        upper_bound = upper_bound,
        maximize = maximize,
    )
        return Vres
    end

    # Try if we can fit permutation indices into shared memory (50% less memory relative to (Tv, Tv)) 
    if try_large_sparse_bellman!(
        Int32,
        Int32,
        workspace,
        strategy_cache,
        Vres,
        V,
        prob,
        stateptr;
        upper_bound = upper_bound,
        maximize = maximize,
    )
        return Vres
    end

    throw(IntervalMDP.OutOfSharedMemory(workspace.max_nonzeros * 2 * sizeof(Int32)))
end

function try_small_sparse_bellman!(
    workspace::CuSparseWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    stateptr;
    upper_bound = false,
    maximize = true,
) where {Tv}
    # Execution plan:
    # - at least 8 states per block
    # - one warp per state
    # - use shared memory to store the values and gap probability
    # - use bitonic sort in a warp to sort values_gaps

    desired_warps = 8
    shmem =
        (workspace.max_nonzeros + workspace.max_actions) * 2 * sizeof(Tv) * desired_warps

    kernel = @cuda launch = false small_sparse_bellman_kernel!(
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        prob,
        stateptr,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv)),
    )

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    if max_threads < desired_warps * 32
        return false
    end

    num_states = length(stateptr) - one(Int32)
    threads = desired_warps * 32
    blocks = min(2^16 - 1, cld(num_states, desired_warps))

    kernel(
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        prob,
        stateptr,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv));
        blocks = blocks,
        threads = threads,
        shmem = shmem,
    )

    return true
end

function small_sparse_bellman_kernel!(
    workspace,
    strategy_cache,
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    stateptr,
    value_lt,
    action_reduce,
) where {Tv}
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())

    action_workspace = CuDynamicSharedArray(Tv, (workspace.max_actions, nwarps))
    value_ws = CuDynamicSharedArray(
        Tv,
        (workspace.max_nonzeros, nwarps),
        workspace.max_actions * nwarps * sizeof(Tv),
    )
    gap_ws = CuDynamicSharedArray(
        Tv,
        (workspace.max_nonzeros, nwarps),
        (workspace.max_nonzeros + workspace.max_actions) * nwarps * sizeof(Tv),
    )

    wid = fld1(threadIdx().x, warpsize())

    @inbounds action_workspace = @view action_workspace[:, wid]
    @inbounds value_ws = @view value_ws[:, wid]
    @inbounds gap_ws = @view gap_ws[:, wid]

    # Grid-stride loop
    num_states = length(stateptr) - one(Int32)
    j = wid + (blockIdx().x - one(Int32)) * nwarps
    while j <= num_states
        state_small_sparse_omaximization!(
            action_workspace,
            value_ws,
            gap_ws,
            strategy_cache,
            Vres,
            V,
            prob,
            stateptr,
            value_lt,
            action_reduce,
            j,
        )
        j += gridDim().x * nwarps
    end
end

@inline function state_small_sparse_omaximization!(
    action_workspace,
    value_ws,
    gap_ws,
    strategy_cache,
    Vres,
    V,
    prob,
    stateptr,
    value_lt,
    action_reduce,
    jₛ,
)
    lane = mod1(threadIdx().x, warpsize())

    s₁, s₂ = stateptr[jₛ], stateptr[jₛ + one(Int32)]
    nactions = s₂ - s₁
    @inbounds action_values = @view action_workspace[1:nactions]

    k = one(Int32)
    @inbounds while k <= nactions
        jₐ = s₁ + k - one(Int32)
        sum_lowerⱼ = sum_lower(prob)[jₐ]

        r = lower(prob).colPtr[jₐ]:(lower(prob).colPtr[jₐ + one(Int32)] - one(Int32))
        lindsⱼ = @view lower(prob).rowVal[r]
        lvalsⱼ = @view lower(prob).nzVal[r]

        r = gap(prob).colPtr[jₐ]:(gap(prob).colPtr[jₐ + one(Int32)] - one(Int32))
        gindsⱼ = @view gap(prob).rowVal[r]
        gvalsⱼ = @view gap(prob).nzVal[r]

        # Use O-maxmization to find the value for the action
        v = state_action_small_sparse_omaximization!(
            value_ws,
            gap_ws,
            V,
            lindsⱼ,
            lvalsⱼ,
            gindsⱼ,
            gvalsⱼ,
            sum_lowerⱼ,
            value_lt,
            lane,
        )

        if lane == one(Int32)
            action_values[k] = v
        end
        sync_warp()

        k += one(Int32)
    end

    # Find the best action
    v = extract_strategy_warp!(strategy_cache, action_values, Vres, jₛ, action_reduce, lane)

    if lane == one(Int32)
        Vres[jₛ] = v
    end
    sync_warp()
end

@inline function state_small_sparse_omaximization!(
    action_workspace,
    value_ws,
    gap_ws,
    strategy_cache::NonOptimizingActiveCache,
    Vres,
    V,
    prob,
    stateptr,
    value_lt,
    action_reduce,
    jₛ,
)
    lane = mod1(threadIdx().x, warpsize())

    @inbounds begin
        s₁ = stateptr[jₛ]
        jₐ = s₁ + strategy_cache[jₛ] - one(Int32)
        sum_lowerⱼ = sum_lower(prob)[jₐ]

        r = lower(prob).colPtr[jₐ]:(lower(prob).colPtr[jₐ + one(Int32)] - one(Int32))
        lindsⱼ = @view lower(prob).rowVal[r]
        lvalsⱼ = @view lower(prob).nzVal[r]

        r = gap(prob).colPtr[jₐ]:(gap(prob).colPtr[jₐ + one(Int32)] - one(Int32))
        gindsⱼ = @view gap(prob).rowVal[r]
        gvalsⱼ = @view gap(prob).nzVal[r]

        # Use O-maxmization to find the value for the action
        v = state_action_small_sparse_omaximization!(
            value_ws,
            gap_ws,
            V,
            lindsⱼ,
            lvalsⱼ,
            gindsⱼ,
            gvalsⱼ,
            sum_lowerⱼ,
            value_lt,
            lane,
        )

        if lane == one(Int32)
            Vres[jₛ] = v
        end
        sync_warp()
    end
end

@inline function state_action_small_sparse_omaximization!(
    value_ws,
    gap_ws,
    V,
    lower_inds,
    lower_vals,
    gap_inds,
    gap_vals,
    sum_lower::Tv,
    value_lt,
    lane,
) where {Tv}
    value = add_lower_mul_V_warp(V, lower_inds, lower_vals, lane)

    small_sparse_initialize_sorting_shared_memory!(V, gap_inds, gap_vals, value_ws, gap_ws)

    @inbounds valueⱼ = @view value_ws[1:length(gap_inds)]
    @inbounds gapⱼ = @view gap_ws[1:length(gap_inds)]
    warp_bitonic_sort!(valueⱼ, gapⱼ, value_lt)

    value += small_add_gap_mul_V_sparse(valueⱼ, gapⱼ, sum_lower, lane)

    return value
end

@inline function add_lower_mul_V_warp(
    V::AbstractVector{Tv},
    lower_inds,
    lower_vals,
    lane,
) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(lower_vals))
    lower_value = zero(Tv)

    s = lane
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        val = if s <= length(lower_vals)
            lower_vals[s] * V[lower_inds[s]]
        else
            zero(Tv)
        end
        lower_value += val

        s += warpsize()
    end

    lower_value = CUDA.reduce_warp(+, lower_value)
    return lower_value
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

@inline function small_add_gap_mul_V_sparse(value, prob, sum_lower::Tv, lane) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(prob))
    @inbounds remaining = one(Tv) - sum_lower
    gap_value = zero(Tv)

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
    return gap_value
end

function try_large_sparse_bellman!(
    ::Type{T1},
    ::Type{T2},
    workspace::CuSparseWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    stateptr;
    upper_bound = false,
    maximize = true,
) where {Tv, T1, T2}
    # Execution plan:
    # - one state per block
    # - use shared memory to store the values/value_perm and gap probability/gap_perm
    # - use bitonic sort in a block to sort the values 

    shmem =
        workspace.max_nonzeros * (sizeof(T1) + sizeof(T2)) +
        workspace.max_actions * sizeof(Tv)

    kernel = @cuda launch = false large_sparse_bellman_kernel!(
        T1,
        T2,
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        prob,
        stateptr,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv)),
    )

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    if max_threads < 32
        return false
    end

    wanted_threads = min(1024, nextwarp(device(), cld(workspace.max_nonzeros, 2)))

    num_states = length(stateptr) - one(Int32)
    threads = min(max_threads, wanted_threads)
    blocks = min(2^16 - 1, num_states)

    kernel(
        T1,
        T2,
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        prob,
        stateptr,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv));
        blocks = blocks,
        threads = threads,
        shmem = shmem,
    )

    return true
end

function large_sparse_bellman_kernel!(
    ::Type{T1},
    ::Type{T2},
    workspace,
    strategy_cache,
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    stateptr,
    value_lt,
    action_reduce,
) where {Tv, T1, T2}
    action_workspace = CuDynamicSharedArray(Tv, workspace.max_actions)
    value_ws =
        CuDynamicSharedArray(T1, workspace.max_nonzeros, workspace.max_actions * sizeof(Tv))
    gap_ws = CuDynamicSharedArray(
        T2,
        workspace.max_nonzeros,
        workspace.max_nonzeros * sizeof(T1) + workspace.max_actions * sizeof(Tv),
    )

    # Grid-stride loop
    num_states = length(stateptr) - one(Int32)
    j = blockIdx().x
    @inbounds while j <= num_states
        state_sparse_omaximization!(
            action_workspace,
            value_ws,
            gap_ws,
            strategy_cache,
            Vres,
            V,
            prob,
            stateptr,
            value_lt,
            action_reduce,
            j,
        )
        j += gridDim().x
    end

    return nothing
end

@inline function state_sparse_omaximization!(
    action_workspace,
    value_ws,
    gap_ws,
    strategy_cache,
    Vres,
    V,
    prob,
    stateptr,
    value_lt,
    action_reduce,
    jₛ,
)
    wid, lane = fldmod1(threadIdx().x, warpsize())

    s₁, s₂ = stateptr[jₛ], stateptr[jₛ + one(Int32)]
    nactions = s₂ - s₁
    @inbounds action_values = @view action_workspace[1:nactions]

    k = one(Int32)
    @inbounds while k <= nactions
        jₐ = s₁ + k - one(Int32)
        sum_lowerⱼ = sum_lower(prob)[jₐ]

        r = lower(prob).colPtr[jₐ]:(lower(prob).colPtr[jₐ + one(Int32)] - one(Int32))
        lindsⱼ = @view lower(prob).rowVal[r]
        lvalsⱼ = @view lower(prob).nzVal[r]

        r = gap(prob).colPtr[jₐ]:(gap(prob).colPtr[jₐ + one(Int32)] - one(Int32))
        gindsⱼ = @view gap(prob).rowVal[r]
        gvalsⱼ = @view gap(prob).nzVal[r]

        # Use O-maxmization to find the value for the action
        v = state_action_sparse_omaximization!(
            value_ws,
            gap_ws,
            V,
            lindsⱼ,
            lvalsⱼ,
            gindsⱼ,
            gvalsⱼ,
            sum_lowerⱼ,
            value_lt,
            wid,
            lane,
        )

        if threadIdx().x == one(Int32)
            action_values[k] = v
        end
        sync_threads()

        k += one(Int32)
    end

    # Find the best action
    if wid == one(Int32)
        v = extract_strategy_warp!(
            strategy_cache,
            action_values,
            Vres,
            jₛ,
            action_reduce,
            lane,
        )

        if threadIdx().x == one(Int32)
            Vres[jₛ] = v
        end
    end
    sync_threads()
end

@inline function state_sparse_omaximization!(
    action_workspace,
    value_ws,
    gap_ws,
    strategy_cache::NonOptimizingActiveCache,
    Vres,
    V,
    prob,
    stateptr,
    value_lt,
    action_reduce,
    jₛ,
)
    wid, lane = fldmod1(threadIdx().x, warpsize())

    @inbounds begin
        s₁ = stateptr[jₛ]
        jₐ = s₁ + strategy_cache[jₛ] - one(Int32)
        sum_lowerⱼ = sum_lower(prob)[jₐ]

        r = lower(prob).colPtr[jₐ]:(lower(prob).colPtr[jₐ + one(Int32)] - one(Int32))
        lindsⱼ = @view lower(prob).rowVal[r]
        lvalsⱼ = @view lower(prob).nzVal[r]

        r = gap(prob).colPtr[jₐ]:(gap(prob).colPtr[jₐ + one(Int32)] - one(Int32))
        gindsⱼ = @view gap(prob).rowVal[r]
        gvalsⱼ = @view gap(prob).nzVal[r]

        # Use O-maxmization to find the value for the action
        v = state_action_sparse_omaximization!(
            value_ws,
            gap_ws,
            V,
            lindsⱼ,
            lvalsⱼ,
            gindsⱼ,
            gvalsⱼ,
            sum_lowerⱼ,
            value_lt,
            wid,
            lane,
        )

        if threadIdx().x == one(Int32)
            Vres[jₛ] = v
        end
        sync_threads()
    end
end

@inline function state_action_sparse_omaximization!(
    value_ws::AbstractVector{Tv},
    gap_ws::AbstractVector{Tv},
    V,
    lower_inds,
    lower_vals,
    gap_inds,
    gap_vals,
    sum_lower::Tv,
    value_lt,
    wid,
    lane,
) where {Tv}
    reduction_ws = CuStaticSharedArray(Tv, 32)

    value = add_lower_mul_V_block(reduction_ws, V, lower_inds, lower_vals, wid, lane)

    ff_sparse_initialize_sorting_shared_memory!(V, gap_inds, gap_vals, value_ws, gap_ws)

    valueⱼ = @view value_ws[1:length(gap_inds)]
    probⱼ = @view gap_ws[1:length(gap_inds)]
    block_bitonic_sort!(valueⱼ, probⱼ, value_lt)
    value += ff_add_gap_mul_V_sparse(reduction_ws, valueⱼ, probⱼ, sum_lower)

    return value
end

@inline function add_lower_mul_V_block(
    reduction_ws,
    V::AbstractVector{Tv},
    lower_inds,
    lower_vals,
    wid,
    lane,
) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(lower_vals))
    lower_value = zero(Tv)

    s = threadIdx().x
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        val = if s <= length(lower_vals)
            lower_vals[s] * V[lower_inds[s]]
        else
            zero(Tv)
        end
        lower_value += val

        s += blockDim().x
    end

    # Warp-reduction
    lower_value = CUDA.reduce_warp(+, lower_value)
    sync_threads()

    # Block-reduction
    if wid == one(Int32)
        reduction_ws[lane] = zero(Tv)
    end
    sync_threads()

    if lane == one(Int32)
        reduction_ws[wid] = lower_value
    end
    sync_threads()

    if wid == one(Int32)
        lower_value = reduction_ws[lane]
        lower_value = CUDA.reduce_warp(+, lower_value)
    end
    sync_threads()

    return lower_value
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

@inline function ff_add_gap_mul_V_sparse(
    reduction_ws,
    value,
    prob,
    sum_lower::Tv,
) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(prob))
    @inbounds remaining = one(Tv) - sum_lower
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
    if wid == one(Int32)
        reduction_ws[lane] = zero(Tv)
    end
    sync_threads()

    if lane == one(Int32)
        reduction_ws[wid] = gap_value
    end
    sync_threads()

    if wid == one(Int32)
        gap_value = reduction_ws[lane]
        gap_value = CUDA.reduce_warp(+, gap_value)
    end
    sync_threads()

    return gap_value
end

@inline function state_action_sparse_omaximization!(
    value::AbstractVector{Tv},
    perm::AbstractVector{Int32},
    V,
    lower_inds,
    lower_vals,
    gap_inds,
    gap_vals,
    sum_lower::Tv,
    value_lt,
    wid,
    lane,
) where {Tv}
    reduction_ws = CuStaticSharedArray(Tv, 32)

    res = add_lower_mul_V_block(reduction_ws, V, lower_inds, lower_vals, wid, lane)

    fi_sparse_initialize_sorting_shared_memory!(V, gap_inds, value, perm)

    valueⱼ = @view value[1:length(gap_inds)]
    permⱼ = @view perm[1:length(gap_inds)]
    block_bitonic_sort!(valueⱼ, permⱼ, value_lt)
    res += fi_add_gap_mul_V_sparse(reduction_ws, valueⱼ, permⱼ, gap_vals, sum_lower)

    return res
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

@inline function fi_add_gap_mul_V_sparse(
    reduction_ws,
    value,
    perm,
    gapvals,
    sum_lower::Tv,
) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(gapvals))
    @inbounds remaining = one(Tv) - sum_lower
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
    end
    sync_threads()

    return gap_value
end

@inline function state_action_sparse_omaximization!(
    Vperm::AbstractVector{Int32},
    Pperm::AbstractVector{Int32},
    V,
    lower_inds,
    lower_vals,
    gap_inds,
    gap_vals,
    sum_lower::Tv,
    value_lt,
    wid,
    lane,
) where {Tv}
    reduction_ws = CuStaticSharedArray(Tv, 32)

    res = add_lower_mul_V_block(reduction_ws, V, lower_inds, lower_vals, wid, lane)

    ii_sparse_initialize_sorting_shared_memory!(gap_inds, Vperm, Pperm)

    Vpermⱼ = @view Vperm[1:length(gap_inds)]
    Ppermⱼ = @view Pperm[1:length(gap_inds)]
    block_bitonic_sortperm!(V, Vpermⱼ, Ppermⱼ, value_lt)

    res += ii_add_gap_mul_V_sparse(reduction_ws, V, Vpermⱼ, Ppermⱼ, gap_vals, sum_lower)

    return res
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

@inline function ii_add_gap_mul_V_sparse(
    reduction_ws,
    value,
    Vperm,
    Pperm,
    gapvals,
    sum_lower::Tv,
) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(gapvals))
    @inbounds remaining = one(Tv) - sum_lower
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
    end
    sync_threads()

    return gap_value
end
