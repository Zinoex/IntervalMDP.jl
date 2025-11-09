function IntervalMDP._bellman_helper!(
    workspace::CuSparseOMaxWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres::AbstractVector{Tv},
    V::AbstractVector{Tv},
    model;
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
        model;
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
        model;
        upper_bound = upper_bound,
        maximize = maximize,
    )
        return Vres
    end

    # Try if we can fit all values and permutation indices into shared memory (25% less memory relative to (Float64, Float64)) 
    if try_large_sparse_bellman!(
        Tv,
        Int32,
        workspace,
        strategy_cache,
        Vres,
        V,
        model;
        upper_bound = upper_bound,
        maximize = maximize,
    )
        return Vres
    end

    # Try if we can fit two permutation indices into shared memory (50% less memory relative to (Float64, Float64)) 
    if try_large_sparse_bellman!(
        Int32,
        Int32,
        workspace,
        strategy_cache,
        Vres,
        V,
        model;
        upper_bound = upper_bound,
        maximize = maximize,
    )
        return Vres
    end

    # Try if we can fit permutation indices into shared memory (75% less memory relative to (Float64, Float64)) 
    if try_large_sparse_bellman!(
        Int32,
        Nothing,
        workspace,
        strategy_cache,
        Vres,
        V,
        model;
        upper_bound = upper_bound,
        maximize = maximize,
    )
        return Vres
    end

    throw(
        IntervalMDP.OutOfSharedMemory(
            workspace.max_support * sizeof(Int32),
            CUDA.limit(CUDA.LIMIT_SHMEM_SIZE),
        ),
    )
end

function try_small_sparse_bellman!(
    workspace::CuSparseOMaxWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres::AbstractVector{Tv},
    V::AbstractVector{Tv},
    model;
    upper_bound = false,
    maximize = true,
) where {Tv}
    # Execution plan:
    # - one warp per state
    # - squeeze as many states as possible in a block
    # - use shared memory to store the values and gap probability
    # - use bitonic sort in a warp to sort values_gaps

    n_actions =
        isa(strategy_cache, IntervalMDP.OptimizingStrategyCache) ? workspace.num_actions : 1
    marginal = marginals(model)[1]
    n_states = source_shape(marginal)[1]

    if IntervalMDP.valuetype(marginal) != Tv
        throw(
            ArgumentError(
                "Value type of the model ($(IntervalMDP.valuetype(marginal))) does not match the value type of the input vector ($Tv).",
            ),
        )
    end

    kernel = @cuda launch = false small_sparse_bellman_kernel!(
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        marginal,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv)),
    )

    function variable_shmem(threads)
        warps = div(threads, 32)
        return (workspace.max_support + n_actions) * 2 * sizeof(Tv) * warps
    end

    config = launch_configuration(kernel.fun; shmem = variable_shmem)

    max_threads = prevwarp(device(), config.threads)
    if max_threads < 32 * 4  # Need at least 4 warps to hide latency - it is better to use a full block (large_sparse_bellman!(Tv, Tv)) than a few warps
        return false
    end

    threads = max_threads
    warps = div(threads, 32)
    blocks = min(2^16 - 1, cld(n_states, warps))
    shmem = variable_shmem(threads)

    kernel(
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        marginal,
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
    Vres::AbstractVector{Tv},
    V,
    marginal,
    value_lt,
    action_reduce,
) where {Tv}
    assume(warpsize() == 32)

    @inbounds action_workspace =
        initialize_small_sparse_action_workspace(workspace, strategy_cache, marginal)
    @inbounds value_ws, gap_ws =
        initialize_small_sparse_value_and_gap(workspace, strategy_cache, V, marginal)

    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())
    jₛ = wid + (blockIdx().x - one(Int32)) * nwarps
    @inbounds while jₛ <= source_shape(marginal)[1]  # Grid-stride loop
        state_small_sparse_omaximization!(
            action_workspace,
            value_ws,
            gap_ws,
            strategy_cache,
            Vres,
            V,
            marginal,
            value_lt,
            action_reduce,
            jₛ,
        )
        jₛ += gridDim().x * nwarps
    end

    return nothing
end

Base.@propagate_inbounds function initialize_small_sparse_action_workspace(
    workspace,
    ::OptimizingActiveCache,
    marginal,
)
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    action_workspace = CuDynamicSharedArray(
        IntervalMDP.valuetype(marginal),
        (workspace.num_actions, nwarps),
    )
    action_workspace = @view action_workspace[:, wid]

    return action_workspace
end

Base.@propagate_inbounds function initialize_small_sparse_action_workspace(
    workspace,
    ::NonOptimizingActiveCache,
    marginal,
)
    return nothing
end

Base.@propagate_inbounds function initialize_small_sparse_value_and_gap(
    workspace,
    ::OptimizingActiveCache,
    V::AbstractVector{Tv},
    marginal,
) where {Tv}
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())
    Tv2 = IntervalMDP.valuetype(marginal)

    value_ws = CuDynamicSharedArray(
        Tv,
        (workspace.max_support, nwarps),
        workspace.num_actions * nwarps * sizeof(Tv2),
    )
    value_ws = @view value_ws[:, wid]

    gap_ws = CuDynamicSharedArray(
        Tv,
        (workspace.max_support, nwarps),
        workspace.num_actions * nwarps * sizeof(Tv2) +
        workspace.max_support * nwarps * sizeof(Tv),
    )
    gap_ws = @view gap_ws[:, wid]

    return value_ws, gap_ws
end

Base.@propagate_inbounds function initialize_small_sparse_value_and_gap(
    workspace,
    ::NonOptimizingActiveCache,
    V::AbstractVector{Tv},
    marginal,
) where {Tv}
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    value_ws = CuDynamicSharedArray(Tv, (workspace.max_support, nwarps))
    value_ws = @view value_ws[:, wid]

    gap_ws = CuDynamicSharedArray(
        Tv,
        (workspace.max_support, nwarps),
        workspace.max_support * nwarps * sizeof(Tv),
    )
    gap_ws = @view gap_ws[:, wid]
    return value_ws, gap_ws
end

Base.@propagate_inbounds function state_small_sparse_omaximization!(
    action_workspace,
    value_ws,
    gap_ws,
    strategy_cache,
    Vres,
    V,
    marginal,
    value_lt,
    action_reduce,
    jₛ,
)
    jₐ = one(Int32)
    while jₐ <= action_shape(marginal)[1]
        ambiguity_set = marginal[(jₐ,), (jₛ,)]

        # Use O-maxmization to find the value for the action
        v = state_action_small_sparse_omaximization!(
            value_ws,
            gap_ws,
            V,
            ambiguity_set,
            value_lt,
        )

        if laneid() == one(Int32)
            action_workspace[jₐ] = v
        end
        sync_warp()

        jₐ += one(Int32)
    end

    # Find the best action
    v = extract_strategy_warp!(strategy_cache, action_workspace, jₛ, action_reduce)

    if laneid() == one(Int32)
        Vres[jₛ] = v
    end
    sync_warp()
end

Base.@propagate_inbounds function state_small_sparse_omaximization!(
    action_workspace,
    value_ws,
    gap_ws,
    strategy_cache::NonOptimizingActiveCache,
    Vres,
    V,
    marginal,
    value_lt,
    action_reduce,
    jₛ,
)
    jₐ = Int32.(strategy_cache[jₛ])
    ambiguity_set = marginal[jₐ, (jₛ,)]

    # Use O-maxmization to find the value for the action
    v = state_action_small_sparse_omaximization!(
        value_ws,
        gap_ws,
        V,
        ambiguity_set,
        value_lt,
    )

    if laneid() == one(Int32)
        Vres[jₛ] = v
    end
    sync_warp()
end

Base.@propagate_inbounds function state_action_small_sparse_omaximization!(
    value_ws,
    gap_ws,
    V,
    ambiguity_set,
    value_lt
)
    value_ws = @view value_ws[1:IntervalMDP.supportsize(ambiguity_set)]
    gap_ws = @view gap_ws[1:IntervalMDP.supportsize(ambiguity_set)]

    small_sparse_initialize_sorting_shared_memory!(V, ambiguity_set, value_ws, gap_ws)
    warp_bitonic_sort!(value_ws, gap_ws, value_lt)

    value, remaining = add_lower_mul_V_warp(V, ambiguity_set)
    value += small_add_gap_mul_V_sparse(value_ws, gap_ws, remaining)

    return value
end

# TODO: Make generic
Base.@propagate_inbounds function small_sparse_initialize_sorting_shared_memory!(
    V,
    ambiguity_set,
    value,
    prob,
)
    assume(warpsize() == 32)
    support = IntervalMDP.support(ambiguity_set)

    # Copy into shared memory
    gap_nonzeros = nonzeros(gap(ambiguity_set))
    s = laneid()
    while s <= IntervalMDP.supportsize(ambiguity_set)
        value[s] = V[support[s]]
        prob[s] = gap_nonzeros[s]
        s += warpsize()
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_warp()
end

Base.@propagate_inbounds function add_lower_mul_V_warp(V::AbstractVector{R}, ambiguity_set) where {R}
    assume(warpsize() == 32)
    warp_aligned_length = kernel_nextwarp(IntervalMDP.supportsize(ambiguity_set))

    used = zero(R)
    lower_value = zero(R)
    support = IntervalMDP.support(ambiguity_set)

    # Add the lower bound multiplied by the value
    lower_nonzeros = nonzeros(lower(ambiguity_set))
    s = laneid()
    while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        if s <= IntervalMDP.supportsize(ambiguity_set)
            l = lower_nonzeros[s]
            lower_value += l * V[support[s]]
            used += l
        end
        s += warpsize()
    end
    used = CUDA.reduce_warp(+, used)
    used = shfl_sync(0xffffffff, used, one(Int32))
    remaining = one(R) - used

    lower_value = CUDA.reduce_warp(+, lower_value)

    return lower_value, remaining
end

Base.@propagate_inbounds function small_add_gap_mul_V_sparse(value, prob, remaining::Tv) where {Tv}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(length(prob))
    gap_value = zero(Tv)

    s = laneid()
    while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= length(prob)
            prob[s]
        else
            # 0 gap is a neural element
            zero(Tv)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_warp(g)

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
    workspace::CuSparseOMaxWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres::AbstractVector{Tv},
    V::AbstractVector{Tv},
    model;
    upper_bound = false,
    maximize = true,
) where {Tv, T1, T2}
    # Execution plan:
    # - one state per block
    # - use shared memory to store the values/value_perm and gap probability/gap_perm
    # - use bitonic sort in a block to sort the values

    n_actions =
        isa(strategy_cache, IntervalMDP.OptimizingStrategyCache) ? workspace.num_actions : 1
    marginal = marginals(model)[1]
    n_states = source_shape(marginal)[1]

    shmem = workspace.max_support * (sizeof(T1) + sizeof(T2)) + n_actions * sizeof(Tv)

    if shmem > CUDA.limit(CUDA.LIMIT_SHMEM_SIZE)  # Early exit if we cannot fit into shared memory
        return false
    end

    kernel = @cuda launch = false large_sparse_bellman_kernel!(
        T1,
        T2,
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        marginal,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv)),
    )

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)

    if max_threads < 32
        return false
    end

    wanted_threads = nextwarp(device(), workspace.max_support)
    threads = min(1024, max_threads, wanted_threads)
    blocks = min(2^16 - 1, n_states)

    kernel(
        T1,
        T2,
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        marginal,
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
    Vres::AbstractVector{Tv},
    V,
    marginal,
    value_lt,
    action_reduce,
) where {Tv, T1, T2}
    @inbounds action_workspace =
        initialize_large_sparse_action_workspace(workspace, strategy_cache, marginal)
    @inbounds value_ws, gap_ws = initialize_large_sparse_value_and_gap(
        T1,
        T2,
        workspace,
        strategy_cache,
        V,
        marginal,
    )

    jₛ = blockIdx().x
    @inbounds while jₛ <= source_shape(marginal)[1]  # Grid-stride loop
        state_sparse_omaximization!(
            action_workspace,
            value_ws,
            gap_ws,
            strategy_cache,
            Vres,
            V,
            marginal,
            value_lt,
            action_reduce,
            jₛ,
        )
        jₛ += gridDim().x
    end

    return nothing
end

Base.@propagate_inbounds function initialize_large_sparse_action_workspace(
    workspace,
    ::OptimizingActiveCache,
    marginal,
)
    action_workspace =
        CuDynamicSharedArray(IntervalMDP.valuetype(marginal), workspace.num_actions)
    return action_workspace
end

Base.@propagate_inbounds function initialize_large_sparse_action_workspace(
    workspace,
    ::NonOptimizingActiveCache,
    marginal,
)
    return nothing
end

Base.@propagate_inbounds function initialize_large_sparse_value_and_gap(
    ::Type{T1},
    ::Type{T2},
    workspace,
    ::OptimizingActiveCache,
    V,
    marginal,
) where {T1, T2}
    Tv = IntervalMDP.valuetype(marginal)

    value_ws =
        CuDynamicSharedArray(T1, workspace.max_support, workspace.num_actions * sizeof(Tv))
    gap_ws = CuDynamicSharedArray(
        T2,
        workspace.max_support,
        workspace.num_actions * sizeof(Tv) + workspace.max_support * sizeof(T1),
    )

    return value_ws, gap_ws
end

Base.@propagate_inbounds function initialize_large_sparse_value_and_gap(
    ::Type{T1},
    ::Type{Nothing},
    workspace,
    ::OptimizingActiveCache,
    V,
    marginal,
) where {T1}
    Tv = IntervalMDP.valuetype(marginal)

    value_ws =
        CuDynamicSharedArray(T1, workspace.max_support, workspace.num_actions * sizeof(Tv))

    return value_ws, nothing
end

Base.@propagate_inbounds function initialize_large_sparse_value_and_gap(
    ::Type{T1},
    ::Type{T2},
    workspace,
    ::NonOptimizingActiveCache,
    V,
    marginal,
) where {T1, T2}
    value_ws = CuDynamicSharedArray(T1, workspace.max_support)
    gap_ws =
        CuDynamicSharedArray(T2, workspace.max_support, workspace.max_support * sizeof(T1))

    return value_ws, gap_ws
end

Base.@propagate_inbounds function initialize_large_sparse_value_and_gap(
    ::Type{T1},
    ::Type{Nothing},
    workspace,
    ::NonOptimizingActiveCache,
    V,
    marginal,
) where {T1}
    value_ws = CuDynamicSharedArray(T1, workspace.max_support)

    return value_ws, nothing
end

Base.@propagate_inbounds function state_sparse_omaximization!(
    action_workspace,
    value_ws,
    gap_ws,
    strategy_cache,
    Vres,
    V,
    marginal,
    value_lt,
    action_reduce,
    jₛ,
)
    assume(warpsize() == 32)

    jₐ = one(Int32)
    while jₐ <= action_shape(marginal)[1]
        ambiguity_set = marginal[(jₐ,), (jₛ,)]

        # Use O-maxmization to find the value for the action
        v = state_action_sparse_omaximization!(value_ws, gap_ws, V, ambiguity_set, value_lt)

        if threadIdx().x == one(Int32)
            action_workspace[jₐ] = v
        end
        sync_threads()

        jₐ += one(Int32)
    end

    # Find the best action
    wid = fld1(threadIdx().x, warpsize())
    if wid == one(Int32)
        v = extract_strategy_warp!(
            strategy_cache,
            action_workspace,
            jₛ,
            action_reduce
        )

        if threadIdx().x == one(Int32)
            Vres[jₛ] = v
        end
    end
    sync_threads()
end

Base.@propagate_inbounds function state_sparse_omaximization!(
    action_workspace,
    value_ws,
    gap_ws,
    strategy_cache::NonOptimizingActiveCache,
    Vres,
    V,
    marginal,
    value_lt,
    action_reduce,
    jₛ,
)
    jₐ = Int32.(strategy_cache[jₛ])
    ambiguity_set = marginal[jₐ, (jₛ,)]

    # Use O-maxmization to find the value for the action
    v = state_action_sparse_omaximization!(value_ws, gap_ws, V, ambiguity_set, value_lt)

    if threadIdx().x == one(Int32)
        Vres[jₛ] = v
    end
    sync_threads()
end

Base.@propagate_inbounds function state_action_sparse_omaximization!(
    value_ws::AbstractVector{Tv},
    gap_ws::AbstractVector{Tv},
    V,
    ambiguity_set,
    value_lt,
) where {Tv}
    value_ws = @view value_ws[1:IntervalMDP.supportsize(ambiguity_set)]
    gap_ws = @view gap_ws[1:IntervalMDP.supportsize(ambiguity_set)]

    ff_sparse_initialize_sorting_shared_memory!(V, ambiguity_set, value_ws, gap_ws)
    block_bitonic_sort!(value_ws, gap_ws, value_lt)

    value, remaining = add_lower_mul_V_block(V, ambiguity_set)
    value += ff_add_gap_mul_V_sparse(value_ws, gap_ws, remaining)

    return value
end

Base.@propagate_inbounds function add_lower_mul_V_block(V::AbstractVector{R}, ambiguity_set) where {R}
    share_ws = CuStaticSharedArray(R, 1)

    supportsize = IntervalMDP.supportsize(ambiguity_set)

    used = zero(R)
    lower_value = zero(R)
    support = IntervalMDP.support(ambiguity_set)

    # Add the lower bound multiplied by the value
    lower_nonzeros = nonzeros(lower(ambiguity_set))
    s = threadIdx().x
    while s <= supportsize
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        l = lower_nonzeros[s]
        lower_value += l * V[support[s]]
        used += l
        s += blockDim().x
    end

    used = reduce_block(+, used, zero(R), Val(true))
    lower_value = reduce_block(+, lower_value, zero(R), Val(true))

    if threadIdx().x == one(Int32)
        share_ws[1] = used  # No need to share lower_value since it is only used by the first thread
    end
    sync_threads()

    used = share_ws[1]
    remaining = one(R) - used

    return lower_value, remaining
end

Base.@propagate_inbounds function ff_sparse_initialize_sorting_shared_memory!(V, ambiguity_set, value, prob)
    support = IntervalMDP.support(ambiguity_set)
    supportsize = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    gap_nonzeros = nonzeros(gap(ambiguity_set))
    s = threadIdx().x
    while s <= supportsize
        idx = support[s]
        value[s] = V[idx]
        prob[s] = gap_nonzeros[s]
        s += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

Base.@propagate_inbounds function ff_add_gap_mul_V_sparse(value, prob, remaining::Tv) where {Tv}
    assume(warpsize() == 32)
    wid = fld1(threadIdx().x, warpsize())
    reduction_ws = CuStaticSharedArray(Tv, 32)

    loop_length = nextmult(blockDim().x, length(prob))
    gap_value = zero(Tv)

    # Block-strided loop and save into register `gap_value`
    s = threadIdx().x
    while s <= loop_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= length(prob)
            prob[s]
        else
            # 0 gap is a neural element
            zero(Tv)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_block(g, reduction_ws, wid)

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

    gap_value = reduce_block(+, gap_value, zero(Tv), Val(true))

    return gap_value
end

Base.@propagate_inbounds function state_action_sparse_omaximization!(
    value::AbstractVector{Tv},
    perm::AbstractVector{Int32},
    V,
    ambiguity_set,
    value_lt,
) where {Tv}
    value = @view value[1:IntervalMDP.supportsize(ambiguity_set)]
    perm = @view perm[1:IntervalMDP.supportsize(ambiguity_set)]

    fi_sparse_initialize_sorting_shared_memory!(V, ambiguity_set, value, perm)
    block_bitonic_sort!(value, perm, value_lt)

    value, remaining = add_lower_mul_V_block(V, ambiguity_set)
    value += fi_add_gap_mul_V_sparse(value, perm, ambiguity_set, remaining)

    return value
end

Base.@propagate_inbounds function fi_sparse_initialize_sorting_shared_memory!(V, ambiguity_set, value, perm)
    support = IntervalMDP.support(ambiguity_set)
    supportsize = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    s = threadIdx().x
    while s <= supportsize
        value[s] = V[support[s]]
        perm[s] = s
        s += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

Base.@propagate_inbounds function fi_add_gap_mul_V_sparse(
    value,
    perm,
    ambiguity_set,
    remaining::Tv,
) where {Tv}
    assume(warpsize() == 32)
    wid = fld1(threadIdx().x, warpsize())
    reduction_ws = CuStaticSharedArray(Tv, 32)

    loop_length = nextmult(blockDim().x, IntervalMDP.supportsize(ambiguity_set))
    gap_value = zero(Tv)

    # Block-strided loop and save into register `gap_value`
    gap_nonzeros = nonzeros(gap(ambiguity_set))
    s = threadIdx().x
    while s <= loop_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= length(value)
            gap_nonzeros[perm[s]]
        else
            # 0 gap is a neural element
            zero(Tv)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_block(g, reduction_ws, wid)

        # Update the remaining probability
        remaining -= cum_gap
        remaining += g

        # Update the probability
        if s <= length(value)
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

    gap_value = reduce_block(+, gap_value, zero(Tv), Val(true))

    return gap_value
end

Base.@propagate_inbounds function state_action_sparse_omaximization!(
    Vperm::AbstractVector{Int32},
    Pperm::AbstractVector{Int32},
    V,
    ambiguity_set,
    value_lt,
)
    ii_sparse_initialize_sorting_shared_memory!(ambiguity_set, Vperm, Pperm)

    Vperm = @view Vperm[1:IntervalMDP.supportsize(ambiguity_set)]
    Pperm = @view Pperm[1:IntervalMDP.supportsize(ambiguity_set)]
    block_bitonic_sortperm!(V, Vperm, Pperm, value_lt)

    value, remaining = add_lower_mul_V_block(V, ambiguity_set)
    value += ii_add_gap_mul_V_sparse(V, Vperm, Pperm, ambiguity_set, remaining)

    return value
end

Base.@propagate_inbounds function ii_sparse_initialize_sorting_shared_memory!(ambiguity_set, Vperm, Pperm)
    support = IntervalMDP.support(ambiguity_set)
    supportsize = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    i = threadIdx().x
    while i <= supportsize
        Vperm[i] = support[i]
        Pperm[i] = i
        i += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

Base.@propagate_inbounds function ii_add_gap_mul_V_sparse(
    value,
    Vperm,
    Pperm,
    ambiguity_set,
    remaining::Tv,
) where {Tv}
    assume(warpsize() == 32)
    wid = fld1(threadIdx().x, warpsize())
    reduction_ws = CuStaticSharedArray(Tv, 32)

    loop_length = nextmult(blockDim().x, IntervalMDP.supportsize(ambiguity_set))
    gap_value = zero(Tv)

    # Block-strided loop and save into register `gap_value`
    gap_nonzeros = nonzeros(gap(ambiguity_set))
    s = threadIdx().x
    @inbounds while s <= loop_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= length(Vperm)
            gap_nonzeros[Pperm[s]]
        else
            # 0 gap is a neural element
            zero(Tv)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_block(g, reduction_ws, wid)

        # Update the remaining probability
        remaining -= cum_gap
        remaining += g

        # Update the probability
        if s <= length(Vperm)
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

    gap_value = reduce_block(+, gap_value, zero(Tv), Val(true))

    return gap_value
end

Base.@propagate_inbounds function state_action_sparse_omaximization!(
    perm::AbstractVector{Int32},
    ::Nothing,
    V,
    ambiguity_set,
    value_lt,
)
    perm = @view perm[1:IntervalMDP.supportsize(ambiguity_set)]

    i_sparse_initialize_sorting_shared_memory!(ambiguity_set, perm)
    block_bitonic_sortperm!(V, perm, nothing, value_lt)

    value, remaining = add_lower_mul_V_block(V, ambiguity_set)
    value += i_add_gap_mul_V_sparse(V, perm, ambiguity_set, remaining)

    return value
end

Base.@propagate_inbounds function i_sparse_initialize_sorting_shared_memory!(ambiguity_set, perm)
    support = IntervalMDP.support(ambiguity_set)
    supportsize = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    i = threadIdx().x
    @inbounds while i <= supportsize
        perm[i] = support[i]
        i += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

Base.@propagate_inbounds function i_add_gap_mul_V_sparse(
    value,
    perm,
    ambiguity_set,
    remaining::Tv,
) where {Tv}
    assume(warpsize() == 32)
    wid = fld1(threadIdx().x, warpsize())
    reduction_ws = CuStaticSharedArray(Tv, 32)

    loop_length = nextmult(blockDim().x, IntervalMDP.supportsize(ambiguity_set))
    gap_value = zero(Tv)

    # Block-strided loop and save into register `gap_value`
    s = threadIdx().x
    while s <= loop_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= length(perm)
            gap(ambiguity_set, perm[s])
        else
            # 0 gap is a neural element
            zero(Tv)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_block(g, reduction_ws, wid)

        # Update the remaining probability
        remaining -= cum_gap
        remaining += g

        # Update the probability
        if s <= length(perm)
            sub = clamp(remaining, zero(Tv), g)
            gap_value += sub * value[perm[s]]
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

    gap_value = reduce_block(+, gap_value, zero(Tv), Val(true))

    return gap_value
end
