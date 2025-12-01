function IntervalMDP._bellman_helper!(
    workspace::CuFactoredOMaxWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres::AbstractArray{Tv},
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{N, M},
    avail_act;
    upper_bound = false,
    maximize = true,
) where {Tv, N, M}
    if IntervalMDP.valuetype(model) != Tv
        throw(
            ArgumentError(
                "Value type of the model ($(IntervalMDP.valuetype(model))) does not match the value type of the input vector ($Tv).",
            ),
        )
    end

    # Execution plan:
    # - 1 block per state
    # - Use the whole block for each action (i.e., iterate over actions sequentially)
    # - For each action, each warp computes the recursive O-maximization for each index in the last marginal's support
    # - For the last marginal, use block-level reduction to compute the final value for the action
    # - Store the action values in shared memory
    # - Finally, use block-level reduction to find the optimal action and value

    # - The task divergence should be minimal as all warps operate on the same marginal synchronously (including sparsity patterns).
    # - The data divergence should also be minimal for the same reason, with the exception of the first level, which will 
    #   access different ranges of V (global mem). However, since the threads in a warp access contiguous elements of V, this will still be coalesced.

    n_actions =
        isa(strategy_cache, IntervalMDP.OptimizingStrategyCache) ? num_actions(model) : 1

    function shmem_func(threads)
        warps = div(threads, 32)
        prealloc_cache_size = sum(workspace.max_support_per_marginal[1:min(2, N - 1)]) * 2 * sizeof(Tv)
        expectation_cache_size = 2 * sum(workspace.max_support_per_marginal) * sizeof(Tv)
        action_cache_size = n_actions * sizeof(Tv)
        return warps * (prealloc_cache_size +
            expectation_cache_size +
            action_cache_size)
    end

    kernel = @cuda launch = false factored_bellman_kernel!(
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        model,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv)),
    )

    config = launch_configuration(kernel.fun; shmem = shmem_func)
    
    max_threads = prevwarp(device(), config.threads)
    warps = div(max_threads, 32)
    threads = warps * 32
    n_states = IntervalMDP.num_source(model)
    blocks = min(2^16 - 1, fld1(n_states, warps))
    shmem = shmem_func(threads)

    kernel(
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        model,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv));
        blocks = blocks,
        threads = threads,
        shmem = shmem,
    )

    return Vres
end

function factored_bellman_kernel!(
    workspace::CuFactoredOMaxWorkspace,
    strategy_cache,
    Vres::AbstractArray{Tv},
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{N, M},
    value_lt,
    action_reduce,
) where {N, M, Tv}
    
    # Prepare action workspace shared memory
    @inbounds action_workspace, offset = initialize_factored_action_workspace(workspace, strategy_cache, model)

    # Prepare sorting shared memory
    @inbounds value_ws, gap_ws, offset = initialize_factored_value_and_gap(workspace, model, offset)

    # Prepare preallocated workspace for ambiguity sets
    @inbounds prealloc_ws, offset = initialize_prealloc_workspace(workspace, model, offset)

    @inbounds factored_omaximization!(
        workspace,
        strategy_cache,
        Vres,
        V,
        model,
        action_workspace,
        value_ws,
        gap_ws,
        prealloc_ws,
        value_lt,
        action_reduce,
    )

    return nothing
end

Base.@propagate_inbounds function initialize_factored_action_workspace(
    workspace,
    ::OptimizingActiveCache,
    model::IntervalMDP.FactoredRMDP{N, M},
) where {N, M}
    assume(warpsize() == 32)

    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    action_workspace = CuDynamicSharedArray(
        IntervalMDP.valuetype(model),
        (action_shape(model)..., nwarps),
    )

    action_workspace = @view action_workspace[(Colon() for _ in 1:M)..., wid]

    return action_workspace, unsafe_trunc(Int32, sizeof(IntervalMDP.valuetype(model)) * num_actions(model) * nwarps)
end

Base.@propagate_inbounds function initialize_factored_action_workspace(
    workspace,
    ::NonOptimizingActiveCache,
    model,
)
    return nothing, zero(Int32)
end

Base.@propagate_inbounds function initialize_factored_value_and_gap(
    workspace,
    model::IntervalMDP.FactoredRMDP{N, M},
    offset
) where {N, M}
    assume(warpsize() == 32)

    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    Tv = IntervalMDP.valuetype(model)

    size = sum(workspace.max_support_per_marginal)

    value_ws = CuDynamicSharedArray(Tv, (size, nwarps), offset)
    offset += sizeof(Tv) * size * nwarps

    gap_ws = CuDynamicSharedArray(Tv, (size, nwarps), offset)
    offset += sizeof(Tv) * size * nwarps

    value_ws = @view value_ws[:, wid]
    gap_ws = @view gap_ws[:, wid]

    return value_ws, gap_ws, offset
end

Base.@propagate_inbounds function initialize_prealloc_workspace(
    workspace,
    model::IntervalMDP.FactoredRMDP{2, M},
    offset
) where {M}
    ws, offset = initialize_prealloc_workspace(
        workspace.max_support_per_marginal[1],
        marginals(model)[1],
        offset,
    )
    return (ws,), offset
end

Base.@propagate_inbounds function initialize_prealloc_workspace(
    workspace,
    model::IntervalMDP.FactoredRMDP{N, M},
    offset
) where {N, M}
    ws1, offset = initialize_prealloc_workspace(
        workspace.max_support_per_marginal[1],
        marginals(model)[1],
        offset,
    )
    ws2, offset = initialize_prealloc_workspace(
        workspace.max_support_per_marginal[2],
        marginals(model)[2],
        offset,
    )
    return (ws1, ws2), offset
end

Base.@propagate_inbounds function initialize_prealloc_workspace(
    marginal_size,
    marginal::Marginal{<:IntervalAmbiguitySets{Tv, <:CuDeviceMatrix}},
    offset
) where {Tv}
    assume(warpsize() == 32)

    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    value_ws = CuDynamicSharedArray(
        IntervalMDP.valuetype(marginal),
        (marginal_size, nwarps),
        offset,
    )
    offset += sizeof(IntervalMDP.valuetype(marginal)) * marginal_size * nwarps

    gap_ws = CuDynamicSharedArray(
        IntervalMDP.valuetype(marginal),
        (marginal_size, nwarps),
        offset,
    )
    offset += sizeof(IntervalMDP.valuetype(marginal)) * marginal_size * nwarps

    value_ws = @view value_ws[:, wid]
    gap_ws = @view gap_ws[:, wid]

    return (value_ws, gap_ws), offset
end

Base.@propagate_inbounds function initialize_prealloc_workspace(
    marginal_size,
    marginal::Marginal{<:IntervalAmbiguitySets{Tv, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}},
    offset
) where {Tv}
    assume(warpsize() == 32)

    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    value_ws = CuDynamicSharedArray(
        IntervalMDP.valuetype(marginal),
        (marginal_size, nwarps),
        offset,
    )
    offset += sizeof(IntervalMDP.valuetype(marginal)) * marginal_size * nwarps

    gap_ws = CuDynamicSharedArray(
        IntervalMDP.valuetype(marginal),
        (marginal_size, nwarps),
        offset,
    )
    offset += sizeof(IntervalMDP.valuetype(marginal)) * marginal_size * nwarps

    index_ws = CuDynamicSharedArray(
        Int32,
        (marginal_size, nwarps),
        offset,
    )

    value_ws = @view value_ws[:, wid]
    index_ws = @view index_ws[:, wid]

    return (value_ws, gap_ws, index_ws), offset + sizeof(Int32) * marginal_size * nwarps
end

Base.@propagate_inbounds function factored_omaximization!(
    workspace::CuFactoredOMaxWorkspace,
    strategy_cache,
    Vres,
    V,
    model::IntervalMDP.FactoredRMDP{N, M},
    action_workspace,
    value_ws,
    gap_ws,
    prealloc_ws,
    value_lt,
    action_reduce,
) where {N, M}
    n_states = IntervalMDP.num_source(model)
    n_states_per_block = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    jₛ = wid + (blockIdx().x - one(Int32)) * n_states_per_block
    while jₛ <= n_states
        I = ind2sub_gpu(source_shape(model), jₛ)
        state_factored_bellman!(
            workspace,
            strategy_cache,
            Vres,
            V,
            model,
            action_workspace,
            value_ws,
            gap_ws,
            prealloc_ws,
            I,
            value_lt,
            action_reduce,
        )
        
        sync_warp()
        jₛ += gridDim().x * n_states_per_block
    end
end

Base.@propagate_inbounds function state_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    strategy_cache::OptimizingActiveCache,
    Vres,
    V,
    model::IntervalMDP.FactoredRMDP{N, M},
    action_workspace,
    value_ws,
    gap_ws,
    prealloc_ws,
    jₛ,
    value_lt,
    action_reduce,
) where {N, M}
    n_actions = num_actions(model)

    jₐ = one(Int32)
    while jₐ <= n_actions
        I = ind2sub_gpu(action_shape(model), jₐ)

        # Use O-maxmization to find the value for the action
        v = state_action_factored_bellman!(
            workspace,
            V,
            model,
            value_ws,
            gap_ws,
            prealloc_ws,
            jₛ,
            I,
            value_lt,
        )

        if laneid() == one(Int32)
            action_workspace[jₐ] = v
        end

        jₐ += one(Int32)
    end

    sync_warp()
    v = extract_strategy_warp!(strategy_cache, action_workspace, jₛ, action_reduce)
    
    if laneid() == one(Int32)
        Vres[jₛ...] = v
    end

    return nothing
end

Base.@propagate_inbounds function state_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    strategy_cache::NonOptimizingActiveCache,
    Vres,
    V,
    model::IntervalMDP.FactoredRMDP{N, M},
    action_workspace,
    value_ws,
    gap_ws,
    prealloc_ws,
    jₛ,
    value_lt,
    action_reduce,
) where {N, M}
    jₐ = strategy_cache[jₛ...]

    # Use O-maxmization to find the value for the action
    v = state_action_factored_bellman!(
        workspace,
        V,
        model,
        value_ws,
        gap_ws,
        prealloc_ws,
        jₛ,
        jₐ,
        value_lt,
    )
    
    if laneid() == one(Int32)
        Vres[jₛ...] = v
    end

    return nothing
end

Base.@propagate_inbounds function state_action_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{2, M},
    value_ws,
    gap_ws,
    prealloc_ws,
    jₛ,
    jₐ,
    value_lt,
) where {M, Tv}
    assume(warpsize() == 32)

    bdgts = budgets(model, jₐ, jₛ)
    ssz = supportsizes(model, jₐ, jₛ)

    value_ws = view_supportsizes(value_ws, ssz)
    gap_ws = view_supportsizes(gap_ws, ssz)

    # Pre-compute the first ambiguity sets, as it is used by far the most
    first_ambiguity_set = preallocate_ambiguity_set(model[one(Int32)][jₐ, jₛ], prealloc_ws[one(Int32)])

    isparse = one(Int32)
    while isparse <= ssz[end]
        I = supports(model, jₐ, jₛ, isparse)

        # For the first dimension, we need to copy the values from V
        factored_initialize_warp_sorting_shared_memory!(@view(V[:, I]), first_ambiguity_set, value_ws[one(Int32)], gap_ws[one(Int32)])
        v = add_lower_mul_V_norem_warp(value_ws[one(Int32)], first_ambiguity_set)

        warp_bitonic_sort!(value_ws[one(Int32)], gap_ws[one(Int32)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[one(Int32)], gap_ws[one(Int32)], bdgts[one(Int32)])

        if laneid() == one(Int32)
            value_ws[Int32(2)][isparse] = v
        end
        sync_warp()
        
        isparse += one(Int32)
    end

    # Final layer reduction
    ambiguity_set = model[Int32(2)][jₐ, jₛ]
    factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[end])
    value = add_lower_mul_V_norem_warp(value_ws[end], ambiguity_set)

    warp_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
    value += small_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], bdgts[end])

    return value
end

Base.@propagate_inbounds function state_action_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{3, M},
    value_ws,
    gap_ws,
    prealloc_ws,
    jₛ,
    jₐ,
    value_lt,
) where {M, Tv}
    bdgts = budgets(model, jₐ, jₛ)
    ssz = supportsizes(model, jₐ, jₛ)

    value_ws = view_supportsizes(value_ws, ssz)
    gap_ws = view_supportsizes(gap_ws, ssz)

    # Pre-compute the first two ambiguity sets, as they are used by far the most
    first_ambiguity_set = preallocate_ambiguity_set(model[one(Int32)][jₐ, jₛ], prealloc_ws[one(Int32)])
    second_ambiguity_set = preallocate_ambiguity_set(model[Int32(2)][jₐ, jₛ], prealloc_ws[Int32(2)])

    Isparse = (one(Int32), one(Int32))
    done = false
    while !done
        I = supports(model, jₐ, jₛ, Isparse)

        # For the first dimension, we need to copy the values from V
        factored_initialize_warp_sorting_shared_memory!(@view(V[:, I...]), first_ambiguity_set, value_ws[one(Int32)], gap_ws[one(Int32)])
        v = add_lower_mul_V_norem_warp(value_ws[one(Int32)], first_ambiguity_set)

        warp_bitonic_sort!(value_ws[one(Int32)], gap_ws[one(Int32)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[one(Int32)], gap_ws[one(Int32)], bdgts[one(Int32)])

        if laneid() == one(Int32)
            value_ws[Int32(2)][Isparse[Int32(1)]] = v
        end
        sync_warp()

        # For the remaining dimensions, if "full", compute expectation and store in the next level
        # The loop over dimensions is unrolled for performance, as N is known at compile time and
        # GPU compiler fails at compiling the loop if N > 3.
        if Isparse[Int32(1)] < ssz[Int32(2)]
            Isparse, done = gpu_nextind(ssz[Int32(2):end], Isparse)
            continue
        end

        factored_initialize_warp_sorting_shared_memory!(second_ambiguity_set, gap_ws[Int32(2)])
        v = add_lower_mul_V_norem_warp(value_ws[Int32(2)], second_ambiguity_set)

        warp_bitonic_sort!(value_ws[Int32(2)], gap_ws[Int32(2)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[Int32(2)], gap_ws[Int32(2)], bdgts[Int32(2)])

        if laneid() == one(Int32)
            value_ws[Int32(3)][Isparse[Int32(2)]] = v
        end
        sync_warp()

        Isparse, done = gpu_nextind(ssz[Int32(2):end], Isparse)
    end

    # Final layer reduction
    ambiguity_set = model[Int32(3)][jₐ, jₛ]
    factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[end])
    value = add_lower_mul_V_norem_warp(value_ws[end], ambiguity_set)

    warp_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
    value += small_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], bdgts[end])

    return value
end

Base.@propagate_inbounds function state_action_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{4, M},
    value_ws,
    gap_ws,
    prealloc_ws,
    jₛ,
    jₐ,
    value_lt,
) where {M, Tv}
    bdgts = budgets(model, jₐ, jₛ)
    ssz = supportsizes(model, jₐ, jₛ)

    value_ws = view_supportsizes(value_ws, ssz)
    gap_ws = view_supportsizes(gap_ws, ssz)

    # Pre-compute the first two ambiguity sets, as they are used by far the most
    first_ambiguity_set = preallocate_ambiguity_set(model[one(Int32)][jₐ, jₛ], prealloc_ws[one(Int32)])
    second_ambiguity_set = preallocate_ambiguity_set(model[Int32(2)][jₐ, jₛ], prealloc_ws[Int32(2)])

    Isparse = (one(Int32), one(Int32), one(Int32))
    done = false
    while !done
        I = supports(model, jₐ, jₛ, Isparse)

        # For the first dimension, we need to copy the values from V
        factored_initialize_warp_sorting_shared_memory!(@view(V[:, I...]), first_ambiguity_set, value_ws[one(Int32)], gap_ws[one(Int32)])
        v = add_lower_mul_V_norem_warp(value_ws[one(Int32)], first_ambiguity_set)

        warp_bitonic_sort!(value_ws[one(Int32)], gap_ws[one(Int32)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[one(Int32)], gap_ws[one(Int32)], bdgts[one(Int32)])

        if laneid() == one(Int32)
            value_ws[Int32(2)][Isparse[Int32(1)]] = v
        end
        sync_warp()

        # For the remaining dimensions, if "full", compute expectation and store in the next level
        # The loop over dimensions is unrolled for performance, as N is known at compile time and
        # GPU compiler fails at compiling the loop if N > 3.
        if Isparse[Int32(1)] < ssz[Int32(2)]
            Isparse, done = gpu_nextind(ssz[Int32(2):end], Isparse)
            continue
        end

        factored_initialize_warp_sorting_shared_memory!(second_ambiguity_set, gap_ws[Int32(2)])
        v = add_lower_mul_V_norem_warp(value_ws[Int32(2)], second_ambiguity_set)

        warp_bitonic_sort!(value_ws[Int32(2)], gap_ws[Int32(2)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[Int32(2)], gap_ws[Int32(2)], bdgts[Int32(2)])

        if laneid() == one(Int32)
            value_ws[Int32(3)][Isparse[Int32(2)]] = v
        end
        sync_warp()

        if Isparse[Int32(2)] < ssz[Int32(3)]
            Isparse, done = gpu_nextind(ssz[Int32(2):end], Isparse)
            continue
        end

        ambiguity_set = model[Int32(3)][jₐ, jₛ]
        factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[Int32(3)])
        v = add_lower_mul_V_norem_warp(value_ws[Int32(3)], ambiguity_set)

        warp_bitonic_sort!(value_ws[Int32(3)], gap_ws[Int32(3)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[Int32(3)], gap_ws[Int32(3)], bdgts[Int32(3)])

        if laneid() == one(Int32)
            value_ws[Int32(4)][Isparse[Int32(3)]] = v
        end
        sync_warp()

        Isparse, done = gpu_nextind(ssz[Int32(2):end], Isparse)
    end

    # Final layer reduction
    ambiguity_set = model[Int32(4)][jₐ, jₛ]
    factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[end])
    value = add_lower_mul_V_norem_warp(value_ws[end], ambiguity_set)

    warp_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
    value += small_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], bdgts[end])

    return value
end

Base.@propagate_inbounds function state_action_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{5, M},
    value_ws,
    gap_ws,
    prealloc_ws,
    jₛ,
    jₐ,
    value_lt,
) where {M, Tv}
    bdgts = budgets(model, jₐ, jₛ)
    ssz = supportsizes(model, jₐ, jₛ)

    value_ws = view_supportsizes(value_ws, ssz)
    gap_ws = view_supportsizes(gap_ws, ssz)

    # Pre-compute the first two ambiguity sets, as they are used by far the most
    first_ambiguity_set = preallocate_ambiguity_set(model[one(Int32)][jₐ, jₛ], prealloc_ws[one(Int32)])
    second_ambiguity_set = preallocate_ambiguity_set(model[Int32(2)][jₐ, jₛ], prealloc_ws[Int32(2)])

    Isparse = (one(Int32), one(Int32), one(Int32), one(Int32))
    done = false
    while !done
        I = supports(model, jₐ, jₛ, Isparse)

        # For the first dimension, we need to copy the values from V
        factored_initialize_warp_sorting_shared_memory!(@view(V[:, I...]), first_ambiguity_set, value_ws[one(Int32)], gap_ws[one(Int32)])  # 13%
        v = add_lower_mul_V_norem_warp(value_ws[one(Int32)], first_ambiguity_set) # 12%

        warp_bitonic_sort!(value_ws[one(Int32)], gap_ws[one(Int32)], value_lt)  # 44%
        v += small_add_gap_mul_V_sparse(value_ws[one(Int32)], gap_ws[one(Int32)], bdgts[one(Int32)])  # 27%

        if laneid() == one(Int32)
            value_ws[Int32(2)][Isparse[Int32(1)]] = v
        end
        sync_warp()

        # For the remaining dimensions, if "full", compute expectation and store in the next level
        # The loop over dimensions is unrolled for performance, as N is known at compile time and
        # GPU compiler fails at compiling the loop if N > 3.
        if Isparse[Int32(1)] < ssz[Int32(2)]
            Isparse, done = gpu_nextind(ssz[Int32(2):end], Isparse)
            continue
        end

        factored_initialize_warp_sorting_shared_memory!(second_ambiguity_set, gap_ws[Int32(2)])
        v = add_lower_mul_V_norem_warp(value_ws[Int32(2)], second_ambiguity_set)

        warp_bitonic_sort!(value_ws[Int32(2)], gap_ws[Int32(2)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[Int32(2)], gap_ws[Int32(2)], bdgts[Int32(2)])

        if laneid() == one(Int32)
            value_ws[Int32(3)][Isparse[Int32(2)]] = v
        end
        sync_warp()

        if Isparse[Int32(2)] < ssz[Int32(3)]
            Isparse, done = gpu_nextind(ssz[Int32(2):end], Isparse)
            continue
        end

        ambiguity_set = model[Int32(3)][jₐ, jₛ]
        factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[Int32(3)])
        v = add_lower_mul_V_norem_warp(value_ws[Int32(3)], ambiguity_set)

        warp_bitonic_sort!(value_ws[Int32(3)], gap_ws[Int32(3)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[Int32(3)], gap_ws[Int32(3)], bdgts[Int32(3)])

        if laneid() == one(Int32)
            value_ws[Int32(4)][Isparse[Int32(3)]] = v
        end
        sync_warp()

        if Isparse[Int32(3)] < ssz[Int32(4)]
            Isparse, done = gpu_nextind(ssz[Int32(2):end], Isparse)
            continue
        end

        ambiguity_set = model[Int32(4)][jₐ, jₛ]
        factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[Int32(4)])
        v = add_lower_mul_V_norem_warp(value_ws[Int32(4)], ambiguity_set)

        warp_bitonic_sort!(value_ws[Int32(4)], gap_ws[Int32(4)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[Int32(4)], gap_ws[Int32(4)], bdgts[Int32(4)])

        if laneid() == one(Int32)
            value_ws[Int32(5)][Isparse[Int32(4)]] = v
        end
        sync_warp()

        Isparse, done = gpu_nextind(ssz[Int32(2):end], Isparse)
    end

    # Final layer reduction
    ambiguity_set = model[Int32(5)][jₐ, jₛ]
    factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[end])
    value = add_lower_mul_V_norem_warp(value_ws[end], ambiguity_set)

    warp_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
    value += small_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], bdgts[end])

    return value
end

Base.@propagate_inbounds function budgets(
    model::IntervalMDP.FactoredRMDP{2, M},
    jₐ,
    jₛ,
) where {M}
    budgets = budget(model[1][jₐ, jₛ]), budget(model[2][jₐ, jₛ])
    return budgets
end

Base.@propagate_inbounds function budgets(
    model::IntervalMDP.FactoredRMDP{3, M},
    jₐ,
    jₛ,
) where {M}
    budgets = budget(model[1][jₐ, jₛ]), budget(model[2][jₐ, jₛ]), budget(model[3][jₐ, jₛ])
    return budgets
end

Base.@propagate_inbounds function budgets(
    model::IntervalMDP.FactoredRMDP{4, M},
    jₐ,
    jₛ,
) where {M}
    budgets = budget(model[1][jₐ, jₛ]), budget(model[2][jₐ, jₛ]), budget(model[3][jₐ, jₛ]), budget(model[4][jₐ, jₛ])
    return budgets
end

Base.@propagate_inbounds function budgets(
    model::IntervalMDP.FactoredRMDP{5, M},
    jₐ,
    jₛ,
) where {M}
    budgets = budget(model[1][jₐ, jₛ]), budget(model[2][jₐ, jₛ]), budget(model[3][jₐ, jₛ]), budget(model[4][jₐ, jₛ]), budget(model[5][jₐ, jₛ])
    return budgets
end

Base.@propagate_inbounds function budgets(
    model::IntervalMDP.FactoredRMDP{6, M},
    jₐ,
    jₛ,
) where {M}
    budgets = budget(model[1][jₐ, jₛ]), budget(model[2][jₐ, jₛ]), budget(model[3][jₐ, jₛ]), budget(model[4][jₐ, jₛ]), budget(model[5][jₐ, jₛ]), budget(model[6][jₐ, jₛ])
    return budgets
end

Base.@propagate_inbounds function budgets(
    model::IntervalMDP.FactoredRMDP{7, M},
    jₐ,
    jₛ,
) where {M}
    budgets = budget(model[1][jₐ, jₛ]), budget(model[2][jₐ, jₛ]), budget(model[3][jₐ, jₛ]), budget(model[4][jₐ, jₛ]), budget(model[5][jₐ, jₛ]), budget(model[6][jₐ, jₛ]), budget(model[7][jₐ, jₛ])
    return budgets
end

Base.@propagate_inbounds function budgets(
    model::IntervalMDP.FactoredRMDP{8, M},
    jₐ,
    jₛ,
) where {M}
    budgets = budget(model[1][jₐ, jₛ]), budget(model[2][jₐ, jₛ]), budget(model[3][jₐ, jₛ]), budget(model[4][jₐ, jₛ]), budget(model[5][jₐ, jₛ]), budget(model[6][jₐ, jₛ]), budget(model[7][jₐ, jₛ]), budget(model[8][jₐ, jₛ])
    return budgets
end

Base.@propagate_inbounds function budgets(
    model::IntervalMDP.FactoredRMDP{9, M},
    jₐ,
    jₛ,
) where {M}
    budgets = budget(model[1][jₐ, jₛ]), budget(model[2][jₐ, jₛ]), budget(model[3][jₐ, jₛ]), budget(model[4][jₐ, jₛ]), budget(model[5][jₐ, jₛ]), budget(model[6][jₐ, jₛ]), budget(model[7][jₐ, jₛ]), budget(model[8][jₐ, jₛ]), budget(model[9][jₐ, jₛ])
    return budgets
end

Base.@propagate_inbounds function budgets(
    model::IntervalMDP.FactoredRMDP{10, M},
    jₐ,
    jₛ,
) where {M}
    budgets = budget(model[1][jₐ, jₛ]), budget(model[2][jₐ, jₛ]), budget(model[3][jₐ, jₛ]), budget(model[4][jₐ, jₛ]), budget(model[5][jₐ, jₛ]), budget(model[6][jₐ, jₛ]), budget(model[7][jₐ, jₛ]), budget(model[8][jₐ, jₛ]), budget(model[9][jₐ, jₛ]), budget(model[10][jₐ, jₛ])
    return budgets
end

Base.@propagate_inbounds function budget(
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}},
) where {Tv}
    assume(warpsize() == 32)

    ssz = IntervalMDP.supportsize(ambiguity_set)
    used = zero(Tv)

    s = laneid()
    while s <= ssz
        used += lower(ambiguity_set, s)
        s += warpsize()
    end

    # TODO: Figure out why this spills into local memory
    used = reduce_warp(+, used) # Reduce within warp
    used = shfl_sync(0xffffffff, used, one(Int32)) # Broadcast to all lanes
    budget = one(Tv) - used

    return budget
end

Base.@propagate_inbounds function budget(
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}},
) where {Tv}
    used = zero(Tv)
    lower_nonzeros = SparseArrays.nonzeros(lower(ambiguity_set))

    s = laneid()
    while s <= length(lower_nonzeros)
        used += lower_nonzeros[s]
        s += warpsize()
    end

    used = reduce_warp(+, used) # Reduce within warp
    used = shfl_sync(0xffffffff, used, one(Int32)) # Broadcast to all lanes
    budget = one(Tv) - used

    return budget
end

Base.@propagate_inbounds function supportsizes(
    model::IntervalMDP.FactoredRMDP{2, M},
    jₐ,
    jₛ,
) where {M}
    ssz = IntervalMDP.supportsize(model[1][jₐ, jₛ]), IntervalMDP.supportsize(model[2][jₐ, jₛ])
    return ssz
end

Base.@propagate_inbounds function supportsizes(
    model::IntervalMDP.FactoredRMDP{3, M},
    jₐ,
    jₛ,
) where {M}
    ssz = IntervalMDP.supportsize(model[1][jₐ, jₛ]), IntervalMDP.supportsize(model[2][jₐ, jₛ]), IntervalMDP.supportsize(model[3][jₐ, jₛ])
    return ssz
end

Base.@propagate_inbounds function supportsizes(
    model::IntervalMDP.FactoredRMDP{4, M},
    jₐ,
    jₛ,
) where {M}
    ssz = IntervalMDP.supportsize(model[1][jₐ, jₛ]), IntervalMDP.supportsize(model[2][jₐ, jₛ]), IntervalMDP.supportsize(model[3][jₐ, jₛ]), IntervalMDP.supportsize(model[4][jₐ, jₛ])
    return ssz
end

Base.@propagate_inbounds function supportsizes(
    model::IntervalMDP.FactoredRMDP{5, M},
    jₐ,
    jₛ,
) where {M}
    ssz = IntervalMDP.supportsize(model[1][jₐ, jₛ]), IntervalMDP.supportsize(model[2][jₐ, jₛ]), IntervalMDP.supportsize(model[3][jₐ, jₛ]), IntervalMDP.supportsize(model[4][jₐ, jₛ]), IntervalMDP.supportsize(model[5][jₐ, jₛ])
    return ssz
end

Base.@propagate_inbounds function supportsizes(
    model::IntervalMDP.FactoredRMDP{6, M},
    jₐ,
    jₛ,
) where {M}
    ssz = IntervalMDP.supportsize(model[1][jₐ, jₛ]), IntervalMDP.supportsize(model[2][jₐ, jₛ]), IntervalMDP.supportsize(model[3][jₐ, jₛ]), IntervalMDP.supportsize(model[4][jₐ, jₛ]), IntervalMDP.supportsize(model[5][jₐ, jₛ]), IntervalMDP.supportsize(model[6][jₐ, jₛ])
    return ssz
end

Base.@propagate_inbounds function supportsizes(
    model::IntervalMDP.FactoredRMDP{7, M},
    jₐ,
    jₛ,
) where {M}
    ssz = IntervalMDP.supportsize(model[1][jₐ, jₛ]), IntervalMDP.supportsize(model[2][jₐ, jₛ]), IntervalMDP.supportsize(model[3][jₐ, jₛ]), IntervalMDP.supportsize(model[4][jₐ, jₛ]), IntervalMDP.supportsize(model[5][jₐ, jₛ]), IntervalMDP.supportsize(model[6][jₐ, jₛ]), IntervalMDP.supportsize(model[7][jₐ, jₛ])
    return ssz
end

Base.@propagate_inbounds function supportsizes(
    model::IntervalMDP.FactoredRMDP{8, M},
    jₐ,
    jₛ,
) where {M}
    ssz = IntervalMDP.supportsize(model[1][jₐ, jₛ]), IntervalMDP.supportsize(model[2][jₐ, jₛ]), IntervalMDP.supportsize(model[3][jₐ, jₛ]), IntervalMDP.supportsize(model[4][jₐ, jₛ]), IntervalMDP.supportsize(model[5][jₐ, jₛ]), IntervalMDP.supportsize(model[6][jₐ, jₛ]), IntervalMDP.supportsize(model[7][jₐ, jₛ]), IntervalMDP.supportsize(model[8][jₐ, jₛ])
    return ssz
end

Base.@propagate_inbounds function supportsizes(
    model::IntervalMDP.FactoredRMDP{9, M},
    jₐ,
    jₛ,
) where {M}
    ssz = IntervalMDP.supportsize(model[1][jₐ, jₛ]), IntervalMDP.supportsize(model[2][jₐ, jₛ]), IntervalMDP.supportsize(model[3][jₐ, jₛ]), IntervalMDP.supportsize(model[4][jₐ, jₛ]), IntervalMDP.supportsize(model[5][jₐ, jₛ]), IntervalMDP.supportsize(model[6][jₐ, jₛ]), IntervalMDP.supportsize(model[7][jₐ, jₛ]), IntervalMDP.supportsize(model[8][jₐ, jₛ]), IntervalMDP.supportsize(model[9][jₐ, jₛ])
    return ssz
end

Base.@propagate_inbounds function supportsizes(
    model::IntervalMDP.FactoredRMDP{10, M},
    jₐ,
    jₛ,
) where {M}
    ssz = IntervalMDP.supportsize(model[1][jₐ, jₛ]), IntervalMDP.supportsize(model[2][jₐ, jₛ]), IntervalMDP.supportsize(model[3][jₐ, jₛ]), IntervalMDP.supportsize(model[4][jₐ, jₛ]), IntervalMDP.supportsize(model[5][jₐ, jₛ]), IntervalMDP.supportsize(model[6][jₐ, jₛ]), IntervalMDP.supportsize(model[7][jₐ, jₛ]), IntervalMDP.supportsize(model[8][jₐ, jₛ]), IntervalMDP.supportsize(model[9][jₐ, jₛ]), IntervalMDP.supportsize(model[10][jₐ, jₛ])
    return ssz
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{2, <:Int32})
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{3, <:Int32})
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{4, <:Int32})
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{5, <:Int32})
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(ws[inds[5]:(inds[6] - one(Int32))])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{6, <:Int32})
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(ws[inds[5]:(inds[6] - one(Int32))]),
           @view(ws[inds[6]:(inds[7] - one(Int32))])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{7, <:Int32})
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(ws[inds[5]:(inds[6] - one(Int32))]),
           @view(ws[inds[6]:(inds[7] - one(Int32))]),
           @view(ws[inds[7]:(inds[8] - one(Int32))])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{8, <:Int32})
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7] + ssz[8])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(ws[inds[5]:(inds[6] - one(Int32))]),
           @view(ws[inds[6]:(inds[7] - one(Int32))]),
           @view(ws[inds[7]:(inds[8] - one(Int32))]),
           @view(ws[inds[8]:(inds[9] - one(Int32))])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{9, <:Int32})
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7] + ssz[8], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7] + ssz[8] + ssz[9])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(ws[inds[5]:(inds[6] - one(Int32))]),
           @view(ws[inds[6]:(inds[7] - one(Int32))]),
           @view(ws[inds[7]:(inds[8] - one(Int32))]),
           @view(ws[inds[8]:(inds[9] - one(Int32))]),
           @view(ws[inds[9]:(inds[10] - one(Int32))])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{10, <:Int32})
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7] + ssz[8], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7] + ssz[8] + ssz[9], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7] + ssz[8] + ssz[9] + ssz[10])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(ws[inds[5]:(inds[6] - one(Int32))]),
           @view(ws[inds[6]:(inds[7] - one(Int32))]),
           @view(ws[inds[7]:(inds[8] - one(Int32))]),
           @view(ws[inds[8]:(inds[9] - one(Int32))]),
           @view(ws[inds[9]:(inds[10] - one(Int32))]),
           @view(ws[inds[10]:(inds[11] - one(Int32))])
end

Base.@propagate_inbounds function supports(
    model::IntervalMDP.FactoredRMDP{2, M},
    jₐ,
    jₛ,
    isparse::Int32,
) where {M}
    supports = IntervalMDP.support(model[2][jₐ, jₛ], isparse)
    return supports
end

Base.@propagate_inbounds function supports(
    model::IntervalMDP.FactoredRMDP{3, M},
    jₐ,
    jₛ,
    Isparse::NTuple{2, Int32},
) where {M}
    supports = IntervalMDP.support(model[2][jₐ, jₛ], Isparse[1]), IntervalMDP.support(model[3][jₐ, jₛ], Isparse[2])
    return supports
end

Base.@propagate_inbounds function supports(
    model::IntervalMDP.FactoredRMDP{4, M},
    jₐ,
    jₛ,
    Isparse::NTuple{3, Int32},
) where {M}
    supports = IntervalMDP.support(model[2][jₐ, jₛ], Isparse[1]), IntervalMDP.support(model[3][jₐ, jₛ], Isparse[2]), IntervalMDP.support(model[4][jₐ, jₛ], Isparse[3])
    return supports
end

Base.@propagate_inbounds function supports(
    model::IntervalMDP.FactoredRMDP{5, M},
    jₐ,
    jₛ,
    Isparse::NTuple{4, Int32},
) where {M}
    supports = IntervalMDP.support(model[2][jₐ, jₛ], Isparse[1]), IntervalMDP.support(model[3][jₐ, jₛ], Isparse[2]), IntervalMDP.support(model[4][jₐ, jₛ], Isparse[3]), IntervalMDP.support(model[5][jₐ, jₛ], Isparse[4])
    return supports
end

Base.@propagate_inbounds function preallocate_ambiguity_set(
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}},
    ws,
) where {Tv}
    assume(warpsize() == 32)

    lower_ws, gap_ws = ws
    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Add the lower bound multiplied by the value
    s = laneid()
    while s <= ssz
        lower_ws[s] = lower(ambiguity_set, s)
        gap_ws[s] = gap(ambiguity_set, s)
        s += warpsize()
    end
    sync_warp()

    return IntervalMDP.IntervalAmbiguitySet(lower_ws, gap_ws)
end

Base.@propagate_inbounds function add_lower_mul_V_norem_warp(V::AbstractVector{Tv}, ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}}) where {Tv}
    assume(warpsize() == 32)

    ssz = IntervalMDP.supportsize(ambiguity_set)
    lower_value = zero(Tv)

    # Add the lower bound multiplied by the value
    s = laneid()
    while s <= ssz
        lower_value += lower(ambiguity_set, s) * V[s]
        s += warpsize()
    end

    lower_value = reduce_warp(+, lower_value)

    return lower_value
end

Base.@propagate_inbounds function add_lower_mul_V_norem_warp(V::AbstractVector{Tv}, ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}}) where {Tv}
    assume(warpsize() == 32)

    lower_nonzeros = SparseArrays.nonzeros(lower(ambiguity_set))
    lower_value = zero(Tv)

    # Add the lower bound multiplied by the value
    s = laneid()
    while s <= length(lower_nonzeros)
        lower_value += lower_nonzeros[s] * V[s]
        s += warpsize()
    end

    lower_value = reduce_warp(+, lower_value)

    return lower_value
end

Base.@propagate_inbounds function factored_initialize_warp_sorting_shared_memory!(
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}},
    prob,
) where {Tv}
    assume(warpsize() == 32)

    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    s = laneid()
    while s <= ssz
        prob[s] = gap(ambiguity_set, s)
        s += warpsize()
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_warp()
end

Base.@propagate_inbounds function factored_initialize_warp_sorting_shared_memory!(
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}},
    prob,
) where {Tv}
    assume(warpsize() == 32)

    # Copy into shared memory
    gap_nonzeros = nonzeros(gap(ambiguity_set))
    s = laneid()
    while s <= length(gap_nonzeros)
        prob[s] = gap_nonzeros[s]
        s += warpsize()
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_warp()
end

Base.@propagate_inbounds function factored_initialize_warp_sorting_shared_memory!(
    V,
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}},
    value,
    prob,
) where {Tv}
    assume(warpsize() == 32)
    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    s = laneid()
    while s <= ssz
        value[s] = V[s]
        prob[s] = gap(ambiguity_set, s)
        s += warpsize()
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_warp()
end

Base.@propagate_inbounds function factored_initialize_warp_sorting_shared_memory!(
    V,
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}},
    value,
    prob,
) where {Tv}
    assume(warpsize() == 32)
    support = IntervalMDP.support(ambiguity_set)

    # Copy into shared memory
    gap_nonzeros = gap(ambiguity_set)
    s = lane
    while s <= length(gap_nonzeros)
        value[s] = V[support[s]]
        prob[s] = gap_nonzeros[s]
        s += warpsize()
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_warp()
end
