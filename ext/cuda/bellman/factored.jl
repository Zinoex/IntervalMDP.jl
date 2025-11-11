function IntervalMDP._bellman_helper!(
    workspace::CuFactoredOMaxWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres::AbstractArray{Tv},
    V::AbstractArray{Tv},
    model;
    upper_bound = false,
    maximize = true,
) where {Tv}
    marginal = marginals(model)[1]

    if IntervalMDP.valuetype(marginal) != Tv
        throw(
            ArgumentError(
                "Value type of the model ($(IntervalMDP.valuetype(marginal))) does not match the value type of the input vector ($Tv).",
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
        expectation_cache_size = 2 * sum(workspace.max_support_per_marginal[1:end - 1]) *
            warps * sizeof(Tv)
        last_expectation_cache_size = 2 * workspace.max_support_per_marginal[end] * sizeof(Tv)
        action_cache_size =
            n_actions * sizeof(Tv)
        return expectation_cache_size +
            last_expectation_cache_size +
            action_cache_size
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
    warps = min(warps, workspace.max_support_per_marginal[end])
    threads = warps * 32
    n_states = IntervalMDP.num_source(model)
    blocks = min(2^16 - 1, n_states)
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
    @inbounds action_workspace = initialize_factored_action_workspace(workspace, strategy_cache, model)

    # Prepare sorting shared memory
    @inbounds value_ws, gap_ws = initialize_factored_value_and_gap(workspace, strategy_cache, model)

    @inbounds factored_omaximization!(
        workspace,
        strategy_cache,
        Vres,
        V,
        model,
        action_workspace,
        value_ws,
        gap_ws,
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
    action_workspace = CuDynamicSharedArray(
        IntervalMDP.valuetype(model),
        action_shape(model),
    )

    return action_workspace
end

Base.@propagate_inbounds function initialize_factored_action_workspace(
    workspace,
    ::NonOptimizingActiveCache,
    model,
)
    return nothing
end

Base.@propagate_inbounds function initialize_factored_value_and_gap(
    workspace,
    ::OptimizingActiveCache,
    model::IntervalMDP.FactoredRMDP{N, M},
) where {N, M}
    assume(warpsize() == 32)

    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    Tv = IntervalMDP.valuetype(model)

    size = sum(workspace.max_support_per_marginal[1:end - 1])
    ws = CuDynamicSharedArray(Tv, (size, Int32(2), nwarps), num_actions(model) * sizeof(Tv))
    final_ws = CuDynamicSharedArray(Tv, (workspace.max_support_per_marginal[end], Int32(2)), num_actions(model) * sizeof(Tv) + size * Int32(2) * nwarps * sizeof(Tv))

    value_ws = @view(ws[:, Int32(1), wid])
    gap_ws = @view(ws[:, Int32(2), wid])

    final_value_ws = @view(final_ws[:, Int32(1)])
    final_gap_ws = @view(final_ws[:, Int32(2)])

    return (value_ws, final_value_ws), (gap_ws, final_gap_ws)
end

Base.@propagate_inbounds function initialize_factored_value_and_gap(
    workspace,
    ::NonOptimizingActiveCache,
    model::IntervalMDP.FactoredRMDP{N, M},
) where {N, M}
    assume(warpsize() == 32)

    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    Tv = IntervalMDP.valuetype(model)

    size = sum(workspace.max_support_per_marginal)
    ws = CuDynamicSharedArray(Tv, (size, Int32(2), nwarps))
    final_ws = CuDynamicSharedArray(Tv, (workspace.max_support_per_marginal[end], Int32(2)), size * Int32(2) * nwarps * sizeof(Tv))

    value_ws = @view(ws[:, Int32(1), wid])
    gap_ws = @view(ws[:, Int32(2), wid])

    final_value_ws = @view(final_ws[:, Int32(1)])
    final_gap_ws = @view(final_ws[:, Int32(2)])

    return (value_ws, final_value_ws), (gap_ws, final_gap_ws)
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
    value_lt,
    action_reduce,
) where {N, M}
    n_states = IntervalMDP.num_source(model)

    jₛ = blockIdx().x
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
            I,
            value_lt,
            action_reduce,
        )
        
        sync_threads()
        jₛ += gridDim().x
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
            jₛ,
            I,
            value_lt,
        )

        if threadIdx().x == one(Int32)
            action_workspace[jₐ] = v
        end

        jₐ += one(Int32)
    end

    sync_threads()
    v = extract_strategy_block!(strategy_cache, action_workspace, jₛ, action_reduce)
    
    if threadIdx().x == one(Int32)
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
        jₛ,
        jₐ,
        value_lt,
    )
    
    if threadIdx().x == one(Int32)
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
    first_ambiguity_set = model[one(Int32)][jₐ, jₛ]

    isparse = fld1(threadIdx().x, warpsize())  # wid
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
        
        isparse += div(blockDim().x, warpsize())  # nwarps
    end

    # Final layer reduction
    value = zero(Tv)
    if div(threadIdx().x, warpsize()) == one(Int32) # wid == 1
        # Only one warp does the reduction, as it is typically so small (<32-64)
        # that block synchronization overhead is not worth it, compared to warp shuffles.
        ambiguity_set = model[Int32(2)][jₐ, jₛ]
        factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[end])
        value = add_lower_mul_V_norem_warp(value_ws[end], ambiguity_set)

        warp_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
        value += small_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], bdgts[end])
    end
    sync_threads()

    return value
end

Base.@propagate_inbounds function state_action_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{3, M},
    value_ws,
    gap_ws,
    jₛ,
    jₐ,
    value_lt,
) where {M, Tv}
    bdgts = budgets(model, jₐ, jₛ)
    ssz = supportsizes(model, jₐ, jₛ)

    value_ws = view_supportsizes(value_ws, ssz)
    gap_ws = view_supportsizes(gap_ws, ssz)

    # Pre-compute the first two ambiguity sets, as they are used by far the most
    first_ambiguity_set = model[one(Int32)][jₐ, jₛ]
    second_ambiguity_set = model[Int32(2)][jₐ, jₛ]

    isparse_last = fld1(threadIdx().x, warpsize())  # wid
    while isparse_last <= ssz[end]

        isparse_inner = one(Int32)
        while isparse_inner <= ssz[Int32(2)]
            Isparse = (isparse_inner, isparse_last)
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

            isparse_inner += one(Int32)

            # For the remaining dimensions, if "full", compute expectation and store in the next level
            # The loop over dimensions is unrolled for performance, as N is known at compile time and
            # GPU compiler fails at compiling the loop if N > 3.
            if Isparse[Int32(1)] < ssz[Int32(2)]
                continue
            end

            factored_initialize_warp_sorting_shared_memory!(second_ambiguity_set, gap_ws[Int32(2)])
            v = add_lower_mul_V_norem_warp(value_ws[Int32(2)], second_ambiguity_set)

            warp_bitonic_sort!(value_ws[Int32(2)], gap_ws[Int32(2)], value_lt)
            v += small_add_gap_mul_V_sparse(value_ws[Int32(2)], gap_ws[Int32(2)], bdgts[Int32(2)])

            if laneid() == one(Int32)
                value_ws[Int(3)][Isparse[Int32(2)]] = v
            end
            sync_warp()
        end

        isparse_last += div(blockDim().x, warpsize())  # nwarps
    end

    # Final layer reduction
    value = zero(Tv)
    if div(threadIdx().x, warpsize()) == one(Int32) # wid == 1
        # Only one warp does the reduction, as it is typically so small (<32-64)
        # that block synchronization overhead is not worth it, compared to warp shuffles.
        ambiguity_set = model[Int32(3)][jₐ, jₛ]
        factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[end])
        value = add_lower_mul_V_norem_warp(value_ws[end], ambiguity_set)

        warp_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
        value += small_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], bdgts[end])
    end
    sync_threads()

    return value
end

Base.@propagate_inbounds function state_action_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{4, M},
    value_ws,
    gap_ws,
    jₛ,
    jₐ,
    value_lt,
) where {M, Tv}
    bdgts = budgets(model, jₐ, jₛ)
    ssz = supportsizes(model, jₐ, jₛ)

    value_ws = view_supportsizes(value_ws, ssz)
    gap_ws = view_supportsizes(gap_ws, ssz)

    # Pre-compute the first two ambiguity sets, as they are used by far the most
    first_ambiguity_set = model[one(Int32)][jₐ, jₛ]
    second_ambiguity_set = model[Int32(2)][jₐ, jₛ]

    isparse_last = fld1(threadIdx().x, warpsize())  # wid
    while isparse_last <= ssz[end]

        isparse_inner = (one(Int32), one(Int32))
        done = false
        while !done
            Isparse = (isparse_inner..., isparse_last)
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
            
            isparse_inner, done = gpu_nextind(ssz[Int32(2):end - one(Int32)], isparse_inner)

            # For the remaining dimensions, if "full", compute expectation and store in the next level
            # The loop over dimensions is unrolled for performance, as N is known at compile time and
            # GPU compiler fails at compiling the loop if N > 3.
            if Isparse[Int32(1)] < ssz[Int32(2)]
                continue
            end

            factored_initialize_warp_sorting_shared_memory!(second_ambiguity_set, gap_ws[Int32(2)])
            v = add_lower_mul_V_norem_warp(value_ws[Int32(2)], second_ambiguity_set)

            warp_bitonic_sort!(value_ws[Int32(2)], gap_ws[Int32(2)], value_lt)
            v += small_add_gap_mul_V_sparse(value_ws[Int32(2)], gap_ws[Int32(2)], bdgts[Int32(2)])

            if laneid() == one(Int32)
                value_ws[Int(3)][Isparse[Int32(2)]] = v
            end
            sync_warp()

            if Isparse[Int32(2)] < ssz[Int32(3)]
                continue
            end

            ambiguity_set = model[Int32(3)][jₐ, jₛ]
            factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[Int32(3)])
            v = add_lower_mul_V_norem_warp(value_ws[Int32(3)], ambiguity_set)

            warp_bitonic_sort!(value_ws[Int32(3)], gap_ws[Int32(3)], value_lt)
            v += small_add_gap_mul_V_sparse(value_ws[Int32(3)], gap_ws[Int32(3)], bdgts[Int32(3)])

            if laneid() == one(Int32)
                value_ws[Int(4)][Isparse[Int32(3)]] = v
            end
            sync_warp()
        end

        isparse_last += div(blockDim().x, warpsize())  # nwarps
    end

    # Final layer reduction
    value = zero(Tv)
    if div(threadIdx().x, warpsize()) == one(Int32) # wid == 1
        # Only one warp does the reduction, as it is typically so small (<32-64)
        # that block synchronization overhead is not worth it, compared to warp shuffles.
        ambiguity_set = model[Int32(4)][jₐ, jₛ]
        factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[end])
        value = add_lower_mul_V_norem_warp(value_ws[end], ambiguity_set)

        warp_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
        value += small_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], bdgts[end])
    end
    sync_threads()

    return value
end

Base.@propagate_inbounds function state_action_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{5, M},
    value_ws,
    gap_ws,
    jₛ,
    jₐ,
    value_lt,
) where {M, Tv}
    bdgts = budgets(model, jₐ, jₛ)
    ssz = supportsizes(model, jₐ, jₛ)

    value_ws = view_supportsizes(value_ws, ssz)
    gap_ws = view_supportsizes(gap_ws, ssz)

    # Pre-compute the first two ambiguity sets, as they are used by far the most
    first_ambiguity_set = model[one(Int32)][jₐ, jₛ]
    second_ambiguity_set = model[Int32(2)][jₐ, jₛ]

    isparse_last = fld1(threadIdx().x, warpsize())  # wid
    while isparse_last <= ssz[end]

        isparse_inner = (one(Int32), one(Int32), one(Int32))
        done = false
        while !done
            Isparse = (isparse_inner..., isparse_last)
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
            
            isparse_inner, done = gpu_nextind(ssz[Int32(2):end - one(Int32)], isparse_inner)

            # For the remaining dimensions, if "full", compute expectation and store in the next level
            # The loop over dimensions is unrolled for performance, as N is known at compile time and
            # GPU compiler fails at compiling the loop if N > 3.
            if Isparse[Int32(1)] < ssz[Int32(2)]
                continue
            end

            factored_initialize_warp_sorting_shared_memory!(second_ambiguity_set, gap_ws[Int32(2)])
            v = add_lower_mul_V_norem_warp(value_ws[Int32(2)], second_ambiguity_set)

            warp_bitonic_sort!(value_ws[Int32(2)], gap_ws[Int32(2)], value_lt)
            v += small_add_gap_mul_V_sparse(value_ws[Int32(2)], gap_ws[Int32(2)], bdgts[Int32(2)])

            if laneid() == one(Int32)
                value_ws[Int(3)][Isparse[Int32(2)]] = v
            end
            sync_warp()

            if Isparse[Int32(2)] < ssz[Int32(3)]
                continue
            end

            ambiguity_set = model[Int32(3)][jₐ, jₛ]
            factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[Int32(3)])
            v = add_lower_mul_V_norem_warp(value_ws[Int32(3)], ambiguity_set)

            warp_bitonic_sort!(value_ws[Int32(3)], gap_ws[Int32(3)], value_lt)
            v += small_add_gap_mul_V_sparse(value_ws[Int32(3)], gap_ws[Int32(3)], bdgts[Int32(3)])

            if laneid() == one(Int32)
                value_ws[Int(4)][Isparse[Int32(3)]] = v
            end
            sync_warp()

            if Isparse[Int32(3)] < ssz[Int32(4)]
                continue
            end

            ambiguity_set = model[Int32(4)][jₐ, jₛ]
            factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[Int32(4)])
            v = add_lower_mul_V_norem_warp(value_ws[Int32(4)], ambiguity_set)

            warp_bitonic_sort!(value_ws[Int32(4)], gap_ws[Int32(4)], value_lt)
            v += small_add_gap_mul_V_sparse(value_ws[Int32(4)], gap_ws[Int32(4)], bdgts[Int32(4)])

            if laneid() == one(Int32)
                value_ws[Int(5)][Isparse[Int32(4)]] = v
            end
            sync_warp()
        end

        isparse_last += div(blockDim().x, warpsize())  # nwarps
    end

    # Final layer reduction
    value = zero(Tv)
    if div(threadIdx().x, warpsize()) == one(Int32) # wid == 1
        # Only one warp does the reduction, as it is typically so small (<32-64)
        # that block synchronization overhead is not worth it, compared to warp shuffles.
        ambiguity_set = model[Int32(5)][jₐ, jₛ]
        factored_initialize_warp_sorting_shared_memory!(ambiguity_set, gap_ws[end])
        value = add_lower_mul_V_norem_warp(value_ws[end], ambiguity_set)

        warp_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
        value += small_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], bdgts[end])
    end
    sync_threads()

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
    ws, final_ws = ws
    inds = (one(Int32), one(Int32) + ssz[1])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(final_ws[1:ssz[end]])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{3, <:Int32})
    ws, final_ws = ws
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(final_ws[1:ssz[end]])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{4, <:Int32})
    ws, final_ws = ws
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(final_ws[1:ssz[end]])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{5, <:Int32})
    ws, final_ws = ws
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(final_ws[1:ssz[end]])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{6, <:Int32})
    ws, final_ws = ws
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(ws[inds[5]:(inds[6] - one(Int32))]),
           @view(final_ws[1:ssz[end]])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{7, <:Int32})
    ws, final_ws = ws
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(ws[inds[5]:(inds[6] - one(Int32))]),
           @view(ws[inds[6]:(inds[7] - one(Int32))]),
           @view(final_ws[1:ssz[end]])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{8, <:Int32})
    ws, final_ws = ws
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(ws[inds[5]:(inds[6] - one(Int32))]),
           @view(ws[inds[6]:(inds[7] - one(Int32))]),
           @view(ws[inds[7]:(inds[8] - one(Int32))]),
           @view(final_ws[1:ssz[end]])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{9, <:Int32})
    ws, final_ws = ws
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7] + ssz[8])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(ws[inds[5]:(inds[6] - one(Int32))]),
           @view(ws[inds[6]:(inds[7] - one(Int32))]),
           @view(ws[inds[7]:(inds[8] - one(Int32))]),
           @view(ws[inds[8]:(inds[9] - one(Int32))]),
           @view(final_ws[1:ssz[end]])
end

Base.@propagate_inbounds function view_supportsizes(ws, ssz::NTuple{10, <:Int32})
    ws, final_ws = ws
    inds = (one(Int32), one(Int32) + ssz[1], one(Int32) + ssz[1] + ssz[2], one(Int32) + ssz[1] + ssz[2] + ssz[3], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7] + ssz[8], one(Int32) + ssz[1] + ssz[2] + ssz[3] + ssz[4] + ssz[5] + ssz[6] + ssz[7] + ssz[8] + ssz[9])
    return @view(ws[inds[1]:(inds[2] - one(Int32))]),
           @view(ws[inds[2]:(inds[3] - one(Int32))]),
           @view(ws[inds[3]:(inds[4] - one(Int32))]),
           @view(ws[inds[4]:(inds[5] - one(Int32))]),
           @view(ws[inds[5]:(inds[6] - one(Int32))]),
           @view(ws[inds[6]:(inds[7] - one(Int32))]),
           @view(ws[inds[7]:(inds[8] - one(Int32))]),
           @view(ws[inds[8]:(inds[9] - one(Int32))]),
           @view(ws[inds[9]:(inds[10] - one(Int32))]),
           @view(final_ws[1:ssz[end]])
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

Base.@propagate_inbounds function add_lower_mul_V_norem_block(V::AbstractVector{Tv}, ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}}) where {Tv}
    supportsize = IntervalMDP.supportsize(ambiguity_set)

    lower_value = zero(Tv)

    # Add the lower bound multiplied by the value
    s = threadIdx().x
    while s <= supportsize
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        lower_value += lower(ambiguity_set, s) * V[s]
        s += blockDim().x
    end

    lower_value = reduce_block(+, lower_value, zero(Tv), Val(true))
    sync_threads()

    return lower_value
end

Base.@propagate_inbounds function add_lower_mul_V_norem_block(V::AbstractVector{Tv}, ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}}) where {Tv}
    supportsize = IntervalMDP.supportsize(ambiguity_set)

    lower_value = zero(Tv)
    support = IntervalMDP.support(ambiguity_set)

    # Add the lower bound multiplied by the value
    lower_nonzeros = nonzeros(lower(ambiguity_set))
    s = threadIdx().x
    while s <= supportsize
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        l = lower_nonzeros[s]
        lower_value += l * V[support[s]]
        s += blockDim().x
    end

    lower_value = reduce_block(+, lower_value, zero(Tv), Val(true))
    sync_threads()

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

Base.@propagate_inbounds function factored_initialize_block_sorting_shared_memory!(
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}},
    prob,
) where {Tv}
    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    s = threadIdx().x
    while s <= ssz
        prob[s] = gap(ambiguity_set, s)
        s += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

Base.@propagate_inbounds function factored_initialize_block_sorting_shared_memory!(
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}},
    prob,
) where {Tv}
    # Copy into shared memory
    gap_nonzeros = nonzeros(gap(ambiguity_set))
    s = threadIdx().x
    while s <= length(gap_nonzeros)
        prob[s] = gap_nonzeros[s]
        s += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
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
