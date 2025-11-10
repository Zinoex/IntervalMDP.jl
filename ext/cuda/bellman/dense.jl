function IntervalMDP._bellman_helper!(
    workspace::CuDenseOMaxWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres::AbstractVector{Tv},
    V::AbstractVector{Tv},
    model;
    upper_bound = false,
    maximize = true,
) where {Tv}
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

    function variable_shmem(threads)
        warps = div(threads, 32)
        return length(V) * (sizeof(Int32) + sizeof(Tv)) + warps * n_actions * sizeof(Tv)
    end        

    kernel = @cuda launch = false dense_bellman_kernel!(
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        marginal,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv)),
    )

    config = launch_configuration(kernel.fun; shmem = variable_shmem)
    max_threads = prevwarp(device(), config.threads)

    # Execution plan:
    # - value assignment: 1 warp per state
    # - reduce over actions
    # - squeeze as many states as possible in a block
    # - use shared memory to store the values and permutation
    # - use bitonic sort to sort the values for all states in a block
    threads_per_state = 32
    states_per_block = min(n_states, div(max_threads, threads_per_state))
    threads = threads_per_state * states_per_block
    blocks = min(2^16 - 1, cld(n_states, states_per_block))
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

    return Vres
end

function dense_bellman_kernel!(
    workspace,
    strategy_cache,
    Vres::AbstractVector{Tv},
    V,
    marginal,
    value_lt,
    action_reduce,
) where {Tv}
    # Prepare action workspace shared memory
    @inbounds action_workspace = initialize_dense_action_workspace(workspace, strategy_cache, V)

    # Prepare sorting shared memory
    @inbounds value, perm = initialize_dense_value_and_perm(workspace, strategy_cache, V, marginal)

    # Perform sorting
    @inbounds dense_initialize_sorting_shared_memory!(V, value, perm)
    @inbounds block_bitonic_sort!(value, perm, value_lt)

    # O-maxmization
    @inbounds dense_omaximization!(
        action_workspace,
        strategy_cache,
        Vres,
        V,
        marginal,
        value,
        perm,
        action_reduce,
    )

    return nothing
end

Base.@propagate_inbounds function initialize_dense_action_workspace(
    workspace,
    ::OptimizingActiveCache,
    V,
)
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())
    action_workspace = CuDynamicSharedArray(
        IntervalMDP.valuetype(V),
        (workspace.num_actions, nwarps),
    )

    action_workspace = @view action_workspace[:, wid]

    return action_workspace
end

Base.@propagate_inbounds function initialize_dense_action_workspace(
    workspace,
    ::NonOptimizingActiveCache,
    V,
)
    return nothing
end

Base.@propagate_inbounds function initialize_dense_value_and_perm(
    workspace,
    ::OptimizingActiveCache,
    V::AbstractVector{Tv},
    marginal,
) where {Tv}
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    Tv2 = IntervalMDP.valuetype(marginal)
    value =
        CuDynamicSharedArray(Tv, length(V), workspace.num_actions * nwarps * sizeof(Tv2))
    perm = CuDynamicSharedArray(
        Int32,
        length(V),
        workspace.num_actions * nwarps * sizeof(Tv2) + length(V) * sizeof(Tv),
    )
    return value, perm
end

Base.@propagate_inbounds function initialize_dense_value_and_perm(
    workspace,
    ::NonOptimizingActiveCache,
    V::AbstractVector{Tv},
    marginal,
) where {Tv}
    value = CuDynamicSharedArray(Tv, length(V))
    perm = CuDynamicSharedArray(Int32, length(V), length(V) * sizeof(Tv))
    return value, perm
end

Base.@propagate_inbounds function dense_initialize_sorting_shared_memory!(V, value, perm)
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

Base.@propagate_inbounds function dense_omaximization!(
    action_workspace,
    strategy_cache,
    Vres,
    V,
    marginal,
    value,
    perm,
    action_reduce,
)
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())
    jₛ = wid + (blockIdx().x - one(Int32)) * nwarps
    while jₛ <= source_shape(marginal)[1]  # Grid-stride loop
        state_dense_omaximization!(
            action_workspace,
            strategy_cache,
            Vres,
            V,
            marginal,
            value,
            perm,
            jₛ,
            action_reduce,
        )
        jₛ += gridDim().x * nwarps
    end

    return nothing
end

Base.@propagate_inbounds function state_dense_omaximization!(
    action_workspace,
    strategy_cache::OptimizingActiveCache,
    Vres::AbstractVector{Tv},
    V,
    marginal,
    value,
    perm,
    jₛ::Int32,
    action_reduce,
) where {Tv}
    jₐ = one(Int32)
    while jₐ <= action_shape(marginal)[1]
        ambiguity_set = marginal[(jₐ,), (jₛ,)]

        # Use O-maxmization to find the value for the action
        v = state_action_dense_omaximization!(V, value, perm, ambiguity_set)

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

Base.@propagate_inbounds function state_dense_omaximization!(
    action_workspace,
    strategy_cache::NonOptimizingActiveCache,
    Vres::AbstractVector{Tv},
    V,
    marginal,
    value,
    perm,
    jₛ::Int32,
    action_reduce,
) where {Tv}
    jₐ = Int32.(strategy_cache[jₛ])
    ambiguity_set = marginal[jₐ, (jₛ,)]

    # Use O-maxmization to find the value for the action
    v = state_action_dense_omaximization!(V, value, perm, ambiguity_set)

    if laneid() == one(Int32)
        Vres[jₛ] = v
    end
    sync_warp()
end

Base.@propagate_inbounds function state_action_dense_omaximization!(
    V::AbstractVector{R},
    value,
    perm,
    ambiguity_set,
) where {R}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(num_target(ambiguity_set))
    used = zero(R)
    res_value = zero(R)

    # Add the lower bound multiplied by the value
    s = laneid()
    while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        if s <= num_target(ambiguity_set)
            l = lower(ambiguity_set, s)
            res_value += l * V[s]
            used += l
        end
        s += warpsize()
    end
    used = reduce_warp(+, used)
    used = shfl_sync(0xffffffff, used, one(Int32))
    remaining = one(R) - used

    # Add the gap multiplied by the value
    s = laneid()
    while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= num_target(ambiguity_set)
            gap(ambiguity_set, perm[s])
        else
            # 0 gap is a neural element
            zero(R)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_warp(g)

        # Update the remaining probability
        remaining -= cum_gap
        remaining += g

        # Update the probability
        if s <= num_target(ambiguity_set)
            g = clamp(remaining, zero(R), g)
            res_value += g * value[s]
            remaining -= g
        end

        # Update the remaining probability from the last thread in the warp
        remaining = shfl_sync(0xffffffff, remaining, warpsize())

        # Early exit if the remaining probability is zero
        if remaining <= zero(R)
            break
        end

        s += warpsize()
    end
    sync_warp()

    res_value = reduce_warp(+, res_value)
    return res_value
end
