function IntervalMDP._bellman_helper!(
    workspace::CuDenseOMaxWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres::AbstractVector{Tv},
    V::AbstractVector{Tv},
    model;
    upper_bound = false,
    maximize = true,
) where {Tv}
    n_actions = isa(strategy_cache, IntervalMDP.OptimizingStrategyCache) ? workspace.num_actions : 1
    marginal = marginals(model)[1]
    n_states = source_shape(marginal)[1]

    if IntervalMDP.valuetype(marginal) != Tv
        throw(ArgumentError("Value type of the model ($(IntervalMDP.valuetype(marginal))) does not match the value type of the input vector ($Tv)."))
    end

    max_states_per_block = 32 # == num_warps
    shmem = length(V) * (sizeof(Int32) + sizeof(Tv)) + max_states_per_block * n_actions * sizeof(Tv)

    kernel = @cuda launch = false dense_bellman_kernel!(
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
    shmem = length(V) * (sizeof(Int32) + sizeof(Tv)) + states_per_block * n_actions * sizeof(Tv)

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
    action_workspace = initialize_action_workspace(workspace, strategy_cache, V)

    # Prepare sorting shared memory
    value, perm = initialize_value_and_perm(workspace, strategy_cache, V, marginal)

    # Perform sorting
    dense_initialize_sorting_shared_memory!(V, value, perm)
    block_bitonic_sort!(value, perm, value_lt)

    # O-maxmization
    dense_omaximization!(
        workspace,
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

@inline function initialize_action_workspace(
    workspace,
    ::OptimizingActiveCache,
    marginal
)
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())
    action_workspace = CuDynamicSharedArray(IntervalMDP.valuetype(marginal), (workspace.num_actions, nwarps))
    @inbounds return @view action_workspace[:, wid]
end

@inline function initialize_action_workspace(
    workspace,
    ::NonOptimizingActiveCache,
    marginal
)
    return nothing
end

@inline function initialize_value_and_perm(
    workspace,
    ::OptimizingActiveCache,
    V::AbstractVector{Tv},
    marginal
) where {Tv}
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    Tv2 = IntervalMDP.valuetype(marginal)
    value = CuDynamicSharedArray(Tv, length(V), workspace.num_actions * nwarps * sizeof(Tv2))
    perm = CuDynamicSharedArray(Int32, length(V), workspace.num_actions * nwarps * sizeof(Tv2) + length(V) * sizeof(Tv))
    return value, perm
end

@inline function initialize_value_and_perm(
    workspace,
    ::NonOptimizingActiveCache,
    V::AbstractVector{Tv},
    marginal
) where {Tv}
    value = CuDynamicSharedArray(Tv, length(V))
    perm = CuDynamicSharedArray(Int32, length(V), length(V) * sizeof(Tv))
    return value, perm
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
    workspace,
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
    @inbounds while jₛ <= source_shape(marginal)[1]  # Grid-stride loop
        state_dense_omaximization!(
            workspace,
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

@inline function state_dense_omaximization!(
    workspace,
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
    assume(warpsize() == 32)
    lane = mod1(threadIdx().x, warpsize())
    nwarps = div(blockDim().x, warpsize())

    jₐ = one(Int32)
    @inbounds while jₐ <= action_shape(marginal)[1]
        ambiguity_set = marginal[(jₐ,), (jₛ,)]

        # Use O-maxmization to find the value for the action
        v = state_action_dense_omaximization!(V, value, perm, ambiguity_set, lane)

        if lane == one(Int32)
            action_workspace[jₐ] = v
        end
        sync_warp()

        jₐ += one(Int32)
    end

    # Find the best action
    v = extract_strategy_warp!(strategy_cache, action_workspace, Vres, jₛ, action_reduce, lane)

    if lane == one(Int32)
        Vres[jₛ] = v
    end
    sync_warp()
end

@inline function state_dense_omaximization!(
    workspace,
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
    lane = mod1(threadIdx().x, warpsize())

    @inbounds begin
        jₐ = Int32.(strategy_cache[jₛ])
        ambiguity_set = marginal[jₐ, (jₛ,)]

        # Use O-maxmization to find the value for the action
        v = state_action_dense_omaximization!(V, value, perm, ambiguity_set, lane)

        if lane == one(Int32)
            Vres[jₛ] = v
        end
        sync_warp()
    end
end

@inline function state_action_dense_omaximization!(
    V,
    value,
    perm,
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{R, MR},
    lane,
) where {R, MR <: AbstractArray}
    assume(warpsize() == 32)

    warp_aligned_length = kernel_nextwarp(IntervalMDP.supportsize(ambiguity_set))
    used = zero(R)
    gap_value = zero(R)

    # Add the lower bound multiplied by the value
    s = lane
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        if s <= IntervalMDP.supportsize(ambiguity_set)
            gap_value += lower(ambiguity_set, s) * V[s]
            used += lower(ambiguity_set, s)
        end
        s += warpsize()
    end
    used = CUDA.reduce_warp(+, used)
    used = shfl_sync(0xffffffff, used, one(Int32))
    remaining = one(R) - used

    # Add the gap multiplied by the value
    s = lane
    @inbounds while s <= warp_aligned_length
        # Find index of the permutation, and lookup the corresponding gap
        g = if s <= IntervalMDP.supportsize(ambiguity_set)
            gap(ambiguity_set, perm[s])
        else
            # 0 gap is a neural element
            zero(R)
        end

        # Cummulatively sum the gap with a tree reduction
        cum_gap = cumsum_warp(g, lane)

        # Update the remaining probability
        remaining -= cum_gap
        remaining += g

        # Update the probability
        if s <= IntervalMDP.supportsize(ambiguity_set)
            g = clamp(remaining, zero(R), g)
            gap_value += g * value[s]
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

    gap_value = CUDA.reduce_warp(+, gap_value)
    return gap_value
end