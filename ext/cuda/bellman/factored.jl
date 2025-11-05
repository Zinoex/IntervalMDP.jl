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
    # - Use the entire block for each action.
    # - Each warp computes the value for one conditional at the top/last level - possibly with a block-stride loop
    # - The reduction in each warp follows the same pattern as the CPU implementation.
    # - The task divergence should be minimal as all warps operate on the same marginal synchronously (including sparsity patterns).
    # - The data divergence should also be minimal for the same reason, with the exception of the first level, which will 
    #   access different ranges of V (global mem). However, since the threads in a warp access contiguous elements of V, this will still be coalesced.

    n_actions =
        isa(strategy_cache, IntervalMDP.OptimizingStrategyCache) ? num_actions(model) : 1

    function shmem_func(threads)
        warps = div(threads, 32)
        expectation_cache_size = 2 * sum(workspace.max_support_per_marginal[1:end - 1]) *
            warps * sizeof(Tv)
        final_level_cache_size =
            2 * workspace.max_support_per_marginal[end] * sizeof(Tv)
        action_cache_size =
            n_actions * sizeof(Tv)
        return expectation_cache_size +
            final_level_cache_size +
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
    action_workspace = initialize_factored_action_workspace(workspace, strategy_cache, model)

    # Prepare sorting shared memory
    value_ws, gap_ws = initialize_factored_value_and_gap(workspace, strategy_cache, model)

    factored_omaximization!(
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

@inline function initialize_factored_action_workspace(
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

@inline function initialize_factored_action_workspace(
    workspace,
    ::NonOptimizingActiveCache,
    model,
)
    return nothing
end

@inline function initialize_factored_value_and_gap(
    workspace,
    ::OptimizingActiveCache,
    model::IntervalMDP.FactoredRMDP{N, M},
) where {N, M}
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    Tv = IntervalMDP.valuetype(model)

    offset = num_actions(model) * sizeof(Tv)

    size = sum(workspace.max_support_per_marginal[1:end-1])
    value = CuDynamicSharedArray(Tv, (size, nwarps), offset)
    offset += size * nwarps * sizeof(Tv)

    gap = CuDynamicSharedArray(Tv, (size, nwarps), offset)
    offset += size * nwarps * sizeof(Tv)

    @inbounds _value = ntuple(i -> @view(value[workspace.workspace_partitioning[i]:(workspace.workspace_partitioning[i + 1] - one(Int32)), wid]), length(workspace.workspace_partitioning) - 2)
    @inbounds _gap = ntuple(i -> @view(gap[workspace.workspace_partitioning[i]:workspace.workspace_partitioning[i + 1] - one(Int32), wid]), length(workspace.workspace_partitioning) - 2)

    value_final = CuDynamicSharedArray(Tv, workspace.max_support_per_marginal[end], offset)
    offset += workspace.max_support_per_marginal[end] * sizeof(Tv)

    gap_final = CuDynamicSharedArray(Tv, workspace.max_support_per_marginal[end], offset)

    return (_value..., value_final), (_gap..., gap_final)
end

@inline function initialize_factored_value_and_gap(
    workspace,
    ::NonOptimizingActiveCache,
    model::IntervalMDP.FactoredRMDP{N, M},
) where {N, M}
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid = fld1(threadIdx().x, warpsize())

    Tv = IntervalMDP.valuetype(model)

    offset = 1

    size = sum(workspace.max_support_per_marginal[1:end-1])
    value = CuDynamicSharedArray(Tv, (size, nwarps), offset)
    offset += size * nwarps * sizeof(Tv)

    gap = CuDynamicSharedArray(TV, (size, nwarps), offset)
    offset += size * nwarps * sizeof(Tv)

    @inbounds _value = ntuple(i -> @view(value[workspace.workspace_partitioning[i]:(workspace.workspace_partitioning[i + 1] - one(Int32)), wid]), length(workspace.workspace_partitioning) - 2)
    @inbounds _gap = ntuple(i -> @view(gap[workspace.workspace_partitioning[i]:workspace.workspace_partitioning[i + 1] - one(Int32), wid]), length(workspace.workspace_partitioning) - 2)

    value_final = CuDynamicSharedArray(Tv, workspace.max_support_per_marginal[end], offset)
    offset += workspace.max_support_per_marginal[end] * sizeof(Tv)

    gap_final = CuDynamicSharedArray(Tv, workspace.max_support_per_marginal[end], offset)

    return (_value..., value_final), (_gap..., gap_final)
end

@inline function factored_omaximization!(
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
    indices = CartesianIndices(source_shape(model))
    n_states = IntervalMDP.num_source(model)

    jₛ = blockIdx().x
    @inbounds while jₛ <= n_states
        I = indices[jₛ]
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

@inline function state_factored_bellman!(
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
    indices = CartesianIndices(action_shape(model))
    n_actions = num_actions(model)

    jₐ = one(Int32)
    while jₐ <= n_actions
        I = indices[jₐ]

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

        sync_threads()
        jₐ += one(Int32)
    end

    v = extract_strategy_block!(strategy_cache, action_workspace, jₛ, action_reduce)
    
    if threadIdx().x == one(Int32)
        Vres[jₛ] = v
    end

    return nothing
end

@inline function state_action_factored_bellman!(
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
    nwarps = div(blockDim().x, warpsize())
    wid, lane = fldmod1(threadIdx().x, warpsize())

    ambiguity_sets = ntuple(r -> marginals(model)[r][jₐ, jₛ], 2)
    used = sum.(lower.(ambiguity_sets))
    budgets = one(Tv) .- used

    supp = IntervalMDP.support(ambiguity_sets[Int32(2)])
    ssz = IntervalMDP.supportsize.(ambiguity_sets)

    @inbounds value_ws = view.(value_ws, Base.OneTo.(ssz))
    @inbounds gap_ws = view.(gap_ws, Base.OneTo.(ssz))

    j_final = wid
    @inbounds while j_final <= ssz[end]
        I = supp[j_final]

        # For the first dimension, we need to copy the values from V
        factored_initialize_warp_sorting_shared_memory!(@view(V[:, I]), ambiguity_sets[one(Int32)], value_ws[one(Int32)], gap_ws[one(Int32)], lane)
        v = add_lower_mul_V_norem_warp(value_ws[one(Int32)], ambiguity_sets[one(Int32)], lane)

        warp_bitonic_sort!(value_ws[one(Int32)], gap_ws[one(Int32)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[one(Int32)], gap_ws[one(Int32)], budgets[one(Int32)], lane)

        if lane == one(Int32)
            value_ws[Int32(2)][j_final] = v
        end

        j_final += nwarps
    end

    sync_threads()

    # Final layer reduction
    value = add_lower_mul_V_norem_block(value_ws[end], ambiguity_sets[end])

    factored_initialize_block_sorting_shared_memory!(ambiguity_sets[end], gap_ws[end])
    block_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
    value += ff_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], budgets[end])

    return value
end

@inline function state_action_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{3, M},
    value_ws,
    gap_ws,
    jₛ,
    jₐ,
    value_lt,
) where {M, Tv}
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid, lane = fldmod1(threadIdx().x, warpsize())

    ambiguity_sets = ntuple(r -> marginals(model)[r][jₐ, jₛ], 3)
    used = sum.(lower.(ambiguity_sets))
    budgets = one(Tv) .- used

    supp = IntervalMDP.support.(ambiguity_sets[Int32(2):end])
    ssz = IntervalMDP.supportsize.(ambiguity_sets)

    @inbounds value_ws = view.(value_ws, Base.OneTo.(ssz))
    @inbounds gap_ws = view.(gap_ws, Base.OneTo.(ssz))

    j_final = wid
    @inbounds while j_final <= ssz[end]
        isparse = one(Int32)
        while isparse <= ssz[Int32(2)]
            Isparse = (isparse, j_final)
            I = getindex.(supp, Isparse)

            # For the first dimension, we need to copy the values from V
            factored_initialize_warp_sorting_shared_memory!(@view(V[:, I...]), ambiguity_sets[one(Int32)], value_ws[one(Int32)], gap_ws[one(Int32)], lane)
            v = add_lower_mul_V_norem_warp(value_ws[one(Int32)], ambiguity_sets[one(Int32)], lane)

            warp_bitonic_sort!(value_ws[one(Int32)], gap_ws[one(Int32)], value_lt)
            v += small_add_gap_mul_V_sparse(value_ws[one(Int32)], gap_ws[one(Int32)], budgets[one(Int32)], lane)

            if lane == one(Int32)
                value_ws[Int32(2)][Isparse[one(Int32)]] = v
            end

            isparse += one(Int32)

            # For the remaining dimensions, if "full", compute expectation and store in the next level
            # The loop over dimensions is unrolled for performance, as N is known at compile time and
            # GPU compiler fails at compiling the loop if N > 3.
            if Isparse[Int32(1)] < ssz[Int32(2)]
                continue
            end

            factored_initialize_warp_sorting_shared_memory!(ambiguity_sets[Int32(2)], gap_ws[Int32(2)], lane)
            v = add_lower_mul_V_norem_warp(value_ws[Int32(2)], ambiguity_sets[Int32(2)], lane)

            warp_bitonic_sort!(value_ws[Int32(2)], gap_ws[Int32(2)], value_lt)
            v += small_add_gap_mul_V_sparse(value_ws[Int32(2)], gap_ws[Int32(2)], budgets[Int32(2)], lane)

            if lane == one(Int32)
                value_ws[Int(3)][Isparse[Int32(2)]] = v
            end
        end

        j_final += nwarps
    end

    sync_threads()

    # Final layer reduction
    value = add_lower_mul_V_norem_block(value_ws[end], ambiguity_sets[end])

    factored_initialize_block_sorting_shared_memory!(ambiguity_sets[end], gap_ws[end])
    block_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
    value += ff_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], budgets[end])

    return value
end

@inline function add_lower_mul_V_norem_warp(V::AbstractVector{Tv}, ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}}, lane) where {Tv}
    assume(warpsize() == 32)

    ssz = IntervalMDP.supportsize(ambiguity_set)
    lower_value = zero(Tv)

    # Add the lower bound multiplied by the value
    s = lane
    @inbounds while s <= ssz
        lower_value += lower(ambiguity_set, s) * V[s]
        s += warpsize()
    end

    lower_value = CUDA.reduce_warp(+, lower_value)

    return lower_value
end

@inline function add_lower_mul_V_norem_warp(V::AbstractVector{Tv}, ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}}, lane) where {Tv}
    assume(warpsize() == 32)

    ssz = IntervalMDP.supportsize(ambiguity_set)
    lower_nonzeros = SparseArrays.nonzeros(lower(ambiguity_set))
    lower_value = zero(Tv)

    # Add the lower bound multiplied by the value
    s = lane
    @inbounds while s <= ssz
        lower_value += lower_nonzeros[s] * V[s]
        s += warpsize()
    end

    lower_value = CUDA.reduce_warp(+, lower_value)

    return lower_value
end

@inline function add_lower_mul_V_norem_block(V::AbstractVector{Tv}, ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}}) where {Tv}
    lower_value = zero(Tv)
    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Add the lower bound multiplied by the value
    s = threadIdx().x
    @inbounds while s <= ssz
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        lower_value += lower(ambiguity_set, s) * V[s]
        s += blockDim().x
    end

    lower_value = CUDA.reduce_block(+, lower_value, zero(Tv), Val(true))

    return lower_value
end

@inline function add_lower_mul_V_norem_block(V::AbstractVector{Tv}, ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}}) where {Tv}
    lower_value = zero(Tv)
    support = IntervalMDP.support(ambiguity_set)
    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Add the lower bound multiplied by the value
    lower_nonzeros = nonzeros(lower(ambiguity_set))
    s = threadIdx().x
    @inbounds while s <= ssz
        # Find index of the permutation, and lookup the corresponding lower bound and multipy by the value
        lower_value += lower_nonzeros[s] * V[s]
        s += blockDim().x
    end

    lower_value = CUDA.reduce_block(+, lower_value, zero(Tv), Val(true))

    return lower_value
end

@inline function factored_initialize_block_sorting_shared_memory!(
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}},
    prob,
) where {Tv}
    support = IntervalMDP.support(ambiguity_set)
    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    s = threadIdx().x
    @inbounds while s <= ssz
        prob[s] = gap(ambiguity_set, s)
        s += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

@inline function factored_initialize_block_sorting_shared_memory!(
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}},
    prob,
) where {Tv}
    support = IntervalMDP.support(ambiguity_set)
    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    gap_nonzeros = nonzeros(gap(ambiguity_set))
    s = threadIdx().x
    @inbounds while s <= ssz
        prob[s] = gap_nonzeros[s]
        s += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

@inline function factored_initialize_warp_sorting_shared_memory!(
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}},
    prob,
    lane
) where {Tv}
    assume(warpsize() == 32)

    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    s = lane
    @inbounds while s <= ssz
        prob[s] = gap(ambiguity_set, s)
        s += warpsize()
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_warp()
end

@inline function factored_initialize_warp_sorting_shared_memory!(
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}},
    prob,
    lane
) where {Tv}
    assume(warpsize() == 32)

    support = IntervalMDP.support(ambiguity_set)
    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    gap_nonzeros = nonzeros(gap(ambiguity_set))
    s = lane
    @inbounds while s <= ssz
        prob[s] = gap_nonzeros[s]
        s += warpsize()
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_warp()
end

@inline function factored_initialize_warp_sorting_shared_memory!(
    V,
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CuDeviceMatrix}},
    value,
    prob,
    lane,
) where {Tv}
    assume(warpsize() == 32)
    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    s = lane
    @inbounds while s <= ssz
        value[s] = V[s]
        prob[s] = gap(ambiguity_set, s)
        s += warpsize()
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_warp()
end

@inline function factored_initialize_warp_sorting_shared_memory!(
    V,
    ambiguity_set::IntervalMDP.IntervalAmbiguitySet{Tv, <:SubArray{Tv, 1, <:CUDA.CUSPARSE.CuSparseDeviceMatrixCSC}},
    value,
    prob,
    lane,
) where {Tv}
    assume(warpsize() == 32)
    support = IntervalMDP.support(ambiguity_set)
    ssz = IntervalMDP.supportsize(ambiguity_set)

    # Copy into shared memory
    gap_nonzeros = gap(ambiguity_set)
    s = lane
    @inbounds while s <= ssz
        value[s] = V[support[s]]
        prob[s] = gap_nonzeros[s]
        s += warpsize()
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_warp()
end
