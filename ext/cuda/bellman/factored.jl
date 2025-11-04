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
    warps = 2
    threads = warps * 32
    n_states = prod(source_shape(model))
    # blocks = min(2^16 - 1, n_states)
    blocks = 1
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
    model,
    value_lt,
    action_reduce,
) where {Tv}
    
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
        source_shape(model),
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
    model,
    action_workspace,
    value_ws,
    gap_ws,
    value_lt,
    action_reduce,
)    
    indices = CartesianIndices(source_shape(model))
    n_states = num_states(model)

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
    model,
    action_workspace,
    value_ws,
    gap_ws,
    jₛ,
    value_lt,
    action_reduce,
)
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
            action_workspace[I] = v
        end

        sync_threads()
        jₐ += one(Int32)
    end

    # TODO: Reduce over actions
    # if wid == one(Int32)
    #     Vres[jₛ] = reduce(action_reduce, action_workspace[1:n_actions])
    # end

    return nothing
end

@inline function state_action_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{N, M},
    value_ws,
    gap_ws,
    jₛ,
    jₐ,
    value_lt,
) where {N, M, Tv}
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid, lane = fldmod1(threadIdx().x, warpsize())

    ambiguity_sets = ntuple(r -> marginals(model)[r][jₐ, jₛ], N)
    used = sum.(lower.(ambiguity_sets))
    budgets = one(Tv) .- used

    supp = IntervalMDP.support.(ambiguity_sets)
    ssz = IntervalMDP.supportsize.(ambiguity_sets)

    @inbounds value_ws = view.(value_ws, Base.OneTo.(ssz))
    @inbounds gap_ws = view.(gap_ws, Base.OneTo.(ssz))

    j_final = wid
    @inbounds while j_final <= ssz[end]
        for Isparse in CartesianIndices(ssz[2:end - one(Int32)])
            Isparse = Tuple(Isparse)
            I = (getindex.(supp[2:end - one(Int32)], Isparse)..., supp[end][j_final])

            # For the first dimension, we need to copy the values from V
            factored_initialize_warp_sorting_shared_memory!(@view(V[:, I...]), ambiguity_sets[1], value_ws[1], gap_ws[1], lane)
            v = add_lower_mul_V_norem_warp(value_ws[1], ambiguity_sets[1], lane)

            warp_bitonic_sort!(value_ws[1], gap_ws[1], value_lt)
            v += small_add_gap_mul_V_sparse(value_ws[1], gap_ws[1], budgets[1], lane)

            idx = N == 2 ? j_final : Isparse[1]
            if lane == 1
                value_ws[2][idx] = v
            end

            # For the remaining dimensions, if "full", compute expectation and store in the next level
            for d in 2:(length(ambiguity_sets) - one(Int32))
                if Isparse[d - one(Int32)] == ssz[d]
                    factored_initialize_warp_sorting_shared_memory!(ambiguity_sets[d], gap_ws[d], lane)
                    v = add_lower_mul_V_norem_warp(value_ws[1], ambiguity_sets[1], lane)

                    warp_bitonic_sort!(value_ws[1], gap_ws[1], value_lt)
                    v += small_add_gap_mul_V_sparse(value_ws[1], gap_ws[1], budgets[1], lane)

                    idx = d == (length(ambiguity_sets) - one(Int32)) ? j_final : Isparse[d]
                    if lane == 1
                        value_ws[d + one(Int32)][idx] = v
                    end
                else
                    break
                end
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
