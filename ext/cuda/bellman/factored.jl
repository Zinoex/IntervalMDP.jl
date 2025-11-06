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
    # - Use one warp each action.
    # - The reduction in each warp follows the same pattern as the CPU implementation.

    # TODO: Reason about divergence more carefully.
    # - The task divergence should be minimal as all warps operate on the same marginal synchronously (including sparsity patterns).
    # - The data divergence should also be minimal for the same reason, with the exception of the first level, which will 
    #   access different ranges of V (global mem). However, since the threads in a warp access contiguous elements of V, this will still be coalesced.

    n_actions =
        isa(strategy_cache, IntervalMDP.OptimizingStrategyCache) ? num_actions(model) : 1

    function shmem_func(threads)
        warps = div(threads, 32)
        expectation_cache_size = 2 * sum(workspace.max_support_per_marginal) *
            warps * sizeof(Tv)
        action_cache_size =
            n_actions * sizeof(Tv)
        return expectation_cache_size +
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

    size = sum(workspace.max_support_per_marginal)
    value = CuDynamicSharedArray(Tv, (size, nwarps), num_actions(model) * sizeof(Tv))
    gap = CuDynamicSharedArray(Tv, (size, nwarps), num_actions(model) * sizeof(Tv) + size * nwarps * sizeof(Tv))

    @inbounds _value = ntuple(i -> @view(value[workspace.workspace_partitioning[i]:(workspace.workspace_partitioning[i + 1] - one(Int32)), wid]), length(workspace.max_support_per_marginal))
    @inbounds _gap = ntuple(i -> @view(gap[workspace.workspace_partitioning[i]:workspace.workspace_partitioning[i + 1] - one(Int32), wid]), length(workspace.max_support_per_marginal))

    return _value, _gap
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

    size = sum(workspace.max_support_per_marginal)
    value = CuDynamicSharedArray(Tv, (size, nwarps))
    gap = CuDynamicSharedArray(Tv, (size, nwarps), size * nwarps * sizeof(Tv))

    @inbounds _value = ntuple(i -> @view(value[workspace.workspace_partitioning[i]:(workspace.workspace_partitioning[i + 1] - one(Int32)), wid]), length(workspace.max_support_per_marginal))
    @inbounds _gap = ntuple(i -> @view(gap[workspace.workspace_partitioning[i]:workspace.workspace_partitioning[i + 1] - one(Int32), wid]), length(workspace.max_support_per_marginal))

    return _value, _gap
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
    while jₛ <= n_states
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
    assume(warpsize() == 32)
    nwarps = div(blockDim().x, warpsize())
    wid, lane = fldmod1(threadIdx().x, warpsize())

    indices = CartesianIndices(action_shape(model))
    n_actions = num_actions(model)

    jₐ = wid
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
            lane,
        )

        if lane == one(Int32)
            action_workspace[jₐ] = v
        end

        jₐ += nwarps
    end

    sync_threads()
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
    lane,
) where {M, Tv}
    @inbounds ambiguity_sets = ntuple(r -> marginals(model)[r][jₐ, jₛ], 2)
    budgets = one(Tv) .- sum.(lower.(ambiguity_sets))

    @inbounds supp = IntervalMDP.support(ambiguity_sets[Int32(2)])
    ssz = IntervalMDP.supportsize.(ambiguity_sets)

    @inbounds value_ws = view.(value_ws, Base.OneTo.(ssz))
    @inbounds gap_ws = view.(gap_ws, Base.OneTo.(ssz))

    isparse = one(Int32)
    @inbounds while isparse <= ssz[end]
        I = supp[isparse]

        # For the first dimension, we need to copy the values from V
        factored_initialize_warp_sorting_shared_memory!(@view(V[:, I]), ambiguity_sets[one(Int32)], value_ws[one(Int32)], gap_ws[one(Int32)], lane)
        v = add_lower_mul_V_norem_warp(value_ws[one(Int32)], ambiguity_sets[one(Int32)], lane)

        warp_bitonic_sort!(value_ws[one(Int32)], gap_ws[one(Int32)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[one(Int32)], gap_ws[one(Int32)], budgets[one(Int32)], lane)

        if lane == one(Int32)
            value_ws[Int32(2)][isparse] = v
        end

        sync_warp()
        isparse += one(Int32)
    end

    sync_warp()

    # Final layer reduction
    @inbounds factored_initialize_warp_sorting_shared_memory!(ambiguity_sets[end], gap_ws[end], lane)
    @inbounds value = add_lower_mul_V_norem_warp(value_ws[end], ambiguity_sets[end], lane)

    @inbounds warp_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
    @inbounds value += small_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], budgets[end], lane)

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
    lane,
) where {M, Tv}
    @inbounds ambiguity_sets = ntuple(r -> marginals(model)[r][jₐ, jₛ], 3)
    budgets = one(Tv) .- sum.(lower.(ambiguity_sets))

    @inbounds supp = IntervalMDP.support.(ambiguity_sets[Int32(2):end])
    ssz = IntervalMDP.supportsize.(ambiguity_sets)

    @inbounds value_ws = view.(value_ws, Base.OneTo.(ssz))
    @inbounds gap_ws = view.(gap_ws, Base.OneTo.(ssz))

    Isparse = (one(Int32), one(Int32))
    done = false
    @inbounds while !done
        I = getindex.(supp, Isparse)

        # For the first dimension, we need to copy the values from V
        factored_initialize_warp_sorting_shared_memory!(@view(V[:, I...]), ambiguity_sets[one(Int32)], value_ws[one(Int32)], gap_ws[one(Int32)], lane)
        v = add_lower_mul_V_norem_warp(value_ws[one(Int32)], ambiguity_sets[one(Int32)], lane)

        warp_bitonic_sort!(value_ws[one(Int32)], gap_ws[one(Int32)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[one(Int32)], gap_ws[one(Int32)], budgets[one(Int32)], lane)

        if lane == one(Int32)
            value_ws[Int32(2)][Isparse[one(Int32)]] = v
        end
        sync_warp()

        Isparse_new, done = gpu_nextind(ssz[Int32(2):end], Isparse)

        # For the remaining dimensions, if "full", compute expectation and store in the next level
        # The loop over dimensions is unrolled for performance, as N is known at compile time and
        # GPU compiler fails at compiling the loop if N > 3.
        if Isparse[Int32(1)] < ssz[Int32(2)]
            Isparse = Isparse_new
            continue
        end

        factored_initialize_warp_sorting_shared_memory!(ambiguity_sets[Int32(2)], gap_ws[Int32(2)], lane)
        v = add_lower_mul_V_norem_warp(value_ws[Int32(2)], ambiguity_sets[Int32(2)], lane)

        warp_bitonic_sort!(value_ws[Int32(2)], gap_ws[Int32(2)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[Int32(2)], gap_ws[Int32(2)], budgets[Int32(2)], lane)

        if lane == one(Int32)
            value_ws[Int(3)][Isparse[Int32(2)]] = v
        end
        sync_warp()

        Isparse = Isparse_new
    end

    # Final layer reduction
    @inbounds factored_initialize_warp_sorting_shared_memory!(ambiguity_sets[end], gap_ws[end], lane)
    @inbounds value = add_lower_mul_V_norem_warp(value_ws[end], ambiguity_sets[end], lane)

    @inbounds warp_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
    @inbounds value += small_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], budgets[end], lane)

    return value
end

@inline function state_action_factored_bellman!(
    workspace::CuFactoredOMaxWorkspace,
    V::AbstractArray{Tv},
    model::IntervalMDP.FactoredRMDP{4, M},
    value_ws,
    gap_ws,
    jₛ,
    jₐ,
    value_lt,
    lane,
) where {M, Tv}
    @inbounds ambiguity_sets = ntuple(r -> marginals(model)[r][jₐ, jₛ], 4)
    budgets = one(Tv) .- sum.(lower.(ambiguity_sets))

    @inbounds supp = IntervalMDP.support.(ambiguity_sets[Int32(2):end])
    ssz = IntervalMDP.supportsize.(ambiguity_sets)

    @inbounds value_ws = view.(value_ws, Base.OneTo.(ssz))
    @inbounds gap_ws = view.(gap_ws, Base.OneTo.(ssz))

    Isparse = (one(Int32), one(Int32), one(Int32))
    done = false
    @inbounds while !done
        I = getindex.(supp, Isparse)

        # For the first dimension, we need to copy the values from V
        factored_initialize_warp_sorting_shared_memory!(@view(V[:, I...]), ambiguity_sets[one(Int32)], value_ws[one(Int32)], gap_ws[one(Int32)], lane)
        v = add_lower_mul_V_norem_warp(value_ws[one(Int32)], ambiguity_sets[one(Int32)], lane)

        warp_bitonic_sort!(value_ws[one(Int32)], gap_ws[one(Int32)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[one(Int32)], gap_ws[one(Int32)], budgets[one(Int32)], lane)

        if lane == one(Int32)
            value_ws[Int32(2)][Isparse[one(Int32)]] = v
        end
        sync_warp()
        
        Isparse_new, done = gpu_nextind(ssz[Int32(2):end], Isparse)

        # For the remaining dimensions, if "full", compute expectation and store in the next level
        # The loop over dimensions is unrolled for performance, as N is known at compile time and
        # GPU compiler fails at compiling the loop if N > 3.

        # N == 2
        if Isparse[Int32(1)] < ssz[Int32(2)]
            Isparse = Isparse_new
            continue
        end

        factored_initialize_warp_sorting_shared_memory!(ambiguity_sets[Int32(2)], gap_ws[Int32(2)], lane)
        v = add_lower_mul_V_norem_warp(value_ws[Int32(2)], ambiguity_sets[Int32(2)], lane)

        warp_bitonic_sort!(value_ws[Int32(2)], gap_ws[Int32(2)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[Int32(2)], gap_ws[Int32(2)], budgets[Int32(2)], lane)

        if lane == one(Int32)
            value_ws[Int(3)][Isparse[Int32(2)]] = v
        end
        sync_warp()

        # N == 3
        if Isparse[Int32(2)] < ssz[Int32(3)]
            Isparse = Isparse_new
            continue
        end

        factored_initialize_warp_sorting_shared_memory!(ambiguity_sets[Int32(3)], gap_ws[Int32(3)], lane)
        v = add_lower_mul_V_norem_warp(value_ws[Int32(3)], ambiguity_sets[Int32(3)], lane)

        warp_bitonic_sort!(value_ws[Int32(3)], gap_ws[Int32(3)], value_lt)
        v += small_add_gap_mul_V_sparse(value_ws[Int32(3)], gap_ws[Int32(3)], budgets[Int32(3)], lane)

        if lane == one(Int32)
            value_ws[Int(4)][Isparse[Int32(3)]] = v
        end
        sync_warp()

        Isparse = Isparse_new
    end

    # Final layer reduction
    @inbounds factored_initialize_warp_sorting_shared_memory!(ambiguity_sets[end], gap_ws[end], lane)
    @inbounds value = add_lower_mul_V_norem_warp(value_ws[end], ambiguity_sets[end], lane)

    @inbounds warp_bitonic_sort!(value_ws[end], gap_ws[end], value_lt)
    @inbounds value += small_add_gap_mul_V_sparse(value_ws[end], gap_ws[end], budgets[end], lane)

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
