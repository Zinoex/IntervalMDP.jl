
struct CuSparseOrdering{T} <: AbstractStateOrdering{T}
    subsets::CuVectorOfVector{T, T}
    rowvals::CuVector{T}
end
# TODO: Add adaptor from SparseOrdering to CuSparseOrdering

function CUDA.unsafe_free!(o::CuSparseOrdering)
    unsafe_free!(o.subsets)
    unsafe_free!(o.rowvals)
    return
end

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, o::CuSparseOrdering)
    return CuSparseDeviceOrdering(adapt(to, o.subsets), adapt(to, o.rowvals))
end

struct CuSparseDeviceOrdering{T, A} <: AbstractStateOrdering{T}
    subsets::CuDeviceVectorOfVector{T, T, A}
    rowvals::CuDeviceVector{T, A}
end

# Permutations are specific to each state
IntervalMDP.perm(order::CuSparseOrdering, state) = order.subsets[state]
IntervalMDP.perm(order::CuSparseDeviceOrdering, state) = order.subsets[state]

function IntervalMDP.sort_states!(order::CuSparseOrdering, V; max = true)
    sort_subsets!(order, V; max = max)

    return order
end

function sort_subsets!(
    order::CuSparseOrdering{Ti},
    V::CuVector{Tv};
    max = true,
) where {Ti, Tv}
    ml = maxlength(order.subsets)

    shmem = ml * (sizeof(Ti) + sizeof(Tv))

    kernel = @cuda launch = false sort_subsets_kernel!(order, V, max ? (>=) : (<=))

    config = launch_configuration(kernel.fun; shmem = shmem)
    max_threads = prevwarp(device(), config.threads)
    wanted_threads = ceil(Ti, ml / 2)
    threads = min(max_threads, wanted_threads)

    blocks = min(65536, length(order.subsets))

    # println(config, ", ", CUDA.registers(kernel), ", ", CUDA.occupancy(kernel.fun, threads; shmem=shmem))

    kernel(order, V, max; blocks = blocks, threads = threads, shmem = shmem)

    return order
end

function sort_subsets_kernel!(
    order::CuSparseDeviceOrdering{Ti, A},
    V::CuDeviceVector{Tv, A},
    lt,
) where {Ti, Tv, A}
    ml = maxlength(order.subsets)

    value = CuDynamicSharedArray(Tv, ml)
    perm = CuDynamicSharedArray(Ti, ml, ml * sizeof(Tv))

    s = blockIdx().x
    while s <= length(order.subsets)  # Grid-stride loop
        @inbounds subset = order.subsets[s]

        @inline initialize_sorting_shared_memory!(order, subset, V, value, perm)
        @inline bitonic_sort!(subset, value, perm, lt)
        @inline copy_perm_to_global!(subset, perm)

        s += gridDim().x
    end

    return nothing
end

@inline function initialize_sorting_shared_memory!(
    order::CuSparseDeviceOrdering{Ti, A},
    subset,
    V,
    value,
    perm,
) where {Ti, A}
    # Copy into shared memory
    i = threadIdx().x
    while i <= length(subset)
        @inbounds value_id = order.rowvals[subset.offset + i - one(Ti)]
        @inbounds value[i] = V[value_id]
        @inbounds perm[i] = i
        i += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    sync_threads()
end

@inline function bitonic_sort!(
    subset::CuDeviceVectorInstance{Ti, Ti, A},
    value,
    perm,
    lt,
) where {Ti, A}
    #### Sort the shared memory with bitonic sort
    nextpow2_subset_length = nextpow(Ti(2), length(subset))

    k = Ti(2)
    while k <= nextpow2_subset_length
        bitonic_sort_major_step!(subset, value, perm, lt, k)

        k *= Ti(2)
    end
end

@inline function bitonic_sort_major_step!(
    subset::CuDeviceVectorInstance{Ti, Ti, A},
    value,
    perm,
    lt,
    k,
) where {Ti, A}
    j = k ÷ Ti(2)
    @inline bitonic_sort_minor_step!(subset, value, perm, lt, merge_other_lane, j)

    j ÷= Ti(2)
    while j >= Ti(1)
        @inline bitonic_sort_minor_step!(
            subset,
            value,
            perm,
            lt,
            compare_and_swap_other_lane,
            j,
        )
        j ÷= Ti(2)
    end
end

@inline function merge_other_lane(j, lane::Ti) where {Ti}
    mask = create_mask(j)

    return (lane - one(Ti)) ⊻ mask + one(Ti)
end

@inline function create_mask(j::Ti) where {Ti}
    mask = Ti(0)
    while j > Ti(0)
        mask |= j
        j ÷= Ti(2)
    end

    return mask
end

@inline function compare_and_swap_other_lane(j, lane)
    return lane + j
end

@inline function bitonic_sort_minor_step!(
    subset::CuDeviceVectorInstance{Ti, Ti, A},
    value,
    perm,
    lt,
    other_lane,
    j,
) where {Ti, A}
    thread = threadIdx().x
    block, lane = fldmod1(thread, j)
    i = (block - one(Ti)) * j * Ti(2) + lane
    l = (block - one(Ti)) * j * Ti(2) + other_lane(j, lane)

    while i <= length(subset)
        if l <= length(subset) && !lt(value[i], value[l])
            @inbounds perm[i], perm[l] = perm[l], perm[i]
            @inbounds value[i], value[l] = value[l], value[i]
        end

        thread += blockDim().x
        block, lane = fldmod1(thread, j)
        i = (block - one(Ti)) * j * Ti(2) + lane
        l = (block - one(Ti)) * j * Ti(2) + other_lane(j, lane)
    end

    sync_threads()
end

@inline function copy_perm_to_global!(subset, perm)
    i = threadIdx().x
    while i <= length(subset)
        @inbounds subset[i] = perm[i]
        i += blockDim().x
    end
end

function IntervalMDP.construct_ordering(T, p::CuSparseMatrixCSC)
    # Assume that input/start state is on the columns and output/target state is on the rows
    vecptr = CuVector{T}(p.colPtr)
    perm = CuVector{T}(SparseArrays.rowvals(p))
    maxlength = maximum(p.colPtr[2:end] - p.colPtr[1:(end - 1)])

    subsets = CuVectorOfVector{T, T}(vecptr, perm, maxlength)
    rowvals = CuVector{T}(SparseArrays.rowvals(p))

    order = CuSparseOrdering(subsets, rowvals)
    return order
end
