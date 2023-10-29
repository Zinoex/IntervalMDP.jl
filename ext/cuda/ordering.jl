
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

function Adapt.adapt_structure(to::CUDA.Adaptor, o::CuSparseOrdering)
    return CuSparseDeviceOrdering(adapt(to, o.subsets), adapt(to, o.rowvals))
end

struct CuSparseDeviceOrdering{T, A} <: AbstractStateOrdering{T}
    subsets::CuDeviceVectorOfVector{T, T, A}
    rowvals::CuDeviceVector{T, A}
end

# Permutations are specific to each state
IMDP.perm(order::CuSparseOrdering, state) = order.subsets[state]
IMDP.perm(order::CuSparseDeviceOrdering, state) = order.subsets[state]

function IMDP.sort_states!(order::CuSparseOrdering, V; max = true)
    sort_subsets!(order, V; max = max)

    return order
end

function sort_subsets!(
    order::CuSparseOrdering{Ti},
    V::CuVector{Tv};
    max = true,
) where {Ti, Tv}
    n = length(order.subsets)

    ml = maxlength(order.subsets)

    threads_per_subset = min(256, ispow2(ml) ? ml ÷ Ti(2) : prevpow(Ti(2), ml))

    threads = threads_per_subset
    blocks = min(65536, n)
    shmem = ml * (sizeof(Ti) + sizeof(Tv))

    @cuda blocks = blocks threads = threads shmem = shmem sort_subsets_kernel!(
        order,
        V,
        max,
    )

    return order
end

function sort_subsets_kernel!(
    order::CuSparseDeviceOrdering{Ti, A},
    V::CuDeviceVector{Tv, A},
    max::Bool,
) where {Ti, Tv, A}
    ml = maxlength(order.subsets)

    value = CuDynamicSharedArray(Tv, ml)
    perm = CuDynamicSharedArray(Ti, ml, ml * sizeof(Tv))

    s = blockIdx().x
    while s <= length(order.subsets)  # Grid-stride loop
        subset = order.subsets[s]

        initialize_sorting_shared_memory!(order, subset, V, value, perm)
        bitonic_sort!(subset, value, perm, max)
        copy_perm_to_global!(subset, perm)

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
        value_id = order.rowvals[subset.offset + i - one(Ti)]
        value[i] = V[value_id]
        perm[i] = i
        i += blockDim().x
    end

    # Need to synchronize to make sure all agree on the shared memory
    return sync_threads()
end

@inline function bitonic_sort!(
    subset::CuDeviceVectorInstance{Ti, Ti, A},
    value,
    perm,
    max
) where {Ti, A}
    #### Sort the shared memory with bitonic sort
    subset_length = length(subset)
    nextpow2_subset_length = nextpow(Ti(2), subset_length)

    # Major step
    k = Ti(2)
    while k <= nextpow2_subset_length

        # Minor step - Merge
        k_half = k ÷ Ti(2)
        i = threadIdx().x - Ti(1)
        i_block, i_lane = fld(i, k_half), mod(i, k_half)
        i_concrete = i_block * k + i_lane
        while i_concrete < subset_length
            l = (i_block + one(Ti)) * k - i_lane - one(Ti)

            if l < subset_length # Ensure only one thread in each pair does the swap
                i1 = i_concrete + Ti(1)
                l1 = l + Ti(1)

                if (value[i1] > value[l1]) != max
                    perm[i1], perm[l1] = perm[l1], perm[i1]
                    value[i1], value[l1] = value[l1], value[i1]
                end
            end

            i += blockDim().x
            i_block, i_lane = fld(i, k_half), mod(i, k_half)
            i_concrete = i_block * k + i_lane
        end

        sync_threads()

        # Minor step - Compare and swap
        j = k ÷ Ti(4)
        while j > Ti(0)

            # Compare and swap
            i = threadIdx().x - Ti(1)
            i_block, i_lane = fld(i, j), mod(i, j)
            i_concrete = i_block * j * Ti(2) + i_lane
            while i_concrete < subset_length
                l = i_concrete + j

                if l < subset_length # Ensure only one thread in each pair does the swap
                    i1 = i_concrete + Ti(1)
                    l1 = l + Ti(1)

                    if (value[i1] > value[l1]) != max
                        perm[i1], perm[l1] = perm[l1], perm[i1]
                        value[i1], value[l1] = value[l1], value[i1]
                    end
                end

                i += blockDim().x
                i_block, i_lane = fld(i, j), mod(i, j)
                i_concrete = i_block * j * Ti(2) + i_lane
            end

            j ÷= Ti(2)

            # Synchronize after minor step to make sure all threads agree on the shared memory
            sync_threads()
        end

        k *= Ti(2)
    end
end

@inline function copy_perm_to_global!(subset, perm)
    i = threadIdx().x
    while i <= length(subset)
        subset[i] = perm[i]
        i += blockDim().x
    end

    # Do I need to synchronize here? Now, we do it for
    # safety's sake.
    return sync_threads()
end

function IMDP.construct_ordering(T, p::CuSparseMatrixCSC)
    # Assume that input/start state is on the columns and output/target state is on the rows
    vecptr = CuVector{T}(p.colPtr)
    perm = CuVector{T}(SparseArrays.rowvals(p))
    maxlength = maximum(p.colPtr[2:end] - p.colPtr[1:(end - 1)])

    subsets = CuVectorOfVector{T, T}(vecptr, perm, maxlength)
    rowvals = CuVector{T}(SparseArrays.rowvals(p))

    order = CuSparseOrdering(subsets, rowvals)
    return order
end
