
struct CuSparseOrdering{T, R} <: AbstractStateOrdering{T}
    subsets::CuPermutationSubsets{R, T}
    rowvals::CuVector{T}
end

function Adapt.adapt_structure(to::CUDA.CuArrayAdaptor, o::SparseOrdering)
    return CuSparseOrdering(
        adapt(to, o.subsets),
        adapt(to, o.rowvals)
    )
end

function CUDA.unsafe_free!(o::CuSparseOrdering)
    unsafe_free!(o.subsets)
    return
end

function Adapt.adapt_structure(to::CUDA.Adaptor, o::CuSparseOrdering)
    return CuSparseDeviceOrdering(
        adapt(to, o.subsets),
        adapt(to, o.rowvals)
    )
end

struct CuSparseDeviceOrdering{T, R, A} <: AbstractStateOrdering{T}
    subsets::CuDevicePermutationSubsets{T, R, A}
    rowvals::CuDeviceVector{T, A}
end

# Permutations are specific to each state
IMDP.perm(order::CuSparseOrdering, state) = order.subsets[state]
IMDP.perm(order::CuSparseDeviceOrdering, state) = order.subsets[state]

function IMDP.sort_states!(order::CuSparseOrdering, V; max = true)
    populate_value_subsets!(order, V)
    initialize_perm_subsets!(order)
    sort_subsets!(order; max = max)

    return order
end

function populate_value_subsets!(order::CuSparseOrdering, V)
    n = length(order.rowvals)

    threads = 1024
    blocks = ceil(Int32, n / threads)

    @cuda blocks = blocks threads = threads populate_value_subsets_kernel!(order, v)

    return order
end

function populate_value_subsets_kernel!(order::CuSparseDeviceOrdering{T, R, A}, V::CuDeviceVector{R, A}) where {T, R, A}
    thread_id = (blockIdx().x - T(1)) * blockDim().x + threadIdx().x
    n = length(order.rowvals)

    if thread_id <= n
        value_id = order.rowvals[thread_id]
        order.subsets.value_subsets.val[thread_id] = V[value_id]
    end

    return nothing
end

function initialize_perm_subsets!(order::CuSparseOrdering)
    n = length(order.subsets) * warpsize(device())

    threads = 256
    blocks = ceil(Int32, n / threads)

    @cuda blocks = blocks threads = threads initialize_perm_subsets_kernel!(order)

    return order
end

function initialize_perm_subsets_kernel!(order::CuSparseDeviceOrdering{T, R, A}) where {T, R, A}
    assume(warpsize() == 32)

    thread_id = (blockIdx().x - T(1)) * blockDim().x + threadIdx().x
    warp_id, lane = fldmod1(thread_id, warpsize())

    n = length(order.subsets)

    if warp_id <= n
        subset = order.subsets[thread_id]

        i = T(lane)
        while i <= length(subset)
            subset.perm_subset.val[value_id] = i
            i += T(warpsize())
        end
    end

    return nothing
end

function sort_subsets!(order::CuSparseOrdering; max = true)
    n = length(order.subsets) * warpsize(device())

    threads = 256
    blocks = ceil(Int32, n / threads)

    @cuda blocks = blocks threads = threads sort_subsets_kernel!(order, max)

    return order
end

function sort_subsets_kernel!(order::CuSparseDeviceOrdering{T, R, A}, max::Bool) where {T, R, A}
    assume(warpsize() == 32)

    thread_id = (blockIdx().x - T(1)) * blockDim().x + threadIdx().x
    warp_id, lane = fldmod1(thread_id, warpsize())

    n = length(order.subsets)

    if warp_id <= n
        subset = order.subsets[thread_id]

        # Copy to CuDynamicSharedArray

        # Sort the permutation subset by the value subset
    end

    return nothing
end

function IMDP.construct_ordering(T, p::CuSparseMatrixCSC)
    # Assume that input/start state is on the columns and output/target state is on the rows
    n, m = size(p)
    perm = CuArray(collect(UnitRange{T}(1, n)))
    state_to_subset = construct_state_to_subset(T, n)

    subsets = Vector{PermutationSubset{T, Vector{T}}}(undef, m)
    colptr = Vector(p.colPtr)
    nzinds = Vector(SparseArrays.rowvals(p))

    for j in eachindex(subsets)
        nrow = colptr[j + 1] - colptr[j]
        subsets[j] = PermutationSubset(T(1), Vector{T}(undef, nrow))

        # This is an ugly way to get the indices of the nonzero elements in the column
        # but it is necessary as subarray does not support SparseArrays.nonzeroinds.
        ids = @view nzinds[colptr[j]:(colptr[j + 1] - 1)]
        for i in ids
            push!(state_to_subset[i], j)
        end
    end

    state_to_subset = cu(state_to_subset)
    subsets = cu(subsets)

    order = CuSparseOrdering(perm, state_to_subset, subsets)
    return order
end
