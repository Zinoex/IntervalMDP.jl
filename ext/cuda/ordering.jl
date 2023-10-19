
struct CuSparseOrdering{T} <: AbstractStateOrdering{T}
    perm::CuVector{T}
    state_to_subsets::CuVectorOfVector{T, T}
    subsets::CuPermutationSubsets{T, T}
end

function Adapt.adapt_structure(to::CUDA.CuArrayAdaptor, o::SparseOrdering)
    return CuSparseOrdering(
        adapt(to, o.perm),
        adapt(to, o.state_to_subsets),
        adapt(to, o.subsets),
    )
end

function CUDA.unsafe_free!(o::CuSparseOrdering)
    unsafe_free!(o.perm)
    unsafe_free!(o.state_to_subsets)
    unsafe_free!(o.subsets)
    return
end

function Adapt.adapt_structure(to::CUDA.Adaptor, o::CuSparseOrdering)
    return CuSparseDeviceOrdering(
        adapt(to, o.perm),
        adapt(to, o.state_to_subsets),
        adapt(to, o.subsets),
    )
end

struct CuSparseDeviceOrdering{T, A} <: AbstractStateOrdering{T}
    perm::CuDeviceVector{T, A}
    state_to_subsets::CuDeviceVectorOfVector{T, T, A}
    subsets::CuDevicePermutationSubsets{T, T, A}
end

# Permutations are specific to each state
IMDP.perm(order::CuSparseOrdering, state) = order.subsets[:, state]
IMDP.perm(order::CuSparseDeviceOrdering, state) = order.subsets[:, state]

function IMDP.sort_states!(order::CuSparseOrdering, V; max = true)
    sortperm!(order.perm, V; rev = max)  # rev=true for maximization
    populate_subsets!(order)

    return order
end

function populate_subsets!(order::CuSparseOrdering)
    reset_subsets!(order.subsets)

    n = maxlength(order.state_to_subsets)

    threads = 256
    blocks = ceil(Int64, n / threads)

    @cuda blocks = blocks threads = threads populate_subsets_kernel!(order)

    return order
end

function populate_subsets_kernel!(order::CuSparseDeviceOrdering{T, A}) where {T, A}
    thread_id = (blockIdx().x - T(1)) * blockDim().x + threadIdx().x
    n = maxlength(order.state_to_subsets)

    if thread_id <= n
        for i in order.perm
            start_states = order.state_to_subsets[i]

            if thread_id <= length(start_states)
                j = start_states[thread_id]
                subset = order.subsets[j]
                Base.push!(subset, i)
            end

            # We need to synchronize threads to preserve the order of the subsets
            # At least I think so...
            sync_threads()
        end
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

function construct_state_to_subset(T, n)
    state_to_subset = Vector{Vector{T}}(undef, n)
    for i in eachindex(state_to_subset)
        state_to_subset[i] = T[]
    end

    return state_to_subset
end
