

struct SparseCudaOrdering{T, VT <: AbstractVector{T}, MT <: AbstractSparseMatrix{T, T}} <: AbstractStateOrdering{T}
    perm::VT
    state_to_subset::MT
    subsets::MT
    ptrs::VT
end

function Adapt.adapt_structure(to::CUDA.Adaptor, x::SparseCudaOrdering)
    return SparseCudaOrdering(
        adapt(to, x.perm),
        adapt(to, x.state_to_subset),
        adapt(to, x.subsets),
        adapt(to, x.ptrs),
    )
end

# Permutations are specific to each state
IMDP.perm(order::SparseCudaOrdering, state) = order.subsets[:, state]

function IMDP.sort_states!(order::SparseCudaOrdering, V; max = true)
    sortperm!(order.perm, V; rev = max)  # rev=true for maximization
    populate_subsets!(order)

    return order
end

function reset_subsets!(order)
    fill!(order.ptrs, 1)
end

function populate_subsets!(order::SparseCudaOrdering)
    reset_subsets!(order)

    n = size(order.state_to_subset, 1)

    threads = 256
    blocks = ceil(Int64, n / threads)

    @cuda blocks=blocks threads=threads populate_subsets_kernel!(order)

    return order
end

function populate_subsets_kernel!(order::SparseCudaOrdering{T}) where {T}
    thread_id = (blockIdx().x - T(1)) * blockDim().x + threadIdx().x
    n = size(order.state_to_subset, 1)

    if thread_id <= n
        colptr = order.state_to_subset.colPtr
        nzs = order.state_to_subset.nzVal

        subsets_colptr = order.subsets.colPtr
        subsets_nz = order.subsets.nzVal

        for i in order.perm
            nrow = colptr[i + T(1)] - colptr[i]

            if thread_id <= nrow
                j = nzs[colptr[i] + thread_id - T(1)]
                ind = subsets_colptr[j] + order.ptrs[j] - T(1)
                subsets_nz[ind] = i
                order.ptrs[j] += T(1)
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

        ids = @view nzinds[colptr[j]:colptr[j + 1] - 1]
        for i in ids
            push!(state_to_subset[i], j)
        end
    end

    max_length = T(maximum(length, state_to_subset))
    state_to_subset = mapreduce(s -> SparseVector(max_length, collect(T, eachindex(s)), s), sparse_hcat, state_to_subset)
    
    subsets = map(s -> s.items, subsets)
    max_length = T(maximum(length, subsets))
    subsets = mapreduce(s -> SparseVector(max_length, collect(T, eachindex(s)), s), sparse_hcat, subsets)
    ptrs = CUDA.ones(T, n)

    state_to_subset = adapt(CuArray, state_to_subset)
    subsets = adapt(CuArray, subsets)

    return SparseCudaOrdering(perm, state_to_subset, subsets, ptrs)
end

function construct_state_to_subset(T, n)
    state_to_subset = Vector{Vector{T}}(undef, n)
    for i in eachindex(state_to_subset)
        state_to_subset[i] = T[]
    end

    return state_to_subset
end