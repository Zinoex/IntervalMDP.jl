# Vector of sparse vectors
function construct_ordering(T, p::VVR) where {VVR <: AbstractVector{<:AbstractSparseVector}}
    # Assume that each vector corresponds to a start state

    n, m = length(first(p)), length(p)
    perm = collect(UnitRange{T}(1, n))
    state_to_subset = construct_state_to_subset(T, n)

    subsets = Vector{PermutationSubset{T, Vector{T}}}(undef, m)
    for j in eachindex(subsets)
        subsets[j] = PermutationSubset(1, Vector{T}(undef, nnz(p[j])))

        ids = SparseArrays.nonzeroinds(p[j])  # This is not exported, but we need the non-zero indices
        for i in ids
            push!(state_to_subset[i].index, j)
        end
    end

    return SparseOrdering(perm, state_to_subset, subsets)
end

# Sparse matrix
function construct_ordering(T, p::AbstractSparseMatrix)
    # Assume that input/start state is on the columns and output/target state is on the rows
    n, m = size(p)
    perm = collect(UnitRange{T}(1, n))
    state_to_subset = construct_state_to_subset(T, n)

    subsets = Vector{PermutationSubset{T, Vector{T}}}(undef, m)
    for j in eachindex(subsets)
        pⱼ = view(p, :, j)
        subsets[j] = PermutationSubset(1, Vector{T}(undef, nnz(pⱼ)))

        ids = SparseArrays.nonzeroinds(pⱼ)  # This is not exported, but we need the non-zero indices
        for i in ids
            push!(state_to_subset[i].index, j)
        end
    end

    return SparseOrdering(perm, state_to_subset, subsets)
    n = size(p, 1)
    return DenseOrdering{T}(n)
end

function construct_state_to_subset(T, n)
    state_to_subset = Vector{Vector{T}}(undef, n)
    for i in eachindex(state_to_subset)
        state_to_subset[i] = Int64[]
    end

    return state_to_subset
end

function sort_states!(order::DenseOrdering, V; max = true)
    sortperm!(order.perm, V; rev = max)  # rev=true for maximization

    return order
end