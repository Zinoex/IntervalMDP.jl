# Vector of sparse vectors
function IMDP.construct_ordering(T, p::VVR) where {VVR <: AbstractVector{<:AbstractSparseVector}}
    # Assume that each vector corresponds to a start state

    n, m = length(first(p)), length(p)
    perm = collect(UnitRange{T}(1, n))
    state_to_subset = construct_state_to_subset(T, n)

    subsets = Vector{PermutationSubset{T, Vector{T}}}(undef, m)
    for j in eachindex(subsets)
        subsets[j] = PermutationSubset(T(1), Vector{T}(undef, nnz(p[j])))

        ids = SparseArrays.nonzeroinds(p[j])  # This is not exported, but we need the non-zero indices
        for i in ids
            push!(state_to_subset[i], j)
        end
    end

    return SparseOrdering(perm, state_to_subset, subsets)
end

# Sparse matrix
function IMDP.construct_ordering(T, p::AbstractSparseMatrix)
    # Assume that input/start state is on the columns and output/target state is on the rows
    n, m = size(p)
    perm = collect(UnitRange{T}(1, n))
    state_to_subset = construct_state_to_subset(T, n)

    subsets = Vector{PermutationSubset{T, Vector{T}}}(undef, m)
    for j in eachindex(subsets)
        pⱼ = view(p, :, j)
        subsets[j] = PermutationSubset(T(1), Vector{T}(undef, nnz(pⱼ)))

        ids = SparseArrays.nonzeroinds(pⱼ)  # This is not exported, but we need the non-zero indices
        for i in ids
            push!(state_to_subset[i], j)
        end
    end

    return SparseOrdering(perm, state_to_subset, subsets)
end

function construct_state_to_subset(T, n)
    state_to_subset = Vector{Vector{T}}(undef, n)
    for i in eachindex(state_to_subset)
        state_to_subset[i] = T[]
    end

    return state_to_subset
end
