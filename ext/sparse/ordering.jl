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
