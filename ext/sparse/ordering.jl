function construct_ordering(T, p::VVR) where {VVR <: AbstractVector{<:AbstractSparseVector}}
    # Assume that each vector corresponds to a start state

    n = length(p)
    perm = collect(UnitRange{T}(1, n + 1))

    state_to_subset = Vector{Vector{T}}(undef, n + 1)
    for i in eachindex(state_to_subset)
        state_to_subset[i] = Int64[]
    end

    subsets = Vector{PermutationSubset{T, Vector{T}}}(undef, n)
    for j in eachindex(subsets)
        subsets[j] = PermutationSubset(1, Vector{T}(undef, nnz(p[j])))

        ids = SparseArrays.nonzeroinds(p[j])  # This is not exported, but we need the non-zero indices
        for i in ids
            push!(state_to_subset[i].index, j)
        end
    end

    return SparseOrdering(perm, state_to_subset, subsets)
end
