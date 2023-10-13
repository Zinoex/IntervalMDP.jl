###################
# Sparse ordering #
###################
struct SparseOrdering{T<:Integer, VT<:AbstractVector{T}} <: AbstractStateOrdering{T}
    perm::VT
    state_to_subset::Vector{Vector{T}}
    subsets::Vector{PermutationSubset{T, VT}}
end

# Permutations are specific to each state
perm(order::SparseOrdering, state) = order.subsets[state].items

# Vector of sparse vectors
mutable struct PermutationSubset{T<:Integer, VT<:AbstractVector{T}}
    ptr::T
    items::VT
end

function Base.empty!(subset::PermutationSubset)
    subset.ptr = 1
end

function Base.push!(subset::PermutationSubset, item)
    subset.items[subset.ptr] = item
    subset.ptr += 1
end

function reset_subsets!(subsets)
    for subset in subsets
        empty!(subset)
    end
end

function populate_subsets!(order::SparseOrdering)
    reset_subsets!(order.subsets)

    for i in order.perm
        for j in order.state_to_subset[i]
            push!(order.subsets[j], i)
        end
    end
end

function construct_ordering{T}(p::VVR) where {VVR<:AbstractVector{<:AbstractSparseVector}}
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

function sort_states!(order::SparseOrdering, V; max=true)
    sortperm!(order.perm, V; rev=rev)  # rev=true for maximization
    populate_subsets!(order)
end