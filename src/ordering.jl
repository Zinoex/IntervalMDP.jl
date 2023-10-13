# Default ordering type
construct_ordering(p) = construct_ordering{Int64}(p)

abstract type AbstractStateOrdering{T} end

##################
# Dense ordering #
##################
struct DenseOrdering{T<:Integer, VT<:AbstractVector{T}} <: AbstractStateOrdering{T}
    perm::VT
end

function DenseOrdering{T}(n)
    return DenseOrdering(collect(UnitRange{T}(1, n)))
end

# Permutations are shared for all states
perm(order::DenseOrdering, state) = order.perm

# Vector of dense vectors
function construct_ordering{T}(p::VV) where {T, VV<:AbstractVector{<:AbstractVector}}
    # Assume that each vector corresponds to a start state
    n = length(p)
    return DenseOrdering{T}(n + 1)
end

# Dense matrix
function construct_ordering{T}(p::AbstractMatrix) where {T}
    # Assume that input/start state is on the columns and output/target state is on the rows
    n = size(p, 1)
    return DenseOrdering{T}(n)
end

function sort_states!(order::DenseOrdering, V; max=true)
    sortperm!(order.perm, V; rev=max)  # rev=true for maximization
end