"""
    Transitions{Tv, Ti <: Integer, V <: AbstractVector{Ti}}

A sparse matrix to represent an upper bound transition probability from a source state or source/action pair to a target state,
if such transition is enabled.

The lower bound is implicitly stored equal to zero. The matrix is stored in the compressed sparse column (CSC) format without `nzval`.

### Fields
- 'colptr::V': The column pointer for the CSC format, pointing to the start of each source state or source/action pair in `rowval`.
- 'rowval::V': The row values for the CSC format, representing the target states.
- 'dims::NTuple{2, Int}': The dimensions of the matrix.

### Examples
# TODO: Update example
```jldoctest
sparse_prob = IntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(15, [4, 10], [0.1, 0.2]),
        SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
    ),
    upper = sparse_hcat(
        SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
        SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
    ),
)
```

"""
struct Transitions{Tv <: Number, Ti <: Integer, V <: AbstractVector{Ti}} <: SparseArrays.AbstractSparseMatrixCSC{Tv, Ti}
    type::Type{Tv}
    colptr::V
    rowval::V
    dims::NTuple{2, Int}
end

# Default to Float64 for the type of the non-zero values
function Transitions(colptr::V, rowval::V, dims::NTuple{2, Int}) where {Ti, V <: AbstractVector{Ti}}        
    return Transitions(Float64, colptr, rowval, dims)
end

"""
TODO: write documentation
"""
function transition_hcat(m, X::AbstractVector{Ti}...) where {Ti}
    # check sizes
    n = length(X)
    maximum(maximum, X) == m || throw(DimensionMismatch("Inconsistent column lengths."))

    tnnz = length(X[1])
    for j = 2:n
        tnnz += length(X[j])
    end

    # construction
    colptr = Vector{Ti}(undef, n + 1)
    nzrow = Vector{Ti}(undef, tnnz)
    roff = 1
    @inbounds for j = 1:n
        xnzind = X[j]
        colptr[j] = roff
        copyto!(nzrow, roff, xnzind)
        roff += length(xnzind)
    end
    colptr[n+1] = roff
    return Transitions(colptr, nzrow, (m, n))
end

function _transition_hcat(X::Transitions{Tv, Ti, V}...) where {Tv <: Number, Ti <: Integer, V <: AbstractVector{Ti}}
    N = length(X)

    # check sizes
    n = sum(x -> size(x, 2), X)
    m = size(X[1], 1)

    tnnz = nnz(X[1])
    for j = 2:N
        size(X[j], 1) == m || throw(DimensionMismatch("Inconsistent column lengths."))
        tnnz += nnz(X[j])
    end

    # construction
    colptr = Vector{Ti}(undef, n + 1)
    nzrow = Vector{Ti}(undef, tnnz)
    roff = 1
    k = 1

    @inbounds for j = 1:N
        xnzind = rowvals(X[j])
        copyto!(nzrow, roff, xnzind)

        @inbounds for i = 1:size(X[j], 2)
            colptr[k] = roff
            roff += nnz(@view(X[j][:, i]))
            k += 1
        end
    end
    colptr[n+1] = roff

    stateptr = Int32[1; 1 .+ cumsum([num_source(p) for p in X])]

    return Transitions(colptr, nzrow, (m, n)), stateptr
end

# Accessors for properties of interval probabilities
Base.size(p::Transitions) = p.dims
Base.size(p::Transitions, dim::Integer) = p.dims[dim]

SparseArrays.getcolptr(p::Transitions) = p.colptr
SparseArrays.rowvals(p::Transitions) = p.rowval

struct Ones{Tv} <: AbstractVector{Tv}
    type::Type{Tv}
    len::Int
end
Base.size(o::Ones) = (o.len,)
Base.size(o::Ones, dim::Integer) = isone(dim) ? o.len : 1
@inline Base.getindex(o::Ones{Tv}, i) where {Tv} = (@boundscheck checkbounds(o, i); one(Tv))

SparseArrays.nonzeros(p::Transitions{Tv}) where {Tv} = Ones(Tv, p.colptr[end] - 1)

"""
    num_source(p::Transitions)

Return the number of source states or source/action pairs.
"""
num_source(p::Transitions) = size(p, 2)

"""
    axes_source(p::Transitions)

Return the valid range of indices for the source states or source/action pairs.
"""
axes_source(p::Transitions) = 1:num_source(p)

"""
    num_target(p::Transitions)

Return the number of target states.
"""
num_target(p::Transitions) = size(p, 1)

stateptr(prob::Transitions{Tv, Ti}) where {Tv, Ti} = UnitRange{Ti}(1, num_source(prob) + one(Ti))
