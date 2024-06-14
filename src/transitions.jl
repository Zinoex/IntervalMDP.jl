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
    return Transitions(Float64, colptr, rowval, dims, nnz)
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
Base.size(o::Ones, dim::Integer) = o.len
@inline Base.getindex(o::Ones{Tv}, i) where {Tv} = (@boundscheck checkbounds(o, i); one(Tv))

SparseArrays.nonzeros(p::Transitions{Tv}) where {Tv} = Ones(Tv, p.colptr[end] - 1)

"""
    num_source(p::Transitions)

Return the number of source states or source/action pairs.
"""
num_source(p::Transitions) = size(gap(p), 2)

"""
    axes_source(p::Transitions)

Return the valid range of indices for the source states or source/action pairs.
"""
axes_source(p::Transitions) = 1:num_source(p)

"""
    num_target(p::Transitions)

Return the number of target states.
"""
num_target(p::Transitions) = size(gap(p), 1)

stateptr(prob::Transitions) = prob.colptr
