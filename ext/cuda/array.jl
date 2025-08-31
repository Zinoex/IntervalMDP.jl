# This is type piracy - please port upstream to CUDA when FixedSparseCSC are stable.
CUDA.CUSPARSE.CuSparseMatrixCSC{Tv, Ti}(M::SparseArrays.FixedSparseCSC) where {Tv, Ti} =
    CuSparseMatrixCSC{Tv, Ti}(
        CuVector{Ti}(M.colptr),
        CuVector{Ti}(M.rowval),
        CuVector{Tv}(M.nzval),
        size(M),
    )

CUDA.CUSPARSE.CuSparseMatrixCSC{Tv, Ti}(M::SparseMatrixCSC) where {Tv, Ti} =
    CuSparseMatrixCSC{Tv, Ti}(
        CuVector{Ti}(M.colptr),
        CuVector{Ti}(M.rowval),
        CuVector{Tv}(M.nzval),
        size(M),
    )

const CuSparseColumnView{Tv, Ti} = SubArray{
    Tv,
    1,
    CuSparseMatrixCSC{Tv, Ti},
    Tuple{Base.Slice{Base.OneTo{Int}}, Int},
    false,
}

function SparseArrays.nnz(x::CuSparseColumnView)
    rowidx, colidx = parentindices(x)
    return length(nzrange(parent(x), colidx))
end
SparseArrays.nzrange(S::CuSparseMatrixCSC, col::Integer) =
    CUDA.@allowscalar(S.colPtr[col]):(CUDA.@allowscalar(S.colPtr[col + 1]) - 1)

Adapt.adapt_storage(
    ::Type{IntervalMDP.CuModelAdaptor{Tv}},
    M::SparseArrays.FixedSparseCSC,
) where {Tv} = CuSparseMatrixCSC{Tv, Int32}(M)

Adapt.adapt_storage(::Type{IntervalMDP.CuModelAdaptor{Tv}}, M::SparseMatrixCSC) where {Tv} =
    CuSparseMatrixCSC{Tv, Int32}(M)

Adapt.adapt_storage(::Type{IntervalMDP.CuModelAdaptor{Tv}}, x::AbstractArray) where {Tv} =
    adapt(CuArray{Tv}, x)

Adapt.adapt_storage(
    ::Type{IntervalMDP.CpuModelAdaptor{Tv}},
    M::CuSparseMatrixCSC,
) where {Tv} = SparseMatrixCSC{Tv, Int32}(M)

Adapt.adapt_storage(::Type{IntervalMDP.CpuModelAdaptor{Tv}}, x::CuArray{Tv}) where {Tv} =
    adapt(Array{Tv}, x)

Adapt.adapt_storage(::Type{IntervalMDP.CpuModelAdaptor{Tv}}, x::CuArray{Int32}) where {Tv} =
    adapt(Array{Int32}, x)
