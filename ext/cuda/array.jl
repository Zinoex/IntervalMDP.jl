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

SparseArrays.nzrange(S::CuSparseMatrixCSC, col::Integer) =
    CUDA.@allowscalar(S.colPtr[col]):(CUDA.@allowscalar(S.colPtr[col + 1]) - 1)

Adapt.adapt_storage(
    ::Type{<:IntervalMDP.CuModelAdaptor{Tv}},
    M::SparseArrays.FixedSparseCSC,
) where {Tv} = CuSparseMatrixCSC{Tv, Int32}(M)

Adapt.adapt_storage(::Type{<:IntervalMDP.CuModelAdaptor{Tv1}}, M::SparseMatrixCSC{Tv2}) where {Tv1, Tv2} =
    CuSparseMatrixCSC{Tv1, Int32}(M)

Adapt.adapt_storage(::Type{<:IntervalMDP.CuModelAdaptor{Tv1}}, x::AbstractArray{Tv2}) where {Tv1, Tv2} =
    adapt(CuArray{Tv1}, x)

Adapt.adapt_storage(::Type{<:IntervalMDP.CuModelAdaptor{Tv1}}, x::AbstractArray{NTuple{N, T}}) where {Tv1, N, T <: Integer} =
    adapt(CuArray{NTuple{N, T}}, x)

Adapt.adapt_storage(
    ::Type{IntervalMDP.CpuModelAdaptor{Tv}},
    M::CuSparseMatrixCSC,
) where {Tv} = SparseMatrixCSC{Tv, Int32}(M)

Adapt.adapt_storage(::Type{<:IntervalMDP.CpuModelAdaptor{Tv1}}, x::CuArray{Tv2}) where {Tv1, Tv2} =
    adapt(Array{Tv1}, x)

Adapt.adapt_storage(::Type{<:IntervalMDP.CpuModelAdaptor{Tv1}}, x::CuArray{NTuple{N, T}}) where {Tv1, N, T <: Integer} =
    adapt(Array{NTuple{N, T}}, x)

const CuSparseDeviceColumnView{Tv, Ti} = SubArray{Tv, 1, <:CuSparseDeviceMatrixCSC{Tv, Ti}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}}
IntervalMDP.support(p::IntervalMDP.IntervalAmbiguitySet{R, <:CuSparseDeviceColumnView{R}}) where {R} = rowvals(p.gap)
IntervalMDP.supportsize(p::IntervalMDP.IntervalAmbiguitySet{R, <:CuSparseDeviceColumnView{R}}) where {R} = nnz(p.gap)