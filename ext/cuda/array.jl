function maxdiff(colptr::CuVector{Int32})
    return reducediff(max, colptr, typemin(Int32))
end

function reducediff(op, colptr::CuVector{Int32}, neutral)
    ret_arr = CuArray{Int32}(undef, 1)
    kernel = @cuda launch = false reducediff_kernel!(op, colptr, neutral, ret_arr)

    config = launch_configuration(kernel.fun)
    max_threads = prevwarp(device(), config.threads)
    wanted_threads = min(1024, nextwarp(device(), length(colptr) - 1))

    threads = min(max_threads, wanted_threads)
    blocks = 1

    kernel(op, colptr, neutral, ret_arr; blocks = blocks, threads = threads)

    return CUDA.@allowscalar ret_arr[1]
end

function reducediff_kernel!(op, colptr, neutral, retarr)
    diff = neutral

    i = threadIdx().x
    @inbounds while i <= length(colptr) - 1
        diff = op(diff, colptr[i + 1] - colptr[i])
        i += blockDim().x
    end

    shuffle = Val(true)
    diff = reduce_block(op, diff, neutral, shuffle)

    if threadIdx().x == 1
        @inbounds retarr[1] = diff
    end

    return
end

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

Adapt.adapt_storage(
    ::Type{<:IntervalMDP.CuModelAdaptor{Tv1}},
    M::SparseMatrixCSC{Tv2},
) where {Tv1, Tv2} = CuSparseMatrixCSC{Tv1, Int32}(M)

Adapt.adapt_storage(
    ::Type{<:IntervalMDP.CuModelAdaptor{Tv1}},
    x::AbstractArray{Tv2},
) where {Tv1, Tv2} = adapt(CuArray{Tv1}, x)

Adapt.adapt_storage(
    ::Type{<:IntervalMDP.CuModelAdaptor{Tv1}},
    x::AbstractArray{NTuple{N, T}},
) where {Tv1, N, T <: Integer} = adapt(CuArray{NTuple{N, T}}, x)

Adapt.adapt_storage(
    ::Type{IntervalMDP.CpuModelAdaptor{Tv}},
    M::CuSparseMatrixCSC,
) where {Tv} = SparseMatrixCSC{Tv, Int32}(M)

Adapt.adapt_storage(
    ::Type{<:IntervalMDP.CpuModelAdaptor{Tv1}},
    x::CuArray{Tv2},
) where {Tv1, Tv2} = adapt(Array{Tv1}, x)

Adapt.adapt_storage(
    ::Type{<:IntervalMDP.CpuModelAdaptor{Tv1}},
    x::CuArray{NTuple{N, T}},
) where {Tv1, N, T <: Integer} = adapt(Array{NTuple{N, T}}, x)
