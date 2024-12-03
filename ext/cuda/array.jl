function IntervalMDP.maxdiff(stateptr::CuVector{Int32})
    return reducediff(max, stateptr, typemin(Int32))
end

function IntervalMDP.mindiff(stateptr::CuVector{Int32})
    return reducediff(min, stateptr, typemax(Int32))
end

function reducediff(op, stateptr::CuVector{Int32}, neutral)
    ret_arr = CuArray{Int32}(undef, 1)
    kernel = @cuda launch = false reducediff_kernel!(op, stateptr, neutral, ret_arr)

    config = launch_configuration(kernel.fun)
    max_threads = prevwarp(device(), config.threads)
    wanted_threads = min(1024, nextwarp(device(), length(stateptr) - 1))

    threads = min(max_threads, wanted_threads)
    blocks = 1

    kernel(op, stateptr, neutral, ret_arr; blocks = blocks, threads = threads)

    return CUDA.@allowscalar ret_arr[1]
end

function reducediff_kernel!(op, stateptr, neutral, retarr)
    diff = neutral

    i = threadIdx().x
    @inbounds while i <= length(stateptr) - 1
        diff = op(diff, stateptr[i + 1] - stateptr[i])
        i += blockDim().x
    end

    shuffle = Val(true)
    diff = CUDA.reduce_block(op, diff, neutral, shuffle)

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
