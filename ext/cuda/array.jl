function IMDP.compute_gap(
    lower::MR,
    upper::MR,
) where {MR <: CuSparseMatrixCSC}
    gap = upper - lower
    lower = upper - gap
    return lower, gap
end

### Vector of Vector
struct CuVectorOfVector{Tv, Ti} <: AbstractVector{AbstractVector{Tv}}
    vecptr::CuVector{Ti}
    val::CuVector{Tv}
    maxlength::Ti
end

function CUDA.unsafe_free!(xs::CuVectorOfVector)
    unsafe_free!(xs.vecptr)
    unsafe_free!(xs.val)
    return
end

maxlength(xs::CuVectorOfVector) = xs.maxlength
Base.length(xs::CuVectorOfVector{Tv, Ti}) where {Tv, Ti} = length(xs.vecptr) - Ti(1)
Base.size(xs::CuVectorOfVector) = (length(xs),)
Base.similar(xs::CuVectorOfVector) = CuVectorOfVector(copy(xs.vecptr), similar(xs.val), maxlength(xs))

function Base.show(io::IO, x::CuVectorOfVector)
    vecptr = Vector(x.vecptr)
    for i in 1:length(x)
        print(io, "[$i]: ")
        show(
            IOContext(io, :typeinfo => eltype(x)),
            Vector(x.val[vecptr[i]:(vecptr[i + 1] - 1)]),
        )
        println(io, "")
    end
end

# CPU to GPU
function CuVectorOfVector(vecs::Vector{Vector{T}}) where {T}
    lengths = convert.(T, map(length, vecs))
    vecptr = [T(1); 1 .+ cumsum(lengths)]

    return CuVectorOfVector(
        CuVector{Cint}(vecptr),
        CuVector{T}(reduce(vcat, vecs)),
        maximum(lengths),
    )
end

Adapt.adapt_storage(::CUDA.CuArrayAdaptor, xs::Vector{Vector{T}}) where {T <: Number} =
    CuVectorOfVector(xs)

# GPU to CPU
function Vector(xs::CuVectorOfVector{T}) where {T}
    vecptr = Vector(xs.vecptr)

    vecs = Vector{Vector{T}}(undef, length(vecptr) - 1)
    for i in eachindex(vecs)
        vecs[i] = Vector{T}(xs.val[vecptr[i]:(vecptr[i + 1] - 1)])
    end

    return vecs
end

Adapt.adapt_storage(::Type{Array}, xs::CuVectorOfVector) = Vector(xs)

# Indexing
Base.getindex(xs::CuVectorOfVector, i::Ti) where {Ti} =
    CuVectorInstance(xs.vecptr[i], xs.vecptr[i + Ti(1)] - xs.vecptr[i], xs.val)

struct CuVectorInstance{Ti, Tv}
    offset::Ti
    length::Ti
    parent::CuVector{Tv}
end
Base.length(x::CuVectorInstance) = x.length
Base.size(x::CuVectorInstance) = (x.length,)
function Base.getindex(xs::CuVectorInstance, i::Ti) where {Ti}
    if i > xs.length
        throw(BoundsError(xs, i))
    end

    return xs.parent[xs.offset + i - Ti(1)]
end

# Device
struct CuDeviceVectorOfVector{Tv, Ti, A} <: AbstractVector{AbstractVector{Tv}}
    vecptr::CuDeviceVector{Ti, A}
    val::CuDeviceVector{Tv, A}
    maxlength::Ti
end
maxlength(xs::CuDeviceVectorOfVector) = xs.maxlength
Base.length(xs::CuDeviceVectorOfVector{Tv, Ti}) where {Tv, Ti} = length(xs.vecptr) - Ti(1)

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuVectorOfVector)
    return CuDeviceVectorOfVector(
        adapt(to, x.vecptr),
        adapt(to, x.val),
        maxlength(x),
    )
end

# Indexing
Base.getindex(xs::CuDeviceVectorOfVector{Tv, Ti, A}, i) where {Tv, Ti, A} =
    CuDeviceVectorInstance{Tv, Ti, A}(xs.vecptr[i], xs.vecptr[i + Ti(1)] - xs.vecptr[i], xs.val)

struct CuDeviceVectorInstance{Tv, Ti, A}
    offset::Ti
    length::Ti
    parent::CuDeviceVector{Tv, A}
end
Base.length(x::CuDeviceVectorInstance) = x.length
Base.size(x::CuDeviceVectorInstance) = (x.length,)
function Base.getindex(xs::CuDeviceVectorInstance{Tv, Ti, A}, i) where {Tv, Ti, A}
    if i > xs.length
        throw(BoundsError(xs, i))
    end

    return xs.parent[xs.offset + i - Ti(1)]
end
function Base.setindex!(xs::CuDeviceVectorInstance{Tv, Ti, A}, v, i) where {Tv, Ti, A}
    if i > xs.length
        throw(BoundsError(xs, i))
    end

    xs.parent[xs.offset + i - Ti(1)] = v
end

# Permutation subsets
struct CuPermutationSubsets{Tv, Ti}
    value_subsets::CuVectorOfVector{Tv, Ti}
    perm_subsets::CuVectorOfVector{Ti, Ti}
end

function CUDA.unsafe_free!(subsets::CuPermutationSubsets)
    unsafe_free!(subsets.value_subsets)
    unsafe_free!(subsets.perm_subsets)
    return
end

Base.length(subsets::CuPermutationSubsets) = length(subsets.value_subsets)
Base.size(subsets::CuPermutationSubsets) = (length(subsets),)
Base.similar(subsets::CuPermutationSubsets) =
    CuPermutationSubsets(similar(subsets.value_subsets), similar(subsets.ptrs))

# Indexing
Base.getindex(subsets::CuPermutationSubsets{Tv, Ti}, i) where {Tv, Ti} =
    CuPermutationSubset(subsets.value_subsets[i], subsets.perm_subsets[i], i)
struct CuPermutationSubset{Tv, Ti}
    value_subset::CuVectorInstance{Tv, Ti}
    perm_subset::CuVectorInstance{Ti, Ti}
    index
end

# Device
struct CuDevicePermutationSubsets{Tv, Ti, A}
    value_subsets::CuDeviceVectorOfVector{Tv, Ti, A}
    perm_subsets::CuDeviceVectorOfVector{Ti, Ti, A}
end

Base.length(subsets::CuDevicePermutationSubsets) = length(subsets.value_subsets)
Base.size(subsets::CuDevicePermutationSubsets) = (length(subsets),)

function Adapt.adapt_structure(to::CUDA.Adaptor, subsets::CuPermutationSubsets)
    return CuDevicePermutationSubsets(adapt(to, subsets.value_subsets), adapt(to, subsets.perm_subsets))
end

# Indexing
struct CuDevicePermutationSubset{Tv, Ti, A}
    value_subset::CuDeviceVectorInstance{Tv, Ti, A}
    perm_subset::CuDeviceVectorInstance{Ti, Ti, A}
    index
end
Base.getindex(subsets::CuDevicePermutationSubsets{Tv, Ti, A}, i) where {Tv, Ti, A} = 
    CuDevicePermutationSubset{Tv, Ti, A}(subsets.value_subsets[i], subsets.perm_subsets[i], i)
Base.length(subset::CuDevicePermutationSubset) = length(subset.value_subset)

# This is type piracy - please port to CUDA when FixedSparseVector and FixedSparseCSC are stable.
CUDA.CUSPARSE.CuSparseVector{T}(Vec::SparseArrays.FixedSparseVector) where {T} =
    CuSparseVector(CuVector{Cint}(Vec.nzind), CuVector{T}(Vec.nzval), length(Vec))
CUDA.CUSPARSE.CuSparseMatrixCSC{T}(Mat::SparseArrays.FixedSparseCSC) where {T} =
    CuSparseMatrixCSC{T}(
        CuVector{Cint}(Mat.colptr),
        CuVector{Cint}(Mat.rowval),
        CuVector{T}(Mat.nzval),
        size(Mat),
    )

Adapt.adapt_storage(::Type{CuArray}, xs::SparseArrays.FixedSparseVector) =
    CuSparseVector(xs)
Adapt.adapt_storage(::Type{CuArray}, xs::SparseArrays.FixedSparseCSC) =
    CuSparseMatrixCSC(xs)
Adapt.adapt_storage(::Type{CuArray{T}}, xs::SparseArrays.FixedSparseVector) where {T} =
    CuSparseVector{T}(xs)
Adapt.adapt_storage(::Type{CuArray{T}}, xs::SparseArrays.FixedSparseCSC) where {T} =
    CuSparseMatrixCSC{T}(xs)
