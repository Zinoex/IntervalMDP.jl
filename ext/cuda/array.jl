### Vector of Vector
struct CuVectorOfVector{Tv, Ti} <: AbstractVector{AbstractVector{Tv}}
    vecPtr::CuVector{Ti}
    lengths::CuVector{Ti}
    val::CuVector{Tv}
end

function CUDA.unsafe_free!(xs::CuVectorOfVector)
    unsafe_free!(xs.vecPtr)
    unsafe_free!(xs.val)
    return
end

maxlength(xs::CuVectorOfVector) = maximum(xs.lengths)
Base.length(xs::CuVectorOfVector{Tv, Ti}) where {Tv, Ti} = length(xs.vecPtr) - Ti(1)
Base.size(xs::CuVectorOfVector) = (xs.length,)
Base.similar(xs::CuVectorOfVector) =
    CuVectorOfVector(copy(xs.vecPtr), copy(xs.lengths), similar(xs.val))

function Base.show(io::IO, x::CuVectorOfVector)
    for i in 1:length(x)
        print(io, "[$i] :")
        CUDA.@allowscalar show(
            IOContext(io, :typeinfo => eltype(x)),
            Vector(x.val[x.vecPtr[i]:(x.vecPtr[i + 1] - 1)]),
        )
        println("")
    end
end

# CPU to GPU
function CuVectorOfVector(vecs::Vector{Vector{T}}) where {T}
    lengths = convert.(T, map(length, vecs))
    vecPtr = [T(1); 1 .+ cumsum(lengths)]

    return CuVectorOfVector(
        CuVector{Cint}(vecPtr),
        CuVector{Cint}(lengths),
        CuVector{T}(reduce(vcat, vecs)),
    )
end

Adapt.adapt_storage(::CUDA.CuArrayAdaptor, xs::Vector{Vector{T}}) where {T <: Number} =
    CuVectorOfVector(xs)

# GPU to CPU
function Vector(xs::CuVectorOfVector{T}) where {T}
    vecPtr = Vector(xs.vecPtr)

    vecs = Vector{Vector{T}}(undef, length(vecPtr) - 1)
    for i in eachindex(vecs)
        vecs[i] = Vector{T}(xs.val[vecPtr[i]:(vecPtr[i + 1] - 1)])
    end

    return vecs
end

Adapt.adapt_storage(::Type{Array}, xs::CuVectorOfVector) = Vector(xs)

# Indexing
Base.getindex(xs::CuVectorOfVector, i::Ti) where {Ti} =
    CuVectorInstance(xs.vecPtr[i], xs.vecPtr[i + Ti(1)] - xs.vecPtr[i], xs.val)

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
    vecPtr::CuDeviceVector{Ti, A}
    lengths::CuDeviceVector{Ti, A}
    val::CuDeviceVector{Tv, A}
    maxlength::Ti
end
maxlength(xs::CuDeviceVectorOfVector) = xs.maxlength
Base.length(xs::CuDeviceVectorOfVector{Tv, Ti}) where {Tv, Ti} = length(xs.vecPtr) - Ti(1)

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuVectorOfVector)
    return CuDeviceVectorOfVector(
        adapt(to, x.vecPtr),
        adapt(to, x.lengths),
        adapt(to, x.val),
        maxlength(x),
    )
end

# Indexing
Base.getindex(xs::CuDeviceVectorOfVector, i::Ti) where {Ti} =
    CuDeviceVectorInstance(xs.vecPtr[i], xs.vecPtr[i + Ti(1)] - xs.vecPtr[i], xs.val)

struct CuDeviceVectorInstance{Ti, Tv, A}
    offset::Ti
    length::Ti
    parent::CuDeviceVector{Tv, A}
end
Base.length(x::CuDeviceVectorInstance) = x.length
Base.size(x::CuDeviceVectorInstance) = (x.length,)
function Base.getindex(xs::CuDeviceVectorInstance, i::Ti) where {Ti}
    if i > xs.length
        throw(BoundsError(xs, i))
    end

    return xs.parent[xs.offset + i - Ti(1)]
end
function Base.setindex!(xs::CuDeviceVectorInstance, v, i::Ti) where {Ti}
    if i > xs.length
        throw(BoundsError(xs, i))
    end

    return xs.parent[xs.offset + i - Ti(1)] = v
end

# Permutation subsets
struct CuPermutationSubsets{Tv, Ti}
    queues::CuVectorOfVector{Tv, Ti}
    ptrs::CuVector{Ti}
end

function CUDA.unsafe_free!(subsets::CuPermutationSubsets)
    unsafe_free!(subsets.queues)
    unsafe_free!(subsets.ptrs)
    return
end

Base.length(subsets::CuPermutationSubsets) = length(subsets.queues)
Base.size(subsets::CuPermutationSubsets) = (length(subsets),)
Base.similar(subsets::CuPermutationSubsets) =
    CuPermutationSubsets(similar(subsets.queues), similar(subsets.ptrs))

# Indexing
Base.getindex(xs::CuPermutationSubsets, i::Ti) where {Ti} =
    CuPermutationSubset(xs.queues[i], xs.ptrs, i)
struct CuPermutationSubset{Tv, Ti}
    queue::CuVectorInstance{Tv, Ti}
    ptrs::CuVector{Ti}
    index::Ti
end
function Base.push!(subset::CuPermutationSubset, item)
    subset.queue[subset.ptrs[subset.index]] = item
    return subset.ptrs[subset.index] += 1
end

function reset_subsets!(subsets::CuPermutationSubsets)
    return fill!(subsets.ptrs, 1)
end

# CPU to GPU
function CuPermutationSubsets(subsets::Vector{<:PermutationSubset{T}}) where {T}
    queues = CuVectorOfVector(map(x -> x.items, subsets))
    ptrs = CUDA.ones(T, length(subsets))

    return CuPermutationSubsets(queues, ptrs)
end

Adapt.adapt_storage(::CUDA.CuArrayAdaptor, subsets::Vector{<:PermutationSubset}) =
    CuPermutationSubsets(subsets)

# GPU to CPU
function Vector(subsets::CuPermutationSubsets{T}) where {T}
    vecs = Vector{PermutationSubset{T, Vector{T}}}(undef, length(subsets.queues))
    for i in eachindex(vecs)
        vecs[i] =
            PermutationSubset{T, Vector{T}}(subsets.ptrs[i], Vector(subsets.queues[i]))
    end

    return vecs
end

Adapt.adapt_storage(::Type{Array}, subsets::CuPermutationSubsets) = Vector(subsets)

# Device
struct CuDevicePermutationSubsets{Tv, Ti, A}
    queues::CuDeviceVectorOfVector{Tv, Ti, A}
    ptrs::CuDeviceVector{Ti, A}
end

Base.length(subsets::CuDevicePermutationSubsets) = length(subsets.queues)
Base.size(subsets::CuDevicePermutationSubsets) = (length(subsets),)

function Adapt.adapt_structure(to::CUDA.Adaptor, subsets::CuPermutationSubsets)
    return CuDevicePermutationSubsets(adapt(to, subsets.queues), adapt(to, subsets.ptrs))
end

# Indexing
struct CuDevicePermutationSubset{Tv, Ti, A}
    queue::CuDeviceVectorInstance{Tv, Ti, A}
    ptrs::CuDeviceVector{Ti, A}
    index::Ti
end
Base.getindex(xs::CuDevicePermutationSubsets, i::Ti) where {Ti} =
    CuDevicePermutationSubset(xs.queues[i], xs.ptrs, i)
Base.getindex(xs::CuDevicePermutationSubset, i) = xs.queue[i]
Base.length(subset::CuDevicePermutationSubset) = length(subset.queue)

function Base.push!(subset::CuDevicePermutationSubset{Tv, Ti}, item::Tv) where {Tv, Ti}
    subset.queue[subset.ptrs[subset.index]] = item
    return subset.ptrs[subset.index] += 1
end

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
