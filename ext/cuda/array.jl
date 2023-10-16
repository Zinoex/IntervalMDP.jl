
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
Base.similar(xs::CuVectorOfVector) = CuVectorOfVector(copy(xs.vecPtr), copy(xs.lengths), similar(xs.val))

# CPU to GPU
function CuVectorOfVector(vecs::Vector{Vector{T}}) where {T}
    lengths = map(length, vecs)
    vecPtr = [1; cumsum(lengths)]
    
    return CuVectorOfVector(CuVector{Cint}(vecPtr), CuVector{Cint}(lengths), CuVector{T}(reduce(vcat, vecs)))
end

Adapt.adapt_storage(::Type{CuArray}, xs::Vector{Vector{T}}) where {T} = CuVectorOfVector(xs)

# GPU to CPU
function Vector(xs::CuVectorOfVector{T}) where {T}
    vecPtr = Vector(xs.vecPtr)

    vecs = Vector{Vector{T}}(undef, length(vecPtr) - 1)
    for i in eachindex(vecs)
        vecs[i] = Vector{T}(xs.val[vecPtr[i]:vecPtr[i + 1] - 1])
    end

    return vecs
end

Adapt.adapt_storage(::Type{Array}, xs::CuVectorOfVector) = Vector(xs)

# Indexing
Base.getindex(xs::CuVectorOfVector, i::Ti) where {Ti} = CuVectorInstance(xs.vecPtr[i], xs.vecPtr[i + Ti(1)] - xs.vecPtr[i], xs.val)

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
        maxlength(x)
    )
end

# Indexing
Base.getindex(xs::CuDeviceVectorOfVector, i::Ti) where {Ti} = CuDeviceVectorInstance(xs.vecPtr[i], xs.vecPtr[i + Ti(1)] - xs.vecPtr[i], xs.val)

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