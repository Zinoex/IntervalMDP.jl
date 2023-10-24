function IMDP.compute_gap(
    lower::MR,
    upper::MR,
) where {MR <: CuSparseMatrixCSC}
    # FIXME: This is an ugly, non-robust hack.
    upper = SparseMatrixCSC(upper)
    lower = SparseMatrixCSC(lower)
    lower, gap = IMDP.compute_gap(lower, upper)
    return cu(lower), cu(gap)
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