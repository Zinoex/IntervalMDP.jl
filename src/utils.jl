@inline recursiveflatten(x::Vector) = collect(Iterators.flatten(map(recursiveflatten, x)))
@inline recursiveflatten(x) = x

@inline eachotherindex(A::AbstractArray, dim) = _eachotherindex(axes(A), dim)
@inline _eachotherindex(t::Tuple{Any}, dim) = [CartesianIndex{0}()]
@inline function _eachotherindex(t::Tuple, dim) 
    t = t[1:end .!= dim]
    return CartesianIndices(t)
end

@inline function selectotherdims(A::AbstractArray, dim::Integer, idxs)
    idxs = Tuple(idxs)
    head, tail = Base.split_rest(idxs, length(idxs) - dim + 1)
    idxs = (head..., Colon(), tail...)

    return view(A, idxs...)
end

@inline function selectotherdims(A::AbstractArray, dims::NTuple{2, Integer}, idxs)
    idxs = Tuple(idxs)

    Ipre = idxs[1:dims[1]-1]
    Ipost = idxs[dims[1]:end]
    Colons = [Colon() for _ in dims[1]:dims[2]]

    idxs = (Ipre..., Colons..., Ipost...)

    return view(A, idxs...)
end

@inline @inbounds maxdiff(x::V) where {V <: AbstractVector} = maximum(x[i + 1] - x[i] for i in 1:(length(x) - 1))
@inline @inbounds mindiff(x::V) where {V <: AbstractVector} = minimum(x[i + 1] - x[i] for i in 1:(length(x) - 1))