@inline recursiveflatten(x::Vector{<:Vector}) = Iterators.flatten(x)
@inline recursiveflatten(x::Vector{<:Number}) = x

@inline eachotherindex(A::AbstractArray, dim) = _eachotherindex(axes(A), dim)
@inline _eachotherindex(t::Tuple{Any}, dim) = []
@inline function _eachotherindex(t::Tuple, dim) 
    t = t[1:end .!= dim]
    return CartesianIndices(t)
end

@inline function selectotherdims(A::AbstractArray, dim, idxs)
    idxs = Tuple(idxs)
    head, tail = Base.split_rest(idxs, length(idxs) - dim + 1)
    idxs = (head..., Colon(), tail...)

    return view(A, idxs...)
end

@inline @inbounds maxdiff(x::V) where {V <: AbstractVector} = maximum(x[i + 1] - x[i] for i in 1:(length(x) - 1))
@inline @inbounds mindiff(x::V) where {V <: AbstractVector} = minimum(x[i + 1] - x[i] for i in 1:(length(x) - 1))