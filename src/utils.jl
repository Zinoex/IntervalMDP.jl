@inline recursiveflatten(x::Tuple) = Tuple(collect(Iterators.flatten(map(recursiveflatten, x))))
@inline recursiveflatten(x) = x

@inline eachotherindex(A::AbstractArray, dim) = _eachotherindex(axes(A), dim)
@inline _eachotherindex(t::Tuple{Any}, dim) = CartesianIndices((),)
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

@inline function selectotherdims(A::AbstractArray, dim, ndims, idxs)
    idxs = Tuple(idxs)
    head, tail = Base.split_rest(idxs, length(idxs) - dim + 1)
    full_size = (Colon() for _ in 1:ndims)
    idxs = (head..., full_size..., tail...)

    return view(A, idxs...)
end

function state_index(workspace, j::Integer, idxs)
    idxs = Tuple(idxs)
    head, tail = Base.split_rest(idxs, length(idxs) - workspace.state_index + 1)
    idx = CartesianIndex(head..., j, tail...)
    return idx
end

function state_index(workspace, j::Tuple, idxs)
    idxs = Tuple(idxs)
    head, tail = Base.split_rest(idxs, length(idxs) - workspace.state_index + 1)
    idx = CartesianIndex(head..., j..., tail...)
    return idx
end

state_index(workspace, j::CartesianIndex, idxs) = state_index(workspace, Tuple(j), idxs)

@inline @inbounds maxdiff(x::V) where {V <: AbstractVector} =
    maximum(x[i + 1] - x[i] for i in 1:(length(x) - 1))
@inline @inbounds mindiff(x::V) where {V <: AbstractVector} =
    minimum(x[i + 1] - x[i] for i in 1:(length(x) - 1))
