recursiveflatten(x::Vector{<:Vector}) = Iterators.flatten(x)
recursiveflatten(x::Vector{<:Number}) = x

@inline eachotherindex(A::AbstractArray) = _eachotherindex(axes(A))
_eachindex(t::Tuple{Any}) = []
_eachindex(t::Tuple) = CartesianIndices(t[2:end])