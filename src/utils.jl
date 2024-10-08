@inline @inbounds maxdiff(x::V) where {V <: AbstractVector} =
    maximum(x[i + 1] - x[i] for i in 1:(length(x) - 1))
@inline @inbounds mindiff(x::V) where {V <: AbstractVector} =
    minimum(x[i + 1] - x[i] for i in 1:(length(x) - 1))
