"""
    OrthogonalIntervalProbabilities{N, P <: IntervalProbabilities}

A tuple of `IntervalProbabilities` transition probabilities from all source states or source/action pairs to the
target states along each axis. 

### Fields
- `probs::NTuple{N, P}`: A tuple of `IntervalProbabilities` transition probabilities along each axis.
- `source_dims::NTuple{N, Int32}`: The dimensions of the orthogonal probabilities for the source axis. This is flattened to a single dimension for indexing.

### Examples
# TODO: Update example
```jldoctest
"""
struct OrthogonalIntervalProbabilities{N, P <: IntervalProbabilities} <:
       AbstractIntervalProbabilities
    probs::NTuple{N, P}
    source_dims::NTuple{N, Int32}
end

"""
    lower(p::OrthogonalIntervalProbabilities, i)

Return the lower bound transition probabilities from a source state or source/action pair to a target state.
"""
lower(p::OrthogonalIntervalProbabilities, i) = p.probs[i].lower

"""
    upper(p::OrthogonalIntervalProbabilities, i)

Return the upper bound transition probabilities from a source state or source/action pair to a target state.

!!! note
    It is not recommended to use this function for the hot loop of O-maximization. Because the [`IntervalProbabilities`](@ref)
    stores the lower and gap transition probabilities, fetching the upper bound requires allocation and computation.
"""
upper(p::OrthogonalIntervalProbabilities, i) = p.probs[i].lower + p.probs[i].gap

"""
    gap(p::OrthogonalIntervalProbabilities, i)

Return the gap between upper and lower bound transition probabilities from a source state or source/action pair to a target state.
"""
gap(p::OrthogonalIntervalProbabilities, i) = p.probs[i].gap

"""
    sum_lower(p::OrthogonalIntervalProbabilities, i) 

Return the sum of lower bound transition probabilities from a source state or source/action pair to all target states.
This is useful in efficiently implementing O-maximization, where we start with a lower bound probability assignment
and iteratively, according to the ordering, adding the gap until the sum of probabilities is 1.
"""
sum_lower(p::OrthogonalIntervalProbabilities, i) = p.probs[i].sum_lower

"""
    num_source(p::OrthogonalIntervalProbabilities)

Return the number of source states or source/action pairs.
"""
num_source(p::OrthogonalIntervalProbabilities) = num_source(first(p.probs))
source_shape(p::OrthogonalIntervalProbabilities) = p.source_dims

"""
    axes_source(p::OrthogonalIntervalProbabilities)

Return the valid range of indices for the source states or source/action pairs.
"""
axes_source(p::OrthogonalIntervalProbabilities) = axes_source(first(p.probs))

num_target(p::OrthogonalIntervalProbabilities{N}) where {N} =
    ntuple(i -> num_target(p[i]), N)
stateptr(p::OrthogonalIntervalProbabilities) = UnitRange{Int32}(1, num_source(p) + 1)
Base.ndims(p::OrthogonalIntervalProbabilities{N}) where {N} = N

Base.getindex(p::OrthogonalIntervalProbabilities, i) = p.probs[i]
Base.lastindex(p::OrthogonalIntervalProbabilities) = ndims(p)
Base.firstindex(p::OrthogonalIntervalProbabilities) = 1
Base.length(p::OrthogonalIntervalProbabilities) = ndims(p)
Base.iterate(p::OrthogonalIntervalProbabilities) = (p[1], 2)
Base.iterate(p::OrthogonalIntervalProbabilities, i) = i > ndims(p) ? nothing : (p[i], i + 1)
