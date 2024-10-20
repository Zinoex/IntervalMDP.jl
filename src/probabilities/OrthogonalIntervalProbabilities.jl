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
```
"""
struct OrthogonalIntervalProbabilities{N, P <: IntervalProbabilities} <:
       AbstractIntervalProbabilities
    probs::NTuple{N, P}
    source_dims::NTuple{N, Int32}

    function OrthogonalIntervalProbabilities(
        probs::NTuple{N, P},
        source_dims::NTuple{N, Int32},
    ) where {N, P}
        source_action_pairs = num_source(first(probs))

        for i in 2:N
            if num_source(probs[i]) != source_action_pairs
                throw(
                    DimensionMismatch(
                        "The number of source states or source/action pairs must be the same for all axes.",
                    ),
                )
            end
        end

        new{N, P}(probs, source_dims)
    end
end

"""
    lower(p::OrthogonalIntervalProbabilities, l)

Return the lower bound transition probabilities from a source state or source/action pair to a target state.
"""
lower(p::OrthogonalIntervalProbabilities, l) = lower(p.probs[l])

"""
    lower(p::OrthogonalIntervalProbabilities, l, i, j)

Return the lower bound transition probabilities from a source state or source/action pair to a target state.
"""
lower(p::OrthogonalIntervalProbabilities, l, i, j) = lower(p.probs[l], i, j)

"""
    upper(p::OrthogonalIntervalProbabilities, l)

Return the upper bound transition probabilities from a source state or source/action pair to a target state.

!!! note
    It is not recommended to use this function for the hot loop of O-maximization. Because the [`IntervalProbabilities`](@ref)
    stores the lower and gap transition probabilities, fetching the upper bound requires allocation and computation.
"""
upper(p::OrthogonalIntervalProbabilities, l) = upper(p.probs[l])

"""
    upper(p::OrthogonalIntervalProbabilities, l, i, j)

Return the upper bound transition probabilities from a source state or source/action pair to a target state.

!!! note
    It is not recommended to use this function for the hot loop of O-maximization. Because the [`IntervalProbabilities`](@ref)
    stores the lower and gap transition probabilities, fetching the upper bound requires allocation and computation.
"""
upper(p::OrthogonalIntervalProbabilities, l, i, j) = upper(p.probs[l], i, j)

"""
    gap(p::OrthogonalIntervalProbabilities, l)

Return the gap between upper and lower bound transition probabilities from a source state or source/action pair to a target state.
"""
gap(p::OrthogonalIntervalProbabilities, l) = gap(p.probs[l])

"""
    gap(p::OrthogonalIntervalProbabilities, l, i, j)

Return the gap between upper and lower bound transition probabilities from a source state or source/action pair to a target state.
"""
gap(p::OrthogonalIntervalProbabilities, l, i, j) = gap(p.probs[l], i, j)

"""
    sum_lower(p::OrthogonalIntervalProbabilities, l) 

Return the sum of lower bound transition probabilities from a source state or source/action pair to all target states.
This is useful in efficiently implementing O-maximization, where we start with a lower bound probability assignment
and iteratively, according to the ordering, adding the gap until the sum of probabilities is 1.
"""
sum_lower(p::OrthogonalIntervalProbabilities, l) = sum_lower(p.probs[l])

"""
    sum_lower(p::OrthogonalIntervalProbabilities, l, j) 

Return the sum of lower bound transition probabilities from a source state or source/action pair to all target states.
This is useful in efficiently implementing O-maximization, where we start with a lower bound probability assignment
and iteratively, according to the ordering, adding the gap until the sum of probabilities is 1.
"""
sum_lower(p::OrthogonalIntervalProbabilities, l, j) = sum_lower(p.probs[l], j)

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
