"""
    OrthogonalIntervalProbabilities{N, P <: IntervalProbabilities}

A tuple of `IntervalProbabilities` for (marginal) transition probabilities from all source/action pairs to the target states along each axis,
with target states/marginals on the rows and source states or source/action pairs on the columns. The source states are ordered in 
a column-major order, i.e., the first axis of source states is the fastest, similar to the ordering of a multi-dimensional array in Julia. 
E.g. for an `OrthogonalIntervalProbabilities` with `source_dims == (3, 3, 3)` and 2 actions for each source state ``\\{a_1, a_2\\}``, 
the columns in order represent the collowing:
```math
    ((1, 1, 1), a_1), ((1, 1, 1), a_2), (2, 1, 1), a_1), ((2, 1, 1), a_2), ..., ((3, 3, 3), a_1), ((3, 3, 3), a_2).
```
The number of target states correspond to the number of rows in the transition probabilities of each axis.


### Fields
- `probs::NTuple{N, P}`: A tuple of `IntervalProbabilities` for (marginal) transition probabilities along each axis.
- `source_dims::NTuple{N, Int32}`: The dimensions of the orthogonal probabilities for the source axis. This is flattened to a single dimension for indexing.

### Examples
An example of OrthogonalIntervalProbabilities with 3 axes and 3 states for each axis, only one action per state. 
Therefore, the `source_dims` is (3, 3, 3) and the number of columns of the transition probabilities is 27.

```jldoctest
lower1 = [
    1/15 3/10 1/15 3/10 1/30 1/3 7/30 4/15 1/6 1/5 1/10 1/5 0 7/30 7/30 1/5 2/15 1/6 1/10 1/30 1/10 1/15 1/10 1/15 4/15 4/15 1/3
    1/5 4/15 1/10 1/5 3/10 3/10 1/10 1/15 3/10 3/10 7/30 1/5 1/10 1/5 1/5 1/30 1/5 3/10 1/5 1/5 1/10 1/30 4/15 1/10 1/5 1/6 7/30
    4/15 1/30 1/5 1/5 7/30 4/15 2/15 7/30 1/5 1/3 2/15 1/6 1/6 1/3 4/15 3/10 1/30 3/10 3/10 1/10 1/15 1/30 2/15 1/6 1/5 1/10 4/15
]
upper1 = [
    7/15 17/30 13/30 3/5 17/30 17/30 17/30 13/30 3/5 2/3 11/30 7/15 0 1/2 17/30 13/30 7/15 13/30 17/30 13/30 2/5 2/5 2/3 2/5 17/30 2/5 19/30
    8/15 1/2 3/5 7/15 8/15 17/30 2/3 17/30 11/30 7/15 19/30 19/30 13/15 1/2 17/30 13/30 3/5 11/30 8/15 7/15 7/15 13/30 8/15 2/5 8/15 17/30 3/5
    11/30 1/3 2/5 8/15 7/15 3/5 2/3 17/30 2/3 8/15 2/15 3/5 2/3 3/5 17/30 2/3 7/15 8/15 2/5 2/5 11/30 17/30 17/30 1/2 2/5 19/30 13/30
]
prob1 = IntervalProbabilities(; lower = lower1, upper = upper1)

lower2 = [
    1/10 1/15 3/10 0 1/6 1/15 1/15 1/6 1/6 1/30 1/10 1/10 1/3 2/15 3/10 4/15 2/15 2/15 1/6 7/30 1/15 2/15 1/10 1/3 7/30 1/30 7/30
    3/10 1/5 3/10 2/15 0 1/30 0 1/15 1/30 7/30 1/30 1/15 7/30 1/15 1/6 1/30 1/10 1/15 3/10 0 3/10 1/6 3/10 1/5 0 7/30 2/15
    3/10 4/15 1/10 3/10 2/15 1/3 3/10 1/10 1/6 3/10 7/30 1/6 1/15 1/15 1/10 1/5 1/5 4/15 1/15 1/3 2/15 1/15 1/5 1/5 1/15 7/30 1/15
]
upper2 = [
    2/5 17/30 3/5 11/30 3/5 7/15 19/30 2/5 3/5 2/3 2/3 8/15 8/15 19/30 8/15 8/15 13/30 13/30 13/30 17/30 17/30 13/30 11/30 19/30 8/15 2/5 8/15
    1/3 13/30 11/30 2/5 2/3 2/3 0 13/30 1/2 17/30 17/30 1/3 2/5 1/3 13/30 11/30 8/15 1/3 1/2 8/15 8/15 8/15 8/15 2/5 3/5 2/3 13/30
    17/30 3/5 8/15 1/2 7/15 1/2 2/3 17/30 11/30 2/5 1/2 7/15 2/5 17/30 11/30 2/5 11/30 2/3 1/3 2/3 17/30 8/15 17/30 3/5 2/5 19/30 11/30
]
prob2 = IntervalProbabilities(; lower = lower2, upper = upper2)

lower3 = [
    4/15 1/5 3/10 3/10 4/15 7/30 1/5 4/15 7/30 1/6 1/5 0 1/15 1/30 3/10 1/3 2/15 1/15 7/30 4/15 1/10 1/3 1/5 7/30 1/30 1/5 7/30
    2/15 4/15 1/10 1/30 7/30 2/15 1/15 1/30 3/10 1/3 1/5 1/10 2/15 1/30 2/15 4/15 0 4/15 1/5 4/15 1/10 1/10 1/3 7/30 3/10 1/3 3/10
    1/5 1/3 3/10 1/10 1/15 1/10 1/30 1/5 2/15 7/30 1/3 2/15 1/10 1/6 3/10 1/5 7/30 1/30 0 1/30 1/15 2/15 1/6 7/30 4/15 4/15 7/30
]
upper3 = [
    3/5 17/30 1/2 3/5 19/30 2/5 8/15 1/3 11/30 2/5 17/30 13/30 2/5 3/5 3/5 11/30 1/2 11/30 2/3 17/30 3/5 7/15 19/30 1/2 3/5 1/3 19/30
    3/5 2/3 13/30 19/30 1/3 2/5 17/30 7/15 11/30 3/5 19/30 7/15 2/5 8/15 17/30 11/30 19/30 13/30 2/3 17/30 8/15 13/30 13/30 3/5 1/2 8/15 8/15
    3/5 2/3 1/2 1/2 2/3 7/15 3/5 3/5 1/2 1/3 2/5 8/15 2/5 11/30 1/3 8/15 7/15 13/30 0 2/5 11/30 19/30 19/30 2/5 1/2 7/15 7/15
]
prob3 = IntervalProbabilities(; lower = lower3, upper = upper3)

prob = OrthogonalIntervalProbabilities((prob1, prob2, prob3), (Int32(3), Int32(3), Int32(3)))
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

Return the lower bound transition probabilities from a source state or source/action pair to a target axis.
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

Return the gap between upper and lower bound transition probabilities from a source state or source/action pair to a target axis.
"""
gap(p::OrthogonalIntervalProbabilities, l) = gap(p.probs[l])

"""
    gap(p::OrthogonalIntervalProbabilities, l, i, j)

Return the gap between upper and lower bound transition probabilities from a source state or source/action pair to a target state.
"""
gap(p::OrthogonalIntervalProbabilities, l, i, j) = gap(p.probs[l], i, j)

"""
    sum_lower(p::OrthogonalIntervalProbabilities, l) 

Return the sum of lower bound transition probabilities from a source state or source/action pair to all target states on one axis.
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

"""
    num_target(p::OrthogonalIntervalProbabilities)

Return the number of target states along each marginal.
"""
num_target(p::OrthogonalIntervalProbabilities) = ntuple(i -> num_target(p[i]), ndims(p))

stateptr(p::OrthogonalIntervalProbabilities) = UnitRange{Int32}(1, num_source(p) + 1)
Base.ndims(p::OrthogonalIntervalProbabilities{N}) where {N} = N

Base.getindex(p::OrthogonalIntervalProbabilities, i) = p.probs[i]
Base.lastindex(p::OrthogonalIntervalProbabilities) = ndims(p)
Base.firstindex(p::OrthogonalIntervalProbabilities) = 1
Base.length(p::OrthogonalIntervalProbabilities) = ndims(p)
Base.iterate(p::OrthogonalIntervalProbabilities) = (p[1], 2)
Base.iterate(p::OrthogonalIntervalProbabilities, i) = i > ndims(p) ? nothing : (p[i], i + 1)
