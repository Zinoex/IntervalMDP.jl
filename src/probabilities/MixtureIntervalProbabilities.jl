"""
    MixtureIntervalProbabilities{N, P <: OrthogonalIntervalProbabilities, Q <: IntervalProbabilities}

A tuple of `OrthogonalIntervalProbabilities` transition probabilities that all share the same source states, or source/action pairs, and target states.

### Fields
- `probs::NTuple{N, P}`: A tuple of `IntervalProbabilities` transition probabilities along each axis.
- `source_dims::NTuple{N, Int32}`: The dimensions of the orthogonal probabilities for the source axis. This is flattened to a single dimension for indexing.

### Examples
# TODO: Update example
```jldoctest
```
"""
struct MixtureIntervalProbabilities{
    N,
    P <: OrthogonalIntervalProbabilities,
    Q <: IntervalProbabilities,
} <: AbstractIntervalProbabilities
    mixture_probs::NTuple{N, P}
    weigthing_probs::Q

    function MixtureIntervalProbabilities(
        mixture_probs::NTuple{N, P},
        weigthing_probs::Q,
    ) where {N, P <: OrthogonalIntervalProbabilities, Q <: IntervalProbabilities}
        _source_shape, _num_source =
            source_shape(first(mixture_probs)), num_source(first(mixture_probs))

        for i in 2:N
            source_shape_i, num_source_i =
                source_shape(mixture_probs[i]), num_source(mixture_probs[i])

            if source_shape_i != _source_shape
                throw(
                    DimensionMismatch(
                        "All mixture probabilities must have the same source shape",
                    ),
                )
            end

            if num_source_i != _num_source
                throw(
                    DimensionMismatch(
                        "All mixture probabilities must have the same number of source/action pairs",
                    ),
                )
            end
        end

        if num_target(weigthing_probs) != N
            throw(
                DimensionMismatch(
                    "The dimensionality of the weigthing ambiguity set must be equal to the number of mixture probabilities",
                ),
            )
        end

        if num_source(weigthing_probs) != _num_source
            throw(
                DimensionMismatch(
                    "The number of source/action pairs in the weigthing ambiguity set must be equal to the number of source/action pairs in the mixture probabilities",
                ),
            )
        end

        new{N, P, Q}(mixture_probs, weigthing_probs)
    end
end

"""
    num_source(p::MixtureIntervalProbabilities)

Return the number of source states or source/action pairs.
"""
num_source(p::MixtureIntervalProbabilities) = num_source(first(p.mixture_probs))
source_shape(p::MixtureIntervalProbabilities) = source_shape(first(p.mixture_probs))

"""
    mixture_probs(p::MixtureIntervalProbabilities)

Return the tuple of `OrthogonalIntervalProbabilities` transition probabilities.
"""
mixture_probs(p::MixtureIntervalProbabilities) = p.mixture_probs

"""
    mixture_probs(p::MixtureIntervalProbabilities, k)

Return the tuple of `OrthogonalIntervalProbabilities` transition probabilities.
"""
mixture_probs(p::MixtureIntervalProbabilities, k) = p.mixture_probs[k]

"""
    weigthing_probs(p::MixtureIntervalProbabilities)

Return the `IntervalProbabilities` weigthing ambiguity set.
"""
weigthing_probs(p::MixtureIntervalProbabilities) = p.weigthing_probs

"""
    axes_source(p::MixtureIntervalProbabilities)

Return the valid range of indices for the source states or source/action pairs.
"""
axes_source(p::MixtureIntervalProbabilities) = axes_source(first(p.mixture_probs))

num_target(p::MixtureIntervalProbabilities) = num_target(first(p.mixture_probs))
stateptr(p::MixtureIntervalProbabilities) = stateptr(first(p.mixture_probs))
Base.ndims(p::MixtureIntervalProbabilities{N}) where {N} = N

Base.getindex(p::MixtureIntervalProbabilities, k) = mixture_probs(p, k)
Base.lastindex(p::MixtureIntervalProbabilities) = ndims(p)
Base.firstindex(p::MixtureIntervalProbabilities) = 1
Base.length(p::MixtureIntervalProbabilities) = ndims(p)
Base.iterate(p::MixtureIntervalProbabilities) = (p[1], 2)
Base.iterate(p::MixtureIntervalProbabilities, k) = k > ndims(p) ? nothing : (p[k], k + 1)
