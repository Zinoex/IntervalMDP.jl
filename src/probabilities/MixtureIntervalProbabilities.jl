"""
    MixtureIntervalProbabilities{N, P <: OrthogonalIntervalProbabilities, Q <: IntervalProbabilities}

A tuple of `OrthogonalIntervalProbabilities` for independent transition probabilities in a mixture that all share
the same source/action pairs, and target states. See [`OrthogonalIntervalProbabilities`](@ref) for more information on the structure of the transition probabilities
for each model in the mixture. The mixture is weighted by an [`IntervalProbabilities`](@ref) ambiguity set, called `weighting_probs`.

### Fields
- `mixture_probs::NTuple{N, P}`: A tuple of `OrthogonalIntervalProbabilities` transition probabilities along each axis.
- `weighting_probs::Q`: The weighting ambiguity set for the mixture.

### Examples
Below is a simple example of a mixture of two `OrthogonalIntervalProbabilities` with one dimension and the same source/action pairs and target states,
and a weighting ambiguity set.
```jldoctest
prob1 = OrthogonalIntervalProbabilities(
    (
        IntervalProbabilities(;
            lower = [
                0.0 0.5
                0.1 0.3
                0.2 0.1
            ],
            upper = [
                0.5 0.7
                0.6 0.5
                0.7 0.3
            ],
        ),
    ),
    (Int32(2),),
)
prob2 = OrthogonalIntervalProbabilities(
    (
        IntervalProbabilities(;
            lower = [
                0.1 0.4
                0.2 0.2
                0.3 0.0
            ],
            upper = [
                0.4 0.6
                0.5 0.4
                0.6 0.2
            ],
        ),
    ),
    (Int32(2),),
)
weighting_probs = IntervalProbabilities(; lower = [
    0.3 0.5
    0.4 0.3
], upper = [
    0.8 0.7
    0.7 0.5
])
mixture_prob = MixtureIntervalProbabilities((prob1, prob2), weighting_probs)
```
"""
struct MixtureIntervalProbabilities{
    N,
    P <: OrthogonalIntervalProbabilities,
    Q <: IntervalProbabilities,
} <: AbstractIntervalProbabilities
    mixture_probs::NTuple{N, P}
    weighting_probs::Q

    function MixtureIntervalProbabilities(
        mixture_probs::NTuple{N, P},
        weighting_probs::Q,
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

        if num_target(weighting_probs) != N
            throw(
                DimensionMismatch(
                    "The dimensionality of the weighting ambiguity set must be equal to the number of mixture probabilities",
                ),
            )
        end

        if num_source(weighting_probs) != _num_source
            throw(
                DimensionMismatch(
                    "The number of source/action pairs in the weighting ambiguity set must be equal to the number of source/action pairs in the mixture probabilities",
                ),
            )
        end

        new{N, P, Q}(mixture_probs, weighting_probs)
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

Return ``k``-th `OrthogonalIntervalProbabilities` transition probabilities.
"""
mixture_probs(p::MixtureIntervalProbabilities, k) = p.mixture_probs[k]

"""
    weighting_probs(p::MixtureIntervalProbabilities)

Return the `IntervalProbabilities` weighting ambiguity set.
"""
weighting_probs(p::MixtureIntervalProbabilities) = p.weighting_probs

"""
    axes_source(p::MixtureIntervalProbabilities)

Return the valid range of indices for the source states or source/action pairs.
"""
axes_source(p::MixtureIntervalProbabilities) = axes_source(first(p.mixture_probs))

"""
    num_target(p::MixtureIntervalProbabilities)

Return the number of target states along each marginal.
"""
num_target(p::MixtureIntervalProbabilities) = num_target(first(p.mixture_probs))

stateptr(p::MixtureIntervalProbabilities) = stateptr(first(p.mixture_probs))
Base.ndims(p::MixtureIntervalProbabilities{N}) where {N} = N

Base.getindex(p::MixtureIntervalProbabilities, k) = mixture_probs(p, k)
Base.lastindex(p::MixtureIntervalProbabilities) = ndims(p)
Base.firstindex(p::MixtureIntervalProbabilities) = 1
Base.length(p::MixtureIntervalProbabilities) = ndims(p)
Base.iterate(p::MixtureIntervalProbabilities) = (p[1], 2)
Base.iterate(p::MixtureIntervalProbabilities, k) = k > ndims(p) ? nothing : (p[k], k + 1)
