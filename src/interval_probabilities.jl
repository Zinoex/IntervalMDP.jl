"""
    IntervalProbabilities{R, VR <: AbstractVector{R}, MR <: AbstractMatrix{R}}

A matrix pair to represent the lower and upper bound transition probabilities from a source state or source/action pair to a target state.
The matrices can be `Matrix{R}` or `SparseMatrixCSC{R}`, or their CUDA equivalents. For memory efficiency, it is recommended to use sparse matrices.

The columns represent the source and the rows represent the target, as if the probability matrix was a linear transformation.
Mathematically, let ``P`` be the probability matrix. Then ``P_{ij}`` represents the probability of transitioning from state ``j`` (or with state/action pair ``j``) to state ``i``.
Due to the column-major format of Julia, this is also a more efficient representation (in terms of cache locality).

The lower bound is explicitly stored, while the upper bound is computed from the lower bound and the gap. This choice is 
because it simplifies repeated probability assignment using O-maximization [1].

### Fields
- `lower::MR`: The lower bound transition probabilities from a source state or source/action pair to a target state.
- `gap::MR`: The gap between upper and lower bound transition probabilities from a source state or source/action pair to a target state.
- `sum_lower::VR`: The sum of lower bound transition probabilities from a source state or source/action pair to all target states.

### Examples
```jldoctest
dense_prob = IntervalProbabilities(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
)

sparse_prob = IntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(15, [4, 10], [0.1, 0.2]),
        SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
    ),
    upper = sparse_hcat(
        SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
        SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
    ),
)
```

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
struct IntervalProbabilities{R, VR <: AbstractVector{R}, MR <: AbstractMatrix{R}}
    lower::MR
    gap::MR

    sum_lower::VR
end

# Constructor from lower and gap with sanity assertions
function IntervalProbabilities(lower::MR, gap::MR) where {R, MR <: AbstractMatrix{R}}
    sum_lower = vec(sum(lower; dims = 1))

    max_lower_bound = maximum(sum_lower)
    @assert max_lower_bound <= 1 "The joint lower bound transition probability per column (max is $max_lower_bound) should be less than or equal to 1."

    sum_upper = vec(sum(lower + gap; dims = 1))

    max_upper_bound = minimum(sum_upper)
    @assert max_upper_bound >= 1 "The joint upper bound transition probability per column (min is $max_upper_bound) should be greater than or equal to 1."

    return IntervalProbabilities(lower, gap, sum_lower)
end

# Keyword constructor from lower and upper
function IntervalProbabilities(; lower::MR, upper::MR) where {MR <: AbstractMatrix}
    lower, gap = compute_gap(lower, upper)
    return IntervalProbabilities(lower, gap)
end

function compute_gap(lower::MR, upper::MR) where {MR <: AbstractMatrix}
    gap = upper - lower
    return lower, gap
end

function compute_gap(
    lower::MR,
    upper::MR,
) where {R, MR <: SparseArrays.AbstractSparseMatrixCSC{R}}
    I, J, _ = findnz(upper)

    gap_nonzeros = Vector{R}(undef, length(I))
    lower_nonzeros = Vector{R}(undef, length(I))

    for (k, (i, j)) in enumerate(zip(I, J))
        gap_nonzeros[k] = upper[i, j] - lower[i, j]
        lower_nonzeros[k] = lower[i, j]
    end

    gap = SparseArrays.FixedSparseCSC(
        size(upper)...,
        upper.colptr,
        upper.rowval,
        gap_nonzeros,
    )
    lower = SparseArrays.FixedSparseCSC(
        size(upper)...,
        upper.colptr,
        upper.rowval,
        lower_nonzeros,
    )
    return lower, gap
end

# Accessors for properties of interval probabilities

Base.size(s::IntervalProbabilities) = size(s.lower)
Base.size(s::IntervalProbabilities, dim::Integer) = size(s.lower, dim)

"""
    lower(s::IntervalProbabilities)

Return the lower bound transition probabilities from a source state or source/action pair to a target state.
"""
lower(s::IntervalProbabilities) = s.lower

"""
    upper(s::IntervalProbabilities)

Return the upper bound transition probabilities from a source state or source/action pair to a target state.

!!! note
    It is not recommended to use this function for the hot loop of O-maximization, because it is not just an accessor and requires allocation and computation.
"""
upper(s::IntervalProbabilities) = s.lower + s.gap

"""
    gap(s::IntervalProbabilities)

Return the gap between upper and lower bound transition probabilities from a source state or source/action pair to a target state.
"""
gap(s::IntervalProbabilities) = s.gap

"""
    sum_lower(s::IntervalProbabilities) 

Return the sum of lower bound transition probabilities from a source state or source/action pair to all target states.
This is useful in efficiently implementing O-maximization, where we start with a lower bound probability assignment
and iteratively, according to the ordering, adding the gap until the sum of probabilities is 1.
"""
sum_lower(s::IntervalProbabilities) = s.sum_lower

"""
    num_source(s::IntervalProbabilities)

Return the number of source states or source/action pairs.
"""
num_source(s::IntervalProbabilities) = size(gap(s), 2)

"""
    num_target(s::IntervalProbabilities)

Return the number of target states.
"""
num_target(s::IntervalProbabilities) = size(gap(s), 1)

function interval_prob_hcat(
    T,
    transition_probs::Vector{<:IntervalProbabilities{R, VR, MR}},
) where {R, VR, MR <: AbstractMatrix{R}}
    l = mapreduce(lower, hcat, transition_probs)
    g = mapreduce(gap, hcat, transition_probs)

    sl = mapreduce(sum_lower, vcat, transition_probs)

    lengths = map(num_source, transition_probs)
    stateptr = T[1; cumsum(lengths) .+ 1]

    return IntervalProbabilities(l, g, sl), stateptr
end
