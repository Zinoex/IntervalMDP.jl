"""
    ominmax(prob, V; max = true)

Compute probability assignment within the interval probabilities `prob` that upper or lower bounds
the expectation of the value function `V` via O-maximization [1]. Whether the expectation is
maximized or minimized is determined by the `max` keyword argument. That is, if `max == true`
then an upper bound is computed and if `max == false` then a lower bound is computed.

### Examples
```jldoctest
prob = IntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(15, [4, 10], [0.1, 0.2]),
        SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
    ),
    upper = sparse_hcat(
        SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
        SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
    ),
)

V = collect(1:15)
p = ominmax(prob, V; max = true)
```

!!! note
    This function will construct a workspace object for the ordering and an output vector.
    For a hot-loop, it is more efficient to use `ominmax!` and pass in pre-allocated objects.
    See [`construct_ordering`](@ref) for how to pre-allocate the workspace.

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function ominmax(prob, V; max = true)
    ordering = construct_ordering(prob)
    return ominmax!(ordering, prob, V; max = max)
end

function ominmax!(ordering::AbstractStateOrdering, prob, V; max = true)
    p = deepcopy(gap(prob))
    return ominmax!(ordering, p, prob, V; max = max)
end

"""
    ominmax!(ordering, p, prob, V; max = true)

Compute in-place the probability assignment within the interval probabilities `prob` that upper or
lower bounds the expectation of the value function `V` via O-maximization [1]. Whether the
expectation is maximized or minimized is determined by the `max` keyword argument. That is, if
`max == true` then an upper bound is computed and if `max == false` then a lower bound is computed.

The output is constructed in the input vector `p` and returned. The ordering workspace object
is also modified.

### Examples

```jldoctest
prob = IntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(15, [4, 10], [0.1, 0.2]),
        SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
    ),
    upper = sparse_hcat(
        SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
        SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
    ),
)

V = collect(1:15)
ordering = construct_ordering(prob)
p = deepcopy(gap(p))

p = ominmax!(ordering, p, prob, V; max = true)
```

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function ominmax!(ordering::AbstractStateOrdering, p, prob, V; max = true)
    sort_states!(ordering, V; max = max)
    probability_assignment!(p, prob, ordering)

    return p
end

"""
    partial_ominmax(prob, V, indices; max = true)

Perform O-maximization on a subset of source states or source/action pairs according to
`indices`. This corresponds to the columns in `prob`. See [`ominmax`](@ref) for more details
on what O-maximization is.

!!! note
    This function will construct a workspace object for the ordering and an output vector.
    For a hot-loop, it is more efficient to use `ominmax!` and pass in pre-allocated objects.
    See [`construct_ordering`](@ref) for how to pre-allocate the workspace.
"""
function partial_ominmax(prob, V, indices; max = true)
    ordering = construct_ordering(prob)
    return partial_ominmax!(ordering, prob, V, indices; max = max)
end

function partial_ominmax!(ordering::AbstractStateOrdering, prob, V, indices; max = true)
    p = deepcopy(gap(prob))
    return partial_ominmax!(ordering, p, prob, V, indices; max = max)
end

"""
    partial_ominmax!(ordering, p, prob, V, indices; max = true)

Perform O-maximization in-place on a subset of source states or source/action pairs according to
`indices`. This corresponds to the columns in `prob`. See [`ominmax`](@ref) for more details
on what O-maximization is.
"""
function partial_ominmax!(ordering::AbstractStateOrdering, p, prob, V, indices; max = true)
    sort_states!(ordering, V; max = max)
    probability_assignment!(p, prob, ordering, indices)

    return p
end

# Assign probabilities to the states in the ordering.
function probability_assignment!(
    p::MR,
    prob::IntervalProbabilities{R},
    ordering::AbstractStateOrdering,
) where {R, MR <: AbstractMatrix{R}}
    return probability_assignment!(p, prob, ordering, axes(p, 2))
end

function probability_assignment!(
    p::MR,
    prob::IntervalProbabilities{R},
    ordering::AbstractStateOrdering,
    indices,
) where {R, MR <: AbstractMatrix{R}}
    @inbounds copyto!(p, lower(prob))

    Threads.@threads for j in indices
        pⱼ = view(p, :, j)
        gⱼ = view(gap(prob), :, j)
        lⱼ = sum_lower(prob)[j]

        add_gap!(pⱼ, gⱼ, lⱼ, perm(ordering, j))
    end

    return p
end

function probability_assignment!(
    p::MR,
    prob::IntervalProbabilities{R},
    ordering::SparseOrdering,
    indices,
) where {R, MR <: AbstractSparseMatrix{R}}
    @inbounds copyto!(nonzeros(p), nonzeros(lower(prob)))
    g = gap(prob)

    Threads.@threads for j in indices
        # p and g must share nonzero structure.
        pⱼ = view(nonzeros(p), p.colptr[j]:(p.colptr[j + 1] - 1))
        gⱼ = view(nonzeros(g), p.colptr[j]:(p.colptr[j + 1] - 1))
        lⱼ = sum_lower(prob)[j]

        add_gap!(pⱼ, gⱼ, lⱼ, perm(ordering, j))
    end

    return p
end

# Shared
function add_gap!(p::VR, gap::VR, sum_lower::R, perm) where {R, VR <: AbstractVector{R}}
    remaining = 1.0 - sum_lower

    for i in perm
        @inbounds p[i] += gap[i]
        @inbounds remaining -= gap[i]
        if remaining < 0.0
            @inbounds p[i] += remaining
            remaining = 0.0
            break
        end
    end

    return p
end
