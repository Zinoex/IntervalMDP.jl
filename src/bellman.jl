"""
    bellman(V, prob; max = true)

Compute robust Bellman update with the value function `V` and the interval probabilities `prob` 
that upper or lower bounds the expectation of the value function `V` via O-maximization [1].
Whether the expectation is maximized or minimized is determined by the `max` keyword argument.
That is, if `max == true` then an upper bound is computed and if `max == false` then a lower
bound is computed.

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

Vprev = collect(1:15)
Vcur = bellman(Vprev, prob; max = true)
```

!!! note
    This function will construct a workspace object for the ordering and an output vector.
    For a hot-loop, it is more efficient to use `bellman!` and pass in pre-allocated objects.
    See [`construct_ordering`](@ref) for how to pre-allocate the workspace.

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function bellman(V, prob; max = true)
    ordering = construct_ordering(prob)
    return bellman!(ordering, V, prob; max = max)
end

function bellman!(ordering::AbstractStateOrdering, V, prob; max = true)
    Vres = similar(V, num_source(prob))
    return bellman!(ordering, Vres, V, prob; max = max)
end

"""
bellman!(ordering, Vres, V, prob; max = true)

Compute in-place robust Bellman update with the value function `V` and the interval probabilities
`prob` that upper or lower bounds the expectation of the value function `V` via O-maximization [1].
Whether the expectation is maximized or minimized is determined by the `max` keyword argument.
That is, if `max == true` then an upper bound is computed and if `max == false` then a lower
bound is computed.

The output is constructed in the input `Vres` and returned. The ordering workspace object
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
Vres = similar(V)

Vres = bellman!(ordering, Vres, V, prob; max = true)
```

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function bellman!(ordering::AbstractStateOrdering, Vres, V, prob; max = true)
    sort_states!(ordering, V; max = max)
    value_assignment!(Vres, V, prob, ordering)

    return Vres
end

# Assign values to the states in the ordering.
function value_assignment!(
    Vres,
    V,
    prob::IntervalProbabilities{R},
    ordering::AbstractStateOrdering,
) where {R}
    return value_assignment!(Vres, V, prob, ordering, axes_source(prob))
end

function value_assignment!(
    Vres,
    V,
    prob::IntervalProbabilities{R},
    ordering::AbstractStateOrdering,
    indices,
) where {R}
    l = lower(prob)
    g = gap(prob)

    @batch for j in indices
        lowerⱼ = @view l[:, j]
        gapⱼ = @view g[:, j]
        remaining = sum_lower(prob)[j]

        Vres[j] = dot(V, lowerⱼ) + gap_value(V, gapⱼ, remaining, perm(ordering, j))
    end

    return Vres
end

function gap_value(V, gap::VR, sum_lower, perm) where {VR <: AbstractVector}
    remaining = 1.0 - sum_lower
    res = 0.0

    @inbounds for i in perm
        p = min(remaining, gap[i])
        res += p * V[i]

        remaining -= p
        if remaining <= 0.0
            break
        end
    end

    return res
end

function gap_value(
    V,
    gap::VR,
    sum_lower,
    perm,
) where {Tv, Ti, VR <: SparseArrays.SparseColumnView{Tv, Ti}}
    remaining = 1.0 - sum_lower
    res = 0.0

    gap_vals, gap_inds = nonzeros(gap), SparseArrays.nonzeroinds(gap)

    @inbounds for i in perm
        p = min(remaining, gap_vals[i])
        res += p * V[gap_inds[i]]

        remaining -= p
        if remaining <= 0.0
            break
        end
    end

    return res
end
