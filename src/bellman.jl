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
    This function will construct a workspace object and an output vector.
    For a hot-loop, it is more efficient to use `bellman!` and pass in pre-allocated objects.
    See [`construct_workspace`](@ref) for how to pre-allocate the workspace.

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function bellman(V, prob; max = true)
    Vres = similar(V, num_source(prob))
    return bellman!(Vres, V, prob; max = max)
end

function bellman!(Vres, V, prob; max = true)
    workspace = construct_workspace(prob)
    return bellman!(workspace, Vres, V, prob; max = max)
end

#########
# Dense #
#########

"""
bellman!(ordering, Vres, V, prob; max = true)

Compute in-place robust Bellman update with the value function `V` and the interval probabilities
`prob` that upper or lower bounds the expectation of the value function `V` via O-maximization [1].
Whether the expectation is maximized or minimized is determined by the `max` keyword argument.
That is, if `max == true` then an upper bound is computed and if `max == false` then a lower
bound is computed.

The output is constructed in the input `Vres` and returned. The workspace object is also modified.

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
workspace = construct_workspace(prob)
Vres = similar(V)

Vres = bellman!(workspace, Vres, V, prob; max = true)
```

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function bellman!(workspace::AbstractDenseWorkspace, Vres, V, prob::IntervalProbabilities; max = true)
    # rev=true for maximization
    sortperm!(workspace.permutation, V; rev = max)
    value_assignment!(Vres, V, prob, workspace, axes_source(prob))

    return Vres
end

function value_assignment!(
    Vres,
    V,
    prob::IntervalProbabilities,
    workspace::DenseWorkspace,
    indices,
)
    l = lower(prob)
    g = gap(prob)

    @inbounds for j in indices
        lowerⱼ = @view l[:, j]
        gapⱼ = @view g[:, j]
        remaining = sum_lower(prob)[j]

        Vres[j] = dot(V, lowerⱼ) + gap_value(V, gapⱼ, remaining, workspace.permutation)
    end

    return Vres
end

function value_assignment!(
    Vres,
    V,
    prob::IntervalProbabilities,
    workspace::ThreadedDenseWorkspace,
    indices,
)
    l = lower(prob)
    g = gap(prob)

    @inbounds Threads.@threads for j in indices
        lowerⱼ = @view l[:, j]
        gapⱼ = @view g[:, j]
        remaining = sum_lower(prob)[j]

        Vres[j] = dot(V, lowerⱼ) + gap_value(V, gapⱼ, remaining, workspace.permutation)
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

##########
# Sparse #
##########

function bellman!(workspace::ThreadedSparseWorkspace, Vres, V, prob; max = true)
    l = lower(prob)
    g = gap(prob)

    @inbounds @threadstid tid for j in axes_source(prob)
        ws = workspace.thread_workspaces[tid]

        lowerⱼ = @view l[:, j]
        gapⱼ = @view g[:, j]
        remaining = sum_lower(prob)[j]

        Vp_workspace = @view ws.values_gaps[1:nnz(gapⱼ)]
        for (i, (V, p)) in enumerate(zip(@view(V[SparseArrays.nonzeroinds(gapⱼ)]), nonzeros(gapⱼ)))
            Vp_workspace[i] = (V, p)
        end

        # rev=true for maximization
        sort!(Vp_workspace; rev = max, by=first)

        Vres[j] = dot(V, lowerⱼ) + gap_value(Vp_workspace, remaining)
    end

    return Vres
end

function bellman!(workspace::SparseWorkspace, Vres, V, prob; max = true)
    l = lower(prob)
    g = gap(prob)

    @inbounds for j in axes_source(prob)
        lowerⱼ = @view l[:, j]
        gapⱼ = @view g[:, j]
        remaining = sum_lower(prob)[j]

        Vp_workspace = @view workspace.values_gaps[1:nnz(gapⱼ)]
        for (i, (V, p)) in enumerate(zip(@view(V[SparseArrays.nonzeroinds(gapⱼ)]), nonzeros(gapⱼ)))
            Vp_workspace[i] = (V, p)
        end

        # rev=true for maximization
        sort!(Vp_workspace; rev = max, by=first)

        Vres[j] = dot(V, lowerⱼ) + gap_value(Vp_workspace, remaining)
    end

    return Vres
end

function gap_value(
    Vp,
    sum_lower,
)
    remaining = 1.0 - sum_lower
    res = 0.0

    @inbounds for (V, p) in Vp
        p = min(remaining, p)
        res += p * V

        remaining -= p
        if remaining <= 0.0
            break
        end
    end

    return res
end
