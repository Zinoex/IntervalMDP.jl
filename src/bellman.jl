"""
    bellman(V, prob; upper_bound = false)

Compute robust Bellman update with the value function `V` and the interval probabilities `prob` 
that upper or lower bounds the expectation of the value function `V` via O-maximization [1].
Whether the expectation is maximized or minimized is determined by the `upper_bound` keyword argument.
That is, if `upper_bound == true` then an upper bound is computed and if `upper_bound == false` then a lower
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
Vcur = bellman(Vprev, prob; upper_bound = false)
```

!!! note
    This function will construct a workspace object and an output vector.
    For a hot-loop, it is more efficient to use `bellman!` and pass in pre-allocated objects.

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function bellman(V, prob; upper_bound = false)
    Vres = similar(V, num_source(prob))
    return bellman!(Vres, V, prob; upper_bound = upper_bound)
end

"""
    bellman!(workspace, strategy_cache, Vres, V, prob, stateptr; upper_bound = false, maximize = true)

Compute in-place robust Bellman update with the value function `V` and the interval probabilities
`prob` that upper or lower bounds the expectation of the value function `V` via O-maximization [1].
Whether the expectation is maximized or minimized is determined by the `upper_bound` keyword argument.
That is, if `upper_bound == true` then an upper bound is computed and if `upper_bound == false` then a lower
bound is computed. 

The output is constructed in the input `Vres` and returned. The workspace object is also modified,
and depending on the type, the strategy cache may be modified as well. See [`construct_workspace`](@ref)
and [`construct_strategy_cache`](@ref) for more details on how to pre-allocate the workspace and strategy cache.

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
strategy_cache = construct_strategy_cache(NoStrategyConfig())
Vres = similar(V)

Vres = bellman!(workspace, strategy_cache, Vres, V, prob; upper_bound = false, maximize = true)
```

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function bellman! end

function bellman!(Vres, V, prob; upper_bound = false)
    workspace = construct_workspace(prob)
    strategy_cache = NoStrategyCache()
    return bellman!(workspace, strategy_cache, Vres, V, prob; upper_bound = upper_bound)
end

function bellman!(workspace, strategy_cache, Vres, V, prob; upper_bound = false)
    return bellman!(
        workspace,
        strategy_cache,
        Vres,
        V,
        prob,
        stateptr(prob);
        upper_bound = upper_bound,
    )
end

#########
# Dense #
#########
function bellman!(
    workspace::DenseWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    prob::IntervalProbabilities,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    # rev=true for upper bound
    sortperm!(workspace.permutation, V; rev = upper_bound)

    for jₛ in 1:(length(stateptr) - 1)
        act, perm = workspace.actions, workspace.permutation
        bellman_dense!(act, perm, strategy_cache, Vres, V, V, prob, stateptr, jₛ, jₛ, maximize)
    end

    return Vres
end

function bellman!(
    workspace::ThreadedDenseWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    prob::IntervalProbabilities,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    # rev=true for upper bound
    sortperm!(workspace.permutation, V; rev = upper_bound)

    @threadstid tid for jₛ in 1:(length(stateptr) - 1)
        @inbounds act, perm = workspace.actions[tid], workspace.permutation
        bellman_dense!(act, perm, strategy_cache, Vres, V, V, prob, stateptr, jₛ, jₛ, maximize)
    end

    return Vres
end

function bellman_dense!(actions, permutation, strategy_cache, Vres, V, Vₒ, prob, stateptr, jₛ, sidx, maximize)
    @inbounds begin
        s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
        actions = @view actions[1:(s₂ - s₁)]
        for (i, jₐ) in enumerate(s₁:(s₂ - 1))
            lowerⱼ = @view lower(prob)[:, jₐ]
            gapⱼ = @view gap(prob)[:, jₐ]
            used = sum_lower(prob)[jₐ]

            actions[i] = dot(Vₒ, lowerⱼ) + gap_value(Vₒ, gapⱼ, used, permutation)
        end

        Vres[sidx] = extract_strategy!(strategy_cache, actions, V, sidx, sidx, s₁, maximize)
    end
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
function bellman!(
    workspace::SparseWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    prob,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    for jₛ in 1:(length(stateptr) - 1)
        bellman_sparse!(workspace, strategy_cache, Vres, V, V, prob, stateptr, jₛ, jₛ, upper_bound, maximize)
    end

    return Vres
end

function bellman!(
    workspace::ThreadedSparseWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    prob,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    @threadstid tid for jₛ in 1:(length(stateptr) - 1)
        @inbounds ws = workspace.thread_workspaces[tid]
        bellman_sparse!(ws, strategy_cache, Vres, V, V, prob, stateptr, jₛ, jₛ, upper_bound, maximize)
    end

    return Vres
end

function bellman_sparse!(workspace, strategy_cache, Vres, V, Vₒ, prob, stateptr, jₛ, sidx, upper_bound, maximize)
    @inbounds begin
        s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
        action_values = @view workspace.actions[1:(s₂ - s₁)]

        for (i, jₐ) in enumerate(s₁:(s₂ - 1))
            lowerⱼ = @view lower(prob)[:, jₐ]
            gapⱼ = @view gap(prob)[:, jₐ]
            used = sum_lower(prob)[jₐ]

            Vp_workspace = @view workspace.values_gaps[1:nnz(gapⱼ)]
            for (i, (V, p)) in
                enumerate(zip(@view(Vₒ[SparseArrays.nonzeroinds(gapⱼ)]), nonzeros(gapⱼ)))
                Vp_workspace[i] = (V, p)
            end

            # rev=true for upper bound
            sort!(Vp_workspace; rev = upper_bound, by = first)

            action_values[i] = dot(Vₒ, lowerⱼ) + gap_value(Vp_workspace, used)
        end

        Vres[sidx] = extract_strategy!(strategy_cache, action_values, V, sidx, sidx, s₁, maximize)
    end
end

function gap_value(Vp, sum_lower)
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

# Dense
function bellman!(
    workspace::DenseOrthogonalWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    prob::OrthogonalIntervalProbabilities,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    @inbounds for other_index in eachotherindex(V, [workspace.state_index + i - 1 for i in 1:ndims(prob)])
        Vₒ = selectotherdims(V, workspace.state_index, ndims(prob), other_index)

        for (jₛ_cart, jₛ_linear) in zip(CartesianIndices(axes(Vₒ)), LinearIndices(axes(Vₒ)))
            sidx_cart = state_index(workspace, jₛ_cart, other_index)
            sidx_linear = state_index(workspace, jₛ_linear, other_index)

            s₁, s₂ = stateptr[jₛ_linear], stateptr[jₛ_linear + 1]
            actions = @view workspace.actions[1:(s₂ - s₁)]
            for (i, jₐ) in enumerate(s₁:(s₂ - 1))
                Vₑ = workspace.expectation_cache

                # TODO: Develop memory efficient version based on recursion
                # - Recursion from the top: right to left axis for cache purposes.
                # - Carry Vₒ through to the lowest level.
                # - At the lowest level: load Vₒ to Vₑ[1], compute expectation, return.
                # - For level i, i ≥ 2: load via lower computation until Vₑ[i] is full. Then compute expectation and store in Vₑ[i + 1]
                # - At the top level: return expectation

                product_inner_bellman!(workspace, Vₒ, Vₑ, prob[1], jₐ, upper_bound)

                for d in 2:ndims(prob)
                    Vᵣ = @view Vₑ[1, (Colon() for _ in 1:ndims(Vₑ) - 1)...]
                    product_inner_bellman!(workspace, Vₑ, Vᵣ, prob[d], jₐ, upper_bound)
                    Vₑ = @view Vₑ[1, (Colon() for _ in 1:ndims(Vₑ) - 1)...]
                end

                actions[i] = Vₑ[1]
            end

            Vres[sidx_cart] = extract_strategy!(strategy_cache, actions, V, sidx_cart, sidx_linear, s₁, maximize)
        end
    end

    return Vres
end

function product_inner_bellman!(workspace::DenseOrthogonalWorkspace, Vₒ::VO, Vₑ::VE, prob::IntervalProbabilities{T}, jₐ::Integer, upper_bound::Bool) where {T, VO <: AbstractArray{T}, VE <: AbstractArray{T}}
    @inbounds for inner_other_index in eachotherindex(Vₒ, 1)
        Vᵢ = @view Vₒ[:, inner_other_index]
        perm = @view workspace.permutation[1:length(Vᵢ)]

        # rev=true for upper bound
        sortperm!(perm, Vᵢ; rev = upper_bound)

        lowerⱼ = @view lower(prob)[:, jₐ]
        gapⱼ = @view gap(prob)[:, jₐ]
        used = sum_lower(prob)[jₐ]

        Vₑ[inner_other_index] = dot(Vᵢ, lowerⱼ) + gap_value(Vᵢ, gapⱼ, used, perm)
    end
end
