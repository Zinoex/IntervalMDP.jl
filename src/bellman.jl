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
    Vres = similar(V, source_shape(prob))
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
    sortperm!(workspace.permutation, V; rev = upper_bound, scratch = workspace.scratch)

    for jₛ in 1:(length(stateptr) - 1)
        act, perm = workspace.actions, workspace.permutation
        bellman_dense!(act, perm, strategy_cache, Vres, V, prob, stateptr, jₛ, maximize)
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
    sortperm!(workspace.permutation, V; rev = upper_bound, scratch = workspace.scratch)

    @threadstid tid for jₛ in 1:(length(stateptr) - 1)
        @inbounds act, perm = workspace.actions[tid], workspace.permutation
        bellman_dense!(act, perm, strategy_cache, Vres, V, prob, stateptr, jₛ, maximize)
    end

    return Vres
end

function bellman_dense!(
    actions,
    permutation,
    strategy_cache,
    Vres,
    V,
    prob,
    stateptr,
    jₛ,
    maximize,
)
    @inbounds begin
        s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
        actions = @view actions[1:(s₂ - s₁)]
        for (i, jₐ) in enumerate(s₁:(s₂ - 1))
            lowerⱼ = @view lower(prob)[:, jₐ]
            gapⱼ = @view gap(prob)[:, jₐ]
            used = sum_lower(prob)[jₐ]

            actions[i] = dot(V, lowerⱼ) + gap_value(V, gapⱼ, used, permutation)
        end

        Vres[jₛ] = extract_strategy!(strategy_cache, actions, V, jₛ, s₁, maximize)
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
        bellman_sparse!(
            workspace,
            strategy_cache,
            Vres,
            V,
            prob,
            stateptr,
            jₛ,
            upper_bound,
            maximize,
        )
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
        bellman_sparse!(
            ws,
            strategy_cache,
            Vres,
            V,
            prob,
            stateptr,
            jₛ,
            upper_bound,
            maximize,
        )
    end

    return Vres
end

function bellman_sparse!(
    workspace,
    strategy_cache,
    Vres,
    V,
    prob,
    stateptr,
    jₛ,
    upper_bound,
    maximize,
)
    @inbounds begin
        s₁, s₂ = stateptr[jₛ], stateptr[jₛ + 1]
        action_values = @view workspace.actions[1:(s₂ - s₁)]

        for (i, jₐ) in enumerate(s₁:(s₂ - 1))
            lowerⱼ = @view lower(prob)[:, jₐ]
            gapⱼ = @view gap(prob)[:, jₐ]
            used = sum_lower(prob)[jₐ]

            Vp_workspace = @view workspace.values_gaps[1:nnz(gapⱼ)]
            for (i, (v, p)) in
                enumerate(zip(@view(V[SparseArrays.nonzeroinds(gapⱼ)]), nonzeros(gapⱼ)))
                Vp_workspace[i] = (v, p)
            end

            # rev=true for upper bound
            sort!(Vp_workspace; rev = upper_bound, by = first, scratch = workspace.scratch)

            action_values[i] = dot(V, lowerⱼ) + gap_value(Vp_workspace, used)
        end

        Vres[jₛ] = extract_strategy!(strategy_cache, action_values, V, jₛ, s₁, maximize)
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

# Dense orthogonal
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
    # Since sorting for the first level is shared among all higher levels, we can precompute it
    product_nstates = num_target(prob)

    # For each higher-level state in the product space
    for I in CartesianIndices(product_nstates[2:end])
        sort_dense_orthogonal(workspace, workspace.first_level_perm, V, I, upper_bound)
    end

    # For each source state
    @inbounds for (jₛ_cart, jₛ_linear) in
                  zip(CartesianIndices(axes(V)), LinearIndices(axes(V)))
        bellman_dense_orthogonal!(
            workspace,
            workspace.first_level_perm,
            strategy_cache,
            Vres,
            V,
            prob,
            stateptr,
            product_nstates,
            jₛ_cart,
            jₛ_linear;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function bellman!(
    workspace::ThreadedDenseOrthogonalWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    prob::OrthogonalIntervalProbabilities,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    # Since sorting for the first level is shared among all higher levels, we can precompute it
    product_nstates = num_target(prob)

    # For each higher-level state in the product space
    @threadstid tid for I in CartesianIndices(product_nstates[2:end])
        ws = workspace.thread_workspaces[tid]
        sort_dense_orthogonal(ws, workspace.first_level_perm, V, I, upper_bound)
    end

    # For each source state
    I_linear = LinearIndices(axes(V))
    @threadstid tid for jₛ_cart in CartesianIndices(axes(V))
        # We can't use @threadstid over a zip, so we need to manually index
        jₛ_linear = I_linear[jₛ_cart]
        
        ws = workspace.thread_workspaces[tid]

        bellman_dense_orthogonal!(
            ws,
            workspace.first_level_perm,
            strategy_cache,
            Vres,
            V,
            prob,
            stateptr,
            product_nstates,
            jₛ_cart,
            jₛ_linear;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function sort_dense_orthogonal(workspace, first_level_perm, V, I, upper_bound)
    @inbounds begin
        perm = @view workspace.permutation[axes(V, 1)]
        sortperm!(perm, @view(V[:, I]); rev = upper_bound, scratch = workspace.scratch)

        copyto!(@view(first_level_perm[:, I]), perm)
    end
end

function bellman_dense_orthogonal!(
    workspace,
    first_level_perm,
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    prob::OrthogonalIntervalProbabilities,
    stateptr,
    product_nstates,
    jₛ_cart,
    jₛ_linear;
    upper_bound = false,
    maximize = true
)
    @inbounds begin
        s₁, s₂ = stateptr[jₛ_linear], stateptr[jₛ_linear + 1]
        actions = @view workspace.actions[1:(s₂ - s₁)]
        for (i, jₐ) in enumerate(s₁:(s₂ - 1))
            Vₑ = workspace.expectation_cache

            # For each higher-level state in the product space
            for I in CartesianIndices(product_nstates[2:end])

                # For the first dimension, we need to copy the values from V
                v = orthogonal_inner_sorted_bellman!(
                    # Use shared first level permutation across threads
                    @view(first_level_perm[:, I]),
                    @view(V[:, I]),
                    prob[1],
                    jₐ,
                )
                Vₑ[1][I[1]] = v

                # For the remaining dimensions, if "full", compute expectation and store in the next level
                for d in 2:(ndims(prob) - 1)
                    if I[d - 1] == product_nstates[d]
                        v = orthogonal_inner_bellman!(
                            workspace,
                            Vₑ[d - 1],
                            prob[d],
                            jₐ,
                            upper_bound,
                        )
                        Vₑ[d][I[d]] = v
                    else
                        break
                    end
                end
            end

            # Last dimension
            v = orthogonal_inner_bellman!(workspace, Vₑ[end], prob[end], jₐ, upper_bound)
            actions[i] = v
        end

        Vres[jₛ_cart] = extract_strategy!(strategy_cache, actions, V, jₛ_cart, s₁, maximize)
    end
end

Base.@propagate_inbounds function orthogonal_inner_bellman!(
    workspace::Union{DenseOrthogonalWorkspace, ThreadDenseOrthogonalWorkspace},
    V,
    prob,
    jₐ,
    upper_bound::Bool,
)
    perm = @view workspace.permutation[1:length(V)]

    # rev=true for upper bound
    sortperm!(perm, V; rev = upper_bound, scratch = workspace.scratch)

    return orthogonal_inner_sorted_bellman!(perm, V, prob, jₐ)
end

Base.@propagate_inbounds function orthogonal_inner_sorted_bellman!(
    perm,
    V::VO,
    prob::IntervalProbabilities{T},
    jₐ::Integer,
) where {T, VO <: AbstractArray{T}}
    lowerⱼ = @view lower(prob)[:, jₐ]
    gapⱼ = @view gap(prob)[:, jₐ]
    used = sum_lower(prob)[jₐ]

    return dot(V, lowerⱼ) + gap_value(V, gapⱼ, used, perm)
end
