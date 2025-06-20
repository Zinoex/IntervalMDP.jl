"""
    bellman(V, model; upper_bound = false)

Compute robust Bellman update with the value function `V` and the model `model`, e.g. [`IntervalMarkovDecisionProcess`](@ref),
that upper or lower bounds the expectation of the value function `V` via O-maximization [1].
Whether the expectation is maximized or minimized is determined by the `upper_bound` keyword argument.
That is, if `upper_bound == true` then an upper bound is computed and if `upper_bound == false` then a lower
bound is computed.

### Examples
```jldoctest
prob1 = IntervalProbabilities(;
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
)

prob2 = IntervalProbabilities(;
    lower = [
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ],
    upper = [
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ],
)

prob3 = IntervalProbabilities(; lower = [
    0.0
    0.0
    1.0
][:, :], upper = [
    0.0
    0.0
    1.0
][:, :])

transition_probs = [prob1, prob2, prob3]
istates = [Int32(1)]

model = IntervalMarkovDecisionProcess(transition_probs, istates)

Vprev = [1, 2, 3]
Vcur = bellman(Vprev, model; upper_bound = false)
```

!!! note
    This function will construct a workspace object and an output vector.
    For a hot-loop, it is more efficient to use `bellman!` and pass in pre-allocated objects.

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function bellman(V, model; upper_bound = false, maximize = true)
    Vres = similar(V, source_shape(model))
    return bellman!(Vres, V, model; upper_bound = upper_bound, maximize = maximize)
end

"""
    bellman!(workspace, strategy_cache, Vres, V, model, stateptr; upper_bound = false, maximize = true)

Compute in-place robust Bellman update with the value function `V` and the model `model`, 
e.g. [`IntervalMarkovDecisionProcess`](@ref), that upper or lower bounds the expectation of the value function `V` via O-maximization [1].
Whether the expectation is maximized or minimized is determined by the `upper_bound` keyword argument.
That is, if `upper_bound == true` then an upper bound is computed and if `upper_bound == false` then a lower
bound is computed. 

The output is constructed in the input `Vres` and returned. The workspace object is also modified,
and depending on the type, the strategy cache may be modified as well. See [`construct_workspace`](@ref)
and [`construct_strategy_cache`](@ref) for more details on how to pre-allocate the workspace and strategy cache.

### Examples

```jldoctest
prob1 = IntervalProbabilities(;
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
)

prob2 = IntervalProbabilities(;
    lower = [
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ],
    upper = [
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ],
)

prob3 = IntervalProbabilities(; lower = [
    0.0
    0.0
    1.0
][:, :], upper = [
    0.0
    0.0
    1.0
][:, :])

transition_probs = [prob1, prob2, prob3]
istates = [Int32(1)]

model = IntervalMarkovDecisionProcess(transition_probs, istates)

V = [1, 2, 3]
workspace = construct_workspace(model)
strategy_cache = construct_strategy_cache(NoStrategyConfig())
Vres = similar(V)

Vres = bellman!(workspace, strategy_cache, Vres, V, model; upper_bound = false, maximize = true)
```

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
function bellman! end

function bellman!(Vres, V, model; upper_bound = false, maximize = true)
    workspace = construct_workspace(model)
    strategy_cache = NoStrategyCache()
    return bellman!(
        workspace,
        strategy_cache,
        Vres,
        V,
        model;
        upper_bound = upper_bound,
        maximize = maximize,
    )
end

function bellman!(
    workspace,
    strategy_cache,
    Vres,
    V,
    model;
    upper_bound = false,
    maximize = true,
)
    return _bellman_helper!(
        workspace,
        strategy_cache,
        Vres,
        V,
        transition_prob(model),
        stateptr(model);
        upper_bound = upper_bound,
        maximize = maximize,
    )
end

######################################################
# Bellman operator for IntervalMarkovDecisionProcess #
######################################################

# Non-threaded
function _bellman_helper!(
    workspace::Union{DenseWorkspace, SparseWorkspace},
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    prob::IntervalProbabilities,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    bellman_precomputation!(workspace, V, upper_bound)

    for jₛ in 1:(length(stateptr) - 1)
        state_bellman!(
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

# Threaded
function _bellman_helper!(
    workspace::Union{ThreadedDenseWorkspace, ThreadedSparseWorkspace},
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    prob::IntervalProbabilities,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    bellman_precomputation!(workspace, V, upper_bound)

    @threadstid tid for jₛ in 1:(length(stateptr) - 1)
        @inbounds ws = workspace[tid]
        state_bellman!(
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

function bellman_precomputation!(
    workspace::Union{DenseWorkspace, ThreadedDenseWorkspace},
    V,
    upper_bound,
)
    # rev=true for upper bound
    sortperm!(permutation(workspace), V; rev = upper_bound, scratch = scratch(workspace))
end

bellman_precomputation!(
    workspace::Union{SparseWorkspace, ThreadedSparseWorkspace},
    V,
    upper_bound,
) = nothing

function state_bellman!(
    workspace::Union{DenseWorkspace, SparseWorkspace},
    strategy_cache::OptimizingStrategyCache,
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
        actions = @view workspace.actions[1:(s₂ - s₁)]

        for (i, jₐ) in enumerate(s₁:(s₂ - 1))
            actions[i] = state_action_bellman(workspace, V, prob, jₐ, upper_bound)
        end

        Vres[jₛ] = extract_strategy!(strategy_cache, actions, V, jₛ, maximize)
    end
end

function state_bellman!(
    workspace::Union{DenseWorkspace, SparseWorkspace},
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    prob,
    stateptr,
    jₛ,
    upper_bound,
    maximize,
)
    @inbounds begin
        s₁ = stateptr[jₛ]
        jₐ = s₁ + strategy_cache[jₛ] - 1
        Vres[jₛ] = state_action_bellman(workspace, V, prob, jₐ, upper_bound)
    end
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::DenseWorkspace,
    V,
    prob,
    jₐ,
    upper_bound,
)
    return dense_sorted_state_action_bellman(V, prob, jₐ, permutation(workspace))
end

Base.@propagate_inbounds function dense_sorted_state_action_bellman(V, prob, jₐ, perm)
    return dot(V, lower(prob, :, jₐ)) +
           gap_value(V, gap(prob, :, jₐ), sum_lower(prob, jₐ), perm)
end

Base.@propagate_inbounds function gap_value(
    V::AbstractVector{T},
    gap::VR,
    sum_lower,
    perm,
) where {T, VR <: AbstractVector}
    remaining = one(T) - sum_lower
    res = zero(T)

    @inbounds for i in perm
        p = min(remaining, gap[i])
        res += p * V[i]

        remaining -= p
        if remaining <= zero(T)
            break
        end
    end

    return res
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::SparseWorkspace,
    V,
    prob,
    jₐ,
    upper_bound,
)
    lowerⱼ = lower(prob, :, jₐ)
    gapⱼ = gap(prob, :, jₐ)
    used = sum_lower(prob)[jₐ]

    Vp_workspace = @view workspace.values_gaps[1:nnz(gapⱼ)]
    for (i, (v, p)) in
        enumerate(zip(@view(V[SparseArrays.nonzeroinds(gapⱼ)]), nonzeros(gapⱼ)))
        Vp_workspace[i] = (v, p)
    end

    # rev=true for upper bound
    sort!(Vp_workspace; rev = upper_bound, by = first, scratch = scratch(workspace))

    return dot(V, lowerⱼ) + gap_value(Vp_workspace, used)
end

Base.@propagate_inbounds function gap_value(
    Vp::VP,
    sum_lower,
) where {T, VP <: AbstractVector{<:Tuple{T, <:Real}}}
    remaining = one(T) - sum_lower
    res = zero(T)

    for (V, p) in Vp
        p = min(remaining, p)
        res += p * V

        remaining -= p
        if remaining <= zero(T)
            break
        end
    end

    return res
end

################################################################
# Bellman operator for OrthogonalIntervalMarkovDecisionProcess #
################################################################
function _bellman_helper!(
    workspace::Union{DenseOrthogonalWorkspace, SparseOrthogonalWorkspace, MixtureWorkspace},
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    prob,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    bellman_precomputation!(workspace, V, prob, upper_bound)

    # For each source state
    @inbounds for (jₛ_cart, jₛ_linear) in zip(
        CartesianIndices(source_shape(prob)),
        LinearIndices(source_shape(prob)),
    )
        state_bellman!(
            workspace,
            strategy_cache,
            Vres,
            V,
            prob,
            stateptr,
            jₛ_cart,
            jₛ_linear;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::Union{
        ThreadedDenseOrthogonalWorkspace,
        ThreadedSparseOrthogonalWorkspace,
        ThreadedMixtureWorkspace,
    },
    strategy_cache::AbstractStrategyCache,
    Vres,
    V,
    prob,
    stateptr;
    upper_bound = false,
    maximize = true,
)
    bellman_precomputation!(workspace, V, prob, upper_bound)

    # For each source state
    I_linear = LinearIndices(source_shape(prob))
    @threadstid tid for jₛ_cart in CartesianIndices(source_shape(prob))
        # We can't use @threadstid over a zip, so we need to manually index
        jₛ_linear = I_linear[jₛ_cart]

        ws = workspace[tid]

        state_bellman!(
            ws,
            strategy_cache,
            Vres,
            V,
            prob,
            stateptr,
            jₛ_cart,
            jₛ_linear;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function bellman_precomputation!(workspace::DenseOrthogonalWorkspace, V, prob, upper_bound)
    # Since sorting for the first level is shared among all higher levels, we can precompute it
    product_nstates = num_target(prob)

    # For each higher-level state in the product space
    for I in CartesianIndices(product_nstates[2:end])
        sort_dense_orthogonal(workspace, V, I, upper_bound)
    end
end

function bellman_precomputation!(
    workspace::ThreadedDenseOrthogonalWorkspace,
    V,
    prob,
    upper_bound,
)
    # Since sorting for the first level is shared among all higher levels, we can precompute it
    product_nstates = num_target(prob)

    # For each higher-level state in the product space
    @threadstid tid for I in CartesianIndices(product_nstates[2:end])
        ws = workspace[tid]
        sort_dense_orthogonal(ws, V, I, upper_bound)
    end
end

bellman_precomputation!(
    workspace::Union{SparseOrthogonalWorkspace, ThreadedSparseOrthogonalWorkspace},
    V,
    prob,
    upper_bound,
) = nothing

function sort_dense_orthogonal(workspace, V, I, upper_bound)
    @inbounds begin
        perm = @view workspace.permutation[axes(V, 1)]
        sortperm!(perm, @view(V[:, I]); rev = upper_bound, scratch = workspace.scratch)

        copyto!(@view(first_level_perm(workspace)[:, I]), perm)
    end
end

function state_bellman!(
    workspace::Union{DenseOrthogonalWorkspace, SparseOrthogonalWorkspace, MixtureWorkspace},
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    prob,
    stateptr,
    jₛ_cart,
    jₛ_linear;
    upper_bound,
    maximize,
)
    @inbounds begin
        s₁, s₂ = stateptr[jₛ_linear], stateptr[jₛ_linear + 1]
        act_vals = @view actions(workspace)[1:(s₂ - s₁)]

        for (i, jₐ) in enumerate(s₁:(s₂ - 1))
            act_vals[i] = state_action_bellman(workspace, V, prob, jₐ, upper_bound)
        end

        Vres[jₛ_cart] = extract_strategy!(strategy_cache, act_vals, V, jₛ_cart, maximize)
    end
end

function state_bellman!(
    workspace::Union{DenseOrthogonalWorkspace, SparseOrthogonalWorkspace, MixtureWorkspace},
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    prob,
    stateptr,
    jₛ_cart,
    jₛ_linear;
    upper_bound,
    maximize,
)
    @inbounds begin
        s₁ = stateptr[jₛ_linear]
        jₐ = s₁ + strategy_cache[jₛ_cart] - 1
        Vres[jₛ_cart] = state_action_bellman(workspace, V, prob, jₐ, upper_bound)
    end
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::DenseOrthogonalWorkspace,
    V,
    prob,
    jₐ,
    upper_bound,
)
    # The only dimension
    if ndims(prob) == 1
        return dense_sorted_state_action_bellman(
            V,
            prob[1],
            jₐ,
            first_level_perm(workspace),
        )
    end

    Vₑ = workspace.expectation_cache
    product_nstates = num_target(prob)

    # For each higher-level state in the product space
    for I in CartesianIndices(product_nstates[2:end])

        # For the first dimension, we need to copy the values from V
        v = dense_sorted_state_action_bellman(
            @view(V[:, I]),
            prob[1],
            jₐ,
            # Use shared first level permutation across threads
            @view(first_level_perm(workspace)[:, I]),
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

    return v
end

Base.@propagate_inbounds function orthogonal_inner_bellman!(
    workspace,
    V,
    prob,
    jₐ,
    upper_bound::Bool,
)
    perm = @view permutation(workspace)[1:length(V)]

    # rev=true for upper bound
    sortperm!(perm, V; rev = upper_bound, scratch = scratch(workspace))

    return dense_sorted_state_action_bellman(V, prob, jₐ, perm)
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::SparseOrthogonalWorkspace,
    V,
    prob,
    jₐ,
    upper_bound,
)
    # This function uses ntuple excessively to avoid allocations (list comprehension requires allocation, while ntuple does not)
    nzinds_first = SparseArrays.nonzeroinds(gap(prob, 1, :, jₐ))
    nzinds_per_prob =
        ntuple(i -> SparseArrays.nonzeroinds(gap(prob, i + 1, :, jₐ)), ndims(prob) - 1)

    lower_nzvals_per_prob = ntuple(i -> nonzeros(lower(prob, i, :, jₐ)), ndims(prob))
    gap_nzvals_per_prob = ntuple(i -> nonzeros(gap(prob, i, :, jₐ)), ndims(prob))
    sum_lower_per_prob = ntuple(i -> sum_lower(prob, i, jₐ), ndims(prob))

    nnz_per_prob = ntuple(i -> nnz(gap(prob, i, :, jₐ)), ndims(prob))
    Vₑ = ntuple(
        i -> @view(workspace.expectation_cache[i][1:nnz_per_prob[i + 1]]),
        ndims(prob) - 1,
    )

    if ndims(prob) == 1
        # The only dimension
        return orthogonal_sparse_inner_bellman!(
            workspace,
            @view(V[nzinds_first]),
            lower_nzvals_per_prob[end],
            gap_nzvals_per_prob[end],
            sum_lower_per_prob[end],
            upper_bound,
        )
    end

    # For each higher-level state in the product space
    for I in CartesianIndices(nnz_per_prob[2:end])
        Isparse = CartesianIndex(ntuple(d -> nzinds_per_prob[d][I[d]], ndims(prob) - 1))

        # For the first dimension, we need to copy the values from V
        v = orthogonal_sparse_inner_bellman!(
            workspace,
            @view(V[nzinds_first, Isparse]),
            lower_nzvals_per_prob[1],
            gap_nzvals_per_prob[1],
            sum_lower_per_prob[1],
            upper_bound,
        )
        Vₑ[1][I[1]] = v

        # For the remaining dimensions, if "full", compute expectation and store in the next level
        for d in 2:(ndims(prob) - 1)
            if I[d - 1] == nnz_per_prob[d]
                v = orthogonal_sparse_inner_bellman!(
                    workspace,
                    Vₑ[d - 1],
                    lower_nzvals_per_prob[d],
                    gap_nzvals_per_prob[d],
                    sum_lower_per_prob[d],
                    upper_bound,
                )
                Vₑ[d][I[d]] = v
            else
                break
            end
        end
    end

    # Last dimension
    v = orthogonal_sparse_inner_bellman!(
        workspace,
        Vₑ[end],
        lower_nzvals_per_prob[end],
        gap_nzvals_per_prob[end],
        sum_lower_per_prob[end],
        upper_bound,
    )

    return v
end

Base.@propagate_inbounds function orthogonal_sparse_inner_bellman!(
    workspace::SparseOrthogonalWorkspace,
    V,
    lower,
    gap,
    sum_lower,
    upper_bound::Bool,
)
    Vp_workspace = @view workspace.values_gaps[1:length(gap)]
    for (i, (v, p)) in enumerate(zip(V, gap))
        Vp_workspace[i] = (v, p)
    end

    # rev=true for upper bound
    sort!(Vp_workspace; rev = upper_bound, scratch = scratch(workspace))

    return dot(V, lower) + gap_value(Vp_workspace, sum_lower)
end

#############################################################
# Bellman operator for MixtureIntervalMarkovDecisionProcess #
#############################################################
bellman_precomputation!(workspace::MixtureWorkspace, V, prob, upper_bound) =
    bellman_precomputation!(workspace.orthogonal_workspace, V, prob, upper_bound)

function bellman_precomputation!(
    workspace::ThreadedMixtureWorkspace{<:DenseOrthogonalWorkspace},
    V,
    prob,
    upper_bound,
)
    # Since sorting for the first level is shared among all higher levels, we can precompute it
    product_nstates = num_target(prob)

    # For each higher-level state in the product space
    @threadstid tid for I in CartesianIndices(product_nstates[2:end])
        ws = workspace[tid]
        sort_dense_orthogonal(ws.orthogonal_workspace, V, I, upper_bound)
    end
end

bellman_precomputation!(
    workspace::ThreadedMixtureWorkspace{<:SparseOrthogonalWorkspace},
    V,
    prob,
    upper_bound,
) = nothing

Base.@propagate_inbounds function state_action_bellman(
    workspace::MixtureWorkspace,
    V,
    prob,
    jₐ,
    upper_bound,
)
    # Value iteration for each model in the mixture (for source-action pair jₐ)
    for (k, p) in enumerate(prob)
        v = state_action_bellman(workspace.orthogonal_workspace, V, p, jₐ, upper_bound)
        workspace.mixture_cache[k] = v
    end

    # Combine mixture with weighting probabilities
    v = orthogonal_inner_bellman!(
        workspace,
        workspace.mixture_cache,
        weighting_probs(prob),
        jₐ,
        upper_bound,
    )

    return v
end
