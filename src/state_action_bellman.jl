# Typed Q-shape primitive: writes Qres[a, s] = opt_γ E_γ[V(·)] for every
# (a, s) the workspace iterates. Caller is responsible for the action
# reduction (`strategy!`). NonOptimizing strategy caches don't fit this
# shape (only one action per state is meaningful).
function bellman_q!(
    workspace,
    strategy_cache::OptimizingStrategyCache,
    Qres::StateActionValueArray,
    V::StateValueArray,
    model::IntervalMarkovProcess,
    update_sequence;
    upper_bound = false,
    maximize = true,
    prop = nothing,
)
    return _bellman_helper!(
        workspace,
        strategy_cache,
        parent(Qres),
        parent(V),
        model,
        update_sequence;
        upper_bound = upper_bound,
        maximize = maximize,
    )
end

###########################################################################
# O-Maximization-based Bellman operator for IntervalMarkovDecisionProcess #
###########################################################################

# Non-threaded
function _bellman_helper!(
    workspace::Union{DenseIntervalOMaxWorkspace, SparseIntervalOMaxWorkspace},
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound = false,
    maximize = true,
)
    bellman_precomputation!(workspace, V, upper_bound)

    marginal = marginals(model)[1]

    @inbounds for (jₐ, jₛ) in update_sequence
        ambiguity_set = marginal[jₐ, jₛ]
        budget = workspace.budget[sub2ind(marginal, jₐ, jₛ)]

        Vres[jₐ, jₛ] =
            state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)
    end

    return Vres
end

function _bellman_helper!(
    workspace::Union{DenseIntervalOMaxWorkspace, SparseIntervalOMaxWorkspace},
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound = false,
    maximize = true,
)
    bellman_precomputation!(workspace, V, upper_bound)

    marginal = marginals(model)[1]

    @inbounds for (jₐ, jₛ) in update_sequence
        ambiguity_set = marginal[jₐ, jₛ]
        budget = workspace.budget[sub2ind(marginal, jₐ, jₛ)]

        Vres[jₛ] =
            state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)
    end

    return Vres
end

# Threaded
function _bellman_helper!(
    workspace::Union{
        ThreadedDenseIntervalOMaxWorkspace,
        ThreadedSparseIntervalOMaxWorkspace,
    },
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound = false,
    maximize = true,
)
    @inbounds bellman_precomputation!(workspace, V, upper_bound)

    marginal = marginals(model)[1]

    @threadstid tid for (jₐ, jₛ) in update_sequence
        @inbounds ws = workspace[tid]

        @inbounds ambiguity_set = marginal[jₐ, jₛ]
        @inbounds budget = ws.budget[sub2ind(marginal, jₐ, jₛ)]
        @inbounds Vres[jₐ, jₛ] =
            state_action_bellman(ws, V, ambiguity_set, budget, upper_bound)
    end

    return Vres
end

function _bellman_helper!(
    workspace::Union{
        ThreadedDenseIntervalOMaxWorkspace,
        ThreadedSparseIntervalOMaxWorkspace,
    },
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound = false,
    maximize = true,
)
    @inbounds bellman_precomputation!(workspace, V, upper_bound)

    marginal = marginals(model)[1]

    @threadstid tid for (jₐ, jₛ) in update_sequence
        @inbounds ws = workspace[tid]

        @inbounds ambiguity_set = marginal[jₐ, jₛ]
        @inbounds budget = ws.budget[sub2ind(marginal, jₐ, jₛ)]
        @inbounds Vres[jₛ] =
            state_action_bellman(ws, V, ambiguity_set, budget, upper_bound)
    end

    return Vres
end

Base.@propagate_inbounds function bellman_precomputation!(
    workspace::Union{DenseIntervalOMaxWorkspace, ThreadedDenseIntervalOMaxWorkspace},
    V,
    upper_bound,
)
    # rev=true for upper bound
    sortperm!(permutation(workspace), V; rev = upper_bound, scratch = scratch(workspace))
end

Base.@propagate_inbounds bellman_precomputation!(
    workspace::Union{SparseIntervalOMaxWorkspace, ThreadedSparseIntervalOMaxWorkspace},
    V,
    upper_bound,
) = nothing



Base.@propagate_inbounds function state_action_bellman(
    workspace::DenseIntervalOMaxWorkspace,
    V,
    ambiguity_set,
    budget,
    upper_bound,
)
    return dot(V, lower(ambiguity_set)) +
           gap_value(V, gap(ambiguity_set), budget, permutation(workspace))
end

Base.@propagate_inbounds function gap_value(
    V::AbstractVector{T},
    gap::VR,
    budget,
    perm,
) where {T, VR <: AbstractVector}
    res = zero(T)

    for i in perm
        p = min(budget, gap[i])
        res += p * V[i]

        budget -= p
        if budget <= zero(T)
            break
        end
    end

    return res
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::SparseIntervalOMaxWorkspace,
    V,
    ambiguity_set,
    budget,
    upper_bound,
)
    Vp_workspace = @view workspace.values_gaps[1:supportsize(ambiguity_set)]
    Vnonzero = @view V[support(ambiguity_set)]
    for (i, (v, p)) in enumerate(zip(Vnonzero, nonzeros(gap(ambiguity_set))))
        Vp_workspace[i] = (v, p)
    end

    # rev=true for upper bound
    sort!(Vp_workspace; rev = upper_bound, by = first, scratch = scratch(workspace))

    return dot(V, lower(ambiguity_set)) + gap_value(Vp_workspace, budget)
end

Base.@propagate_inbounds function gap_value(
    Vp::VP,
    budget,
) where {T <: Real, VP <: AbstractVector{<:Tuple{T, T}}}
    res = zero(T)

    for (V, p) in Vp
        p = min(budget, p)
        res += p * V

        budget -= p
        if budget <= zero(T)
            break
        end
    end

    return res
end

##########################################################
# McCormick relaxation-based Bellman operator for fIMDPs #
##########################################################

# Non-threaded — Q-shape (default for OptimizingStrategyCache, including NoStrategyCache).
# Writes Vres[a, s] for every available (a, s); the action reduction is the
# caller's responsibility (`strategy!`). The matching V-shape primitive is
# `bellman_v!`, which calls `state_bellman_v!` instead.
function _bellman_helper!(
    workspace::FactoredIntervalMcCormickWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @inbounds for jₛ in CartesianIndices(source_shape(model))
        state_bellman_q!(
            workspace,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ,
            upper_bound,
            maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::ThreadedFactoredIntervalMcCormickWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @threadstid tid for jₛ in CartesianIndices(source_shape(model))
        @inbounds ws = workspace[tid]
        @inbounds state_bellman_q!(
            ws,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ,
            upper_bound,
            maximize,
        )
    end

    return Vres
end

# Non-optimizing (given strategy) — only the chosen action is computed, so
# there is no Q-array to populate. Writes Vres[s] (V-shape).
function _bellman_helper!(
    workspace::FactoredIntervalMcCormickWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @inbounds for jₛ in CartesianIndices(source_shape(model))
        state_bellman_v!(
            workspace,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ,
            upper_bound,
            maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::ThreadedFactoredIntervalMcCormickWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @threadstid tid for jₛ in CartesianIndices(source_shape(model))
        @inbounds ws = workspace[tid]
        @inbounds state_bellman_v!(
            ws,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ,
            upper_bound,
            maximize,
        )
    end

    return Vres
end

# Common per-state inner loop: populate `workspace.actions[a]` for every
# available action. Used by both Q-shape and V-shape paths; the difference
# is what they do with `workspace.actions` afterward.
Base.@propagate_inbounds function _populate_actions!(
    workspace::FactoredIntervalMcCormickWorkspace,
    V,
    model,
    jₛ,
    upper_bound,
)
    for jₐ in available(model, jₛ)
        ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
        workspace.actions[jₐ] =
            state_action_bellman(workspace, V, ambiguity_sets, upper_bound)
    end
end

# Q-shape: copy per-action bellmans into Vres[:, jₛ].
Base.@propagate_inbounds function state_bellman_q!(
    workspace::FactoredIntervalMcCormickWorkspace,
    ::OptimizingStrategyCache,
    Qres,
    V,
    model,
    jₛ,
    upper_bound,
    maximize,
)
    _populate_actions!(workspace, V, model, jₛ, upper_bound)
    @inbounds for jₐ in available(model, jₛ)
        Qres[jₐ, jₛ] = workspace.actions[jₐ]
    end
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::FactoredIntervalMcCormickWorkspace,
    V::AbstractArray{R},
    ambiguity_sets,
    upper_bound,
) where {R}
    V = @view V[map(support, ambiguity_sets)...]

    model = workspace.model
    JuMP.empty!(model)

    # Recursively add McCormick variables and constraints for each ambiguity set
    p, _, _ = mccormick_branch(model, ambiguity_sets)

    if upper_bound
        @objective(model, Max, sum(V[I] * p[I] for I in CartesianIndices(p)))
    else
        @objective(model, Min, sum(V[I] * p[I] for I in CartesianIndices(p)))
    end

    JuMP.optimize!(model)
    return JuMP.objective_value(model)
end

Base.@propagate_inbounds function marginal_lp_constraints(
    model,
    ambiguity_set::IntervalAmbiguitySet{R},
) where {R}
    p = @variable(model, [1:supportsize(ambiguity_set)])
    p_lower = map(i -> lower(ambiguity_set, i), support(ambiguity_set))
    p_upper = map(i -> upper(ambiguity_set, i), support(ambiguity_set))
    for i in eachindex(p)
        set_lower_bound(p[i], p_lower[i])
        set_upper_bound(p[i], p_upper[i])
    end
    @constraint(model, sum(p) == one(R))

    return p, p_lower, p_upper
end

Base.@propagate_inbounds function mccormick_branch(model, ambiguity_sets)
    if length(ambiguity_sets) == 1
        return marginal_lp_constraints(model, ambiguity_sets[1])
    else
        if length(ambiguity_sets) == 2
            p, p_lower, p_upper = marginal_lp_constraints(model, ambiguity_sets[1])
            q, q_lower, q_upper = marginal_lp_constraints(model, ambiguity_sets[2])
        else
            mid = fld(length(ambiguity_sets), 2) + 1
            p, p_lower, p_upper = mccormick_branch(model, ambiguity_sets[1:mid])
            q, q_lower, q_upper = mccormick_branch(model, ambiguity_sets[(mid + 1):end])
        end

        # McCormick envelopes
        sizes = (size(p)..., size(q)...)
        w = Array{VariableRef}(undef, sizes)
        w_lower = Array{eltype(p_lower)}(undef, sizes)
        w_upper = Array{eltype(p_upper)}(undef, sizes)
        for J in CartesianIndices(q)
            for I in CartesianIndices(p)
                w_lower[I, J] = p_lower[I] * q_lower[J]
                w_upper[I, J] = p_upper[I] * q_upper[J]

                w[I, J] = @variable(
                    model,
                    lower_bound = w_lower[I, J],
                    upper_bound = w_upper[I, J]
                )
                @constraint(
                    model,
                    w[I, J] >=
                    p[I] * q_lower[J] + q[J] * p_lower[I] − p_lower[I] * q_lower[J]
                )
                @constraint(
                    model,
                    w[I, J] >=
                    p[I] * q_upper[J] + q[J] * p_upper[I] − p_upper[I] * q_upper[J]
                )
                @constraint(
                    model,
                    w[I, J] <=
                    p[I] * q_upper[J] + q[J] * p_lower[I] − p_lower[I] * q_upper[J]
                )
                @constraint(
                    model,
                    w[I, J] <=
                    p[I] * q_lower[J] + q[J] * p_upper[I] − p_upper[I] * q_lower[J]
                )
            end
        end
        @constraint(model, sum(w) == one(eltype(p_lower)))

        return w, w_lower, w_upper
    end
end

####################################################
# O-Maximization-based Bellman operator for fIMDPs #
####################################################
function _bellman_helper!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @inbounds for jₛ in CartesianIndices(source_shape(model))
        state_bellman_q!(
            workspace,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::ThreadedFactoredIntervalOMaxWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @threadstid tid for jₛ in CartesianIndices(source_shape(model))
        @inbounds ws = workspace[tid]
        @inbounds state_bellman_q!(
            ws,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @inbounds for jₛ in CartesianIndices(source_shape(model))
        state_bellman_v!(
            workspace,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::ThreadedFactoredIntervalOMaxWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @threadstid tid for jₛ in CartesianIndices(source_shape(model))
        @inbounds ws = workspace[tid]
        @inbounds state_bellman_v!(
            ws,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ;
            upper_bound = upper_bound,
            maximize = maximize,
        )
    end

    return Vres
end

Base.@propagate_inbounds function _populate_actions!(
    workspace::FactoredIntervalOMaxWorkspace,
    V,
    model,
    jₛ,
    upper_bound,
)
    for jₐ in available(model, jₛ)
        ambiguity_sets = map(marginal -> marginal[jₐ, jₛ], marginals(model))
        inds = map(marginal -> sub2ind(marginal, jₐ, jₛ), marginals(model))
        budgets = getindex.(workspace.budgets, inds)

        workspace.actions[jₐ] = state_action_bellman(
            workspace,
            V,
            model,
            ambiguity_sets,
            budgets,
            upper_bound,
        )
    end
end

Base.@propagate_inbounds function state_bellman_q!(
    workspace::FactoredIntervalOMaxWorkspace,
    ::OptimizingStrategyCache,
    Qres,
    V,
    model::FactoredRMDP{N},
    jₛ;
    upper_bound,
    maximize,
) where {N}
    _populate_actions!(workspace, V, model, jₛ, upper_bound)
    @inbounds for jₐ in available(model, jₛ)
        Qres[jₐ, jₛ] = workspace.actions[jₐ]
    end
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::FactoredIntervalOMaxWorkspace,
    V,
    model,
    ambiguity_sets,
    budgets,
    upper_bound,
)
    Vₑ = workspace.bellman_cache
    R = valuetype(model)

    ssize = supportsize.(ambiguity_sets)

    # For each higher-level state in the product space
    for Isparse in CartesianIndices(ssize[2:end])
        I = CartesianIndex(support.(ambiguity_sets[2:end], Tuple(Isparse)))

        # For the first dimension, we need to copy the values from V
        v = orthogonal_inner_bellman!(
            workspace,
            @view(V[:, I]),
            ambiguity_sets[1],
            budgets[1],
            upper_bound,
        )
        Vₑ[1][I[1]] = v

        # For the remaining dimensions, if "full", compute bellman and store in the next level
        for d in 2:(length(ambiguity_sets) - 1)
            if Isparse[d - 1] == ssize[d]
                v = orthogonal_inner_bellman!(
                    workspace,
                    Vₑ[d - 1],
                    ambiguity_sets[d],
                    budgets[d],
                    upper_bound,
                )
                fill!(Vₑ[d - 1], zero(R))
                Vₑ[d][I[d]] = v
            else
                break
            end
        end
    end

    # Last dimension
    v = orthogonal_inner_bellman!(
        workspace,
        Vₑ[end],
        ambiguity_sets[end],
        budgets[end],
        upper_bound,
    )
    fill!(Vₑ[end], zero(R))

    return v
end

Base.@propagate_inbounds function orthogonal_inner_bellman!(
    workspace,
    V,
    ambiguity_set,
    budget,
    upper_bound::Bool,
)
    Vp_workspace = @view workspace.values_gaps[1:supportsize(ambiguity_set)]
    @inbounds for (i, j) in enumerate(support(ambiguity_set))
        Vp_workspace[i] = (V[j], gap(ambiguity_set, j))
    end

    # rev=true for upper bound
    sort!(Vp_workspace; rev = upper_bound, by = first, scratch = scratch(workspace))

    return dot(V, lower(ambiguity_set)) + gap_value(Vp_workspace, budget)
end

##########################################################
# Vertex enumeration-based Bellman operator for fIMDPs #
##########################################################

function _bellman_helper!(
    workspace::FactoredVertexIteratorWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @inbounds for jₛ in CartesianIndices(source_shape(model))
        state_bellman_q!(
            workspace,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ,
            upper_bound,
            maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::ThreadedFactoredVertexIteratorWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @threadstid tid for jₛ in CartesianIndices(source_shape(model))
        @inbounds ws = workspace[tid]
        @inbounds state_bellman_q!(
            ws,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ,
            upper_bound,
            maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::FactoredVertexIteratorWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @inbounds for jₛ in CartesianIndices(source_shape(model))
        state_bellman_v!(
            workspace,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ,
            upper_bound,
            maximize,
        )
    end

    return Vres
end

function _bellman_helper!(
    workspace::ThreadedFactoredVertexIteratorWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
)
    @threadstid tid for jₛ in CartesianIndices(source_shape(model))
        @inbounds ws = workspace[tid]
        @inbounds state_bellman_v!(
            ws,
            strategy_cache,
            Vres,
            V,
            model,
            jₛ,
            upper_bound,
            maximize,
        )
    end

    return Vres
end

Base.@propagate_inbounds function _populate_actions!(
    workspace::FactoredVertexIteratorWorkspace,
    V,
    model,
    jₛ,
    upper_bound,
)
    for jₐ in available(model, jₛ)
        ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
        workspace.actions[jₐ] =
            state_action_bellman(workspace, V, ambiguity_sets, upper_bound)
    end
end

Base.@propagate_inbounds function state_bellman_q!(
    workspace::FactoredVertexIteratorWorkspace,
    ::OptimizingStrategyCache,
    Vres,
    V,
    model,
    jₛ,
    upper_bound,
    maximize,
)
    _populate_actions!(workspace, V, model, jₛ, upper_bound)
    @inbounds for jₐ in available(model, jₛ)
        Vres[jₐ, jₛ] = workspace.actions[jₐ]
    end
end