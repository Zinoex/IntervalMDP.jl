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

#############################################################################
# State-action sweep (sa_sweep!)
#
# Walks a (jₐ, jₛ)-yielding update sequence and updates V[s] and the
# strategy cache asynchronously via `relax!`. Unlike `_bellman_helper!`
# this path never materializes a (action × state)-shaped Q buffer: each
# per-(s, a) bellman is computed into a workspace-local scalar and
# immediately consumed. States not visited by the sequence retain their
# V_prev value.
#############################################################################

# Single-threaded dense / sparse flat IMDP
function sa_sweep!(
    workspace::Union{DenseIntervalOMaxWorkspace, SparseIntervalOMaxWorkspace},
    strategy_cache::AbstractStrategyCache,
    Vres::AbstractArray,
    V::AbstractArray,
    model,
    update_sequence;
    upper_bound = false,
    maximize = true,
)
    bellman_precomputation!(workspace, V, upper_bound)

    marginal = marginals(model)[1]
    visited = falses(size(Vres))
    copy!(Vres, V)

    @inbounds for (jₐ, jₛ) in update_sequence
        ambiguity_set = marginal[jₐ, jₛ]
        budget = workspace.budget[sub2ind(marginal, jₐ, jₛ)]
        q = state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)
        relax!(strategy_cache, Vres, visited, jₛ, jₐ, q, maximize)
    end

    return Vres
end

# Threaded dense / sparse: Phase 2 does not thread the sa_sweep inner loop
# (see plan §3 — parallelism is opt-in, capped, and off by default for
# trajectory-based samplers). Fall back to thread 1's workspace.
sa_sweep!(
    workspace::Union{
        ThreadedDenseIntervalOMaxWorkspace,
        ThreadedSparseIntervalOMaxWorkspace,
    },
    strategy_cache::AbstractStrategyCache,
    Vres::AbstractArray,
    V::AbstractArray,
    model,
    update_sequence;
    upper_bound = false,
    maximize = true,
) = sa_sweep!(
    workspace[1],
    strategy_cache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound = upper_bound,
    maximize = maximize,
)

# Factored O-Max (single-threaded). Per-(s, a) ambiguity-set bellman
# uses the workspace's existing `state_action_bellman` for factored
# O-Max, which sequentially marginalizes the per-dim ambiguity sets via
# `orthogonal_inner_bellman!`. The relax-based update is identical in
# spirit to the flat path.
function sa_sweep!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres::AbstractArray,
    V::AbstractArray,
    model,
    update_sequence;
    upper_bound = false,
    maximize = true,
)
    visited = falses(size(Vres))
    copy!(Vres, V)

    @inbounds for (jₐ, jₛ) in update_sequence
        ambiguity_sets = map(marginal -> marginal[jₐ, jₛ], marginals(model))
        inds = map(marginal -> sub2ind(marginal, jₐ, jₛ), marginals(model))
        budgets = getindex.(workspace.budgets, inds)
        q = state_action_bellman(
            workspace,
            V,
            model,
            ambiguity_sets,
            budgets,
            upper_bound,
        )
        relax!(strategy_cache, Vres, visited, jₛ, jₐ, q, maximize)
    end

    return Vres
end

# Threaded factored O-Max — same convention as flat (sequential by
# default for trajectory-style samplers; thread 1's workspace).
sa_sweep!(
    workspace::ThreadedFactoredIntervalOMaxWorkspace,
    strategy_cache::AbstractStrategyCache,
    Vres::AbstractArray,
    V::AbstractArray,
    model,
    update_sequence;
    upper_bound = false,
    maximize = true,
) = sa_sweep!(
    workspace[1],
    strategy_cache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound = upper_bound,
    maximize = maximize,
)

#############################################################################
# bellman_v! — V-shape (state-indexed) primitive.
#
# Always writes V[s], one value per state. No (action × state) Q-array is
# ever materialized: per-state action scratch lives in `workspace.actions`,
# is reused across states, and is reduced to V[s] either by
# `extract_strategy!` (state-outer full sweep) or by `relax!` (per-(s, a)
# asynchronous sweep). The choice is dispatched on the update sequence's
# `sequence_shape`:
#
#   - `StateUpdateSequence` (yields bare `s`): per-state full sweep over
#     `available(s)` — used by `RobustValueIteration`. Calls the workspace's
#     existing `state_bellman_v!`.
#   - `StateActionUpdateSequence` (yields `(a, s)`): per-(s, a) sweep, with
#     `relax!` driving strictly-better updates over visited actions — used
#     by sampling-based / partial-update algorithms like `GenSamplingDP`.
#     Calls `sa_sweep!`.
#
# `prop` is accepted but ignored at this layer; it is only meaningful for
# `ProductWorkspace`, which has its own bespoke entry in `bellman!`.
#############################################################################

# Dense / sparse flat IMDP: dispatch on `sequence_shape(update_sequence)`.
function _bellman_v_kernel!(
    workspace::Union{
        DenseIntervalOMaxWorkspace,
        SparseIntervalOMaxWorkspace,
        ThreadedDenseIntervalOMaxWorkspace,
        ThreadedSparseIntervalOMaxWorkspace,
    },
    strategy_cache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound = false,
    maximize = true,
    prop = nothing,
)
    return _bellman_v_dispatch!(
        sequence_shape(update_sequence),
        workspace,
        strategy_cache,
        Vres,
        V,
        model,
        update_sequence;
        upper_bound = upper_bound,
        maximize = maximize,
    )
end

# `(a, s)` sequence — async sweep (sa_sweep! / relax! semantics).
function _bellman_v_dispatch!(
    ::StateActionUpdateSequence,
    workspace,
    strategy_cache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    return sa_sweep!(
        workspace,
        strategy_cache,
        Vres,
        V,
        model,
        update_sequence;
        upper_bound = upper_bound,
        maximize = maximize,
    )
end

# State sequence + Optimizing — per-state full sweep over available actions
# into `ws.actions`, then `extract_strategy!` to reduce to V[s] and (if the
# cache is a synthesis cache) record the argmax.
function _bellman_v_dispatch!(
    ::StateUpdateSequence,
    workspace::Union{DenseIntervalOMaxWorkspace, SparseIntervalOMaxWorkspace},
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    bellman_precomputation!(workspace, V, upper_bound)
    marginal = marginals(model)[1]
    @inbounds for jₛ in update_sequence
        for jₐ in available(model, jₛ)
            ambiguity_set = marginal[jₐ, jₛ]
            budget = workspace.budget[sub2ind(marginal, jₐ, jₛ)]
            workspace.actions[jₐ] =
                state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)
        end
        Vres[jₛ] = extract_strategy!(
            strategy_cache,
            workspace.actions,
            available(model, jₛ),
            jₛ,
            maximize,
        )
    end
    return Vres
end

# State sequence + NonOptimizing — given-strategy evaluation. Each state has
# exactly one prescribed action `σ(s)`; compute its bellman directly.
function _bellman_v_dispatch!(
    ::StateUpdateSequence,
    workspace::Union{DenseIntervalOMaxWorkspace, SparseIntervalOMaxWorkspace},
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    bellman_precomputation!(workspace, V, upper_bound)
    marginal = marginals(model)[1]
    @inbounds for jₛ in update_sequence
        jₐ = CartesianIndex(strategy_cache[jₛ])
        ambiguity_set = marginal[jₐ, jₛ]
        budget = workspace.budget[sub2ind(marginal, jₐ, jₛ)]
        Vres[jₛ] =
            state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)
    end
    return Vres
end

# Threaded versions — partition states across threads; each thread reuses its
# own `ws.actions` buffer.
function _bellman_v_dispatch!(
    ::StateUpdateSequence,
    workspace::Union{
        ThreadedDenseIntervalOMaxWorkspace,
        ThreadedSparseIntervalOMaxWorkspace,
    },
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @inbounds bellman_precomputation!(workspace, V, upper_bound)
    marginal = marginals(model)[1]
    @threadstid tid for jₛ in update_sequence
        @inbounds ws = workspace[tid]
        @inbounds for jₐ in available(model, jₛ)
            ambiguity_set = marginal[jₐ, jₛ]
            budget = ws.budget[sub2ind(marginal, jₐ, jₛ)]
            ws.actions[jₐ] =
                state_action_bellman(ws, V, ambiguity_set, budget, upper_bound)
        end
        @inbounds Vres[jₛ] = extract_strategy!(
            strategy_cache,
            ws.actions,
            available(model, jₛ),
            jₛ,
            maximize,
        )
    end
    return Vres
end

function _bellman_v_dispatch!(
    ::StateUpdateSequence,
    workspace::Union{
        ThreadedDenseIntervalOMaxWorkspace,
        ThreadedSparseIntervalOMaxWorkspace,
    },
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @inbounds bellman_precomputation!(workspace, V, upper_bound)
    marginal = marginals(model)[1]
    @threadstid tid for jₛ in update_sequence
        @inbounds ws = workspace[tid]
        @inbounds jₐ = CartesianIndex(strategy_cache[jₛ])
        @inbounds ambiguity_set = marginal[jₐ, jₛ]
        @inbounds budget = ws.budget[sub2ind(marginal, jₐ, jₛ)]
        @inbounds Vres[jₛ] =
            state_action_bellman(ws, V, ambiguity_set, budget, upper_bound)
    end
    return Vres
end

# `bellman_v!` always writes V-shape Vres[s]. For factored workspaces
# the matching `_bellman_helper!` now writes Q-shape (so that
# `bellman!` is shape-consistent across all workspace types), so the
# V-shape entry point cannot delegate to it — it has to drive the
# per-state V-reducing helper (`state_bellman_v!`) directly.
function _bellman_v_kernel!(
    workspace::FactoredIntervalMcCormickWorkspace,
    strategy_cache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
    prop = nothing,
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

function _bellman_v_kernel!(
    workspace::ThreadedFactoredIntervalMcCormickWorkspace,
    strategy_cache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
    prop = nothing,
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

function _bellman_v_kernel!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
    prop = nothing,
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

function _bellman_v_kernel!(
    workspace::ThreadedFactoredIntervalOMaxWorkspace,
    strategy_cache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
    prop = nothing,
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

function _bellman_v_kernel!(
    workspace::FactoredVertexIteratorWorkspace,
    strategy_cache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
    prop = nothing,
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

function _bellman_v_kernel!(
    workspace::ThreadedFactoredVertexIteratorWorkspace,
    strategy_cache,
    Vres,
    V,
    model,
    _update_sequence;
    upper_bound = false,
    maximize = true,
    prop = nothing,
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

# V-shape: reduce per-action bellmans to Vres[jₛ] via extract_strategy!.
Base.@propagate_inbounds function state_bellman_v!(
    workspace::FactoredIntervalMcCormickWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    jₛ,
    upper_bound,
    maximize,
)
    _populate_actions!(workspace, V, model, jₛ, upper_bound)
    Vres[jₛ] = extract_strategy!(
        strategy_cache,
        workspace.actions,
        available(model, jₛ),
        jₛ,
        maximize,
    )
end

Base.@propagate_inbounds function state_bellman_v!(
    workspace::FactoredIntervalMcCormickWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    jₛ,
    upper_bound,
    maximize,
)
    jₐ = CartesianIndex(strategy_cache[jₛ])
    ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
    Vres[jₛ] = state_action_bellman(workspace, V, ambiguity_sets, upper_bound)
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
    Vres,
    V,
    model::FactoredRMDP{N},
    jₛ;
    upper_bound,
    maximize,
) where {N}
    _populate_actions!(workspace, V, model, jₛ, upper_bound)
    @inbounds for jₐ in available(model, jₛ)
        Vres[jₐ, jₛ] = workspace.actions[jₐ]
    end
end

Base.@propagate_inbounds function state_bellman_v!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model::FactoredRMDP{N},
    jₛ;
    upper_bound,
    maximize,
) where {N}
    _populate_actions!(workspace, V, model, jₛ, upper_bound)
    Vres[jₛ] = extract_strategy!(
        strategy_cache,
        workspace.actions,
        available(model, jₛ),
        jₛ,
        maximize,
    )
end

Base.@propagate_inbounds function state_bellman_v!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model::FactoredRMDP{N},
    jₛ;
    upper_bound,
    maximize,
) where {N}
    jₐ = CartesianIndex(strategy_cache[jₛ])
    ambiguity_sets = map(marginal -> marginal[jₐ, jₛ], marginals(model))
    inds = map(marginal -> sub2ind(marginal, jₐ, jₛ), marginals(model))
    budgets = getindex.(workspace.budgets, inds)
    Vres[jₛ] =
        state_action_bellman(workspace, V, model, ambiguity_sets, budgets, upper_bound)
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

Base.@propagate_inbounds function state_bellman_v!(
    workspace::FactoredVertexIteratorWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    jₛ,
    upper_bound,
    maximize,
)
    _populate_actions!(workspace, V, model, jₛ, upper_bound)
    Vres[jₛ] = extract_strategy!(
        strategy_cache,
        workspace.actions,
        available(model, jₛ),
        jₛ,
        maximize,
    )
end

Base.@propagate_inbounds function state_bellman_v!(
    workspace::FactoredVertexIteratorWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    jₛ,
    upper_bound,
    maximize,
)
    jₐ = CartesianIndex(strategy_cache[jₛ])
    ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
    Vres[jₛ] = state_action_bellman(workspace, V, ambiguity_sets, upper_bound)
end

Base.@propagate_inbounds function state_action_bellman(
    workspace::FactoredVertexIteratorWorkspace,
    V::AbstractArray{R},
    ambiguity_sets,
    upper_bound,
) where {R}
    iterators = vertex_generator.(ambiguity_sets, workspace.result_vectors)

    optval = upper_bound ? typemin(R) : typemax(R)
    optfunc = upper_bound ? max : min

    for marginal_vertices in Iterators.product(iterators...)
        v = sum(
            V[I] * prod(r -> marginal_vertices[r][I[r]], eachindex(ambiguity_sets)) for
            I in CartesianIndices(num_target.(ambiguity_sets))
        )
        optval = optfunc(optval, v)
    end

    return optval
end
