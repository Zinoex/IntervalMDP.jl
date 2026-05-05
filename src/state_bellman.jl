

# Typed V-shape primitive: writes Vres[s] = opt_a opt_γ E_γ[V(·)] for
# every state the workspace visits. Both Optimizing and NonOptimizing
# strategy caches are supported (the NonOptimizing case takes the action
# from the cache).
function bellman_v!(
    workspace,
    strategy_cache,
    Vres::StateValueArray,
    V::StateValueArray,
    model::IntervalMarkovProcess,
    update_sequence = sample(default_sampling_strategy(), model, strategy_cache);
    upper_bound = false,
    maximize = true,
    prop = nothing,
)
    return _bellman_v_kernel!(
        workspace,
        strategy_cache,
        parent(Vres),
        parent(V),
        model,
        update_sequence;
        upper_bound = upper_bound,
        maximize = maximize,
    )
end


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
