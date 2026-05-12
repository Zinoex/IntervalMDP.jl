# State-value Bellman primitive (`bellman_v!`).
#
#   V_cur[s] = opt_{a ∈ A(s)} opt_{γ ∈ Γ_{s,a}} E_{s' ~ γ}[V_prev(s')]
#
# for every state `s` in the update sequence (a *state* sequence, yielding
# bare CartesianIndex values). The strategy cache is updated in
# strictly-better fashion: a state's chosen action is replaced only when
# the new candidate strictly beats the previous one (or no previous action
# was stored for that state). For NonOptimizing caches the action is read
# from the cache and there is no comparison.
#
# This file only contains the public typed entry point and the
# per-(workspace × strategy_cache) state-iteration dispatchers. The scalar
# kernels and the per-state `_populate_actions!` helper live in
# `bellman_kernels.jl`. The action reduction is delegated to
# `extract_strategy!` (strategy_cache.jl), whose strictly-better semantics
# match the spec above for all three optimizing-cache flavours.

"""
    bellman_v!(workspace, strategy_cache, Vres::StateValueArray, V::StateValueArray,
               model, update_sequence; upper_bound=false, maximize=true, prop=nothing)

Per-state Bellman update over a state-shape update sequence. Writes
`Vres[s] = opt_a opt_γ E_γ[V(·)]` for every `s` yielded by
`update_sequence`. States not visited retain their `Vres[s]` entry; the
caller is responsible for any pre-loop initialisation.
"""
function bellman_v!(
    workspace,
    strategy_cache,
    Vres::StateValueArray,
    V::StateValueArray,
    model::Union{IntervalMarkovProcess, AbstractAmbiguitySets},
    update_sequence = sample(AllStatesSweep(), model, strategy_cache);
    upper_bound = false,
    maximize = true,
    prop = nothing,
)
    return _bellman_v!(
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

###############################################################################
# Dense / sparse flat IMDP                                                    #
###############################################################################
#
# Per-state outer loop: for each `s` in `update_sequence`, populate
# `workspace.actions` for every available `a` (via `_populate_actions!`),
# then either reduce via `extract_strategy!` (Optimizing — strictly-better
# update on the chosen action) or read the prescribed action from the
# cache and write `Vres[s]` directly (NonOptimizing).

function _bellman_v!(
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

    @inbounds for jₛ in update_sequence
        _populate_actions!(workspace, V, model, jₛ, upper_bound)
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

function _bellman_v!(
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
        Vres[jₛ] = state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)
    end
    return Vres
end

# Threaded variants — partition states across threads, each thread reuses
# its own `ws.actions`/scratch buffer.
function _bellman_v!(
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

    @threadstid tid for jₛ in update_sequence
        @inbounds ws = workspace[tid]
        @inbounds _populate_actions!(ws, V, model, jₛ, upper_bound)
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

function _bellman_v!(
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
        @inbounds Vres[jₛ] = state_action_bellman(ws, V, ambiguity_set, budget, upper_bound)
    end
    return Vres
end

###############################################################################
# Factored: O-Max / McCormick / Vertex enumeration                            #
###############################################################################
#
# All three factored backends share the same outer skeleton: per state,
# `_populate_actions!` (kernel-specific) fills `workspace.actions` for
# every available action, then `extract_strategy!` reduces. NonOptimizing
# variants call the per-(s, a) factored scalar kernel for the prescribed
# action only.

function _bellman_v!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @inbounds for jₛ in update_sequence
        _populate_actions!(workspace, V, model, jₛ, upper_bound)
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

function _bellman_v!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @inbounds for jₛ in update_sequence
        jₐ = CartesianIndex(strategy_cache[jₛ])
        ambiguity_sets = map(marginal -> marginal[jₐ, jₛ], marginals(model))
        inds = map(marginal -> sub2ind(marginal, jₐ, jₛ), marginals(model))
        budgets = getindex.(workspace.budgets, inds)
        Vres[jₛ] =
            state_action_bellman(workspace, V, model, ambiguity_sets, budgets, upper_bound)
    end
    return Vres
end

function _bellman_v!(
    workspace::ThreadedFactoredIntervalOMaxWorkspace,
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @threadstid tid for jₛ in update_sequence
        @inbounds ws = workspace[tid]
        @inbounds _populate_actions!(ws, V, model, jₛ, upper_bound)
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

function _bellman_v!(
    workspace::ThreadedFactoredIntervalOMaxWorkspace,
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @threadstid tid for jₛ in update_sequence
        @inbounds ws = workspace[tid]
        @inbounds jₐ = CartesianIndex(strategy_cache[jₛ])
        @inbounds ambiguity_sets = map(marginal -> marginal[jₐ, jₛ], marginals(model))
        @inbounds inds = map(marginal -> sub2ind(marginal, jₐ, jₛ), marginals(model))
        @inbounds budgets = getindex.(ws.budgets, inds)
        @inbounds Vres[jₛ] =
            state_action_bellman(ws, V, model, ambiguity_sets, budgets, upper_bound)
    end
    return Vres
end

# McCormick / Vertex enumeration — no per-marginal budget, scalar kernel
# takes (workspace, V, ambiguity_sets, upper_bound).
function _bellman_v!(
    workspace::Union{FactoredIntervalMcCormickWorkspace, FactoredVertexIteratorWorkspace},
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @inbounds for jₛ in update_sequence
        _populate_actions!(workspace, V, model, jₛ, upper_bound)
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

function _bellman_v!(
    workspace::Union{FactoredIntervalMcCormickWorkspace, FactoredVertexIteratorWorkspace},
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @inbounds for jₛ in update_sequence
        jₐ = CartesianIndex(strategy_cache[jₛ])
        ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
        Vres[jₛ] = state_action_bellman(workspace, V, ambiguity_sets, upper_bound)
    end
    return Vres
end

function _bellman_v!(
    workspace::Union{
        ThreadedFactoredIntervalMcCormickWorkspace,
        ThreadedFactoredVertexIteratorWorkspace,
    },
    strategy_cache::OptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @threadstid tid for jₛ in update_sequence
        @inbounds ws = workspace[tid]
        @inbounds _populate_actions!(ws, V, model, jₛ, upper_bound)
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

function _bellman_v!(
    workspace::Union{
        ThreadedFactoredIntervalMcCormickWorkspace,
        ThreadedFactoredVertexIteratorWorkspace,
    },
    strategy_cache::NonOptimizingStrategyCache,
    Vres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @threadstid tid for jₛ in update_sequence
        @inbounds ws = workspace[tid]
        @inbounds jₐ = CartesianIndex(strategy_cache[jₛ])
        @inbounds ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
        @inbounds Vres[jₛ] = state_action_bellman(ws, V, ambiguity_sets, upper_bound)
    end
    return Vres
end
