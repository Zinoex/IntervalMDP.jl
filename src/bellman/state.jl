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
    update_sequence = sample(ExhaustiveState(), model, strategy_cache);
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
    _threaded_sweep_pays_off(model, update_sequence, _sweep_actions(model)) || return _bellman_v!(
        workspace[1],
        strategy_cache,
        Vres,
        V,
        model,
        update_sequence;
        upper_bound = upper_bound,
        maximize = maximize,
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
    _threaded_sweep_pays_off(model, update_sequence, 1) || return _bellman_v!(
        workspace[1],
        strategy_cache,
        Vres,
        V,
        model,
        update_sequence;
        upper_bound = upper_bound,
        maximize = maximize,
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

# Spawning and joining the thread tasks of `@threadstid` costs several µs per sweep, more
# than a whole serial sweep of a short update sequence (e.g. the ~8 states a GSRDP
# trajectory iteration relaxes on a model with small supports). Short sweeps therefore
# run serially on the first thread's workspace, which shares the precomputation buffers
# and runs the identical per-state code. Each state's result depends only on `V`, the
# model and its own strategy-cache entry, so the schedule cannot change any value.
# `work` estimates the serial cost in units of roughly 10 ns (stored entries scanned per
# (state, action) plus a per-action overhead); a pure speed heuristic.
const _THREADED_SWEEP_MIN_WORK = 1000

function _threaded_sweep_pays_off(model, update_sequence, actions_per_state)
    support = _mean_support(marginals(model)[1])
    support === nothing && return true
    work = length(update_sequence) * actions_per_state * (support + 4)
    return work >= _THREADED_SWEEP_MIN_WORK
end

_mean_support(marginal::Marginal) = _mean_support(ambiguity_sets(marginal))
function _mean_support(p::IntervalAmbiguitySets{R, MR}) where {R, MR <: AbstractSparseMatrix}
    n = num_sets(p)
    return n == 0 ? 0.0 : nnz(p.gap) / n
end
_mean_support(p::IntervalAmbiguitySets{R, MR}) where {R, MR <: AbstractMatrix} =
    Float64(num_target(p))
_mean_support(_) = nothing

_sweep_actions(model::FactoredRMDP) = num_actions(model)
_sweep_actions(_) = 1

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
