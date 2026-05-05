# State-action-value Bellman primitive (`bellman_q!`).
#
#   Q_cur[s, a] = opt_{γ ∈ Γ_{s,a}} E_{s' ~ γ}[V_prev(s')]
#
# for every (s, a) pair in the update sequence (a *state-action* sequence,
# yielding `(jₐ, jₛ)` tuples). The strategy cache is **not** updated
# here — different (s, a) pairs for the same `s` may be visited from
# different threads, and the action reduction's race-on-`strategy[s]`
# would force a serialisation we'd rather avoid. Callers that need a
# strategy can run a separate `strategy!` pass after `bellman_q!` returns.
#
# This file only contains the public typed entry point and the
# per-workspace (a, s)-iteration dispatchers. The scalar kernels live in
# `bellman_kernels.jl`. Each dispatcher is a tight `for (jₐ, jₛ) in
# update_sequence` that calls one `state_action_bellman` per pair and
# stores the result in `Qres[jₐ, jₛ]`.
#
# Only `OptimizingStrategyCache` is supported — a Q-shape output for a
# given strategy is degenerate (Q[a, s] is meaningful only for the chosen
# a) and almost certainly indicates a misuse of the API.

"""
    bellman_q!(workspace, strategy_cache::OptimizingStrategyCache,
               Qres::StateActionValueArray, V::StateValueArray, model,
               update_sequence; upper_bound=false, maximize=true, prop=nothing)

Per-(s, a) Bellman update over a state-action-shape update sequence.
Writes `Qres[a, s] = opt_γ E_γ[V(·)]` for every `(a, s)` yielded by
`update_sequence`. Pairs not visited retain their `Qres[a, s]` entry; the
caller is responsible for any pre-loop initialisation. The strategy cache
is *not* modified — call `strategy!` separately if you need to fold Q into
V and a stored policy.
"""
function bellman_q!(
    workspace,
    strategy_cache::OptimizingStrategyCache,
    Qres::StateActionValueArray,
    V::StateValueArray,
    model::Union{IntervalMarkovProcess, AbstractAmbiguitySets},
    update_sequence = sample(default_sampling_strategy(), model, strategy_cache);
    upper_bound = false,
    maximize = true,
    prop = nothing,
)
    return _bellman_q!(
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

function bellman_q!(
    ::Any,
    ::NonOptimizingStrategyCache,
    ::StateActionValueArray,
    ::StateValueArray,
    ::Union{IntervalMarkovProcess, AbstractAmbiguitySets},
    args...;
    kwargs...,
)
    throw(
        ArgumentError(
            "bellman_q!: NonOptimizingStrategyCache (given strategy) is not compatible with a Q-shape output — there is only one action per state. Use bellman_v! with a V-shape buffer.",
        ),
    )
end

###############################################################################
# Dense / sparse flat IMDP                                                    #
###############################################################################

function _bellman_q!(
    workspace::Union{DenseIntervalOMaxWorkspace, SparseIntervalOMaxWorkspace},
    strategy_cache,
    Qres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    bellman_precomputation!(workspace, V, upper_bound)
    marginal = marginals(model)[1]

    @inbounds for (jₐ, jₛ) in update_sequence
        ambiguity_set = marginal[jₐ, jₛ]
        budget = workspace.budget[sub2ind(marginal, jₐ, jₛ)]
        Qres[jₐ, jₛ] =
            state_action_bellman(workspace, V, ambiguity_set, budget, upper_bound)
    end
    return Qres
end

# Threaded — each (jₐ, jₛ) pair is unique, so different threads writing to
# different (a, s) cells don't race on `Qres`. They do share `V` and the
# model (read-only), and each thread reads from its own per-thread
# workspace's `budget` / scratch (so no contention on those either).
function _bellman_q!(
    workspace::Union{
        ThreadedDenseIntervalOMaxWorkspace,
        ThreadedSparseIntervalOMaxWorkspace,
    },
    strategy_cache,
    Qres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @inbounds bellman_precomputation!(workspace, V, upper_bound)
    marginal = marginals(model)[1]

    @threadstid tid for (jₐ, jₛ) in update_sequence
        @inbounds ws = workspace[tid]
        @inbounds ambiguity_set = marginal[jₐ, jₛ]
        @inbounds budget = ws.budget[sub2ind(marginal, jₐ, jₛ)]
        @inbounds Qres[jₐ, jₛ] =
            state_action_bellman(ws, V, ambiguity_set, budget, upper_bound)
    end
    return Qres
end

###############################################################################
# Factored: O-Max                                                             #
###############################################################################

function _bellman_q!(
    workspace::FactoredIntervalOMaxWorkspace,
    strategy_cache,
    Qres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @inbounds for (jₐ, jₛ) in update_sequence
        ambiguity_sets = map(marginal -> marginal[jₐ, jₛ], marginals(model))
        inds = map(marginal -> sub2ind(marginal, jₐ, jₛ), marginals(model))
        budgets = getindex.(workspace.budgets, inds)
        Qres[jₐ, jₛ] = state_action_bellman(
            workspace,
            V,
            model,
            ambiguity_sets,
            budgets,
            upper_bound,
        )
    end
    return Qres
end

function _bellman_q!(
    workspace::ThreadedFactoredIntervalOMaxWorkspace,
    strategy_cache,
    Qres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @threadstid tid for (jₐ, jₛ) in update_sequence
        @inbounds ws = workspace[tid]
        @inbounds ambiguity_sets = map(marginal -> marginal[jₐ, jₛ], marginals(model))
        @inbounds inds = map(marginal -> sub2ind(marginal, jₐ, jₛ), marginals(model))
        @inbounds budgets = getindex.(ws.budgets, inds)
        @inbounds Qres[jₐ, jₛ] = state_action_bellman(
            ws,
            V,
            model,
            ambiguity_sets,
            budgets,
            upper_bound,
        )
    end
    return Qres
end

###############################################################################
# Factored: McCormick / Vertex enumeration                                    #
###############################################################################
#
# These two share an outer loop — the only difference is which scalar
# kernel `state_action_bellman` resolves to via dispatch on the workspace
# type. McCormick is non-thread-safe inside `state_action_bellman`
# (mutates `workspace.model`, the JuMP problem), so the threaded variant
# uses per-thread workspaces.

function _bellman_q!(
    workspace::Union{FactoredIntervalMcCormickWorkspace, FactoredVertexIteratorWorkspace},
    strategy_cache,
    Qres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @inbounds for (jₐ, jₛ) in update_sequence
        ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
        Qres[jₐ, jₛ] =
            state_action_bellman(workspace, V, ambiguity_sets, upper_bound)
    end
    return Qres
end

function _bellman_q!(
    workspace::Union{
        ThreadedFactoredIntervalMcCormickWorkspace,
        ThreadedFactoredVertexIteratorWorkspace,
    },
    strategy_cache,
    Qres,
    V,
    model,
    update_sequence;
    upper_bound,
    maximize,
)
    @threadstid tid for (jₐ, jₛ) in update_sequence
        @inbounds ws = workspace[tid]
        @inbounds ambiguity_sets = getindex.(marginals(model), jₐ, jₛ)
        @inbounds Qres[jₐ, jₛ] =
            state_action_bellman(ws, V, ambiguity_sets, upper_bound)
    end
    return Qres
end
