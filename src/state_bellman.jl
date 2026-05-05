

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