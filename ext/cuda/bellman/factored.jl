function IntervalMDP._bellman_helper!(
    workspace::CuFactoredOMaxWorkspace,
    strategy_cache::IntervalMDP.AbstractStrategyCache,
    Vres::AbstractArray{Tv},
    V::AbstractArray{Tv},
    model;
    upper_bound = false,
    maximize = true,
) where {Tv}
    marginal = marginals(model)[1]

    if IntervalMDP.valuetype(marginal) != Tv
        throw(
            ArgumentError(
                "Value type of the model ($(IntervalMDP.valuetype(marginal))) does not match the value type of the input vector ($Tv).",
            ),
        )
    end

    # Execution plan:
    # - 1 block per state
    # - Use the entire block for each action.
    # - Each warp computes the value for one conditional at the top/last level - possibly with a block-stride loop
    # - The reduction in each warp follows the same pattern as the CPU implementation.
    # - The task divergence should be minimal as all warps operate on the same marginal synchronously (including sparsity patterns).
    # - The data divergence should also be minimal for the same reason, with the exception of the first level, which will 
    #   access different ranges of V (global mem). However, since the threads in a warp access contiguous elements of V, this will still be coalesced.

    n_actions =
        isa(strategy_cache, IntervalMDP.OptimizingStrategyCache) ? num_actions(model) : 1

    function shmem_func(threads)
        warps = cld(threads, 32)
        expectation_cache_size = 2 * sum(workspace.max_support_per_marginal[1:end - 1]) *
            warps * sizeof(Tv)
        final_level_cache_size =
            2 * workspace.max_support_per_marginal[end] * sizeof(Tv)
        action_cache_size =
            n_actions * sizeof(Tv)
        return expectation_cache_size +
            final_level_cache_size +
            action_cache_size
    end

    kernel = @cuda launch = false factored_bellman_kernel!(
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        model,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv)),
    )

    config = launch_configuration(kernel.fun; shmem = shmem_func)
    
    max_threads = prevwarp(device(), config.threads)
    warps = div(max_threads, 32)
    warps = min(warps, workspace.max_support_per_marginal[end])
    threads = warps * 32
    n_states = prod(source_shape(model))
    blocks = min(2^16 - 1, n_states)
    shmem = shmem_func(threads)

    kernel(
        workspace,
        active_cache(strategy_cache),
        Vres,
        V,
        model,
        upper_bound ? (>=) : (<=),
        maximize ? (max, >, typemin(Tv)) : (min, <, typemax(Tv));
        blocks = blocks,
        threads = threads,
        shmem = shmem,
    )

    return Vres
end

function factored_bellman_kernel!(
    workspace::CuFactoredOMaxWorkspace,
    strategy_cache,
    Vres::AbstractArray{Tv},
    V::AbstractArray{Tv},
    model,
    value_lt,
    action_reduce,
) where {Tv}
    # Kernel implementation goes here
end