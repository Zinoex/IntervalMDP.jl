"""
    control_synthesis(problem::Problem; callback=nothing)

Compute the optimal control strategy for the given problem (system + specification). If the specification is finite time, then the strategy is time-varying,
with the returned strategy being in step order (i.e., the first element of the returned vector is the strategy for the first time step).
If the specification is infinite time, then the strategy is stationary and only a single vector of length `num_states(system)` is returned.

It is possible to provide a callback function that will be called at each iteration with the current value function and
iteration count. The callback function should have the signature `callback(V::AbstractArray, k::Int)`.
"""
function control_synthesis(problem::Problem; callback=nothing)
    spec = specification(problem)
    prop = system_property(spec)

    if !(strategy(problem) isa NoStrategy)
        @warn "The strategy is given to `control_synthesis`. Ignoring the strategy and synthesizing a new one."
    end

    strategy_config =
        isfinitetime(prop) ? TimeVaryingStrategyConfig() : StationaryStrategyConfig()
    V, k, res, strategy_cache = _value_iteration!(strategy_config, problem; callback=callback)

    return cachetostrategy(strategy_cache), V, k, res
end
