"""
    control_synthesis(problem::Problem)

Compute the optimal control strategy for the given problem (system + specification). If the specification is finite time, then the strategy is time-varying,
with the returned strategy being in step order (i.e., the first element of the returned vector is the strategy for the first time step).
If the specification is infinite time, then the strategy is stationary and only a single vector of length `num_states(system)` is returned.
"""
function control_synthesis(problem::Problem)
    spec = specification(problem)
    prop = system_property(spec)

    strategy_config =
        isfinitetime(prop) ? TimeVaryingStrategyConfig() : StationaryStrategyConfig()
    V, k, res, strategy_cache = _value_iteration!(strategy_config, problem)

    return cachetostrategy(strategy_cache), V, k, res
end
