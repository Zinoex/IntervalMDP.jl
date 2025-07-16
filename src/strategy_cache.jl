abstract type AbstractStrategyCache end
abstract type NonOptimizingStrategyCache <: AbstractStrategyCache end
abstract type OptimizingStrategyCache <: AbstractStrategyCache end

"""
    construct_strategy_cache(mp_or_problem)

Construct a strategy cache for a given problem. The resuling cache type depends on
the specification and the device to store the strategy depends on the device of the Markov process.
"""
function construct_strategy_cache end

# Strategy cache for not storing policies - useful for dispatching
struct NoStrategyCache <: OptimizingStrategyCache end

function construct_strategy_cache(
    ::Union{
        IntervalProbabilities,
        OrthogonalIntervalProbabilities,
        MixtureIntervalProbabilities,
        StochasticProcess,
    },
)
    return NoStrategyCache()
end

construct_strategy_cache(::VerificationProblem{S, F, <:NoStrategy}) where {S, F} =
    NoStrategyCache()

function extract_strategy!(::NoStrategyCache, values, V, j, maximize)
    return maximize ? maximum(values) : minimum(values)
end
step_postprocess_strategy_cache!(::NoStrategyCache) = nothing

# Strategy cache for applying given policies - useful for dispatching
struct GivenStrategyCache{S <: AbstractStrategy} <: NonOptimizingStrategyCache
    strategy::S
end

construct_strategy_cache(problem::VerificationProblem{S, F, C}) where {S, F, C} =
    GivenStrategyCache(strategy(problem))
time_length(cache::GivenStrategyCache) = time_length(cache.strategy)

struct ActiveGivenStrategyCache{A <: AbstractArray{Int32}} <: NonOptimizingStrategyCache
    strategy::A
end
Base.getindex(cache::GivenStrategyCache, k) = ActiveGivenStrategyCache(cache.strategy[k])
Base.getindex(cache::ActiveGivenStrategyCache, j) = cache.strategy[j]

function extract_strategy!(::GivenStrategyCache, values, V, j, maximize)
    throw(ArgumentError("The strategy is given and not supposed to optimize over actions."))
end
function extract_strategy!(::ActiveGivenStrategyCache, values, V, j, maximize)
    throw(ArgumentError("The strategy is given and not supposed to optimize over actions."))
end
step_postprocess_strategy_cache!(::GivenStrategyCache) = nothing
step_postprocess_strategy_cache!(::ActiveGivenStrategyCache) = nothing

construct_strategy_cache(problem::ControlSynthesisProblem) = construct_strategy_cache(
    problem,
    Val(isfinitetime(system_property(specification(problem)))),
)

# Strategy cache for storing time-varying policies
struct TimeVaryingStrategyCache{A <: AbstractArray{Int32}} <: OptimizingStrategyCache
    cur_strategy::A
    strategy::Vector{A}
end

function TimeVaryingStrategyCache(cur_strategy::A) where {A}
    return TimeVaryingStrategyCache(cur_strategy, Vector{A}())
end

function construct_strategy_cache(problem::ControlSynthesisProblem, time_varying::Val{true})
    mp = system(problem)
    cur_strategy = arrayfactory(mp, Int32, product_num_states(mp))
    return TimeVaryingStrategyCache(cur_strategy)
end

function replacezerobyone!(array)
    array[array .== 0] .= 1
    return array
end

cachetostrategy(strategy_cache::TimeVaryingStrategyCache) = TimeVaryingStrategy([
    replacezerobyone!(indices) for indices in reverse(strategy_cache.strategy)
])

function extract_strategy!(
    strategy_cache::TimeVaryingStrategyCache,
    values::AbstractArray{R},
    V,
    j,
    maximize,
) where {R <: Real}
    opt_val = maximize ? typemin(R) : typemax(R)
    opt_index = 1
    neutral = (opt_val, opt_index)

    return _extract_strategy!(strategy_cache.cur_strategy, values, neutral, j, maximize)
end
function step_postprocess_strategy_cache!(strategy_cache::TimeVaryingStrategyCache)
    push!(strategy_cache.strategy, copy(strategy_cache.cur_strategy))
end

# Strategy cache for storing stationary policies
struct StationaryStrategyCache{A <: AbstractArray{Int32}} <: OptimizingStrategyCache
    strategy::A
end

function construct_strategy_cache(
    problem::ControlSynthesisProblem,
    time_varying::Val{false},
)
    mp = system(problem)
    strategy = arrayfactory(mp, Int32, product_num_states(mp))
    return StationaryStrategyCache(strategy)
end

cachetostrategy(strategy_cache::StationaryStrategyCache) =
    StationaryStrategy(replacezerobyone!(strategy_cache.strategy))

function extract_strategy!(
    strategy_cache::StationaryStrategyCache,
    values::AbstractArray{R},
    V,
    j,
    maximize,
) where {R <: Real}
    neutral = if iszero(strategy_cache.strategy[j])
        maximize ? typemin(R) : typemax(R), 1
    else
        V[j], strategy_cache.strategy[j]
    end

    return _extract_strategy!(strategy_cache.strategy, values, neutral, j, maximize)
end
step_postprocess_strategy_cache!(::StationaryStrategyCache) = nothing

# Shared between stationary and time-varying strategies
function _extract_strategy!(cur_strategy, values, neutral, j, maximize)
    gt = maximize ? (>) : (<)

    opt_val, opt_index = neutral

    for (i, v) in enumerate(values)
        if gt(v, opt_val)
            opt_val = v
            opt_index = i
        end
    end

    @inbounds cur_strategy[j] = opt_index
    return opt_val
end
