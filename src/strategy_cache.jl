abstract type AbstractStrategyCache end
abstract type NonOptimizingStrategyCache <: AbstractStrategyCache end
abstract type OptimizingStrategyCache <: AbstractStrategyCache end

select_strategy_cache(strategy_cache::OptimizingStrategyCache, k) = strategy_cache
select_strategy_cache(strategy_cache::NonOptimizingStrategyCache, k) =
    strategy_cache[time_length(strategy_cache) - k]

"""
    construct_strategy_cache(mp_or_problem)

Construct a strategy cache for a given problem. The resuling cache type depends on
the specification and the device to store the strategy depends on the device of the Markov process.
"""
function construct_strategy_cache end

# Strategy cache for not storing policies - useful for dispatching
struct NoStrategyCache <: OptimizingStrategyCache end

function construct_strategy_cache(::Union{<:AbstractAmbiguitySets, <:StochasticProcess})
    return NoStrategyCache()
end

construct_strategy_cache(::VerificationProblem{S, F, <:NoStrategy}) where {S, F} =
    NoStrategyCache()

function extract_strategy!(::NoStrategyCache, values, available_actions, j, maximize)
    values = @view values[available_actions]
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

struct ActiveGivenStrategyCache{N, A <: AbstractArray{NTuple{N, Int32}}} <:
       NonOptimizingStrategyCache
    strategy::A
end
Base.getindex(cache::GivenStrategyCache, k) = ActiveGivenStrategyCache(cache.strategy[k])
Base.getindex(cache::ActiveGivenStrategyCache, j) = cache.strategy[j]

step_postprocess_strategy_cache!(::GivenStrategyCache) = nothing

construct_strategy_cache(problem::ControlSynthesisProblem) = construct_strategy_cache(
    problem,
    Val(isfinitetime(system_property(specification(problem)))),
)

# Strategy cache for storing time-varying policies
struct TimeVaryingStrategyCache{N, A <: AbstractArray{NTuple{N, Int32}}} <:
       OptimizingStrategyCache
    cur_strategy::A
    strategy::Vector{A}
end

function TimeVaryingStrategyCache(
    cur_strategy::A,
) where {N, A <: AbstractArray{NTuple{N, Int32}}}
    return TimeVaryingStrategyCache(cur_strategy, Vector{A}())
end

function construct_strategy_cache(problem::ControlSynthesisProblem, time_varying::Val{true})
    mp = system(problem)
    N = length(action_values(mp))
    cur_strategy = arrayfactory(mp, NTuple{N, Int32}, source_shape(mp))
    cur_strategy .= (ntuple(_ -> 0, N),)
    return TimeVaryingStrategyCache(cur_strategy)
end

cachetostrategy(strategy_cache::TimeVaryingStrategyCache) =
    TimeVaryingStrategy(collect(reverse(strategy_cache.strategy)))

function extract_strategy!(
    strategy_cache::TimeVaryingStrategyCache,
    values::AbstractArray{R},
    available_actions,
    jₛ,
    maximize,
) where {R <: Real}
    opt_val = maximize ? typemin(R) : typemax(R)
    opt_index = Tuple(first(available_actions))
    neutral = (opt_val, opt_index)

    return _extract_strategy!(
        strategy_cache.cur_strategy,
        values,
        available_actions,
        neutral,
        jₛ,
        maximize,
    )
end
function step_postprocess_strategy_cache!(strategy_cache::TimeVaryingStrategyCache)
    push!(strategy_cache.strategy, copy(strategy_cache.cur_strategy))
end

# Strategy cache for storing stationary policies
struct StationaryStrategyCache{N, A <: AbstractArray{NTuple{N, Int32}}} <:
       OptimizingStrategyCache
    strategy::A
end

function construct_strategy_cache(
    problem::ControlSynthesisProblem,
    time_varying::Val{false},
)
    mp = system(problem)
    N = length(action_values(mp))
    strategy = arrayfactory(mp, NTuple{N, Int32}, source_shape(mp))
    strategy .= (ntuple(_ -> 0, N),)
    return StationaryStrategyCache(strategy)
end

cachetostrategy(strategy_cache::StationaryStrategyCache) =
    StationaryStrategy(strategy_cache.strategy)

function extract_strategy!(
    strategy_cache::StationaryStrategyCache,
    values::AbstractArray{R},
    available_actions,
    jₛ,
    maximize,
) where {R <: Real}
    neutral = if all(iszero.(strategy_cache.strategy[jₛ])) || jₛ ∉ available_actions
        maximize ? typemin(R) : typemax(R), Tuple(first(available_actions))
    else
        s = strategy_cache.strategy[jₛ]
        values[CartesianIndex(s)], s
    end

    return _extract_strategy!(
        strategy_cache.strategy,
        values,
        available_actions,
        neutral,
        jₛ,
        maximize,
    )
end
step_postprocess_strategy_cache!(::StationaryStrategyCache) = nothing

# Shared between stationary and time-varying strategies
function _extract_strategy!(cur_strategy, values, available_actions, neutral, jₛ, maximize)
    gt = maximize ? (>) : (<)

    opt_val, opt_index = neutral

    for jₐ in available_actions
        v = values[jₐ]
        if gt(v, opt_val)
            opt_val = v
            opt_index = Tuple(jₐ)
        end
    end

    @inbounds cur_strategy[jₛ] = opt_index
    return opt_val
end

function strategy!(
    strategy_cache::OptimizingStrategyCache,
    Vres::AbstractArray{R},
    Q::AbstractArray{R},
    model,
    maximize,
) where {R <: Real}

    #TODO: can be threaded?
    @inbounds for jₛ in CartesianIndices(source_shape(model))
        Vres[jₛ] = extract_strategy!(
            strategy_cache,
            @view(Q[:, jₛ]),
            available(model, jₛ),
            jₛ,
            maximize,
        )
    end
end

########################################################################
# Per-(s, a) relax! primitive for sampling-based value iteration.
#
# The contract is: each state the sampler visits is updated asynchronously
# via a sequence of `relax!` calls. The `visited` scratch tracks states
# touched in the current iteration:
#   - First visit to a state this iteration: Vres[s] is overwritten with q
#     and the strategy is assigned unconditionally. This ensures the sweep
#     starts from the correct per-action "neutral" for the direction,
#     independent of V_prev's sign (works for monotone-up reachability and
#     monotone-down safety alike).
#   - Subsequent visits: relax strictly-better — Vres[s] ← q and strategy ← a
#     iff `q` beats the current best (`maximize ? q > Vres[s] : q < Vres[s]`).
#
# Callers who know that every state in the sweep is visited at most once
# (e.g. on-policy `ZipIterator`) can use `relax_nooptim!`, which skips the
# comparison and just assigns.
########################################################################

Base.@propagate_inbounds function relax!(
    strategy_cache::OptimizingStrategyCache,
    Vres::AbstractArray{R},
    visited::AbstractArray{Bool},
    jₛ,
    jₐ,
    q::R,
    maximize::Bool,
) where {R <: Real}
    if !visited[jₛ]
        Vres[jₛ] = q
        _set_strategy!(strategy_cache, jₛ, jₐ)
        visited[jₛ] = true
    else
        better = maximize ? (q > Vres[jₛ]) : (q < Vres[jₛ])
        if better
            Vres[jₛ] = q
            _set_strategy!(strategy_cache, jₛ, jₐ)
        end
    end
    return nothing
end

# Non-optimizing strategy cache: the sampler yields the given policy's
# action for each state, so just assign (no comparison, no strategy update).
Base.@propagate_inbounds function relax!(
    ::NonOptimizingStrategyCache,
    Vres::AbstractArray{R},
    visited::AbstractArray{Bool},
    jₛ,
    jₐ,
    q::R,
    maximize::Bool,
) where {R <: Real}
    Vres[jₛ] = q
    visited[jₛ] = true
    return nothing
end

_set_strategy!(::NoStrategyCache, jₛ, jₐ) = nothing
Base.@propagate_inbounds _set_strategy!(sc::TimeVaryingStrategyCache, jₛ, jₐ) =
    (sc.cur_strategy[jₛ] = Tuple(jₐ))
Base.@propagate_inbounds _set_strategy!(sc::StationaryStrategyCache, jₛ, jₐ) =
    (sc.strategy[jₛ] = Tuple(jₐ))
