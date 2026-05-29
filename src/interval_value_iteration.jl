"""
    solve(problem::AbstractIntervalMDPProblem, alg::IntervalValueIteration; callback=nothing)

Solve reach-avoid specification problems using interval value iteration (IVI) for interval Markov processes.

In contrast to [`RobustValueIteration`](@ref), IVI propagates both a lower and an upper bound on the
reach-avoid satisfaction probability simultaneously. The lower bound is initialized to zero everywhere
except on the reach set, where it is set to one. The upper bound is initialized to one everywhere
except on the avoid set, where it is set to zero. At each iteration, the strategy is synthesized with
respect to one of the two bounds — the lower bound when the satisfaction mode is `Pessimistic` and the
upper bound when it is `Optimistic` — and the synthesized strategy is then applied directly to the
other bound. For infinite-horizon problems, value iteration terminates when the gap between the upper
and lower bound over the set of initial states drops below `convergence_eps`.

IVI is only defined for reach-avoid properties (see [`FiniteTimeReachAvoid`](@ref) and
[`InfiniteTimeReachAvoid`](@ref)).

It is possible to provide a callback function that will be called at each iteration with the lower
and upper bound value functions and the iteration count. The callback should have the signature
`callback(V_lower::AbstractArray, V_upper::AbstractArray, k::Int)`.

The returned [`VerificationSolution`](@ref) or [`ControlSynthesisSolution`](@ref) stores the
satisfaction-mode bound as its `value_function` (`V_lower` for `Pessimistic`, `V_upper` for
`Optimistic`), the gap `V_upper - V_lower` as its `residual`, and the other bound in `additional_data`.
"""
function solve(problem::VerificationProblem, alg::IntervalValueIteration; kwargs...)
    V_lower, V_upper, k, gap, _ = _interval_value_iteration!(problem, alg; kwargs...)
    return _ivi_verification_solution(specification(problem), V_lower, V_upper, gap, k)
end

function solve(problem::ControlSynthesisProblem, alg::IntervalValueIteration; kwargs...)
    V_lower, V_upper, k, gap, strategy_cache = _interval_value_iteration!(problem, alg; kwargs...)
    strat = cachetostrategy(strategy_cache)
    return _ivi_control_synthesis_solution(specification(problem), strat, V_lower, V_upper, gap, k)
end

function _ivi_verification_solution(spec, V_lower, V_upper, gap, k)
    if ispessimistic(spec)
        return VerificationSolution(V_lower, gap, k, V_upper)
    else
        return VerificationSolution(V_upper, gap, k, V_lower)
    end
end

function _ivi_control_synthesis_solution(spec, strat, V_lower, V_upper, gap, k)
    if ispessimistic(spec)
        return ControlSynthesisSolution(strat, V_lower, gap, k, V_upper)
    else
        return ControlSynthesisSolution(strat, V_upper, gap, k, V_lower)
    end
end

function _interval_value_iteration!(
    problem::AbstractIntervalMDPProblem,
    alg::IntervalValueIteration;
    callback = nothing,
)
    mp = system(problem)
    spec = specification(problem)
    checkivisupported(system_property(spec))

    term_criteria = ivi_termination_criteria(spec)

    workspace = construct_workspace(mp, bellman_algorithm(alg))
    strategy_cache = construct_ivi_strategy_cache(problem)

    V_lower = ValueFunction(problem)
    V_upper = ValueFunction(problem)
    initialize_ivi!(V_lower, V_upper, spec)
    nextiteration!(V_lower)
    nextiteration!(V_upper)

    ivi_step!(workspace, strategy_cache, V_lower, V_upper, 0, mp, spec)
    k = 1

    if !isnothing(callback)
        callback(V_lower.current, V_upper.current, k)
    end

    while !term_criteria(V_lower, V_upper, k, mp)
        nextiteration!(V_lower)
        nextiteration!(V_upper)

        ivi_step!(workspace, strategy_cache, V_lower, V_upper, k, mp, spec)
        k += 1

        if !isnothing(callback)
            callback(V_lower.current, V_upper.current, k)
        end
    end

    postprocess_value_function!(V_lower, spec)
    postprocess_value_function!(V_upper, spec)

    gap = V_upper.previous
    gap .= V_upper.current .- V_lower.current

    return V_lower.current, V_upper.current, k, gap, strategy_cache
end

# Termination criteria
function ivi_termination_criteria(spec::Specification)
    prop = system_property(spec)
    ft = isfinitetime(prop)
    return ivi_termination_criteria(prop, Val(ft))
end

struct IVIFixedIterationsCriteria{T <: Integer} <: TerminationCriteria
    n::T
end
(f::IVIFixedIterationsCriteria)(V_lower, V_upper, k, mp) = k >= f.n
ivi_termination_criteria(prop, ::Val{true}) = IVIFixedIterationsCriteria(time_horizon(prop))

struct IVIInitialGapCriteria{T <: Real} <: TerminationCriteria
    tol::T
end
function (f::IVIInitialGapCriteria)(V_lower, V_upper, k, mp)
    return max_initial_gap(V_lower.current, V_upper.current, initial_states(mp)) < f.tol
end
ivi_termination_criteria(prop, ::Val{false}) = IVIInitialGapCriteria(convergence_eps(prop))

function max_initial_gap(V_lower, V_upper, ::AllStates)
    diff = zero(eltype(V_lower))
    @inbounds for i in eachindex(V_lower, V_upper)
        d = V_upper[i] - V_lower[i]
        if d > diff
            diff = d
        end
    end
    return diff
end

function max_initial_gap(V_lower, V_upper, initial::AbstractVector)
    diff = zero(eltype(V_lower))
    @inbounds for s in initial
        ci = CartesianIndex(s)
        d = V_upper[ci] - V_lower[ci]
        if d > diff
            diff = d
        end
    end
    return diff
end

# One IVI step: synthesize on the primary bound, apply the synthesized strategy to
# the other bound. Both Bellman calls use the same nature (`upper_bound` flag)
# determined by the satisfaction mode — V_lower and V_upper are under- and
# over-approximations of the same value function, so they must follow the same
# Bellman operator for the gap to vanish in the limit.
function ivi_step!(workspace, strategy_cache, V_lower, V_upper, k, mp, spec)
    prop = system_property(spec)
    maximize = ismaximize(spec)
    upper_bound = isoptimistic(spec)
    model = select_model(mp, k)
    primary_cache = select_strategy_cache(strategy_cache, k)

    # The "primary" bound is the one used to synthesize the strategy:
    # the lower bound for Pessimistic, the upper bound for Optimistic.
    primary_current, primary_previous, secondary_current, secondary_previous =
        if ispessimistic(spec)
            V_lower.current, V_lower.previous, V_upper.current, V_upper.previous
        else
            V_upper.current, V_upper.previous, V_lower.current, V_lower.previous
        end

    # Synthesize the strategy on the primary bound.
    bellman!(
        workspace,
        primary_cache,
        primary_current,
        primary_previous,
        model;
        upper_bound = upper_bound,
        maximize = maximize,
        prop = prop,
    )

    # Apply the synthesized strategy to the other bound.
    bellman!(
        workspace,
        applied_strategy_cache(primary_cache),
        secondary_current,
        secondary_previous,
        model;
        upper_bound = upper_bound,
        maximize = maximize,
        prop = prop,
    )

    step_postprocess_value_function!(V_lower, spec)
    step_postprocess_value_function!(V_upper, spec)
    step_postprocess_strategy_cache!(strategy_cache)
end
