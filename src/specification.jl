### Property types

"""
    Property

Super type for all system Property
"""
abstract type Property end

function checktimehorizon!(prop, ::StationaryIntervalMarkovProcess)
    @assert time_horizon(prop) > 0
end

function checktimehorizon!(prop, system::TimeVaryingIntervalMarkovProcess)
    @assert time_horizon(prop) > 0
    # It is not meaningful to check a property with a different time horizon.
    @assert time_horizon(prop) == time_length(system) "The time horizon of the property does not match the time length of the system"
end

function checkconvergence!(prop)
    @assert convergence_eps(prop) > 0
end

## Temporal logics

"""
    AbstractTemporalLogic

Super type for temporal logic property
"""
abstract type AbstractTemporalLogic <: Property end

"""
    LTLFormula

Linear Temporal Logic (LTL) property (first-order logic + next and until operators) [1].
Let ``ϕ`` denote the formula and ``M`` denote an interval Markov process. Then compute ``M ⊧ ϕ``.

[1] Vardi, M.Y. (1996). An automata-theoretic approach to linear temporal logic. In: Moller, F., Birtwistle, G. (eds) Logics for Concurrency. Lecture Notes in Computer Science, vol 1043. Springer, Berlin, Heidelberg.
"""
struct LTLFormula <: AbstractTemporalLogic
    formula::String
end

"""
    isfinitetime(prop::LTLFormula)

Return `false` for an LTL formula. LTL formulas are not finite time property.
"""
isfinitetime(prop::LTLFormula) = false

"""
    LTLfFormula

An LTL formula over finite traces [1]. See [`LTLFormula`](@ref) for the structure of LTL formulas.
Let ``ϕ`` denote the formula, ``M`` denote an interval Markov process, and ``H`` the time horizon.
Then compute ``M ⊧ ϕ`` within traces of length ``H``.

### Fields
- `formula::String`: LTL formula
- `time_horizon::T`: Time horizon of the finite traces 

[1] Giuseppe De Giacomo and Moshe Y. Vardi. 2013. Linear temporal logic and linear dynamic logic on finite traces. In Proceedings of the Twenty-Third international joint conference on Artificial Intelligence (IJCAI '13). AAAI Press, 854–860.
"""
struct LTLfFormula{T <: Integer} <: AbstractTemporalLogic
    formula::String
    time_horizon::T
end

"""
    isfinitetime(spec::LTLfFormula)

Return `true` for an LTLf formula. LTLf formulas are specifically over finite traces.
"""
isfinitetime(prop::LTLfFormula) = true

"""
    time_horizon(spec::LTLfFormula)

Return the time horizon of an LTLf formula.
"""
time_horizon(prop::LTLfFormula) = prop.time_horizon

"""
    PCTLFormula

A Probabilistic Computation Tree Logic (PCTL) formula [1].
Let ``ϕ`` denote the formula and ``M`` denote an interval Markov process. Then compute ``M ⊧ ϕ``.

[1] M. Lahijanian, S. B. Andersson and C. Belta, "Formal Verification and Synthesis for Discrete-Time Stochastic Systems," in IEEE Transactions on Automatic Control, vol. 60, no. 8, pp. 2031-2045, Aug. 2015, doi: 10.1109/TAC.2015.2398883.

"""
struct PCTLFormula <: AbstractTemporalLogic
    formula::String
end

## Reachability

"""
    AbstractReachability

Super type for all reachability-like property.
"""
abstract type AbstractReachability <: Property end

function initialize!(value_function, prop::AbstractReachability)
    @inbounds value_function.prev[reach(prop)] .= 1
end

function postprocess!(value_function, prop::AbstractReachability)
    @inbounds value_function.cur[reach(prop)] .= 1
end

"""
    FiniteTimeReachability{T <: Integer, VT <: AbstractVector{T}}

Finite time reachability specified by a set of target/terminal states and a time horizon. 
That is, if ``T`` is the set of target states and ``H`` is the time horizon, compute
``ℙ(∃k = 0…H, s_k ∈ T)``.
"""
struct FiniteTimeReachability{T <: Integer, VT <: AbstractVector{T}} <: AbstractReachability
    terminal_states::VT
    time_horizon::Any
end

function checkproperty!(prop::FiniteTimeReachability, system::IntervalMarkovProcess)
    checkterminal!(terminal_states(prop), system)
    checktimehorizon!(prop, system)
end

"""
    isfinitetime(prop::FiniteTimeReachability)

Return `true` for FiniteTimeReachability.
"""
isfinitetime(prop::FiniteTimeReachability) = true

"""
    time_horizon(prop::FiniteTimeReachability)

Return the time horizon of a finite time reachability property.
"""
time_horizon(prop::FiniteTimeReachability) = prop.time_horizon

"""
    terminal_states(spec::FiniteTimeReachability)

Return the set of terminal states of a finite time reachability property.
"""
terminal_states(prop::FiniteTimeReachability) = prop.terminal_states

"""
    terminal_states(prop::FiniteTimeReachability)

Return the set of states with which to compute reachbility for a finite time reachability prop.
This is equivalent for [`terminal_states(prop::FiniteTimeReachability)`](@ref) for a regular reachability
property. See [`FiniteTimeReachAvoid`](@ref) for a more complex property where the reachability and
terminal states differ.
"""
reach(prop::FiniteTimeReachability) = prop.terminal_states

"""
    InfiniteTimeReachability{R <: Real, T <: Integer, VT <: AbstractVector{T}} 
 
`InfiniteTimeReachability` is similar to [`FiniteTimeReachability`](@ref) except that the time horizon is infinite.
The convergence threshold is that the largest value of the most recent Bellman residual is less than `eps`.
"""
struct InfiniteTimeReachability{R <: Real, T <: Integer, VT <: AbstractVector{T}} <:
       AbstractReachability
    terminal_states::VT
    convergence_eps::R
end

function checkproperty!(
    prop::InfiniteTimeReachability,
    system::StationaryIntervalMarkovProcess,
)
    checkterminal!(terminal_states(prop), system)
    checkconvergence!(prop)
end

function checkproperty!(::InfiniteTimeReachability, ::TimeVaryingIntervalMarkovProcess)
    @assert false "Time-varying interval Markov processes are not supported for infinite time properties."
end

"""
    isfinitetime(prop::InfiniteTimeReachability)

Return `false` for InfiniteTimeReachability.
"""
isfinitetime(prop::InfiniteTimeReachability) = false

"""
    convergence_eps(prop::InfiniteTimeReachability)

Return the convergence threshold of an infinite time reachability property.
"""
convergence_eps(prop::InfiniteTimeReachability) = prop.convergence_eps

"""
    terminal_states(prop::InfiniteTimeReachability)

Return the set of terminal states of an infinite time reachability property.
"""
terminal_states(prop::InfiniteTimeReachability) = prop.terminal_states

"""
    reach(prop::InfiniteTimeReachability)

Return the set of states with which to compute reachbility for a infinite time reachability property.
This is equivalent for [`terminal_states(prop::InfiniteTimeReachability)`](@ref) for a regular reachability
property. See [`InfiniteTimeReachAvoid`](@ref) for a more complex property where the reachability and
terminal states differ.
"""
reach(prop::InfiniteTimeReachability) = prop.terminal_states

## Reach-avoid

"""
    AbstractReachAvoid

A property of reachability that includes a set of states to avoid.
"""
abstract type AbstractReachAvoid <: AbstractReachability end

function postprocess!(value_function, prop::AbstractReachAvoid)
    @inbounds value_function.cur[reach(prop)] .= 1
    @inbounds value_function.cur[avoid(prop)] .= 0
end

"""
    FiniteTimeReachAvoid{T <: Integer, VT <: AbstractVector{T}}

Finite time reach-avoid specified by a set of target/terminal states, a set of avoid states, and a time horizon.
That is, if ``T`` is the set of target states, ``A`` is the set of states to avoid, and ``H`` is the time horizon, compute
``ℙ(∃k = 0…H, s_k ∈ T and ∀k' = 0…k, s_k' ∉ A)``.
"""
struct FiniteTimeReachAvoid{T <: Integer, VT <: AbstractVector{T}} <: AbstractReachAvoid
    reach::VT
    avoid::VT
    time_horizon::Any
end

function checkproperty!(prop::FiniteTimeReachAvoid, system::IntervalMarkovProcess)
    checkterminal!(terminal_states(prop), system)
    checkdisjoint!(reach(prop), avoid(prop))
    checktimehorizon!(prop, system)
end

"""
    isfinitetime(prop::FiniteTimeReachAvoid)

Return `true` for FiniteTimeReachAvoid.
"""
isfinitetime(prop::FiniteTimeReachAvoid) = true

"""
    time_horizon(prop::FiniteTimeReachAvoid)

Return the time horizon of a finite time reach-avoid property.
"""
time_horizon(prop::FiniteTimeReachAvoid) = prop.time_horizon

"""
    terminal_states(prop::FiniteTimeReachAvoid)

Return the set of terminal states of a finite time reach-avoid property.
That is, the union of the reach and avoid sets.
"""
terminal_states(prop::FiniteTimeReachAvoid) = [prop.reach; prop.avoid]

"""
    reach(prop::FiniteTimeReachAvoid)

Return the set of target states.
"""
reach(prop::FiniteTimeReachAvoid) = prop.reach

"""
    avoid(prop::FiniteTimeReachAvoid)

Return the set of states to avoid.
"""
avoid(prop::FiniteTimeReachAvoid) = prop.avoid

"""
    InfiniteTimeReachAvoid{R <: Real, T <: Integer, VT <: AbstractVector{T}}

`InfiniteTimeReachAvoid` is similar to [`FiniteTimeReachAvoid`](@ref) except that the time horizon is infinite.
"""
struct InfiniteTimeReachAvoid{R <: Real, T <: Integer, VT <: AbstractVector{T}} <:
       AbstractReachAvoid
    reach::VT
    avoid::VT
    convergence_eps::R
end

function checkproperty!(
    prop::InfiniteTimeReachAvoid,
    system::StationaryIntervalMarkovProcess,
)
    checkterminal!(terminal_states(prop), system)
    checkdisjoint!(reach(prop), avoid(prop))
    checkconvergence!(prop)
end

function checkproperty!(::InfiniteTimeReachAvoid, ::TimeVaryingIntervalMarkovProcess)
    @assert false "Time-varying interval Markov processes are not supported for infinite time properties."
end

"""
    isfinitetime(prop::InfiniteTimeReachAvoid)

Return `false` for InfiniteTimeReachAvoid.
"""
isfinitetime(prop::InfiniteTimeReachAvoid) = false

"""
    convergence_eps(prop::InfiniteTimeReachAvoid)

Return the convergence threshold of an infinite time reach-avoid property.
"""
convergence_eps(prop::InfiniteTimeReachAvoid) = prop.convergence_eps

"""
    terminal_states(prop::InfiniteTimeReachAvoid)

Return the set of terminal states of an infinite time reach-avoid property.
That is, the union of the reach and avoid sets.
"""
terminal_states(prop::InfiniteTimeReachAvoid) = [prop.reach; prop.avoid]

"""
    reach(prop::InfiniteTimeReachAvoid)

Return the set of target states.
"""
reach(prop::InfiniteTimeReachAvoid) = prop.reach

"""
    avoid(prop::InfiniteTimeReachAvoid)

Return the set of states to avoid.
"""
avoid(prop::InfiniteTimeReachAvoid) = prop.avoid

function checkterminal!(terminal_states, system)
    nstates = num_states(system)
    for j in terminal_states
        @assert 1 <= j <= nstates "The terminal state $j is not a valid state"
    end
end

function checkdisjoint!(reach, avoid)
    @assert isdisjoint(reach, avoid) "The reach and avoid sets are not disjoint"
end

## Reward
"""
    AbstractReward{R <: Real}

Super type for all reward specifications.
"""
abstract type AbstractReward{R <: Real} <: Property end

function initialize!(value_function, prop::AbstractReward)
    value_function.prev .= reward(prop)
end

function postprocess!(value_function, prop::AbstractReward)
    rmul!(value_function.cur, discount(prop))
    value_function.cur += reward(prop)
end

function checkreward!(prop::AbstractReward, system::IntervalMarkovProcess)
    checkdevice!(reward(prop), transition_prob(system))
    @assert length(reward(prop)) == num_states(system)
    @assert 0 < discount(prop)  # Discount must be non-negative.
end

function checkdevice!(v::AbstractVector, p::IntervalProbabilities)
    # Lower and gap are required to be the same type.
    checkdevice!(v, lower(p))
end

function checkdevice!(::AbstractVector, ::AbstractMatrix)
    # Both arguments are on the CPU (technically in RAM).
    return nothing
end

"""
    FiniteTimeReward{R <: Real, T <: Integer, VR <: AbstractVector{R}}

`FiniteTimeReward` is a property of rewards assigned to each state at each iteration
and a discount factor. The time horizon is finite, so the discount factor is optional and 
the optimal policy will be time-varying.
"""
struct FiniteTimeReward{R <: Real, T <: Integer, VR <: AbstractVector{R}} <:
       AbstractReward{R}
    reward::VR
    discount::R
    time_horizon::T
end

function checkproperty!(prop::FiniteTimeReward, system::IntervalMarkovProcess)
    checkreward!(prop, system)
    checktimehorizon!(prop, system)
end

"""
    isfinitetime(prop::FiniteTimeReward)

Return `true` for FiniteTimeReward.
"""
isfinitetime(prop::FiniteTimeReward) = true

"""
    reward(prop::FiniteTimeReward)

Return the reward vector of a finite time reward optimization.
"""
reward(prop::FiniteTimeReward) = prop.reward

"""
    discount(prop::FiniteTimeReward)

Return the discount factor of a finite time reward optimization.
"""
discount(prop::FiniteTimeReward) = prop.discount

"""
    time_horizon(prop::FiniteTimeReward)

Return the time horizon of a finite time reward optimization.
"""
time_horizon(prop::FiniteTimeReward) = prop.time_horizon

"""
    InfiniteTimeReward{R <: Real, VR <: AbstractVector{R}}

`InfiniteTimeReward` is a property of rewards assigned to each state at each iteration
and a discount factor for guaranteed convergence. The time horizon is infinite, so the optimal
policy will be stationary.
"""
struct InfiniteTimeReward{R <: Real, VR <: AbstractVector{R}} <: AbstractReward{R}
    reward::VR
    discount::R
    convergence_eps::R
end

function checkproperty!(prop::InfiniteTimeReward, system::StationaryIntervalMarkovProcess)
    checkreward!(prop, system)
    @assert discount(prop) < 1
    checkconvergence!(prop)
end

function checkproperty!(::InfiniteTimeReward, ::TimeVaryingIntervalMarkovProcess)
    @assert false "Time-varying interval Markov processes are not supported for infinite time properties."
end

"""
    isfinitetime(prop::InfiniteTimeReward)

Return `false` for InfiniteTimeReward.
"""
isfinitetime(prop::InfiniteTimeReward) = false

"""
    reward(prop::FiniteTimeReward)

Return the reward vector of a finite time reward optimization.
"""
reward(prop::InfiniteTimeReward) = prop.reward

"""
    discount(prop::FiniteTimeReward)

Return the discount factor of a finite time reward optimization.
"""
discount(prop::InfiniteTimeReward) = prop.discount

"""
    convergence_eps(prop::InfiniteTimeReward)

Return the convergence threshold of an infinite time reward optimization.
"""
convergence_eps(prop::InfiniteTimeReward) = prop.convergence_eps

## Problem

"""
    SatisfactionMode

When computing the satisfaction probability of a property over an interval Markov process,
be it IMC or IMDP, the desired satisfaction probability to verify can either be `Optimistic` or
`Pessimistic`. That is, upper and lower bounds on the satisfaction probability within
the probability uncertainty.
"""
@enum SatisfactionMode Pessimistic Optimistic

"""
    StrategyMode

When computing the satisfaction probability of a property over an IMDP, the strategy
can either maximize or minimize the satisfaction probability (wrt. the satisfaction mode).
"""
@enum StrategyMode Maximize Minimize

"""
    Specification{F <: Property}

A specfication is a property together with a satisfaction mode and a strategy mode. 
The satisfaction mode is either `Optimistic` or `Pessimistic`. See [`SatisfactionMode`](@ref) for more details.
The strategy  mode is either `Maxmize` or `Minimize`. See [`StrategyMode`](@ref) for more details.

### Fields
- `prop::F`: verification property (either temporal logic or reachability-like).
- `satisfaction::SatisfactionMode`: satisfaction mode (either optimistic or pessimistic). Default is pessimistic.
- `strategy::StrategyMode`: strategy mode (either maximize or minimize). Default is maximize.
"""
struct Specification{F <: Property}
    prop::F
    satisfaction::SatisfactionMode
    strategy::StrategyMode
end

Specification(prop::Property) = Specification(prop, Pessimistic)
Specification(prop::Property, satisfaction::SatisfactionMode) =
    Specification(prop, satisfaction, Maximize)

initialize!(value_function, spec::Specification) =
    initialize!(value_function, system_property(spec))
postprocess!(value_function, spec::Specification) =
    postprocess!(value_function, system_property(spec))

function checkspecification!(spec::Specification, system)
    return checkproperty!(system_property(spec), system)
end

"""
    system_property(spec::Specification)
"""
system_property(spec::Specification) = spec.prop

"""
    satisfaction_mode(spec::Specification)

Return the satisfaction mode of a specification.
"""
satisfaction_mode(spec::Specification) = spec.satisfaction

"""
    strategy_mode(spec::Specification)

Return the strategy mode of a specification.
"""
strategy_mode(spec::Specification) = spec.strategy

"""
    Problem{S <: IntervalMarkovProcess, F <: Specification}

A problem is a tuple of an interval Markov process and a specification.

### Fields
- `system::S`: interval Markov process.
- `spec::F`: specification (either temporal logic or reachability-like).
"""
struct Problem{S <: IntervalMarkovProcess, F <: Specification}
    system::S
    spec::F

    function Problem(
        system::S,
        spec::F,
    ) where {S <: IntervalMarkovProcess, F <: Specification}
        checkspecification!(spec, system)
        return new{S, F}(system, spec)
    end
end
"""
    system(prob::Problem)

Return the system of a problem.
"""
system(prob::Problem) = prob.system

"""
    specification(prob::Problem)

Return the specification of a problem.
"""
specification(prob::Problem) = prob.spec
