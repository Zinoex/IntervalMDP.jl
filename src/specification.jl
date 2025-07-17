### Property types

"""
    Property

Super type for all system Property
"""
abstract type Property end

function checkmodelpropertycompatibility(prop, system)
    throw(
        ArgumentError(
            "the property $(typeof(prop)) is not compatible with the system $(typeof(system))",
        ),
    )
end

"""
    BasicProperty

A basic property that applies to a "raw" [`IntervalMarkovProcess`](@ref).
"""
abstract type BasicProperty <: Property end

function checkmodelpropertycompatibility(::BasicProperty, ::IntervalMarkovProcess)
    return nothing
end

"""
    ProductProperty

A property that applies to a [`ProductProcess`](@ref).
"""
abstract type ProductProperty <: Property end

function checkmodelpropertycompatibility(::ProductProperty, ::ProductProcess)
    return nothing
end

function checktimehorizon(prop, ::AbstractStrategy)
    if time_horizon(prop) < 1
        throw(
            DomainError(
                time_horizon(prop),
                "the time horizon of the property must be greater than 0",
            ),
        )
    end
end

function checktimehorizon(prop, strategy::TimeVaryingStrategy)
    if time_horizon(prop) < 1
        throw(
            DomainError(
                time_horizon(prop),
                "the time horizon of the property must be greater than 0",
            ),
        )
    end

    # It is not meaningful to check a property with a different time horizon.
    if time_horizon(prop) != time_length(strategy)
        throw(
            ArgumentError(
                "the time horizon of the property ($(time_horizon(prop))) does not match the time length of the strategy ($(time_length(strategy)))",
            ),
        )
    end
end

function checkconvergence(prop, ::AbstractStrategy)
    if convergence_eps(prop) <= 0
        throw(
            DomainError(
                convergence_eps(prop),
                "the convergence threshold of the property must be greater than 0",
            ),
        )
    end
end

function checkconvergence(prop, ::TimeVaryingStrategy)
    throw(
        ArgumentError(
            "time-varying strategies are not supported for infinite time properties.",
        ),
    )
end

## DFA reachability

"""
    AbstractDFAReachability

Super type for all reachability-like properties.
"""
abstract type AbstractDFAReachability <: ProductProperty end

function initialize!(value_function, prop::AbstractDFAReachability)
    @inbounds selectdim(
        value_function.current,
        ndims(value_function.current),
        reach(prop),
    ) .= 1.0
end

function step_specification!(value_function, prop::AbstractDFAReachability)
    @inbounds selectdim(
        value_function.current,
        ndims(value_function.current),
        reach(prop),
    ) .= 1.0
end

postprocess_value_function!(value_function, ::AbstractDFAReachability) = nothing

"""
    FiniteTimeDFAReachability{VT <: Vector{<:Int32}, T <: Integer}

Finite time reachability specified by a set of target/terminal states and a time horizon. 
That is, denote a trace by ``z_1 z_2 z_3 \\cdots`` with ``z_k = (s_k, q_k)`` then if ``T`` is the set of target states and ``H`` is the time horizon,
the property is 
```math
    \\mathbb{P}(\\exists k = \\{0, \\ldots, H\\}, q_k \\in T).
```
"""
struct FiniteTimeDFAReachability{VT <: Vector{<:Int32}, T <: Integer} <:
       AbstractDFAReachability
    terminal_states::VT
    time_horizon::T
end

function FiniteTimeDFAReachability(terminal_states::Vector{<:Integer}, time_horizon)
    terminal_states = Int32.(terminal_states)
    return FiniteTimeDFAReachability(terminal_states, time_horizon)
end

function checkproperty(prop::FiniteTimeDFAReachability, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::FiniteTimeDFAReachability, system)
    checkstatebounds(terminal_states(prop), system)
end

"""
    isfinitetime(prop::FiniteTimeDFAReachability)

Return `true` for FiniteTimeDFAReachability.
"""
isfinitetime(prop::FiniteTimeDFAReachability) = true

"""
    time_horizon(prop::FiniteTimeDFAReachability)

Return the time horizon of a finite time reachability property.
"""
time_horizon(prop::FiniteTimeDFAReachability) = prop.time_horizon

"""
    terminal_states(spec::FiniteTimeDFAReachability)

Return the set of terminal states of a finite time reachability property.
"""
terminal_states(prop::FiniteTimeDFAReachability) = prop.terminal_states

"""
    reach(prop::FiniteTimeDFAReachability)

Return the set of states with which to compute reachbility for a finite time reachability prop.
This is equivalent for [`terminal_states(prop::FiniteTimeDFAReachability)`](@ref) for a DFA reachability
property. 
"""
reach(prop::FiniteTimeDFAReachability) = prop.terminal_states

"""
    InfiniteTimeDFAReachability{R <: Real, VT <: Vector{<:Int32}} 
 
`InfiniteTimeDFAReachability` is similar to [`FiniteTimeDFAReachability`](@ref) except that the time horizon is infinite, i.e., ``H = \\infty``.
In practice it means, performing the value iteration until the value function has converged, defined by some threshold `convergence_eps`.
The convergence threshold is that the largest value of the most recent Bellman residual is less than `convergence_eps`.
"""
struct InfiniteTimeDFAReachability{R <: Real, VT <: Vector{<:Int32}} <:
       AbstractDFAReachability
    terminal_states::VT
    convergence_eps::R
end

function InfiniteTimeDFAReachability(terminal_states::Vector{<:Integer}, convergence_eps)
    terminal_states = Int32.(terminal_states)
    return InfiniteTimeDFAReachability(terminal_states, convergence_eps)
end

function checkproperty(prop::InfiniteTimeDFAReachability, system, strategy)
    checkconvergence(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::InfiniteTimeDFAReachability, system)
    checkstatebounds(terminal_states(prop), system)
end

"""
    isfinitetime(prop::InfiniteTimeDFAReachability)

Return `false` for InfiniteTimeDFAReachability.
"""
isfinitetime(prop::InfiniteTimeDFAReachability) = false

"""
    convergence_eps(prop::InfiniteTimeDFAReachability)

Return the convergence threshold of an infinite time reachability property.
"""
convergence_eps(prop::InfiniteTimeDFAReachability) = prop.convergence_eps

"""
    terminal_states(prop::InfiniteTimeDFAReachability)

Return the set of terminal states of an infinite time reachability property.
"""
terminal_states(prop::InfiniteTimeDFAReachability) = prop.terminal_states

"""
    reach(prop::InfiniteTimeDFAReachability)

Return the set of states with which to compute reachbility for a infinite time reachability property.
This is equivalent for [`terminal_states(prop::InfiniteTimeDFAReachability)`](@ref) for a DFA reachability
property.
"""
reach(prop::InfiniteTimeDFAReachability) = prop.terminal_states

## Reachability

"""
    AbstractReachability

Super type for all reachability-like properties.
"""
abstract type AbstractReachability <: BasicProperty end

function initialize!(value_function, prop::AbstractReachability)
    @inbounds value_function.current[reach(prop)] .= 1.0
end

function step_specification!(value_function, prop::AbstractReachability)
    @inbounds value_function.current[reach(prop)] .= 1.0
end

postprocess_value_function!(value_function, ::AbstractReachability) = nothing

"""
    FiniteTimeReachability{VT <: Vector{<:CartesianIndex}, T <: Integer}

Finite time reachability specified by a set of target/terminal states and a time horizon. 
That is, denote a trace by ``s_1 s_2 s_3 \\cdots``, then if ``T`` is the set of target states and ``H`` is the time horizon,
the property is 
```math
    \\mathbb{P}(\\exists k = \\{0, \\ldots, H\\}, s_k \\in T).
```
"""
struct FiniteTimeReachability{VT <: Vector{<:CartesianIndex}, T <: Integer} <:
       AbstractReachability
    terminal_states::VT
    time_horizon::T
end

function FiniteTimeReachability(terminal_states::Vector{<:UnionIndex}, time_horizon)
    terminal_states = CartesianIndex.(terminal_states)
    return FiniteTimeReachability(terminal_states, time_horizon)
end

function checkproperty(prop::FiniteTimeReachability, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::FiniteTimeReachability, system)
    checkstatebounds(terminal_states(prop), system)
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
    reach(prop::FiniteTimeReachability)

Return the set of states with which to compute reachbility for a finite time reachability prop.
This is equivalent for [`terminal_states(prop::FiniteTimeReachability)`](@ref) for a regular reachability
property. See [`FiniteTimeReachAvoid`](@ref) for a more complex property where the reachability and
terminal states differ.
"""
reach(prop::FiniteTimeReachability) = prop.terminal_states

"""
    InfiniteTimeReachability{R <: Real, VT <: Vector{<:CartesianIndex}} 
 
`InfiniteTimeReachability` is similar to [`FiniteTimeReachability`](@ref) except that the time horizon is infinite, i.e., ``H = \\infty``.
In practice it means, performing the value iteration until the value function has converged, defined by some threshold `convergence_eps`.
The convergence threshold is that the largest value of the most recent Bellman residual is less than `convergence_eps`.
"""
struct InfiniteTimeReachability{R <: Real, VT <: Vector{<:CartesianIndex}} <:
       AbstractReachability
    terminal_states::VT
    convergence_eps::R
end

function InfiniteTimeReachability(terminal_states::Vector{<:UnionIndex}, convergence_eps)
    terminal_states = CartesianIndex.(terminal_states)
    return InfiniteTimeReachability(terminal_states, convergence_eps)
end

function checkproperty(prop::InfiniteTimeReachability, system, strategy)
    checkconvergence(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::InfiniteTimeReachability, system)
    checkstatebounds(terminal_states(prop), system)
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

"""
    ExactTimeReachability{VT <: Vector{<:CartesianIndex}, T <: Integer}

Exact time reachability specified by a set of target/terminal states and a time horizon. 
That is, denote a trace by ``s_1 s_2 s_3 \\cdots``, then if ``T`` is the set of target states and ``H`` is the time horizon,
the property is 
```math
    \\mathbb{P}(s_H \\in T).
```
"""
struct ExactTimeReachability{VT <: Vector{<:CartesianIndex}, T <: Integer} <:
       AbstractReachability
    terminal_states::VT
    time_horizon::T
end

function ExactTimeReachability(terminal_states::Vector{<:UnionIndex}, time_horizon)
    terminal_states = CartesianIndex.(terminal_states)
    return ExactTimeReachability(terminal_states, time_horizon)
end

function checkproperty(prop::ExactTimeReachability, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::ExactTimeReachability, system)
    checkstatebounds(terminal_states(prop), system)
end

function step_specification!(_, ::ExactTimeReachability)
    return nothing
end

"""
    isfinitetime(prop::ExactTimeReachability)

Return `true` for ExactTimeReachability.
"""
isfinitetime(prop::ExactTimeReachability) = true

"""
    time_horizon(prop::ExactTimeReachability)

Return the time horizon of an exact time reachability property.
"""
time_horizon(prop::ExactTimeReachability) = prop.time_horizon

"""
    terminal_states(spec::ExactTimeReachability)

Return the set of terminal states of an exact time reachability property.
"""
terminal_states(prop::ExactTimeReachability) = prop.terminal_states

"""
    reach(prop::ExactTimeReachability)

Return the set of states with which to compute reachbility for an exact time reachability prop.
This is equivalent for [`terminal_states(prop::ExactTimeReachability)`](@ref) for a regular reachability
property. See [`ExactTimeReachAvoid`](@ref) for a more complex property where the reachability and
terminal states differ.
"""
reach(prop::ExactTimeReachability) = prop.terminal_states

## Reach-avoid

"""
    AbstractReachAvoid

A property of reachability that includes a set of states to avoid.
"""
abstract type AbstractReachAvoid <: AbstractReachability end

function step_specification!(value_function, prop::AbstractReachAvoid)
    @inbounds value_function.current[reach(prop)] .= 1.0
    @inbounds value_function.current[avoid(prop)] .= 0.0
end

"""
    FiniteTimeReachAvoid{VT <: AbstractVector{<:CartesianIndex}}, T <: Integer}

Finite time reach-avoid specified by a set of target/terminal states, a set of avoid states, and a time horizon.
That is, denote a trace by ``s_1 s_2 s_3 \\cdots``, then if ``T`` is the set of target states, ``A`` is the set of states to avoid,
and ``H`` is the time horizon, the property is 
```math
    \\mathbb{P}(\\exists k = \\{0, \\ldots, H\\}, s_k \\in T, \\text{ and } \\forall k' = \\{0, \\ldots, k\\}, s_k' \\notin A).
```
"""
struct FiniteTimeReachAvoid{VT <: AbstractVector{<:CartesianIndex}, T <: Integer} <:
       AbstractReachAvoid
    reach::VT
    avoid::VT
    time_horizon::T
end

function FiniteTimeReachAvoid(
    reach::Vector{<:UnionIndex},
    avoid::Vector{<:UnionIndex},
    time_horizon,
)
    reach = CartesianIndex.(reach)
    avoid = CartesianIndex.(avoid)
    return FiniteTimeReachAvoid(reach, avoid, time_horizon)
end

function checkproperty(prop::FiniteTimeReachAvoid, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::FiniteTimeReachAvoid, system)
    checkstatebounds(terminal_states(prop), system)
    checkdisjoint(reach(prop), avoid(prop))
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
    InfiniteTimeReachAvoid{R <: Real, VT <: AbstractVector{<:CartesianIndex}}

`InfiniteTimeReachAvoid` is similar to [`FiniteTimeReachAvoid`](@ref) except that the time horizon is infinite, i.e., ``H = \\infty``.
"""
struct InfiniteTimeReachAvoid{R <: Real, VT <: AbstractVector{<:CartesianIndex}} <:
       AbstractReachAvoid
    reach::VT
    avoid::VT
    convergence_eps::R
end

function InfiniteTimeReachAvoid(
    reach::Vector{<:UnionIndex},
    avoid::Vector{<:UnionIndex},
    convergence_eps,
)
    reach = CartesianIndex.(reach)
    avoid = CartesianIndex.(avoid)
    return InfiniteTimeReachAvoid(reach, avoid, convergence_eps)
end

function checkproperty(prop::InfiniteTimeReachAvoid, system, strategy)
    checkconvergence(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::InfiniteTimeReachAvoid, system)
    checkstatebounds(terminal_states(prop), system)
    checkdisjoint(reach(prop), avoid(prop))
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

"""
    ExactTimeReachAvoid{VT <: AbstractVector{<:CartesianIndex}}, T <: Integer}

Exact time reach-avoid specified by a set of target/terminal states, a set of avoid states, and a time horizon.
That is, denote a trace by ``s_1 s_2 s_3 \\cdots``, then if ``T`` is the set of target states, ``A`` is the set of states to avoid,
and ``H`` is the time horizon, the property is 
```math
    \\mathbb{P}(s_H \\in T, \\text{ and } \\forall k = \\{0, \\ldots, H\\}, s_k \\notin A).
```
"""
struct ExactTimeReachAvoid{VT <: AbstractVector{<:CartesianIndex}, T <: Integer} <:
       AbstractReachAvoid
    reach::VT
    avoid::VT
    time_horizon::T
end

function ExactTimeReachAvoid(
    reach::Vector{<:UnionIndex},
    avoid::Vector{<:UnionIndex},
    time_horizon,
)
    reach = CartesianIndex.(reach)
    avoid = CartesianIndex.(avoid)
    return ExactTimeReachAvoid(reach, avoid, time_horizon)
end

function checkproperty(prop::ExactTimeReachAvoid, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::ExactTimeReachAvoid, system)
    checkstatebounds(terminal_states(prop), system)
    checkdisjoint(reach(prop), avoid(prop))
end

function step_specification!(value_function, prop::ExactTimeReachAvoid)
    @inbounds value_function.current[avoid(prop)] .= 0.0
end

"""
    isfinitetime(prop::ExactTimeReachAvoid)

Return `true` for ExactTimeReachAvoid.
"""
isfinitetime(prop::ExactTimeReachAvoid) = true

"""
    time_horizon(prop::ExactTimeReachAvoid)

Return the time horizon of an exact time reach-avoid property.
"""
time_horizon(prop::ExactTimeReachAvoid) = prop.time_horizon

"""
    terminal_states(prop::ExactTimeReachAvoid)

Return the set of terminal states of an exact time reach-avoid property.
That is, the union of the reach and avoid sets.
"""
terminal_states(prop::ExactTimeReachAvoid) = [prop.reach; prop.avoid]

"""
    reach(prop::ExactTimeReachAvoid)

Return the set of target states.
"""
reach(prop::ExactTimeReachAvoid) = prop.reach

"""
    avoid(prop::ExactTimeReachAvoid)

Return the set of states to avoid.
"""
avoid(prop::ExactTimeReachAvoid) = prop.avoid

function checkstatebounds(states, system::IntervalMarkovProcess)
    pns = product_num_states(system)
    for j in states
        j = Tuple(j)

        if length(j) != length(pns)
            throw(StateDimensionMismatch(j, length(pns)))
        end

        if any(j .< 1) || any(j .> pns)
            throw(InvalidStateError(j, pns))
        end
    end
end

checkstatebounds(states, system::ProductProcess) =
    checkstatebounds(states, automaton(system))

function checkstatebounds(states, system::DeterministicAutomaton)
    for state in states
        if state < 1 || state > num_states(system)
            throw(InvalidStateError(state, num_states(system)))
        end
    end
end

function checkdisjoint(reach, avoid)
    if !isdisjoint(reach, avoid)
        throw(DomainError((reach, avoid), "reach and avoid sets are not disjoint"))
    end
end

## Safety

"""
    AbstractSafety

Super type for all safety properties.
"""
abstract type AbstractSafety <: BasicProperty end

function initialize!(value_function, prop::AbstractSafety)
    @inbounds value_function.current[avoid(prop)] .= -1.0
end

function step_specification!(value_function, prop::AbstractSafety)
    @inbounds value_function.current[avoid(prop)] .= -1.0
end

function postprocess_value_function!(value_function, ::AbstractSafety)
    value_function.current .+= 1.0
end

"""
    FiniteTimeSafety{VT <: Vector{<:CartesianIndex}, T <: Integer}

Finite time safety specified by a set of avoid states and a time horizon. 
That is, denote a trace by ``s_1 s_2 s_3 \\cdots``, then if ``A`` is the set of avoid states and ``H`` is the time horizon,
the property is 
```math
    \\mathbb{P}(\\forall k = \\{0, \\ldots, H\\}, s_k \\notin A).
```
"""
struct FiniteTimeSafety{VT <: Vector{<:CartesianIndex}, T <: Integer} <: AbstractSafety
    avoid_states::VT
    time_horizon::T
end

function FiniteTimeSafety(avoid_states::Vector{<:UnionIndex}, time_horizon)
    avoid_states = CartesianIndex.(avoid_states)
    return FiniteTimeSafety(avoid_states, time_horizon)
end

function checkproperty(prop::FiniteTimeSafety, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::FiniteTimeSafety, system)
    checkstatebounds(terminal_states(prop), system)
end

"""
    isfinitetime(prop::FiniteTimeSafety)

Return `true` for FiniteTimeSafety.
"""
isfinitetime(prop::FiniteTimeSafety) = true

"""
    time_horizon(prop::FiniteTimeSafety)

Return the time horizon of a finite time safety property.
"""
time_horizon(prop::FiniteTimeSafety) = prop.time_horizon

"""
    terminal_states(spec::FiniteTimeSafety)

Return the set of terminal states of a finite time safety property.
"""
terminal_states(prop::FiniteTimeSafety) = prop.avoid_states

"""
    avoid(prop::FiniteTimeSafety)

Return the set of states with which to compute reachbility for a finite time reachability prop.
This is equivalent for [`terminal_states(prop::FiniteTimeSafety)`](@ref).
"""
avoid(prop::FiniteTimeSafety) = prop.avoid_states

"""
    InfiniteTimeSafety{R <: Real, VT <: Vector{<:CartesianIndex}} 
 
`InfiniteTimeSafety` is similar to [`FiniteTimeSafety`](@ref) except that the time horizon is infinite, i.e., ``H = \\infty``.
In practice it means, performing the value iteration until the value function has converged, defined by some threshold `convergence_eps`.
The convergence threshold is that the largest value of the most recent Bellman residual is less than `convergence_eps`.
"""
struct InfiniteTimeSafety{R <: Real, VT <: Vector{<:CartesianIndex}} <: AbstractSafety
    avoid_states::VT
    convergence_eps::R
end

function InfiniteTimeSafety(avoid_states::Vector{<:UnionIndex}, convergence_eps)
    avoid_states = CartesianIndex.(avoid_states)
    return InfiniteTimeSafety(avoid_states, convergence_eps)
end

function checkproperty(prop::InfiniteTimeSafety, system, strategy)
    checkconvergence(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::InfiniteTimeSafety, system)
    checkstatebounds(terminal_states(prop), system)
end

"""
    isfinitetime(prop::InfiniteTimeSafety)

Return `false` for InfiniteTimeSafety.
"""
isfinitetime(prop::InfiniteTimeSafety) = false

"""
    convergence_eps(prop::InfiniteTimeSafety)

Return the convergence threshold of an infinite time safety property.
"""
convergence_eps(prop::InfiniteTimeSafety) = prop.convergence_eps

"""
    terminal_states(prop::InfiniteTimeSafety)

Return the set of terminal states of an infinite time safety property.
"""
terminal_states(prop::InfiniteTimeSafety) = prop.avoid_states

"""
    avoid(prop::InfiniteTimeSafety)

Return the set of states with which to compute safety for a infinite time safety property.
This is equivalent for [`terminal_states(prop::InfiniteTimeSafety)`](@ref).
"""
avoid(prop::InfiniteTimeSafety) = prop.avoid_states

## Reward

"""
    AbstractReward{R <: Real}

Super type for all reward properties.
"""
abstract type AbstractReward{R <: Real} <: BasicProperty end

function initialize!(value_function, prop::AbstractReward)
    value_function.current .= reward(prop)
end

function step_specification!(value_function, prop::AbstractReward)
    rmul!(value_function.current, discount(prop))
    value_function.current .+= reward(prop)
end
postprocess_value_function!(value_function, ::AbstractReward) = value_function

function checkreward(prop::AbstractReward, system)
    checkdevice(reward(prop), system)

    pns = product_num_states(system)
    if size(reward(prop)) != pns
        throw(
            DimensionMismatch(
                "the reward array must have the same dimensions $(size(reward(prop))) as the number of states along each axis $pns",
            ),
        )
    end

    if discount(prop) <= 0
        throw(DomainError(discount(prop), "the discount factor must be greater than 0"))
    end
end

"""
    FiniteTimeReward{R <: Real, AR <: AbstractArray{R}, T <: Integer}

`FiniteTimeReward` is a property of rewards ``R : S \\to \\mathbb{R}`` assigned to each state at each iteration
and a discount factor ``\\gamma``. The time horizon ``H`` is finite, so the discount factor is optional and 
the optimal policy will be time-varying. Given a strategy ``\\pi : S \\to A``, the property is
```math
    V(s_0) = \\mathbb{E}\\left[\\sum_{k=0}^{H} \\gamma^k R(s_k) \\mid s_0, \\pi\\right].
```
"""
struct FiniteTimeReward{R <: Real, AR <: AbstractArray{R}, T <: Integer} <:
       AbstractReward{R}
    reward::AR
    discount::R
    time_horizon::T
end

function checkproperty(prop::FiniteTimeReward, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::FiniteTimeReward, system)
    checkreward(prop, system)
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
    InfiniteTimeReward{R <: Real, AR <: AbstractArray{R}}

`InfiniteTimeReward` is a property of rewards assigned to each state at each iteration
and a discount factor for guaranteed convergence. The time horizon is infinite, i.e. ``H = \\infty``, so the optimal
policy will be stationary.
"""
struct InfiniteTimeReward{R <: Real, AR <: AbstractArray{R}} <: AbstractReward{R}
    reward::AR
    discount::R
    convergence_eps::R
end

function checkproperty(prop::InfiniteTimeReward, system, strategy)
    checkconvergence(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::InfiniteTimeReward, system)
    checkreward(prop, system)
    checkdiscountupperbound(prop)
end

function checkdiscountupperbound(prop::InfiniteTimeReward)
    if discount(prop) >= 1
        throw(
            DomainError(
                discount(prop),
                "the discount factor must be less than 1 for infinite horizon discounted rewards",
            ),
        )
    end
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

## Hitting time
"""
    AbstractHittingTime

Super type for all HittingTime properties.
"""
abstract type AbstractHittingTime <: BasicProperty end

postprocess_value_function!(value_function, ::AbstractHittingTime) = value_function

"""
    ExpectedExitTime{R <: Real, VT <: Vector{<:CartesianIndex}}

`ExpectedExitTime` is a property of hitting time with respect to an unsafe set. An equivalent
characterization is that of the expected number of steps in the safe set until reaching the unsafe set.
The time horizon is infinite, i.e., ``H = \\infty``, thus the package performs value iteration until the value function
has converged. The convergence threshold is that the largest value of the most recent Bellman residual is less than `convergence_eps`.
As this is an infinite horizon property, the resulting optimal policy will be stationary.
In formal language, given a strategy ``\\pi : S \\to A`` and an unsafe set ``O``, the property is defined as
```math
    V(s_0) = \\mathbb{E}\\left[\\lvert \\omega_{0:k-1} \\rvert \\mid s_0, \\pi, \\omega_{0:k-1} \\notin O, \\omega_k \\in O \\right]
```
where ``\\omega = s_0 s_1 \\ldots s_k`` is the trajectory of the system, ``\\omega_{0:k-1} = s_0 s_1 \\ldots s_{k-1}`` denotes the subtrajectory
excluding the final state, and ``\\omega_k = s_k``.
"""
struct ExpectedExitTime{R <: Real, VT <: Vector{<:CartesianIndex}} <: AbstractHittingTime
    avoid_states::VT
    convergence_eps::R
end

function ExpectedExitTime(avoid_states::Vector{<:UnionIndex}, convergence_eps)
    avoid_states = CartesianIndex.(avoid_states)
    return ExpectedExitTime(avoid_states, convergence_eps)
end

function checkproperty(prop::ExpectedExitTime, system, strategy)
    checkconvergence(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::ExpectedExitTime, system)
    checkstatebounds(avoid(prop), system)
end

function initialize!(value_function, prop::ExpectedExitTime)
    value_function.current .= 1.0
    value_function.current[avoid(prop)] .= 0.0
end

function step_specification!(value_function, prop::ExpectedExitTime)
    value_function.current .+= 1.0
    value_function.current[avoid(prop)] .= 0.0
end

"""
    isfinitetime(prop::ExpectedExitTime)

Return `true` for ExpectedExitTime.
"""
isfinitetime(prop::ExpectedExitTime) = false

"""
    terminal_states(prop::ExpectedExitTime)

Return the set of terminal states of an expected hitting time property.
"""
terminal_states(prop::ExpectedExitTime) = prop.avoid_states

"""
    avoid(prop::ExpectedExitTime)

Return the set of unsafe states that we compute the expected hitting time with respect to.
This is equivalent for [`terminal_states(prop::ExpectedExitTime)`](@ref).
"""
avoid(prop::ExpectedExitTime) = prop.avoid_states

"""
    convergence_eps(prop::ExpectedExitTime)

Return the convergence threshold of an expected exit time.
"""
convergence_eps(prop::ExpectedExitTime) = prop.convergence_eps

## Problem

"""
    SatisfactionMode

When computing the satisfaction probability of a property over an interval Markov process,
be it IMC or IMDP, the desired satisfaction probability to verify can either be `Optimistic` or
`Pessimistic`. That is, upper and lower bounds on the satisfaction probability within
the probability uncertainty.
"""
@enum SatisfactionMode Pessimistic Optimistic
ispessimistic(mode::SatisfactionMode) = mode == Pessimistic
isoptimistic(mode::SatisfactionMode) = mode == Optimistic

Base.:!(mode::SatisfactionMode) = ispessimistic(mode) ? Optimistic : Pessimistic

"""
    StrategyMode

When computing the satisfaction probability of a property over an IMDP, the strategy
can either maximize or minimize the satisfaction probability (wrt. the satisfaction mode).
"""
@enum StrategyMode Maximize Minimize
ismaximize(mode::StrategyMode) = mode == Maximize
isminimize(mode::StrategyMode) = mode == Minimize

Base.:!(mode::StrategyMode) = ismaximize(mode) ? Minimize : Maximize

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
step_specification!(value_function, spec::Specification) =
    step_specification!(value_function, system_property(spec))
postprocess_value_function!(value_function, spec::Specification) =
    postprocess_value_function!(value_function, system_property(spec))

function checkspecification(spec::Specification, system, strategy)
    checkmodelpropertycompatibility(system_property(spec), system)
    checkproperty(system_property(spec), system, strategy)
end

function checkspecification(spec::Specification, system)
    checkmodelpropertycompatibility(system_property(spec), system)
    checkproperty(system_property(spec), system)
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
ispessimistic(spec::Specification) = ispessimistic(satisfaction_mode(spec))
isoptimistic(spec::Specification) = isoptimistic(satisfaction_mode(spec))

"""
    strategy_mode(spec::Specification)

Return the strategy mode of a specification.
"""
strategy_mode(spec::Specification) = spec.strategy
ismaximize(spec::Specification) = ismaximize(strategy_mode(spec))
isminimize(spec::Specification) = isminimize(strategy_mode(spec))
