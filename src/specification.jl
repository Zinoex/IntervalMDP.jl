### Property types

abstract type Property end

function checkmodelpropertycompatibility(prop, system)
    throw(
        ArgumentError(
            "the property $(typeof(prop)) is not compatible with the system $(typeof(system))",
        ),
    )
end

Base.show(io::IO, mime::MIME"text/plain", prop::Property) = showproperty(io, "", "", prop)

abstract type BasicProperty <: Property end
function checkmodelpropertycompatibility(::BasicProperty, ::IntervalMarkovProcess)
    return nothing
end

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

Super type for all DFA reachability-like properties.
"""
abstract type AbstractDFAReachability <: ProductProperty end

function initialize!(value_function, prop::AbstractDFAReachability)
    @inbounds selectdim(
        value_function.current,
        ndims(value_function.current),
        reach(prop),
    ) .= 1.0
end

function step_postprocess_value_function!(value_function, prop::AbstractDFAReachability)
    @inbounds selectdim(
        value_function.current,
        ndims(value_function.current),
        reach(prop),
    ) .= 1.0
end

postprocess_value_function!(value_function, ::AbstractDFAReachability) = nothing

"""
    FiniteTimeDFAReachability{VT <: Vector{<:Integer}, T <: Integer}

Finite time reachability specified by a set of target/terminal states and a time horizon. 
That is, denote a trace by ``z_1 z_2 z_3 \\cdots`` with ``z_k = (s_k, q_k)`` then if ``T`` is the set of target states and ``H`` is the time horizon,
the property is 
```math
    \\mathbb{P}(\\exists k = \\{0, \\ldots, H\\}, q_k \\in T).
```
"""
struct FiniteTimeDFAReachability{VT <: Vector{<:Int32}, T <: Integer} <:
       AbstractDFAReachability
    reach::VT
    time_horizon::T
end

function FiniteTimeDFAReachability(reach::Vector{<:Integer}, time_horizon)
    reach = Int32.(reach)
    return FiniteTimeDFAReachability(reach, time_horizon)
end

function checkproperty(prop::FiniteTimeDFAReachability, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::FiniteTimeDFAReachability, system)
    checkstatebounds(reach(prop), system)
end

isfinitetime(prop::FiniteTimeDFAReachability) = true

"""
    time_horizon(prop::FiniteTimeDFAReachability)

Return the time horizon of a finite time DFA reachability property.
"""
time_horizon(prop::FiniteTimeDFAReachability) = prop.time_horizon

"""
    reach(prop::FiniteTimeDFAReachability)

Return the set of DFA states with respect to which to compute reachbility for a finite time DFA reachability property.
"""
reach(prop::FiniteTimeDFAReachability) = prop.reach
terminal(prop::FiniteTimeDFAReachability) = reach(prop)

function showproperty(io::IO, first_prefix, prefix, prop::FiniteTimeDFAReachability)
    println(io, first_prefix, styled"{code:FiniteTimeDFAReachability}")
    println(io, prefix, styled"├─ Time horizon: {magenta:$(time_horizon(prop))}")
    println(io, prefix, styled"└─ Reach states: {magenta:$(reach(prop))}")
end

"""
    InfiniteTimeDFAReachability{VT <: Vector{<:Integer}, R <: Real} 
 
`InfiniteTimeDFAReachability` is similar to [`FiniteTimeDFAReachability`](@ref) except that the time horizon is infinite, i.e., ``H = \\infty``.
In practice it means, performing the value iteration until the value function has converged, defined by some threshold `convergence_eps`.
The convergence threshold is that the largest value of the most recent Bellman residual is less than `convergence_eps`.
"""
struct InfiniteTimeDFAReachability{VT <: Vector{<:Int32}, R <: Real} <:
       AbstractDFAReachability
    reach::VT
    convergence_eps::R
end

function InfiniteTimeDFAReachability(reach::Vector{<:Integer}, convergence_eps)
    reach = Int32.(reach)
    return InfiniteTimeDFAReachability(reach, convergence_eps)
end

function checkproperty(prop::InfiniteTimeDFAReachability, system, strategy)
    checkconvergence(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::InfiniteTimeDFAReachability, system)
    checkstatebounds(reach(prop), system)
end

isfinitetime(prop::InfiniteTimeDFAReachability) = false

"""
    convergence_eps(prop::InfiniteTimeDFAReachability)

Return the convergence threshold of an infinite time DFA reachability property.
"""
convergence_eps(prop::InfiniteTimeDFAReachability) = prop.convergence_eps

"""
    reach(prop::InfiniteTimeDFAReachability)

Return the set of DFA states with respect to which to compute reachbility for a infinite time DFA reachability property.
"""
reach(prop::InfiniteTimeDFAReachability) = prop.reach
terminal(prop::InfiniteTimeDFAReachability) = reach(prop)

function showproperty(io::IO, first_prefix, prefix, prop::InfiniteTimeDFAReachability)
    println(io, first_prefix, styled"{code:InfiniteTimeDFAReachability}")
    println(
        io,
        prefix,
        styled"├─ Convergence threshold: {magenta:$(convergence_eps(prop))}",
    )
    println(io, prefix, styled"└─ Reach states: {magenta:$(reach(prop))}")
end

## DFA Safety

"""
    AbstractDFASafety

Super type for all DFA safety-like properties.
"""
abstract type AbstractDFASafety <: ProductProperty end

function initialize!(value_function, prop::AbstractDFASafety)
    @inbounds selectdim(
        value_function.current,
        ndims(value_function.current),
        avoid(prop),
    ) .= -1.0
end

function step_postprocess_value_function!(value_function, prop::AbstractDFASafety)
    @inbounds selectdim(
        value_function.current,
        ndims(value_function.current),
        avoid(prop),
    ) .= -1.0
end

function postprocess_value_function!(value_function, ::AbstractDFASafety)
    value_function.current .+= 1.0
end

"""
    FiniteTimeDFASafety{VT <: Vector{<:Integer}, T <: Integer}

Finite time Safety specified by a set of target/terminal states and a time horizon. 
That is, denote a trace by ``z_1 z_2 z_3 \\cdots`` with ``z_k = (s_k, q_k)`` then if ``T`` is the set of target states and ``H`` is the time horizon,
the property is 
```math
    \\mathbb{P}(\\exists k = \\{0, \\ldots, H\\}, q_k \\in T).
```
"""
struct FiniteTimeDFASafety{VT <: Vector{<:Int32}, T <: Integer} <: AbstractDFASafety
    avoid::VT
    time_horizon::T
end

function FiniteTimeDFASafety(avoid::Vector{<:Integer}, time_horizon)
    avoid = Int32.(avoid)
    return FiniteTimeDFASafety(avoid, time_horizon)
end

function checkproperty(prop::FiniteTimeDFASafety, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::FiniteTimeDFASafety, system)
    checkstatebounds(avoid(prop), system)
end

isfinitetime(prop::FiniteTimeDFASafety) = true

"""
    time_horizon(prop::FiniteTimeDFASafety)

Return the time horizon of a finite time DFA safety property.
"""
time_horizon(prop::FiniteTimeDFASafety) = prop.time_horizon

"""
    avoid(prop::FiniteTimeDFASafety)

Return the set of DFA states with respect to which to compute safety for a finite time DFA safety property.
"""
avoid(prop::FiniteTimeDFASafety) = prop.avoid
terminal(prop::FiniteTimeDFASafety) = avoid(prop)

function showproperty(io::IO, first_prefix, prefix, prop::FiniteTimeDFASafety)
    println(io, first_prefix, styled"{code:FiniteTimeDFASafety}")
    println(io, prefix, styled"├─ Time horizon: {magenta:$(time_horizon(prop))}")
    println(io, prefix, styled"└─ Avoid states: {magenta:$(avoid(prop))}")
end

"""
    InfiniteTimeDFASafety{VT <: Vector{<:Integer}, R <: Real} 
 
`InfiniteTimeDFASafety` is similar to [`FiniteTimeDFASafety`](@ref) except that the time horizon is infinite, i.e., ``H = \\infty``.
In practice it means, performing the value iteration until the value function has converged, defined by some threshold `convergence_eps`.
The convergence threshold is that the largest value of the most recent Bellman residual is less than `convergence_eps`.
"""
struct InfiniteTimeDFASafety{VT <: Vector{<:Int32}, R <: Real} <: AbstractDFASafety
    avoid::VT
    convergence_eps::R
end

function InfiniteTimeDFASafety(avoid::Vector{<:Integer}, convergence_eps)
    avoid = Int32.(avoid)
    return InfiniteTimeDFASafety(avoid, convergence_eps)
end

function checkproperty(prop::InfiniteTimeDFASafety, system, strategy)
    checkconvergence(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::InfiniteTimeDFASafety, system)
    checkstatebounds(avoid(prop), system)
end

isfinitetime(prop::InfiniteTimeDFASafety) = false

"""
    convergence_eps(prop::InfiniteTimeDFASafety)

Return the convergence threshold of an infinite time DFA safety property.
"""
convergence_eps(prop::InfiniteTimeDFASafety) = prop.convergence_eps

"""
    avoid(prop::InfiniteTimeDFASafety)

Return the set of DFA states with respect to which to compute safety for a infinite time DFA safety property.
"""
avoid(prop::InfiniteTimeDFASafety) = prop.avoid
terminal(prop::InfiniteTimeDFASafety) = avoid(prop)

function showproperty(io::IO, first_prefix, prefix, prop::InfiniteTimeDFASafety)
    println(io, first_prefix, styled"{code:InfiniteTimeDFASafety}")
    println(
        io,
        prefix,
        styled"├─ Convergence threshold: {magenta:$(convergence_eps(prop))}",
    )
    println(io, prefix, styled"└─ Avoid states: {magenta:$(avoid(prop))}")
end

## Reachability

"""
    AbstractReachability

Super type for all reachability-like properties.
"""
abstract type AbstractReachability <: BasicProperty end

function initialize!(value_function, prop::AbstractReachability)
    @inbounds value_function.current[reach(prop)] .= 1.0
end

function step_postprocess_value_function!(value_function, prop::AbstractReachability)
    @inbounds value_function.current[reach(prop)] .= 1.0
end

postprocess_value_function!(value_function, ::AbstractReachability) = nothing

"""
    FiniteTimeReachability{VT <: Vector{Union{<:Integer, <:Tuple, <:CartesianIndex}}, T <: Integer}

Finite time reachability specified by a set of target/terminal states and a time horizon. 
That is, denote a trace by ``\\omega = s_1 s_2 s_3 \\cdots``, then if ``G`` is the set of target states and ``K`` is the time horizon,
the property is 
```math
    \\mathbb{P}^{\\pi, \\eta}_{\\mathrm{reach}}(G, K) = \\mathbb{P}^{\\pi, \\eta} \\left[\\omega \\in \\Omega : \\exists k \\in \\{0, \\ldots, K\\}, \\, \\omega[k] \\in G \\right].
```
"""
struct FiniteTimeReachability{VT <: Vector{<:CartesianIndex}, T <: Integer} <:
       AbstractReachability
    reach::VT
    time_horizon::T
end

function FiniteTimeReachability(reach::Vector{<:UnionIndex}, time_horizon)
    reach = CartesianIndex.(reach)
    return FiniteTimeReachability(reach, time_horizon)
end

function checkproperty(prop::FiniteTimeReachability, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::FiniteTimeReachability, system)
    checkstatebounds(reach(prop), system)
end

isfinitetime(prop::FiniteTimeReachability) = true

"""
    time_horizon(prop::FiniteTimeReachability)

Return the time horizon of a finite time reachability property.
"""
time_horizon(prop::FiniteTimeReachability) = prop.time_horizon

"""
    reach(prop::FiniteTimeReachability)

Return the set of states with respect to which to compute reachbility for a finite time reachability property.
"""
reach(prop::FiniteTimeReachability) = prop.reach

function showproperty(io::IO, first_prefix, prefix, prop::FiniteTimeReachability)
    println(io, first_prefix, styled"{code:FiniteTimeReachability}")
    println(io, prefix, styled"├─ Time horizon: {magenta:$(time_horizon(prop))}")
    println(io, prefix, styled"└─ Reach states: {magenta:$(reach(prop))}")
end

"""
    InfiniteTimeReachability{VT <: Vector{Union{<:Integer, <:Tuple, <:CartesianIndex}}, R <: Real}

`InfiniteTimeReachability` is similar to [`FiniteTimeReachability`](@ref) except that the time horizon is infinite, i.e., ``K = \\infty``.
In practice it means, performing the value iteration until the value function has converged, defined by some threshold `convergence_eps`.
The convergence threshold is that the largest value of the most recent Bellman residual is less than `convergence_eps`.
"""
struct InfiniteTimeReachability{VT <: Vector{<:CartesianIndex}, R <: Real} <:
       AbstractReachability
    reach::VT
    convergence_eps::R
end

function InfiniteTimeReachability(reach::Vector{<:UnionIndex}, convergence_eps)
    reach = CartesianIndex.(reach)
    return InfiniteTimeReachability(reach, convergence_eps)
end

function checkproperty(prop::InfiniteTimeReachability, system, strategy)
    checkconvergence(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::InfiniteTimeReachability, system)
    checkstatebounds(reach(prop), system)
end

isfinitetime(prop::InfiniteTimeReachability) = false

"""
    convergence_eps(prop::InfiniteTimeReachability)

Return the convergence threshold of an infinite time reachability property.
"""
convergence_eps(prop::InfiniteTimeReachability) = prop.convergence_eps

"""
    reach(prop::InfiniteTimeReachability)

Return the set of states with which to compute reachbility for a infinite time reachability property.
"""
reach(prop::InfiniteTimeReachability) = prop.reach

function showproperty(io::IO, first_prefix, prefix, prop::InfiniteTimeReachability)
    println(io, first_prefix, styled"{code:InfiniteTimeReachability}")
    println(
        io,
        prefix,
        styled"├─ Convergence threshold: {magenta:$(convergence_eps(prop))}",
    )
    println(io, prefix, styled"└─ Reach states: {magenta:$(reach(prop))}")
end

"""
    ExactTimeReachability{VT <: Vector{Union{<:Integer, <:Tuple, <:CartesianIndex}}, T <: Integer}

Exact time reachability specified by a set of target/terminal states and a time horizon. 
That is, denote a trace by ``\\omega = s_1 s_2 s_3 \\cdots``, then if ``G`` is the set of target states and ``K`` is the time horizon,
the property is 
```math
    \\mathbb{P}^{\\pi, \\eta}_{\\mathrm{exact-reach}}(G, K) = \\mathbb{P}^{\\pi, \\eta} \\left[\\omega \\in \\Omega : \\omega[K] \\in G \\right].
```
"""
struct ExactTimeReachability{VT <: Vector{<:CartesianIndex}, T <: Integer} <:
       AbstractReachability
    reach::VT
    time_horizon::T
end

function ExactTimeReachability(reach::Vector{<:UnionIndex}, time_horizon)
    reach = CartesianIndex.(reach)
    return ExactTimeReachability(reach, time_horizon)
end

function checkproperty(prop::ExactTimeReachability, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::ExactTimeReachability, system)
    checkstatebounds(reach(prop), system)
end

function step_postprocess_value_function!(_, ::ExactTimeReachability)
    return nothing
end

isfinitetime(prop::ExactTimeReachability) = true

"""
    time_horizon(prop::ExactTimeReachability)

Return the time horizon of an exact time reachability property.
"""
time_horizon(prop::ExactTimeReachability) = prop.time_horizon

"""
    reach(prop::ExactTimeReachability)

Return the set of states with which to compute reachbility for an exact time reachability prop.
"""
reach(prop::ExactTimeReachability) = prop.reach

function showproperty(io::IO, first_prefix, prefix, prop::ExactTimeReachability)
    println(io, first_prefix, styled"{code:ExactTimeReachability}")
    println(io, prefix, styled"├─ Time horizon: {magenta:$(time_horizon(prop))}")
    println(io, prefix, styled"└─ Reach states: {magenta:$(reach(prop))}")
end

## Reach-avoid

"""
    AbstractReachAvoid

A property of reachability that includes a set of states to avoid.
"""
abstract type AbstractReachAvoid <: AbstractReachability end

function step_postprocess_value_function!(value_function, prop::AbstractReachAvoid)
    @inbounds value_function.current[reach(prop)] .= 1.0
    @inbounds value_function.current[avoid(prop)] .= 0.0
end

function checkstatebounds(states, system::IntervalMarkovProcess)
    pns = state_values(system)
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

"""
    FiniteTimeReachAvoid{VT <: Vector{Union{<:Integer, <:Tuple, <:CartesianIndex}}}, T <: Integer}

Finite time reach-avoid specified by a set of target/terminal states, a set of avoid states, and a time horizon.
That is, denote a trace by ``\\omega = s_1 s_2 s_3 \\cdots``, then if ``G`` is the set of target states, ``O`` is the set of states to avoid,
and ``K`` is the time horizon, the property is 
```math
    \\mathbb{P}^{\\pi, \\eta}_{\\mathrm{reach-avoid}}(G, O, K) = \\mathbb{P}^{\\pi, \\eta} \\left[\\omega \\in \\Omega : \\exists k \\in \\{0, \\ldots, K\\}, \\, \\omega[k] \\in G, \\; \\forall k' \\in \\{0, \\ldots, k' \\}, \\, \\omega[k] \\notin O \\right].
```
"""
struct FiniteTimeReachAvoid{VT <: Vector{<:CartesianIndex}, T <: Integer} <:
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
    checkstatebounds(reach(prop), system)
    checkstatebounds(avoid(prop), system)
    checkdisjoint(reach(prop), avoid(prop))
end

isfinitetime(prop::FiniteTimeReachAvoid) = true

"""
    time_horizon(prop::FiniteTimeReachAvoid)

Return the time horizon of a finite time reach-avoid property.
"""
time_horizon(prop::FiniteTimeReachAvoid) = prop.time_horizon

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

function showproperty(io::IO, first_prefix, prefix, prop::FiniteTimeReachAvoid)
    println(io, first_prefix, styled"{code:FiniteTimeReachAvoid}")
    println(io, prefix, styled"├─ Time horizon: {magenta:$(time_horizon(prop))}")
    println(io, prefix, styled"├─ Reach states: {magenta:$(reach(prop))}")
    println(io, prefix, styled"└─ Avoid states: {magenta:$(avoid(prop))}")
end

"""
    InfiniteTimeReachAvoid{VT <: Vector{Union{<:Integer, <:Tuple, <:CartesianIndex}}, R <: Real}

`InfiniteTimeReachAvoid` is similar to [`FiniteTimeReachAvoid`](@ref) except that the time horizon is infinite, i.e., ``K = \\infty``.
"""
struct InfiniteTimeReachAvoid{VT <: Vector{<:CartesianIndex}, R <: Real} <:
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
    checkstatebounds(reach(prop), system)
    checkstatebounds(avoid(prop), system)
    checkdisjoint(reach(prop), avoid(prop))
end

isfinitetime(prop::InfiniteTimeReachAvoid) = false

"""
    convergence_eps(prop::InfiniteTimeReachAvoid)

Return the convergence threshold of an infinite time reach-avoid property.
"""
convergence_eps(prop::InfiniteTimeReachAvoid) = prop.convergence_eps

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

function showproperty(io::IO, first_prefix, prefix, prop::InfiniteTimeReachAvoid)
    println(io, first_prefix, styled"{code:InfiniteTimeReachAvoid}")
    println(
        io,
        prefix,
        styled"├─ Convergence threshold: {magenta:$(convergence_eps(prop))}",
    )
    println(io, prefix, styled"├─ Reach states: {magenta:$(reach(prop))}")
    println(io, prefix, styled"└─ Avoid states: {magenta:$(avoid(prop))}")
end

"""
    ExactTimeReachAvoid{VT <: Vector{Union{<:Integer, <:Tuple, <:CartesianIndex}}}, T <: Integer}

Exact time reach-avoid specified by a set of target/terminal states, a set of avoid states, and a time horizon.
That is, denote a trace by ``\\omega = s_1 s_2 s_3 \\cdots``, then if ``G`` is the set of target states, ``O`` is the set of states to avoid,
and ``K`` is the time horizon, the property is 
```math
    \\mathbb{P}^{\\pi, \\eta}_{\\mathrm{exact-reach-avoid}}(G, O, K) = \\mathbb{P}^{\\pi, \\eta} \\left[\\omega \\in \\Omega : \\omega[K] \\in G, \\; \\forall k \\in \\{0, \\ldots, K\\}, \\, \\omega[k] \\notin O \\right].
```
"""
struct ExactTimeReachAvoid{VT <: Vector{<:CartesianIndex}, T <: Integer} <:
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
    checkstatebounds(reach(prop), system)
    checkstatebounds(avoid(prop), system)
    checkdisjoint(reach(prop), avoid(prop))
end

function step_postprocess_value_function!(value_function, prop::ExactTimeReachAvoid)
    @inbounds value_function.current[avoid(prop)] .= 0.0
end

isfinitetime(prop::ExactTimeReachAvoid) = true

"""
    time_horizon(prop::ExactTimeReachAvoid)

Return the time horizon of an exact time reach-avoid property.
"""
time_horizon(prop::ExactTimeReachAvoid) = prop.time_horizon

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

function showproperty(io::IO, first_prefix, prefix, prop::ExactTimeReachAvoid)
    println(io, first_prefix, styled"{code:ExactTimeReachAvoid}")
    println(io, prefix, styled"├─ Time horizon: {magenta:$(time_horizon(prop))}")
    println(io, prefix, styled"├─ Reach states: {magenta:$(reach(prop))}")
    println(io, prefix, styled"└─ Avoid states: {magenta:$(avoid(prop))}")
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

function step_postprocess_value_function!(value_function, prop::AbstractSafety)
    @inbounds value_function.current[avoid(prop)] .= -1.0
end

function postprocess_value_function!(value_function, ::AbstractSafety)
    value_function.current .+= 1.0
end

"""
    FiniteTimeSafety{VT <: Vector{Union{<:Integer, <:Tuple, <:CartesianIndex}}, T <: Integer}

Finite time safety specified by a set of avoid states and a time horizon. 
That is, denote a trace by ``\\omega = s_1 s_2 s_3 \\cdots``, then if ``O`` is the set of avoid states and ``K`` is the time horizon,
the property is 
```math
    \\mathbb{P}^{\\pi, \\eta}_{\\mathrm{safe}}(O, K) = \\mathbb{P}^{\\pi, \\eta} \\left[\\omega \\in \\Omega : \\forall k \\in \\{0, \\ldots, K\\}, \\, \\omega[k] \\notin O \\right].
```
"""
struct FiniteTimeSafety{VT <: Vector{<:CartesianIndex}, T <: Integer} <: AbstractSafety
    avoid::VT
    time_horizon::T
end

function FiniteTimeSafety(avoid::Vector{<:UnionIndex}, time_horizon)
    avoid = CartesianIndex.(avoid)
    return FiniteTimeSafety(avoid, time_horizon)
end

function checkproperty(prop::FiniteTimeSafety, system, strategy)
    checktimehorizon(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::FiniteTimeSafety, system)
    checkstatebounds(avoid(prop), system)
end

isfinitetime(prop::FiniteTimeSafety) = true

"""
    time_horizon(prop::FiniteTimeSafety)

Return the time horizon of a finite time safety property.
"""
time_horizon(prop::FiniteTimeSafety) = prop.time_horizon

"""
    avoid(prop::FiniteTimeSafety)

Return the set of states with which to compute reachbility for a finite time reachability prop.
"""
avoid(prop::FiniteTimeSafety) = prop.avoid

function showproperty(io::IO, first_prefix, prefix, prop::FiniteTimeSafety)
    println(io, first_prefix, styled"{code:FiniteTimeSafety}")
    println(io, prefix, styled"├─ Time horizon: {magenta:$(time_horizon(prop))}")
    println(io, prefix, styled"└─ Avoid states: {magenta:$(avoid(prop))}")
end

"""
    InfiniteTimeSafety{VT <: Vector{Union{<:Integer, <:Tuple, <:CartesianIndex}}, R <: Real} 
 
`InfiniteTimeSafety` is similar to [`FiniteTimeSafety`](@ref) except that the time horizon is infinite, i.e., ``K = \\infty``.
In practice it means, performing the value iteration until the value function has converged, defined by some threshold `convergence_eps`.
The convergence threshold is that the largest value of the most recent Bellman residual is less than `convergence_eps`.
"""
struct InfiniteTimeSafety{VT <: Vector{<:CartesianIndex}, R <: Real} <: AbstractSafety
    avoid::VT
    convergence_eps::R
end

function InfiniteTimeSafety(avoid::Vector{<:UnionIndex}, convergence_eps)
    avoid = CartesianIndex.(avoid)
    return InfiniteTimeSafety(avoid, convergence_eps)
end

function checkproperty(prop::InfiniteTimeSafety, system, strategy)
    checkconvergence(prop, strategy)
    checkproperty(prop, system)
end

function checkproperty(prop::InfiniteTimeSafety, system)
    checkstatebounds(avoid(prop), system)
end

isfinitetime(prop::InfiniteTimeSafety) = false

"""
    convergence_eps(prop::InfiniteTimeSafety)

Return the convergence threshold of an infinite time safety property.
"""
convergence_eps(prop::InfiniteTimeSafety) = prop.convergence_eps

"""
    avoid(prop::InfiniteTimeSafety)

Return the set of states with which to compute safety for a infinite time safety property.
"""
avoid(prop::InfiniteTimeSafety) = prop.avoid

function showproperty(io::IO, first_prefix, prefix, prop::InfiniteTimeSafety)
    println(io, first_prefix, styled"{code:InfiniteTimeSafety}")
    println(
        io,
        prefix,
        styled"├─ Convergence threshold: {magenta:$(convergence_eps(prop))}",
    )
    println(io, prefix, styled"└─ Avoid states: {magenta:$(avoid(prop))}")
end

## Reward

"""
    AbstractReward{R <: Real}

Super type for all reward properties.
"""
abstract type AbstractReward{R <: Real} <: BasicProperty end

function initialize!(value_function, prop::AbstractReward)
    value_function.current .= reward(prop)
end

function step_postprocess_value_function!(value_function, prop::AbstractReward)
    rmul!(value_function.current, discount(prop))
    value_function.current .+= reward(prop)
end
postprocess_value_function!(value_function, ::AbstractReward) = value_function

function checkreward(prop::AbstractReward, system)
    checkdevice(reward(prop), system)

    pns = state_values(system)
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

`FiniteTimeReward` is a property of rewards ``r : S \\to \\mathbb{R}`` assigned to each state at each iteration
and a discount factor ``\\nu``. The time horizon ``K`` is finite, so the discount factor can be greater than or equal to one. The property is
```math
    \\mathbb{E}^{\\pi,\\eta}_{\\mathrm{reward}}(r, \\nu, K) = \\mathbb{E}^{\\pi,\\eta}\\left[\\sum_{k=0}^{K} \\nu^k r(\\omega[k]) \\right].
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

function showproperty(io::IO, first_prefix, prefix, prop::FiniteTimeReward)
    println(io, first_prefix, styled"{code:FiniteTimeReward}")
    println(io, prefix, styled"├─ Time horizon: {magenta:$(time_horizon(prop))}")
    println(io, prefix, styled"├─ Discount factor: {magenta:$(discount(prop))}")
    println(
        io,
        prefix,
        styled"└─ Reward storage: {magenta:$(eltype(reward(prop))), $(size(reward(prop)))}",
    )
end

"""
    InfiniteTimeReward{R <: Real, AR <: AbstractArray{R}}

`InfiniteTimeReward` is a property of rewards assigned to each state at each iteration
and a discount factor for guaranteed convergence. The time horizon is infinite, i.e. ``K = \\infty``.
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

function showproperty(io::IO, first_prefix, prefix, prop::InfiniteTimeReward)
    println(io, first_prefix, styled"{code:InfiniteTimeReward}")
    println(
        io,
        prefix,
        styled"├─ Convergence threshold: {magenta:$(convergence_eps(prop))}",
    )
    println(io, prefix, styled"├─ Discount factor: {magenta:$(discount(prop))}")
    println(
        io,
        prefix,
        styled"└─ Reward storage: {magenta:$(eltype(reward(prop))), $(size(reward(prop)))}",
    )
end

## Hitting time
"""
    AbstractHittingTime

Super type for all HittingTime properties.
"""
abstract type AbstractHittingTime <: BasicProperty end

postprocess_value_function!(value_function, ::AbstractHittingTime) = value_function

"""
    ExpectedExitTime{VT <: Vector{Union{<:Integer, <:Tuple, <:CartesianIndex}}, R <: Real}

`ExpectedExitTime` is a property of hitting time with respect to an unsafe set. An equivalent
characterization is that of the expected number of steps in the safe set until reaching the unsafe set.
The time horizon is infinite, i.e., ``K = \\infty``, thus the package performs value iteration until the value function
has converged. The convergence threshold is that the largest value of the most recent Bellman residual is less than `convergence_eps`.
Given an unsafe set ``O``, the property is defined as
```math
    \\mathbb{E}^{\\pi,\\eta}_{\\mathrm{exit}}(O) = \\mathbb{E}^{\\pi,\\eta}\\left[k : \\omega[k] \\in O, \\, \\forall k' \\in \\{0, \\ldots, k - 1\\}, \\, \\omega[k'] \\notin O \\right].
```
where ``\\omega = s_0 s_1 \\ldots s_k`` is a trace of the system.
"""
struct ExpectedExitTime{VT <: Vector{<:CartesianIndex}, R <: Real} <: AbstractHittingTime
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

function step_postprocess_value_function!(value_function, prop::ExpectedExitTime)
    value_function.current .+= 1.0
    value_function.current[avoid(prop)] .= 0.0
end

isfinitetime(prop::ExpectedExitTime) = false

"""
    avoid(prop::ExpectedExitTime)

Return the set of unsafe states that we compute the expected hitting time with respect to.
"""
avoid(prop::ExpectedExitTime) = prop.avoid_states

"""
    convergence_eps(prop::ExpectedExitTime)

Return the convergence threshold of an expected exit time.
"""
convergence_eps(prop::ExpectedExitTime) = prop.convergence_eps

function showproperty(io::IO, first_prefix, prefix, prop::ExpectedExitTime)
    println(io, first_prefix, styled"{code:ExpectedExitTime}")
    println(
        io,
        prefix,
        styled"├─ Convergence threshold: {magenta:$(convergence_eps(prop))}",
    )
    println(io, prefix, styled"└─ Avoid states: {magenta:$(avoid(prop))}")
end

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
step_postprocess_value_function!(value_function, spec::Specification) =
    step_postprocess_value_function!(value_function, system_property(spec))
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

Base.show(io::IO, spec::Specification) = showspecification(io, "", "", spec)

function showspecification(io::IO, first_prefix, prefix, spec::Specification)
    println(io, first_prefix, styled"{code:Specification}")
    println(io, prefix, styled"├─ Satisfaction mode: {magenta:$(satisfaction_mode(spec))}")
    println(io, prefix, styled"├─ Strategy mode: {magenta:$(strategy_mode(spec))}")
    showproperty(io, prefix * "└─ Property: ", prefix * "   ", system_property(spec))
end
