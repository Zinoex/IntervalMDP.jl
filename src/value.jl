@enum IntervalMode Upper Lower
isupper(mode::IntervalMode) = mode == Upper
islower(mode::IntervalMode) = mode == Lower

#######################################################################
# Semantic array wrappers
#
# Many primitives in this package take a result array whose intended
# meaning (a state-value V[s] or a state-action-value Q[a, s]) is not
# expressible in plain `Array{T,N}`. These wrapper types make the
# distinction explicit at the type level, so a primitive that must
# write Q cannot silently accept a V-shaped buffer (and vice versa).
#
# Both wrappers are thin: they hold a single `data::AbstractArray` and
# forward the AbstractArray interface. Users can construct them from
# raw arrays via the unary constructor, or let the public
# `expectation!` / `expectation` entry points do the lifting based on
# the buffer's shape.
#######################################################################

"""
    StateValueArray{T,N,A} <: AbstractArray{T,N}

Wraps an array whose axes index *states only* (V-shape). Has the same
shape and `ndims` as the system's state space. Used as the canonical
type for state-value functions in primitive APIs.
"""
struct StateValueArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
end
StateValueArray(a::AbstractArray) = StateValueArray{eltype(a), ndims(a), typeof(a)}(a)

"""
    StateActionValueArray{T,N,A} <: AbstractArray{T,N}

Wraps an array whose axes index *both actions and states* (Q-shape):
`(action_shape..., state_shape...)`. Has strictly more axes than the
matching `StateValueArray`. Used as the canonical type for
state-action-value functions in primitive APIs.
"""
struct StateActionValueArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
end
StateActionValueArray(a::AbstractArray) =
    StateActionValueArray{eltype(a), ndims(a), typeof(a)}(a)

# Forward AbstractArray interface to `.data` for both wrappers.
for W in (:StateValueArray, :StateActionValueArray)
    @eval begin
        Base.size(x::$W) = size(x.data)
        Base.IndexStyle(::Type{$W{T, N, A}}) where {T, N, A} = IndexStyle(A)
        Base.@propagate_inbounds Base.getindex(x::$W, I::Int...) = getindex(x.data, I...)
        Base.@propagate_inbounds Base.getindex(x::$W, I::CartesianIndex) =
            getindex(x.data, I)
        Base.@propagate_inbounds Base.getindex(x::$W, I::Vararg{Any, M}) where {M} =
            getindex(x.data, I...)
        Base.@propagate_inbounds Base.setindex!(x::$W, v, I::Int...) =
            (setindex!(x.data, v, I...); x)
        Base.@propagate_inbounds Base.setindex!(x::$W, v, I::CartesianIndex) =
            (setindex!(x.data, v, I); x)
        Base.@propagate_inbounds Base.setindex!(x::$W, v, I::Vararg{Any, M}) where {M} =
            (setindex!(x.data, v, I...); x)
        Base.similar(x::$W, ::Type{S}, dims::Dims) where {S} = $W(similar(x.data, S, dims))
    end
end

# `parent` lets primitives reach the underlying array for code paths
# that need to interact with raw-array APIs (e.g. `selectdim`).
Base.parent(x::StateValueArray) = x.data
Base.parent(x::StateActionValueArray) = x.data

abstract type ValueFunction end

# State-value (V) function: holds the per-state value array across iterations.
# Per project convention there is no per-iteration Q-array on this type — the
# `(action × state)` Q-values are transient, computed into per-thread
# `workspace.actions` scratch on demand by `expectation_v!`. See plan §1.
struct StateValueFunction{R, A1 <: AbstractArray{R}} <: ValueFunction
    previous::A1
    current::A1
    interval::IntervalMode
end

StateValueFunction(problem::AbstractIntervalMDPProblem) = StateValueFunction(problem, Lower)

function StateValueFunction(problem::AbstractIntervalMDPProblem, mode::IntervalMode)
    mp = system(problem)
    previous = arrayfactory(mp, valuetype(mp), state_values(mp))
    previous .= zero(valuetype(mp))
    current = copy(previous)

    return StateValueFunction(previous, current, mode)
end

function lastdiff!(V::StateValueFunction{R}) where {R}
    # Reuse prev to store the latest difference
    V.previous .-= V.current
    rmul!(V.previous, -one(R))

    return V.previous
end

function nextiteration!(V::StateValueFunction)
    copy!(V.previous, V.current)

    return V
end

islower(V::StateValueFunction) = islower(V.interval)
isupper(V::StateValueFunction) = isupper(V.interval)

struct StateActionValueFunction{R, A1 <: AbstractArray{R}, A2 <: AbstractArray{R}} <:
       ValueFunction
    previous::A1
    current::A1
    intermediate_state_value::A2
    interval::IntervalMode
end

function StateActionValueFunction(problem::AbstractIntervalMDPProblem)
    mp = system(problem)
    dim = (action_values(mp)..., state_values(mp)...)
    # TODO: works for IMDP, need to check for fIMDP
    previous = arrayfactory(mp, valuetype(mp), dim)
    previous .= zero(valuetype(mp))
    current = copy(previous)

    intermediate_state_value = arrayfactory(mp, valuetype(mp), state_values(mp))
    intermediate_state_value .= zero(valuetype(mp))

    return StateActionValueFunction(previous, current, intermediate_state_value, Lower)
end

function StateActionValueFunction(problem::AbstractIntervalMDPProblem, mode::IntervalMode)
    mp = system(problem)
    dim = (action_values(mp)..., state_values(mp)...)
    # TODO: works for IMDP, need to check for fIMDP
    previous = arrayfactory(mp, valuetype(mp), dim)
    previous .= zero(valuetype(mp))
    current = copy(previous)

    intermediate_state_value = arrayfactory(mp, valuetype(mp), state_values(mp))
    intermediate_state_value .= zero(valuetype(mp))

    return StateActionValueFunction(previous, current, intermediate_state_value, mode)
end

function lastdiff!(V::StateActionValueFunction{R}) where {R}
    # Reuse prev to store the latest difference
    V.previous .-= V.current
    rmul!(V.previous, -one(R))

    return V.previous
end

function nextiteration!(V::StateActionValueFunction)
    copy!(V.previous, V.current)

    return V
end

islower(V::StateActionValueFunction) = islower(V.interval)
isupper(V::StateActionValueFunction) = isupper(V.interval)

struct IntervalValueFunction{V <: ValueFunction} <: ValueFunction
    lower::V
    upper::V
end

lower(V::IntervalValueFunction) = V.lower
upper(V::IntervalValueFunction) = V.upper

function lastdiff!(V::IntervalValueFunction)
    return (lastdiff!(V.lower), lastdiff!(V.upper))
end

function nextiteration!(V::IntervalValueFunction)
    nextiteration!(V.lower)
    nextiteration!(V.upper)

    return V
end

function gap(V::IntervalValueFunction)
    return abs.(V.lower.current .- V.upper.current)
end

function initialize!(value_function::IntervalValueFunction, prop::AbstractReachability)
    initialize!(value_function.lower, prop, Val(false))
    initialize!(value_function.upper, prop, Val(true))
end

#################
# Algorithms    #
#################
construct_value_function(::RobustValueIteration, problem) = StateValueFunction(problem)
construct_value_function(::IntervalValueIteration, problem) = IntervalValueFunction(
    lower = StateValueFunction(problem),
    upper = StateValueFunction(problem),
)
construct_value_function(::GeneralizedSamplingbasedRobustDynamicProgramming, problem) =
    StateValueFunction(problem)
