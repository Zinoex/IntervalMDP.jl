@enum IntervalMode Upper Lower
isupper(mode::IntervalMode) = mode == Upper
islower(mode::IntervalMode) = mode == Lower

abstract type ValueFunction end

struct StateValueFunction{R, A1 <: AbstractArray{R}, A2 <: AbstractArray{R}} <: ValueFunction
    previous::A1
    current::A1
    intermediate_state_action_value::A2
    interval::IntervalMode
end

function StateValueFunction(problem::AbstractIntervalMDPProblem)
    mp = system(problem)
    previous = arrayfactory(mp, valuetype(mp), state_values(mp))
    previous .= zero(valuetype(mp))
    current = copy(previous)

    dim = (action_values(mp)..., state_values(mp)...)
    # concat gives shape: (a1, a2) , (s1, s2) => (a1, a2, s1, s2)
    # (a, s) to access a more frequently due to column major
    # TODO: works for IMDP, need to check for fIMDP
    intermediate_state_action_value = arrayfactory(mp, valuetype(mp), dim)
    intermediate_state_action_value .= zero(valuetype(mp))

    return StateValueFunction(previous, current, intermediate_state_action_value, Lower)
end

function StateValueFunction(problem::AbstractIntervalMDPProblem, mode::IntervalMode)
    mp = system(problem)
    previous = arrayfactory(mp, valuetype(mp), state_values(mp))
    previous .= zero(valuetype(mp))
    current = copy(previous)

    dim = (action_values(mp)..., state_values(mp)...)
    # concat gives shape: (a1, a2) , (s1, s2) => (a1, a2, s1, s2)
    # (a, s) to access a more frequently due to column major
    # TODO: works for IMDP, need to check for fIMDP
    intermediate_state_action_value = arrayfactory(mp, valuetype(mp), dim)
    intermediate_state_action_value .= zero(valuetype(mp))

    return StateValueFunction(previous, current, intermediate_state_action_value, mode)
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


struct StateActionValueFunction{R, A1 <: AbstractArray{R}, A2 <: AbstractArray{R}} <: ValueFunction
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
construct_value_function(::IntervalValueIteration, problem) = IntervalValueFunction(lower=StateValueFunction(problem), upper=StateValueFunction(problem))
construct_value_function(::GeneralizedSamplingbasedRobustDynamicProgramming, problem) = IntervalValueFunction(lower=StateActionValueFunction(problem), upper=StateActionValueFunction(problem))