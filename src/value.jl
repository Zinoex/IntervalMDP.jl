abstract type ValueFunction end

struct StateValueFunction{R, A <: AbstractArray{R}} <: ValueFunction
    previous::A
    current::A
    intermediate_state_action_value::A
end

function StateValueFunction(problem::AbstractIntervalMDPProblem)
    mp = system(problem)
    previous = arrayfactory(mp, valuetype(mp), state_values(mp))
    previous .= zero(valuetype(mp))
    current = copy(previous)

    intermediate_state_action_value = arrayfactory(mp, valuetype(mp), state_values(mp) .* action_values(mp))
    intermediate_state_action_value .= zero(valuetype(mp))

    return StateValueFunction(previous, current, intermediate_state_action_value)
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


struct StateActionValueFunction{R, A <: AbstractArray{R}} <: ValueFunction
    previous::A
    current::A
    intermediate_state_value::A
end

function StateActionValueFunction(problem::AbstractIntervalMDPProblem)
    mp = system(problem)
    previous = arrayfactory(mp, valuetype(mp), state_values(mp) .* action_values(mp))
    previous .= zero(valuetype(mp))
    current = copy(previous)

    intermediate_state_value = arrayfactory(mp, valuetype(mp), state_values(mp))
    intermediate_state_value .= zero(valuetype(mp))

    return StateActionValueFunction(previous, current, intermediate_state_value)
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