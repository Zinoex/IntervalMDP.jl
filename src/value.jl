abstract type ValueFunction end

struct StateValueFunction{R, A1 <: AbstractArray{R}, A2 <: AbstractArray{R}} <: ValueFunction
    previous::A1
    current::A1
    intermediate_state_action_value::A2
end

function StateValueFunction(problem::AbstractIntervalMDPProblem)
    mp = system(problem)
    previous = arrayfactory(mp, valuetype(mp), state_values(mp))
    previous .= zero(valuetype(mp))
    current = copy(previous)

    dim = Tuple(Iterators.flatten(zip(action_values(mp), state_values(mp))))
    # interleaved concat gives shape: (a1, a2) , (s1, s2) => (a1, s1, a2, s2)
    # (a, s) to access s more frequently due to column major
    # TODO: works for IMDP, need to check for fIMDP
    intermediate_state_action_value = arrayfactory(mp, valuetype(mp), dim)
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


struct StateActionValueFunction{R, A1 <: AbstractArray{R}, A2 <: AbstractArray{R}} <: ValueFunction
    previous::A1
    current::A1
    intermediate_state_value::A2
end

function StateActionValueFunction(problem::AbstractIntervalMDPProblem)
    mp = system(problem)
    dim = Tuple(Iterators.flatten(zip(action_values(mp), state_values(mp))))
    # TODO: works for IMDP, need to check for fIMDP
    previous = arrayfactory(mp, valuetype(mp), dim)
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