struct ValueFunction{R, A <: AbstractArray{R}}
    previous::A
    current::A
end

function ValueFunction(problem::AbstractIntervalMDPProblem)
    mp = system(problem)
    previous = arrayfactory(mp, valuetype(mp), product_num_states(mp))
    current = copy(previous)

    return ValueFunction(previous, current)
end

function lastdiff!(V::ValueFunction{R}) where {R}
    # Reuse prev to store the latest difference
    V.previous .-= V.current
    rmul!(V.previous, -one(R))

    return V.previous
end

function diff!(residual, V_primal::ValueFunction, V_dual::ValueFunction)
    copyto!(residual, V_primal.current)
    residual .-= V_dual.current

    return residual
end

function nextiteration!(V)
    copy!(V.previous, V.current)

    return V
end