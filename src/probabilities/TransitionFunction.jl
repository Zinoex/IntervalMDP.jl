"""
    Transition matrix => |2^L| x |Z| 
        column: alphabet
        row: states
"""

struct TransitionFunction{R, MR <: AbstractMatrix{R}}
    transition::MR
end

function TransitionFunction(transition::MR) where {R, MR <: AbstractMatrix{R}}

    # checktransition!(transition) 

    return TransitionFunction(transition)
end

"""
Check given transition func valid
"""
function checktransition!(transition::AbstractMatrix)
    # check only transition to valid states
    @assert all(transition .>= 1) "all transitioned states exists"
    @assert all(transition .<= size(transition, 2)) "all transitioned states exists"
end