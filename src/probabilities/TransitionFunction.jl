"""
    struct TransitionFunction{
        T<:Unsigned, 
        MT <: AbstractMatrix{T}
    }

A type representing the determininistic transition function of a DFA.

Formally, let ``T : |2^{AP}| \\times |Z| => |Z|`` be a transition function, where 
- ``Z`` is the set of DFA states, and
- ``2^{AP}`` is the power set of atomic propositions

Then the ```TransitionFunction``` type is defined as matrix which stores the mapping. The column indices are the alphabet indices and the row indices represent the states.  

### Fields
- `transition::MT`: transition function.

"""

struct TransitionFunction{T<:Unsigned, MT <: AbstractMatrix{T}}
    transition::MT
end

function TransitionFunction(transition::MT) where {T<:Unsigned, MT <: AbstractMatrix{T}}

    checktransition!(transition) 

    return TransitionFunction(transition)
end

"""
Check given transition func valid
"""
function checktransition!(transition::AbstractMatrix)
    # check only transition to valid states
    # @assert all(transition .>= 1) "all transitioned states exists"
    # @assert all(transition .<= size(transition, 2)) "all transitioned states exists"

    if !all(transition .>= 1)
        throw(
            throw(ArgumentError("Transitioned state index cannot be zero or negative"))
        )
    end

    if !all(transition .<= size(transition, 2))
        throw(
            throw(ArgumentError("Transitioned state index cannot be larger than total number of states"))
        )
    end
end