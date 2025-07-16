"""
    struct TransitionFunction{
        T <: Unsigned, 
        MT <: AbstractMatrix{T}
    }

A type representing the determininistic transition function of a DFA.

Formally, let ``T : |Q| \\times |2^{AP}| => |Q|`` be a transition function, where 
- ``Q`` is the set of DFA states, and
- ``2^{AP}`` is the power set of atomic propositions

Then the ```TransitionFunction``` type is defined as matrix which stores the mapping. The row indices are the alphabet indices and the column indices represent the states.  

### Fields
- `transition::MT`: transition functions encoded as matrix with labels on the rows, source states on the columns, and integer values for the destination.

The choice to have labels on the rows is due to the column-major storage of matrices in Julia and the fact that we want the outer loop over DFA source states 
in the Bellman operator `bellman!`.

We check that the transition matrix is valid, i.e. that all indices are positive and do not exceed the number of states.

"""
struct TransitionFunction{T <: Integer, MT <: AbstractMatrix{T}}
    transition::MT

    function TransitionFunction(
        transition::MT,
    ) where {T <: Integer, MT <: AbstractMatrix{T}}
        checktransition(transition)

        return new{T, MT}(transition)
    end
end

function checktransition(transition::AbstractMatrix)
    # Check that transitions only happens to valid states
    if any(transition .< 1)
        throw(ArgumentError("Transitioned state index cannot be less than 1"))
    end

    num_states = size(transition, 2)
    if any(transition .> num_states)
        throw(
            ArgumentError(
                "Transitioned state index cannot be larger than total number of states",
            ),
        )
    end
end

"""
    transition(transition_func::TransitionFunction)

Return the transition matrix of the transition function. 
"""
transition(transition_func::TransitionFunction) = transition_func.transition

"""
    getindex(tf::TransitionFunction, z, w)

Return the next state for source state ``z`` and input ``w`` of the transition function. 
"""
Base.getindex(tf::TransitionFunction, z, w) = tf.transition[w, z]

Base.size(tf::TransitionFunction) = size(tf.transition)
Base.size(tf::TransitionFunction, i) = size(tf.transition, i)

"""
    num_states(tf::TransitionFunction)
Return the number of states ``|Q|`` of the transition function.
"""
num_states(tf::TransitionFunction) = size(tf.transition, 2)

"""
    num_labels(tf::TransitionFunction)
Return the number of labels (DFA inputs) in the transition function.
"""
num_labels(tf::TransitionFunction) = size(tf.transition, 1)
