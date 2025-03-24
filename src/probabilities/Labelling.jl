abstract type AbstractLabelling end

"""
    struct LabellingFunction{
        T  <:Unsigned, 
        VT <: AbstractVector{T}
    }

A type representing the labelling of IMDP states into DFA inputs.

Formally, let ``L : S => 2^{AP}`` be a labelling function, where 
- ``S`` is the set of IMDP states, and
- ``2^{AP}`` is the power set of atomic propositions

Then the ```LabellingFunction``` type is defined as vector which stores the mapping. 

### Fields
- `map::VT`: mapping function where indices are IMDP states and stored values are DFA inputs.
- `num_inputs::Int32`: number of IMDP states accounted for in mapping.
- `num_outputs::Int32`: number of DFA inputs accounted for in mapping.

"""

struct LabellingFunction{T<:Unsigned, VT <: AbstractVector{T}
} <: AbstractLabelling
    map::VT,
    num_inputs::Int32,
    num_outputs::Int32,
end

function LabellingFunction(map::VT) where {T<:Unsigned, VT <: AbstractVector{T}}

    num_inputs, num_outputs = count_mapping!(map) 

    return LabellingFunction(map, num_inputs, num_outputs)
end

"""
Find size of input and output space of function
"""
function count_mapping!(map::AbstractVector)
    num_inputs = length(map)
    num_outputs = maximum(map)

    return num_inputs, num_outputs
end