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
struct LabellingFunction{T <: Unsigned, AT <: AbstractArray{T}} <: AbstractLabelling
    map::AT
    num_inputs::Int32
    num_outputs::Int32
end

function LabellingFunction(map::AT) where {T <: Unsigned, AT <: AbstractArray{T}}
    num_inputs, num_outputs = count_mapping(map)

    return LabellingFunction(map, Int32(num_inputs), Int32(num_outputs))
end

"""
Find size of input and output space of function
"""
function count_mapping(map::AbstractArray)
    num_inputs = length(map)
    num_outputs = maximum(map)

    return num_inputs, num_outputs
end

"""
    mapping(labelling_func::LabellingFunction)

Return the mapping array of the labelling function. 
"""
mapping(labelling_func::LabellingFunction) = labelling_func.map

"""
    size(labelling_func::LabellingFunction)

Returns ``|S|`` and ``|2^{AP}|`` of the labeling function ``L: S => 2^{AP}`` . 
"""
Base.size(labelling_func::LabellingFunction) =
    (labelling_func.num_inputs, labelling_func.num_outputs)

"""
    getindex(lf::LabellingFunction, s::Int)

Return the label for state s. 
"""
Base.getindex(lf::LabellingFunction, s::Int) = lf.map[s]
