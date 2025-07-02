abstract type AbstractLabelling end

"""
    struct LabellingFunction{
        T  <: Integer, 
        VT <: AbstractVector{T}
    }

A type representing the labelling of IMDP states into DFA inputs.

Formally, let ``L : S \\to 2^{AP}`` be a labelling function, where 
- ``S`` is the set of IMDP states, and
- ``2^{AP}`` is the power set of atomic propositions

Then the ```LabellingFunction``` type is defined as vector which stores the mapping. 

### Fields
- `map::VT`: mapping function where indices are (factored) IMDP states and stored values are DFA inputs.
- `num_inputs::Int32`: number of IMDP states accounted for in mapping.
- `num_outputs::Int32`: number of DFA inputs accounted for in mapping.

"""
struct LabellingFunction{T <: Integer, AT <: AbstractArray{T}} <: AbstractLabelling
    map::AT
    num_outputs::Int32
end

function LabellingFunction(map::AT) where {T <: Integer, AT <: AbstractArray{T}}
    num_outputs = checklabelling(map)

    return LabellingFunction(map, Int32(num_outputs))
end

function checklabelling(map::AbstractArray{<:Integer})
    labels = unique(map)

    if any(labels .< 1)
        throw(ArgumentError("Labelled state index cannot be less than 1"))
    end

    # Check that labels are consecutive integers
    sort!(labels)
    if any(diff(labels) .!= 1)
        throw(ArgumentError("Labelled state indices must be consecutive integers"))
    end

    return last(labels)
end

"""
    mapping(labelling_func::LabellingFunction)

Return the mapping array of the labelling function. 
"""
mapping(labelling_func::LabellingFunction) = labelling_func.map

"""
    size(labelling_func::LabellingFunction)

Returns the shape of the input range of the labeling function ``L : S \\to 2^{AP}``, which can be multiple dimensions in case of factored IMDPs. 
"""
Base.size(labelling_func::LabellingFunction) = size(labelling_func.map)

"""
    num_labels(labelling_func::LabellingFunction)
Return the number of labels (DFA inputs) in the labelling function.
"""
num_labels(labelling_func::LabellingFunction) = labelling_func.num_outputs

"""
    getindex(lf::LabellingFunction, s...)

Return the label for state s.
"""
Base.getindex(lf::LabellingFunction, s...) = lf.map[s...]
