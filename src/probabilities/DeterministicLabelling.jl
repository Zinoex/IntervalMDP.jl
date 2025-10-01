"""
    struct DeterministicLabelling{
        T  <: Integer, 
        AT <: AbstractArray{T}
    }

A type representing the labelling of IMDP states into DFA inputs.

Formally, let ``L : S \\to 2^{AP}`` be a labelling function, where 
- ``S`` is the set of IMDP states, and
- ``2^{AP}`` is the power set of atomic propositions

Then the ```DeterministicLabelling``` type is defined as vector which stores the mapping. 

### Fields
- `map::AT`: mapping function where indices are (factored) IMDP states and stored values are DFA inputs.
- `num_outputs::Int32`: number of labels accounted for in mapping.

"""
struct DeterministicLabelling{T <: Integer, AT <: AbstractArray{T}} <: AbstractLabelling
    map::AT
    num_outputs::Int32

    function DeterministicLabelling(map::AT) where {T <: Integer, AT <: AbstractArray{T}}
        num_outputs = checklabelling(map)

        return new{T, AT}(map, Int32(num_outputs))
    end
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
    mapping(dl::DeterministicLabelling)

Return the mapping array of the labelling function. 
"""
mapping(dl::DeterministicLabelling) = dl.map

"""
    size(dl::DeterministicLabelling)

Returns the shape of the input range of the labeling function ``L : S \\to 2^{AP}``, which can be multiple dimensions in case of factored IMDPs. 
"""
Base.size(dl::DeterministicLabelling) = size(dl.map)

"""
    num_labels(dl::DeterministicLabelling)
Return the number of labels (DFA inputs) in the labelling function.
"""
num_labels(dl::DeterministicLabelling) = dl.num_outputs

"""
    state_values(dl::DeterministicLabelling)
Return a tuple with the number of states for each state variable of the labeling function ``L : S \\to 2^{AP}``, which can be multiple dimensions in case of factored IMDPs. 
"""
state_values(dl::DeterministicLabelling) = size(dl.map)

"""
    num_states(dl::DeterministicLabelling)
Return the number of states of the labeling function ``L : S \\to 2^{AP}``
"""
num_states(dl::DeterministicLabelling) = prod(state_values(dl))

"""
    getindex(dl::DeterministicLabelling, s...)

Return the label for state s.
"""
Base.getindex(dl::DeterministicLabelling, s...) = dl.map[s...]
