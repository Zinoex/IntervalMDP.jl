abstract type AbstractLabelling end

"""
    struct LabellingFunction{
        T  <: Integer, 
        AT <: AbstractArray{T}
    }

A type representing the labelling of IMDP states into DFA inputs.

Formally, let ``L : S \\to 2^{AP}`` be a labelling function, where 
- ``S`` is the set of IMDP states, and
- ``2^{AP}`` is the power set of atomic propositions

Then the ```LabellingFunction``` type is defined as vector which stores the mapping. 

### Fields
- `map::AT`: mapping function where indices are (factored) IMDP states and stored values are DFA inputs.
- `num_outputs::Int32`: number of labels accounted for in mapping.

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
    num_states(labelling_func::LabellingFunction)
Return the number of states (input range) of the labeling function ``L : S \\to 2^{AP}``, which can be multiple dimensions in case of factored IMDPs. 
"""
num_states(labelling_func::LabellingFunction) = size(labelling_func.map)

"""
    getindex(lf::LabellingFunction, s...)

Return the label for state s.
"""
Base.getindex(lf::LabellingFunction, s...) = lf.map[s...]




"""
    struct ProbabilisticLabelling{
        R <: AbstractFloat, 
        MR <: AbstractMatrix{R}
    }

A type representing the Probabilistic labelling of IMDP states into DFA inputs. Each labelling is assigned a probability.

Formally, let ``L : S \\times 2^{AP} \\to [0, 1]`` be a labelling function, where 
- ``S`` is the set of IMDP states, and
- ``2^{AP}`` is the power set of atomic propositions

Then the ```ProbabilisticLabelling``` type is defined as matrix which stores the mapping. 

### Fields
- `map::MT`: mapping function encoded as matrix with labels on the rows, IMDP states on the columns, and valid probability values for the destination.

The choice to have labels on the rows is due to the column-major storage of matrices in Julia and the fact that we want the inner loop over DFA target states 
in the Bellman operator `bellman!`.

"""
struct ProbabilisticLabelling{R <: AbstractFloat, MR <: AbstractMatrix{R}} <: AbstractLabelling
    map::MR
end

function ProbabilisticLabelling(map::MR) where {R <: AbstractFloat, MR <: AbstractMatrix{R}}
    checklabelling(map)

    return ProbabilisticLabelling(map)
end

function checklabelling(map::AbstractMatrix{<:AbstractFloat})

    # check for each state, all the labels probabilities sum to 1
    if any(sum(map, dim=1) .!= 1)
        throw(ArgumentError("For each IMDP state, probabilities over label states must sum to 1"))
    end
end


"""
    mapping(pl::ProbabilisticLabelling)

Return the mapping matrix of the probabilistic labelling function. 
"""
mapping(pl::ProbabilisticLabelling) = pl.map


Base.size(pl::ProbabilisticLabelling) = size(pl.map)
Base.size(pl::ProbabilisticLabelling, i) = size(pl.map, i)


"""
    getindex(pl::ProbabilisticLabelling, s, l)

Return the probabilities for labelling l from state s.
"""
Base.getindex(pl::ProbabilisticLabelling, s, l) = pl.map[l, s]

"""
    getindex(pl::ProbabilisticLabelling, s)

Return the probabilities over labels from state s.
"""
Base.getindex(pl::ProbabilisticLabelling, s) = pl.map[:, s]


"""
    num_labels(pl::ProbabilisticLabelling)
Return the number of labels (DFA inputs) in the probabilistic labelling function.
"""
num_labels(pl::ProbabilisticLabelling) = size(pl.map, 1)

"""
    num_states(pl::ProbabilisticLabelling)
Return the number of states (input range) of the probabilistic labelling function, , which can be multiple dimensions in case of factored IMDPs.
"""
num_states(pl::ProbabilisticLabelling) = Base.tail(size(pl.map))

