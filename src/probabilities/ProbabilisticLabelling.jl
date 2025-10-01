"""
    struct ProbabilisticLabelling{
        R <: Real, 
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
struct ProbabilisticLabelling{R <: Real, MR <: AbstractMatrix{R}} <: AbstractLabelling
    map::MR

    function ProbabilisticLabelling(map::MR) where {R <: Real, MR <: AbstractMatrix{R}}
        checklabellingprobs(map)

        return new{R, MR}(map)
    end
end

function checklabellingprobs(map::AbstractMatrix{<:Real})

    # check for each state, all the labels probabilities sum to 1
    if any(sum(map; dims=1) .!= 1)
        throw(
            ArgumentError(
                "For each IMDP state, probabilities over label states must sum to 1",
            ),
        )
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
Base.getindex(pl::ProbabilisticLabelling, s) = @view(pl.map[:, s])

"""
    num_labels(pl::ProbabilisticLabelling)
Return the number of labels (DFA inputs) in the probabilistic labelling function.
"""
num_labels(pl::ProbabilisticLabelling) = size(pl.map, 1)

"""
    state_values(pl::ProbabilisticLabelling)
Return a tuple with the number of states for each state variable of the labeling function ``L : S \\to 2^{AP}``, which can be multiple dimensions in case of factored IMDPs. 
"""
state_values(pl::ProbabilisticLabelling) = Base.tail(size(pl.map))

"""
    num_states(pl::ProbabilisticLabelling)
Return the number of states of the labeling function ``L : S \\to 2^{AP}``
"""
num_states(pl::ProbabilisticLabelling) = prod(state_values(pl))
