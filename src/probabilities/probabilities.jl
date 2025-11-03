# Ambiguity sets
abstract type AbstractAmbiguitySets end
abstract type PolytopicAmbiguitySets <: AbstractAmbiguitySets end

abstract type AbstractAmbiguitySet end
abstract type PolytopicAmbiguitySet <: AbstractAmbiguitySet end

abstract type VertexIterator end

"""
    num_sets(ambiguity_sets::AbstractAmbiguitySets)

Return the number of ambiguity sets in the AbstractAmbiguitySets object.
"""
function num_sets end
export num_sets

"""
    support(ambiguity_set::AbstractAmbiguitySet)

Return the support (set of indices with non-zero probability) of the ambiguity set.
"""
function support end
export support

include("IntervalAmbiguitySets.jl")
export IntervalAmbiguitySets, lower, upper, gap

abstract type AbstractIsInterval end
struct IsInterval <: AbstractIsInterval end
struct IsNotInterval <: AbstractIsInterval end

isinterval(::AbstractAmbiguitySets) = IsNotInterval()
isinterval(::IntervalAmbiguitySets) = IsInterval()

# Marginals
include("Marginal.jl")
export Marginal,
    ambiguity_sets,
    state_variables,
    action_variables,
    source_shape,
    action_shape,
    num_target

isinterval(marginal::Marginal) = isinterval(ambiguity_sets(marginal))

# DFA
include("TransitionFunction.jl")
export TransitionFunction, transition

include("Labelling.jl")

include("DeterministicLabelling.jl")
export DeterministicLabelling, mapping, num_labels, state_values, num_states

include("ProbabilisticLabelling.jl")
export ProbabilisticLabelling, mapping, num_labels, state_values, num_states
