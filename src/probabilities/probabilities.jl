abstract type AbstractAmbiguitySets end
abstract type PolytopicAmbiguitySets <: AbstractAmbiguitySets end

abstract type AbstractAmbiguitySet end
abstract type PolytopicAmbiguitySet <: AbstractAmbiguitySet end

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

abstract type AbstractMarginal end
include("Marginal.jl")
export SARectangularMarginal, Marginal, ambiguity_sets, state_variables, action_variables, source_shape, action_shape, num_target

include("TransitionFunction.jl")
export TransitionFunction, transition

include("Labelling.jl")
export LabellingFunction, mapping, num_labels
