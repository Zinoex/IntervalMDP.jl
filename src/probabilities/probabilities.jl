abstract type AbstractMarginal end

export lower, upper, gap

include("IntervalAmbiguitySets.jl")
export IntervalAmbiguitySets

include("TransitionFunction.jl")
export TransitionFunction, transition

include("Labelling.jl")
export LabellingFunction, mapping, num_labels
