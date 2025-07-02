abstract type AbstractIntervalProbabilities end
export lower, upper, gap, sum_lower
export num_source, axes_source, num_target, axes_target

include("IntervalProbabilities.jl")
export IntervalProbabilities

include("OrthogonalIntervalProbabilities.jl")
export OrthogonalIntervalProbabilities

include("MixtureIntervalProbabilities.jl")
export MixtureIntervalProbabilities, mixture_probs, weighting_probs

include("TransitionFunction.jl")
export TransitionFunction, transition

include("Labelling.jl")
export LabellingFunction, mapping, num_labels
