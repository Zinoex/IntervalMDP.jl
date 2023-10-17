module IMDP

using LinearAlgebra

include("interval_probabilities.jl")
export StateIntervalProbabilities, MatrixIntervalProbabilities
export gap, lower, sum_lower

include("ordering.jl")
export construct_ordering, sort_states!, perm
export AbstractStateOrdering, DenseOrdering, SparseOrdering, PermutationSubset

include("ominmax.jl")
export ominmax, ominmax!
export partial_ominmax, partial_ominmax!
export probability_assignment!, probability_assignment_from!

include("value_iteration.jl")
export interval_value_iteration
export TerminationCriteria, FixedIterationsCriteria, CovergenceCriteria

end
