module IMDP

using LinearAlgebra


include("interval_probabilities.jl")
export StateIntervalProbabilities, MatrixIntervalProbabilities
export gap, lower, sum_lower

include("models.jl")
export System, IntervalMarkovChain
export transition_prob, terminal_states, num_states

include("specification.jl")
export Specification, LTLFormula, LTLfFormula, PTCLFormula, FiniteTimeReachability
export Problem, SatisfactionMode, Pessimistic, Optimistic, time_horizon

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

include("certify.jl")
export satisfaction_probability

end
