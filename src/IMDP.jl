module IMDP

using LinearAlgebra

include("interval_probabilities.jl")
export StateIntervalProbabilities, MatrixIntervalProbabilities
export gap, lower, sum_lower, num_src

include("models.jl")
export System, IntervalMarkovChain, IntervalMarkovDecisionProcess
export transition_prob, num_states, initial_state

include("specification.jl")
export Specification, LTLFormula, LTLfFormula, PTCLFormula
export FiniteTimeReachability,
    InfiniteTimeReachability, FiniteTimeReachAvoid, InfiniteTimeReachAvoid
export Problem, SatisfactionMode, Pessimistic, Optimistic
export reach, avoid, terminal_states, time_horizon, eps

include("ordering.jl")
export construct_ordering, sort_states!, perm
export AbstractStateOrdering, DenseOrdering, SparseOrdering, PermutationSubset

include("ominmax.jl")
export ominmax, ominmax!
export partial_ominmax, partial_ominmax!
export probability_assignment!, probability_assignment_from!

include("value_iteration.jl")
export interval_value_iteration, termination_criteria
export TerminationCriteria, FixedIterationsCriteria, CovergenceCriteria

include("certify.jl")
export satisfaction_probability

end
