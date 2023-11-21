module IMDP

using LinearAlgebra, SparseArrays

include("interval_probabilities.jl")
export IntervalProbabilities
export lower, upper, gap, sum_lower, num_source, num_target

include("models.jl")
export IntervalMarkovProcess, IntervalMarkovChain, IntervalMarkovDecisionProcess
export transition_prob, num_states, initial_state, actions

include("specification.jl")
export Specification, LTLFormula, LTLfFormula, PCTLFormula
export AbstractReachability,
    FiniteTimeReachability,
    InfiniteTimeReachability,
    FiniteTimeReachAvoid,
    InfiniteTimeReachAvoid
export reach, avoid, terminal_states, time_horizon, eps
export Problem, SatisfactionMode, Pessimistic, Optimistic
export system, specification, satisfaction_mode

include("ordering.jl")
export construct_ordering, sort_states!, perm
export AbstractStateOrdering, DenseOrdering, SparseOrdering, PermutationSubset

include("ominmax.jl")
export ominmax, ominmax!
export partial_ominmax, partial_ominmax!
export probability_assignment!, probability_assignment_from!

include("value_iteration.jl")
export value_iteration, termination_criteria
export TerminationCriteria, FixedIterationsCriteria, CovergenceCriteria

include("certify.jl")
export satisfaction_probability

include("cuda.jl")

include("Data/Data.jl")

end
