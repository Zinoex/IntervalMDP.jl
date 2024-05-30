module IntervalMDP

using LinearAlgebra, SparseArrays
using Polyester: @batch

include("interval_probabilities.jl")
export IntervalProbabilities
export lower, upper, gap, sum_lower
export num_source, axes_source, num_target, axes_target

include("models/IntervalMarkovProcess.jl")
include("models/IntervalMarkovChain.jl")
include("models/IntervalMarkovDecisionProcess.jl")
include("models/TimeVaryingIntervalMarkovChain.jl")
export IntervalMarkovProcess,
    StationaryIntervalMarkovProcess, TimeVaryingIntervalMarkovProcess
export IntervalMarkovChain, IntervalMarkovDecisionProcess
export TimeVaryingIntervalMarkovChain
export transition_prob,
    num_states, initial_states, actions, num_choices, tomarkovchain, time_length

include("specification.jl")
export Property, LTLFormula, LTLfFormula, PCTLFormula

export AbstractReachability, FiniteTimeReachability, InfiniteTimeReachability
export AbstractReachAvoid, FiniteTimeReachAvoid, InfiniteTimeReachAvoid
export AbstractReward, FiniteTimeReward, InfiniteTimeReward

export isfinitetime
export reach, avoid, terminal_states, time_horizon, convergence_eps, reward, discount

export SatisfactionMode, Pessimistic, Optimistic
export StrategyMode, Maximize, Minimize
export Specification, Problem
export system, specification, system_property, satisfaction_mode, strategy_mode

include("policy.jl")

include("ordering.jl")
export construct_ordering, sort_states!, perm
export AbstractStateOrdering, DenseOrdering, SparseOrdering, PermutationSubset

include("bellman.jl")
export bellman, bellman!

include("value_iteration.jl")
export value_iteration, termination_criteria
export TerminationCriteria, FixedIterationsCriteria, CovergenceCriteria

include("certify.jl")
export satisfaction_probability

include("synthesis.jl")
export control_synthesis

include("cuda.jl")

include("Data/Data.jl")

end
