module IntervalMDP

using LinearAlgebra, SparseArrays

const UnionIndex = Union{<:Integer, <:Tuple}

include("threading.jl")
include("utils.jl")
include("errors.jl")
export InvalidStateError, StateDimensionMismatch

include("interval_probabilities.jl")
export IntervalProbabilities, OrthogonalIntervalProbabilities
export lower, upper, gap, sum_lower
export num_source, axes_source, num_target, axes_target

include("models/IntervalMarkovProcess.jl")
include("models/IntervalMarkovDecisionProcess.jl")
include("models/OrthogonalIntervalMarkovDecisionProcess.jl")
export IntervalMarkovProcess
export AllStates
export IntervalMarkovDecisionProcess, IntervalMarkovChain
export OrthogonalIntervalMarkovDecisionProcess, OrthogonalIntervalMarkovChain
export transition_prob, num_states, initial_states, stateptr, tomarkovchain, time_length

include("strategy.jl")
export GivenStrategyConfig, NoStrategyConfig, TimeVaryingStrategyConfig, StationaryStrategyConfig
export StationaryStrategy, TimeVaryingStrategy
export construct_strategy_cache

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
export system, specification, system_property, stategy, satisfaction_mode, strategy_mode

include("workspace.jl")
export construct_workspace

include("bellman.jl")
export bellman, bellman!

include("value_iteration.jl")
export value_iteration, termination_criteria
export TerminationCriteria, FixedIterationsCriteria, CovergenceCriteria

include("synthesis.jl")
export control_synthesis

include("cuda.jl")

include("Data/Data.jl")

end
