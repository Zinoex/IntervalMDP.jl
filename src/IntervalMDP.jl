module IntervalMDP

using LinearAlgebra, SparseArrays

const UnionIndex = Union{<:Integer, <:Tuple}

include("threading.jl")
include("utils.jl")
include("errors.jl")
export InvalidStateError, StateDimensionMismatch

include("probabilities/probabilities.jl")
include("models/models.jl")

include("strategy.jl")
export GivenStrategyConfig,
    NoStrategyConfig, TimeVaryingStrategyConfig, StationaryStrategyConfig
export StationaryStrategy, TimeVaryingStrategy
export construct_strategy_cache, time_length

include("specification.jl")
export Property, LTLFormula, LTLfFormula, PCTLFormula

export AbstractReachability, FiniteTimeReachability, InfiniteTimeReachability, ExactTimeReachability
export AbstractReachAvoid, FiniteTimeReachAvoid, InfiniteTimeReachAvoid, ExactTimeReachAvoid
export AbstractSafety, FiniteTimeSafety, InfiniteTimeSafety
export AbstractReward, FiniteTimeReward, InfiniteTimeReward
export AbstractHittingTime, ExpectedExitTime

export isfinitetime
export reach, avoid, safe, terminal_states, time_horizon, convergence_eps, reward, discount

export SatisfactionMode, Pessimistic, Optimistic, ispessimistic, isoptimistic
export StrategyMode, Maximize, Minimize, ismaximize, isminimize
export Specification, Problem
export system, specification, system_property, strategy, satisfaction_mode, strategy_mode

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
