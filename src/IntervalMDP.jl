# IntervalMDP.jl - A Julia package for solving Interval Markov Decision Processes (IMDPs)
module IntervalMDP

# General solve interface
import CommonSolve: solve, solve!, init
export solve

# Import necessary libraries
using LinearAlgebra, SparseArrays

### Utilities
const UnionIndex = Union{<:Integer, <:Tuple}

include("errors.jl")
export InvalidStateError, StateDimensionMismatch

### Modelling
include("probabilities/probabilities.jl")
include("models/models.jl")

include("strategy.jl")
export StationaryStrategy, TimeVaryingStrategy
export time_length

include("specification.jl")
export Property, BasicProperty, ProductProperty

export FiniteTimeDFAReachability, InfiniteTimeDFAReachability
export FiniteTimeReachability, InfiniteTimeReachability, ExactTimeReachability
export FiniteTimeReachAvoid, InfiniteTimeReachAvoid, ExactTimeReachAvoid
export FiniteTimeSafety, InfiniteTimeSafety
export FiniteTimeReward, InfiniteTimeReward
export ExpectedExitTime

export isfinitetime
export reach, avoid, safe, terminal_states, time_horizon, convergence_eps, reward, discount

export SatisfactionMode, Pessimistic, Optimistic, ispessimistic, isoptimistic
export StrategyMode, Maximize, Minimize, ismaximize, isminimize
export Specification
export system, specification, system_property, strategy, satisfaction_mode, strategy_mode

include("problem.jl")
export VerificationProblem, ControlSynthesisProblem
export value_function, residual, num_iterations

include("cuda.jl")

### Solving
include("utils.jl")
include("threading.jl")

include("algorithms/algorithms.jl")
export RobustValueIteration

### Saving and loading models
include("Data/Data.jl")

end
