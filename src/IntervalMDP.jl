# IntervalMDP.jl - A Julia package for solving Interval Markov Decision Processes (IMDPs)
module IntervalMDP

# General solve interface
import CommonSolve: solve, solve!, init
export solve

# Import necessary libraries
using LinearAlgebra, SparseArrays
using JuMP, HiGHS
using Combinatorics: permutations, Permutations
using StyledStrings

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
public cu, cpu

### Solving
include("algorithms.jl")
export OMaximization, LPMcCormickRelaxation
export RobustValueIteration

include("utils.jl")
include("threading.jl")
include("workspace.jl")
include("strategy_cache.jl")
include("bellman.jl")

include("robust_value_iteration.jl")

### Saving and loading models
include("Data/Data.jl")

end
