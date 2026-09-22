# IntervalMDP.jl - A Julia package for solving Interval Markov Decision Processes (IMDPs)
module IntervalMDP

# General solve interface
import CommonSolve: solve, solve!, init
export solve

# Import necessary libraries
using LinearAlgebra, SparseArrays
using JuMP, HiGHS
using StyledStrings
import Flux

### Utilities
const UnionIndex = Union{<:Integer, <:Tuple}

include("errors.jl")
export InvalidStateError, StateDimensionMismatch, InvertedBracketError

### Modelling
include("probabilities/probabilities.jl")
include("available_actions.jl")
export AllAvailableActions, ListAvailableActions, TimeVaryingAvailableActions
include("models/models.jl")

include("strategy.jl")
export StationaryStrategy, TimeVaryingStrategy
export time_length

include("specification.jl")
export Property, BasicProperty, ProductProperty

export FiniteTimeDFAReachability, InfiniteTimeDFAReachability
export FiniteTimeDFASafety, InfiniteTimeDFASafety
export FiniteTimeReachability, InfiniteTimeReachability, ExactTimeReachability
export FiniteTimeReachAvoid, InfiniteTimeReachAvoid, ExactTimeReachAvoid
export FiniteTimeSafety, InfiniteTimeSafety
export FiniteTimeReward, InfiniteTimeReward
export ExpectedExitTime

export reach, avoid, safe, time_horizon, convergence_eps, reward, discount

export SatisfactionMode, Pessimistic, Optimistic, ispessimistic, isoptimistic
export StrategyMode, Maximize, Minimize, ismaximize, isminimize
export Specification
export system,
    specification,
    system_property,
    strategy,
    satisfaction_mode,
    strategy_mode,
    restrict_to_initial

include("problem.jl")
export VerificationProblem, ControlSynthesisProblem
export value_function, residual, num_iterations

include("cuda.jl")
public cu, cpu

### Solving
include("value.jl")
include("algorithms.jl")
export OMaximization, LPMcCormickRelaxation, VertexEnumeration
export RobustValueIteration, GeneralizedSamplingbasedRobustDynamicProgramming
export default_algorithm, default_bellman_algorithm, bellman_algorithm

include("utils.jl")
include("threading.jl")
include("workspace.jl")
include("strategy_cache.jl")
include("termination.jl")

# Bellman primitives: scalar kernels in `bellman/kernels.jl`, then the
# state-shape (`bellman_v!`) and state-action-shape (`bellman_q!`) sweeps
# in their own files. Both call into the kernels.
include("bellman/kernels.jl")
include("bellman/state.jl")
include("bellman/state_action.jl")
include("bellman/product.jl")

# `rnd.jl` provides the Random Network Distillation novelty model used by
# `PriorityQueueSampling.RNDPriority`. It comes before `prioritysweeping.jl`
# because that submodule imports the `_rnd_*` hooks by name at
# module-definition time.
include("rnd.jl")

# `sampling.jl` defines `SamplingStrategy`s and the `sample` dispatcher.
# Comes after the algorithm-defining files because `sampling_strategy`
# methods dispatch on the algorithm types.
include("sampling.jl")

# The configurable trajectory sampler. Split out of `sampling.jl` because the
# configuration vocabulary (policies, score functions, termination rules) is
# substantial on its own; the `TrajectorySamplingStrategy` category and the
# O-max primitives it builds on stay in `sampling.jl`, since the
# priority-queue strategies share them.
include("trajectorysampling.jl")

# The configurable priority-queue sampler, split out for the same reason. It
# comes after `trajectorysampling.jl` because it imports that submodule's
# selection policies and temperature schedules by name at module-definition
# time, rather than restating them.
include("prioritysweeping.jl")

export SamplingStrategy, PriorityQueueSamplingStrategy
public AllSampling,
    AllStatesSweep,
    RandomSubsetStateActions,
    RandomSubsetState,
    ValueFunctionOrderedSampling,
    TrajectorySampling,
    PriorityQueueSampling

include("robust_value_iteration.jl")
include("gsrdp.jl")

### Saving and loading models
include("Data/Data.jl")

end
