# Solve Interface

```@docs
solve
residual
num_iterations
value_function
strategy(res::IntervalMDP.ControlSynthesisSolution)
StationaryStrategy
TimeVaryingStrategy
```

## VI-like Algorithms

```@docs
RobustValueIteration
```

## Sampling-based Algorithms

```@docs
GeneralizedSamplingbasedRobustDynamicProgramming
```

## Sampling strategies

```@docs
SamplingStrategy
IntervalMDP.AllSamplingStrategy
IntervalMDP.AllSampling
IntervalMDP.AllStatesSweep
IntervalMDP.RandomSamplingStrategy
IntervalMDP.RandomSubsetStateActions
IntervalMDP.RandomSubsetState
IntervalMDP.RoundRobinSamplingStrategy
PriorityQueueSamplingStrategy
IntervalMDP.ValueFunctionOrderedSampling
IntervalMDP.GivenSequence
IntervalMDP.CompositeSamplingStrategy
```

### Trajectory sampling

Trajectory sampling simulates a trajectory through the model — from an initial
state, repeatedly select an action, realize a concrete transition distribution
out of the ambiguity set, and draw a successor from it — and relaxes the states
visited along the way. It is one strategy configured by a product of orthogonal
choices, rather than a family of fixed strategies.

```@docs
IntervalMDP.TrajectorySamplingStrategy
IntervalMDP.TrajectorySampling.TrajectorySampling
```

#### Selection policies

```@docs
IntervalMDP.TrajectorySampling.SelectionPolicy
IntervalMDP.TrajectorySampling.EpsilonGreedy
IntervalMDP.TrajectorySampling.Boltzmann
```

#### Temperature schedules

```@docs
IntervalMDP.TrajectorySampling.TemperatureSchedule
IntervalMDP.TrajectorySampling.FixedTemperature
IntervalMDP.TrajectorySampling.GapDecayTemperature
IntervalMDP.TrajectorySampling.UpdateDecayTemperature
IntervalMDP.TrajectorySampling.TemperatureContext
```

#### Action scores

```@docs
IntervalMDP.TrajectorySampling.ActionScore
IntervalMDP.TrajectorySampling.UpperBoundScore
IntervalMDP.TrajectorySampling.LowerBoundScore
IntervalMDP.TrajectorySampling.WeightedAverageScore
IntervalMDP.TrajectorySampling.LogScore
```

#### Concrete transition

```@docs
IntervalMDP.TrajectorySampling.ConcreteTransition
```

#### Successor scores

```@docs
IntervalMDP.TrajectorySampling.StateScore
IntervalMDP.TrajectorySampling.GreedyScore
IntervalMDP.TrajectorySampling.ExplorationScore
IntervalMDP.TrajectorySampling.GapWeightedExplorationScore
IntervalMDP.TrajectorySampling.GapFunction
IntervalMDP.TrajectorySampling.ExponentialGap
IntervalMDP.TrajectorySampling.PolynomialGap
```

#### Termination rules

```@docs
IntervalMDP.TrajectorySampling.TerminationRule
IntervalMDP.TrajectorySampling.MaxSteps
IntervalMDP.TrajectorySampling.ExpectedGapStop
IntervalMDP.TrajectorySampling.PredicateStop
IntervalMDP.TrajectorySampling.terminate_pre
IntervalMDP.TrajectorySampling.terminate_post
```

### Priority-queue sampling

Priority-queue sampling ranks every state by a priority, relaxes the top `k`,
and then repairs only the priorities that batch could have made stale — the
predecessors of what was relaxed, which is where a change in the value function
can propagate. Like trajectory sampling, it is one strategy configured by a
product of orthogonal choices.

The two are alternatives to the same question, and which suits a problem depends
on its termination criterion: a trajectory concentrates on the states reachable
from the initial state, while a queue covers the whole state space, which is
what `GapTerminationCriteria` stops on unless the property sets
`restrict_to_initial`.

```@docs
IntervalMDP.PriorityQueueSampling.PrioritizedSweep
IntervalMDP.compute_priority
```

#### State priorities

```@docs
IntervalMDP.PriorityQueueSampling.StatePriority
IntervalMDP.PriorityQueueSampling.state_priority
IntervalMDP.PriorityQueueSampling.GapPriority
IntervalMDP.PriorityQueueSampling.BoundPriority
IntervalMDP.PriorityQueueSampling.WeightedPriority
IntervalMDP.PriorityQueueSampling.ResidualPriority
IntervalMDP.PriorityQueueSampling.ActionUncertaintyPriority
IntervalMDP.PriorityQueueSampling.RNDPriority
IntervalMDP.PriorityQueueSampling._novelty_floor
IntervalMDP.PriorityQueueSampling.initialize_priority_state!
IntervalMDP.PriorityQueueSampling.on_selected!
IntervalMDP.PriorityQueueSampling.reset_priority!
```

#### Residual functions

```@docs
IntervalMDP.PriorityQueueSampling.bellman_residual_delta
IntervalMDP.PriorityQueueSampling.gap_delta
IntervalMDP.PriorityQueueSampling.action_uncertainty_delta
```

#### Propagation rules

```@docs
IntervalMDP.PriorityQueueSampling.PropagationRule
IntervalMDP.PriorityQueueSampling._combine
IntervalMDP.PriorityQueueSampling.MaxPropagation
IntervalMDP.PriorityQueueSampling.AdditivePropagation
IntervalMDP.PriorityQueueSampling.Recompute
```

#### Selection

Selection reuses the trajectory sampler's [selection policies](#Selection-policies)
and [temperature schedules](#Temperature-schedules), generalized from one draw to
`k` without replacement, plus one member of its own.

```@docs
IntervalMDP.PriorityQueueSampling.TopK
IntervalMDP.PriorityQueueSampling._top_k
```

#### Admission rules

```@docs
IntervalMDP.PriorityQueueSampling.AdmissionRule
IntervalMDP.PriorityQueueSampling.admit
IntervalMDP.PriorityQueueSampling.ConvergedSkip
IntervalMDP.PriorityQueueSampling.PredicateSkip
```

#### Deprecated constructors

Each is one configuration of [`PrioritizedSweep`](@ref
IntervalMDP.PriorityQueueSampling.PrioritizedSweep), kept so existing call sites
keep working. New code should construct a `PrioritizedSweep` directly.

```@docs
IntervalMDP.PriorityQueueSampling.GapPriorityQueueSampling
IntervalMDP.PriorityQueueSampling.UpperBoundPriorityQueueSampling
IntervalMDP.PriorityQueueSampling.ActionUncertaintyPriorityQueueSampling
IntervalMDP.PriorityQueueSampling.RNDPriorityQueueSampling
```