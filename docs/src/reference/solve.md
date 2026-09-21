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