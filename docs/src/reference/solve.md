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
IntervalMDP.AllSampling
IntervalMDP.AllStatesSweep
IntervalMDP.RandomSubsetStateActions
IntervalMDP.RandomSubsetState
ValueBasedSamplingStrategy
IntervalMDP.ValueFunctionOrderedSampling
```