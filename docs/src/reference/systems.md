# System representation

```@docs
IntervalMarkovProcess
num_states(s::IntervalMarkovProcess)
initial_states(s::IntervalMarkovProcess)
```

## Stationary Markov Processes
```@docs
StationaryIntervalMarkovProcess
transition_prob(mp::StationaryIntervalMarkovProcess)
IntervalMarkovChain
IntervalMarkovDecisionProcess
stateptr(mdp::IntervalMarkovDecisionProcess)
tomarkovchain(mdp::IntervalMarkovDecisionProcess, strategy::AbstractVector)
tomarkovchain(mdp::IntervalMarkovDecisionProcess, strategy::AbstractVector{<:AbstractVector})
```

## Time-varying Markov Processes
```@docs
TimeVaryingIntervalMarkovProcess
transition_prob(mp::TimeVaryingIntervalMarkovProcess, t)
time_length(mp::TimeVaryingIntervalMarkovProcess)
TimeVaryingIntervalMarkovChain
TimeVaryingIntervalMarkovDecisionProcess
stateptr(mdp::TimeVaryingIntervalMarkovDecisionProcess)
tomarkovchain(mdp::TimeVaryingIntervalMarkovDecisionProcess, strategy::AbstractVector{<:AbstractVector})
```

## Composite Markov Processes
```@docs
CompositeIntervalMarkovProcess
ProductIntervalMarkovProcess
ParallelProduct
```

## Probability representation
```@docs
IntervalProbabilities
lower(p::IntervalProbabilities)
upper(p::IntervalProbabilities)
gap(p::IntervalProbabilities)
sum_lower(p::IntervalProbabilities)
num_source(p::IntervalProbabilities)
num_target(p::IntervalProbabilities)
axes_source(p::IntervalProbabilities)
```
