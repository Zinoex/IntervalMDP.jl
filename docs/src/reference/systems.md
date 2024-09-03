# System representation

```@docs
IntervalMarkovProcess
num_states(s::IntervalMarkovProcess)
initial_states(s::IntervalMarkovProcess)
AllStates
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
OrthogonalIntervalMarkovChain
OrthogonalIntervalMarkovDecisionProcess
stateptr(mdp::OrthogonalIntervalMarkovDecisionProcess)
tomarkovchain(mdp::OrthogonalIntervalMarkovDecisionProcess, strategy::AbstractVector)
tomarkovchain(mdp::OrthogonalIntervalMarkovDecisionProcess, strategy::AbstractVector{<:AbstractVector})
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

## Probability representation
```@docs
IntervalProbabilities
OrthogonalIntervalProbabilities
lower
upper
gap
sum_lower
num_source
num_target
axes_source
```
