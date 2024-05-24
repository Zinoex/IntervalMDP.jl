# System representation

```@docs
IntervalMarkovProcess
num_states(mp::IntervalMarkovProcess)
initial_states(mp::IntervalMarkovProcess)
StationaryIntervalMarkovProcess
transition_prob(mp::StationaryIntervalMarkovProcess)
TimeVaryingIntervalMarkovProcess
transition_prob(mp::TimeVaryingIntervalMarkovProcess, t)
time_length(mp::TimeVaryingIntervalMarkovProcess)
```

## Markov chain
```@docs
IntervalMarkovChain
TimeVaryingIntervalMarkovProcess
```

## Markov decision process
```@docs
IntervalMarkovDecisionProcess
actions(mdp::IntervalMarkovDecisionProcess)
num_choices(mdp::IntervalMarkovDecisionProcess)
tomarkovchain
```

## Probability representation
```@docs
IntervalProbabilities
lower(s::IntervalProbabilities)
upper(s::IntervalProbabilities)
gap(s::IntervalProbabilities)
sum_lower(s::IntervalProbabilities)
num_source(s::IntervalProbabilities)
num_target(s::IntervalProbabilities)
axes_source(s::IntervalProbabilities)
axes_target(s::IntervalProbabilities)
```
