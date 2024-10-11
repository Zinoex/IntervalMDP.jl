# System representation

```@docs
IntervalMarkovProcess
num_states(s::IntervalMarkovProcess)
initial_states(s::IntervalMarkovProcess)
AllStates
transition_prob(mp::IntervalMarkovProcess)
IntervalMarkovChain
IntervalMarkovDecisionProcess
stateptr(mdp::IntervalMarkovDecisionProcess)
OrthogonalIntervalMarkovChain
OrthogonalIntervalMarkovDecisionProcess
stateptr(mdp::OrthogonalIntervalMarkovDecisionProcess)
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
