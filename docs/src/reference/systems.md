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
MixtureIntervalMarkovChain
MixtureIntervalMarkovDecisionProcess
stateptr(mdp::MixtureIntervalMarkovDecisionProcess)
```

## Probability representation
```@docs
IntervalProbabilities
OrthogonalIntervalProbabilities
MixtureIntervalProbabilities
lower
upper
gap
sum_lower
num_source
num_target
axes_source
mixture_probs
weighting_probs
```
