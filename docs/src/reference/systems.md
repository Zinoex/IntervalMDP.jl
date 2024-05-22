# System representation

```@docs
IntervalMarkovProcess
```

## Markov chain
```@docs
IntervalMarkovChain
transition_prob(mc::IntervalMarkovChain)
num_states(mc::IntervalMarkovChain)
initial_states(mc::IntervalMarkovChain)
```

## Markov decision process
```@docs
IntervalMarkovDecisionProcess
transition_prob(mdp::IntervalMarkovDecisionProcess)
num_states(mdp::IntervalMarkovDecisionProcess)
initial_states(mdp::IntervalMarkovDecisionProcess)
actions(mdp::IntervalMarkovDecisionProcess)
num_choices(mdp::IntervalMarkovDecisionProcess)
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
