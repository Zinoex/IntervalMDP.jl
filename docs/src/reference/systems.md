# System representation

```@docs
IntervalMarkovProcess
num_states(s::IntervalMarkovProcess)
initial_states(s::IntervalMarkovProcess)
AllStates
transition_prob(mp::IntervalMarkovProcess)
IntervalMarkovDecisionProcess
IntervalMarkovChain
stateptr(mdp::IntervalMarkovDecisionProcess)
OrthogonalIntervalMarkovDecisionProcess
OrthogonalIntervalMarkovChain
stateptr(mdp::OrthogonalIntervalMarkovDecisionProcess)
MixtureIntervalMarkovDecisionProcess
MixtureIntervalMarkovChain
stateptr(mdp::MixtureIntervalMarkovDecisionProcess)
DFA
num_states(dfa::DFA)
num_labels(dfa::DFA)
transition(dfa::DFA)
labelmap(dfa::DFA)
initial_state(dfa::DFA)
accepting_states(dfa::DFA)
ProductProcess
markov_process(proc::ProductProcess)
automaton(proc::ProductProcess)
labelling_function(proc::ProductProcess)
```

## Probability representation

### Interval ambiguity sets
```@docs
IntervalProbabilities
lower(p::IntervalProbabilities)
lower(p::IntervalProbabilities, i, j)
upper(p::IntervalProbabilities)
upper(p::IntervalProbabilities, i, j)
gap(p::IntervalProbabilities)
gap(p::IntervalProbabilities, i, j)
sum_lower(p::IntervalProbabilities)
sum_lower(p::IntervalProbabilities, j)
num_source(p::IntervalProbabilities)
num_target(p::IntervalProbabilities)
axes_source(p::IntervalProbabilities)
```

### Marginal interval ambiguity sets
```@docs
OrthogonalIntervalProbabilities
lower(p::OrthogonalIntervalProbabilities, l)
lower(p::OrthogonalIntervalProbabilities, l, i, j)
upper(p::OrthogonalIntervalProbabilities, l)
upper(p::OrthogonalIntervalProbabilities, l, i, j)
gap(p::OrthogonalIntervalProbabilities, l)
gap(p::OrthogonalIntervalProbabilities, l, i, j)
sum_lower(p::OrthogonalIntervalProbabilities, l)
sum_lower(p::OrthogonalIntervalProbabilities, l, j)
num_source(p::OrthogonalIntervalProbabilities)
num_target(p::OrthogonalIntervalProbabilities)
axes_source(p::OrthogonalIntervalProbabilities)
```

### Mixtures of marginal interval ambiguity sets
```@docs
MixtureIntervalProbabilities
num_source(p::MixtureIntervalProbabilities)
num_target(p::MixtureIntervalProbabilities)
axes_source(p::MixtureIntervalProbabilities)
mixture_probs
weighting_probs
```

### Labelling of IMDP states to Automaton alphabet
```@docs
LabellingFunction
mapping(labelling_func::LabellingFunction)
num_labels(labelling_func::LabellingFunction)
```

### Transition function for DFA
```@docs
TransitionFunction
transition(transition_func::TransitionFunction)
num_states(tf::TransitionFunction)
num_labels(tf::TransitionFunction)
```