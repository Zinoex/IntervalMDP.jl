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
letters2alphabet(letters::AbstractVector{String})
alphabet2index(alphabet::AbstractVector{String})
transition(dfa::DFA)
alphabetptr(dfa::DFA)
initial_state(dfa::DFA)
accepting_states(dfa::DFA)
ProductIntervalMarkovDecisionProcessDFA
imdp(md::ProductIntervalMarkovDecisionProcessDFA)
automaton(md::ProductIntervalMarkovDecisionProcessDFA)
labelling_function(md::ProductIntervalMarkovDecisionProcessDFA)
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
count_mapping(map::AbstractArray)
mapping(labelling_func::LabellingFunction)
```

### Transition function for DFA
```@docs
TransitionFunction
transition(transition_func::TransitionFunction)
```