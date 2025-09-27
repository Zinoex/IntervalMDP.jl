# System representation

```@docs
num_states
num_actions
initial_states
AllStates
```

## [Factored RMDPs](@id api-frmdp)
```@docs
FactoredRobustMarkovDecisionProcess
state_values(s::FactoredRobustMarkovDecisionProcess)
action_values(s::FactoredRobustMarkovDecisionProcess)
marginals(s::FactoredRobustMarkovDecisionProcess)
```

### Convenience constructors for subclasses of fRMDPs
```@docs
IntervalMarkovChain
IntervalMarkovDecisionProcess
```

## Probability representation
```@docs
Marginal
ambiguity_sets(m::Marginal)
state_variables(m::Marginal)
action_variables(m::Marginal)
source_shape(m::Marginal)
action_shape(m::Marginal)
getindex(p::Marginal, action, source)

num_sets
num_target
support
```

### Interval ambiguity sets
```@docs
IntervalAmbiguitySets
lower
upper
gap
```

## Deterministic Finite Automaton (DFA)
```@docs
DFA
num_states(dfa::DFA)
num_labels(dfa::DFA)
transition(dfa::DFA)
labelmap(dfa::DFA)
initial_state(dfa::DFA)
ProductProcess
markov_process(proc::ProductProcess)
automaton(proc::ProductProcess)
labelling_function(proc::ProductProcess)
```

### Transition function for DFA
```@docs
TransitionFunction
transition(transition_func::TransitionFunction)
num_states(tf::TransitionFunction)
num_labels(tf::TransitionFunction)
```

### Labelling of IMDP states to Automaton alphabet
```@docs
LabellingFunction
mapping(labelling_func::LabellingFunction)
num_labels(labelling_func::LabellingFunction)
```