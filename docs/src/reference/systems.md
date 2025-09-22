# System representation

```@docs
IntervalMarkovProcess
num_states
num_actions
initial_states(mp::IntervalMarkovProcess)
AllStates
```

## [Factored RMDPs](@id api-frmdp)
```@docs
FactoredRobustMarkovDecisionProcess
state_variables(s::FactoredRobustMarkovDecisionProcess)
action_variables(s::FactoredRobustMarkovDecisionProcess)
marginals(s::FactoredRobustMarkovDecisionProcess)
```

## Convenience constructors for subclasses of fRMDPs
```@docs
IntervalMarkovChain
IntervalMarkovDecisionProcess
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

## Probability representation
```@docs
Marginal
ambiguity_sets(m::Marginal)
state_variables(m::Marginal)
action_variables(m::Marginal)
source_shape(m::Marginal)
action_shape(m::Marginal)
num_target(m::Marginal)
getindex(p::Marginal, action, source)

num_sets
support
```

### Interval ambiguity sets
```@docs
IntervalAmbiguitySets
num_sets(p::IntervalAmbiguitySets)
num_target(p::IntervalAmbiguitySets)
lower
upper
gap
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