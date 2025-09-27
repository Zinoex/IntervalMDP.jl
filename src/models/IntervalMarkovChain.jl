function IntervalMarkovChain(marginal::Marginal{<:IntervalAmbiguitySets}, initial_states=AllStates())
    state_vars = (Int32(num_target(marginal)),)
    source_dims = source_shape(marginal)

    if action_shape(marginal) != (1,)
        throw(DimensionMismatch("The action shape of the marginal must be (1,) for an IntervalMarkovChain. Got $(action_shape(marginal))."))
    end

    action_vars = (Int32(1),)

    return FactoredRMDP( # wrap in a FactoredRMDP for consistency
        state_vars,
        action_vars,
        source_dims,
        (marginal,),
        initial_states,
    )
end

"""
    IntervalMarkovChain(ambiguity_set::IntervalAmbiguitySets, initial_states=AllStates())

A convenience constructor for a [`FactoredRobustMarkovDecisionProcess`](@ref) representing an interval Markov chain,
as IMCs are a subclass of fRMDPs, from a single [`IntervalAmbiguitySets`](@ref) object.

Formally, an IMC ``M`` is a tuple ``M = (S, S_0, \\Gamma)``, where

- ``S`` is a finite set of states,
- ``S_0 \\subseteq S`` is a set of initial states,
- ``\\Gamma = \\{\\Gamma_{s}\\}_{s \\in S}`` is a set of ambiguity sets for source state ``s``,
    where each ``\\Gamma_{s}`` is an _interval_ ambiguity set over ``S``.

Notice also that an IMC is an [`IntervalMarkovDecisionProcess`](@ref) with a single action.

### Example
```jldoctest
using IntervalMDP

prob = IntervalAmbiguitySets(;
    lower = [
        0     1/2   0
        1/10  3/10  0
        1/5   1/10  1
    ],
    upper = [
        1/2   7/10  0
        3/5   1/2   0
        7/10  3/10  1
    ],
)

initial_states = [1]
mc = IntervalMarkovChain(prob, initial_states)

# output

FactoredRobustMarkovDecisionProcess
├─ 1 state variables with cardinality: (3,)
├─ 1 action variables with cardinality: (1,)
├─ Initial states: [1]
├─ Transition marginals:
│  └─ Marginal 1:
│     ├─ Conditional variables: states = (1,), actions = (1,)
│     └─ Ambiguity set type: Interval (dense, Matrix{Float64})
└─Inferred properties
   ├─Model type: Interval MDP
   ├─Number of states: 3
   ├─Number of actions: 1
   ├─Default model checking algorithm: Robust Value Iteration
   └─Default Bellman operator algorithm: O-Maximization
```
"""
function IntervalMarkovChain(ambiguity_set::IntervalAmbiguitySets, initial_states=AllStates())
    source_dims = (num_sets(ambiguity_set),)
    action_vars = (1,)
    marginal = Marginal(ambiguity_set, source_dims, action_vars)

    return IntervalMarkovChain(marginal, initial_states)
end