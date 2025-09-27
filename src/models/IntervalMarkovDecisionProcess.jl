function IntervalMarkovDecisionProcess(marginal::Marginal{<:IntervalAmbiguitySets}, initial_states::InitialStates = AllStates())
    state_vars = (Int32(num_target(marginal)),)
    action_vars = action_shape(marginal)
    source_dims = source_shape(marginal)
    transition = (marginal,)

    return FactoredRMDP(
        state_vars,
        action_vars,
        source_dims,
        transition,
        initial_states
    )
end

"""
    IntervalMarkovDecisionProcess(ambiguity_set::IntervalAmbiguitySets, num_actions::Integer, initial_states::InitialStates = AllStates())

A convenience constructor for a [`FactoredRobustMarkovDecisionProcess`](@ref) representing an interval Markov decision process,
as IMDPs are a subclass of fRMDPs, from a single [`IntervalAmbiguitySets`](@ref) object and a specified number of actions.

Formally, an IMDP ``M`` is a tuple ``M = (S, S_0, A, \\Gamma)``, where

- ``S`` is a finite set of states,
- ``S_0 \\subseteq S`` is a set of initial states,
- ``A`` is a finite set of actions,
- ```\\Gamma = \\{\\Gamma_{s,a}\\}_{s \\in S,a \\in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, where each ``\\Gamma_{s,a}`` is an _interval_ ambiguity set over ``S``.

### Example
```jldoctest
using IntervalMDP

prob1 = IntervalAmbiguitySets(;
    lower = [
        0    1/2
        1/10 3/10
        1/5  1/10
    ],
    upper = [
        1/2  7/10
        3/5  1/2
        7/10 3/10
    ],
)

prob2 = IntervalAmbiguitySets(;
    lower = [
        1/10 1/5
        1/5  3/10
        3/10 2/5
    ],
    upper = [
        3/5 3/5
        1/2 1/2
        2/5 2/5
    ],
)

prob3 = IntervalAmbiguitySets(;
    lower = Float64[
        0 0
        0 0
        1 1
    ],
    upper = Float64[
        0 0
        0 0
        1 1
    ]
)

initial_states = [1]
mdp = IntervalMarkovDecisionProcess([prob1, prob2, prob3], initial_states)

# output

FactoredRobustMarkovDecisionProcess
├─ 1 state variables with cardinality: (3,)
├─ 1 action variables with cardinality: (2,)
├─ Initial states: [1]
├─ Transition marginals:
│  └─ Marginal 1:
│     ├─ Conditional variables: states = (1,), actions = (1,)
│     └─ Ambiguity set type: Interval (dense, Matrix{Float64})
└─Inferred properties
   ├─Model type: Interval MDP
   ├─Number of states: 3
   ├─Number of actions: 2
   ├─Default model checking algorithm: Robust Value Iteration
   └─Default Bellman operator algorithm: O-Maximization
```

"""
function IntervalMarkovDecisionProcess(ambiguity_set::IntervalAmbiguitySets, num_actions::Integer, initial_states::InitialStates = AllStates())
    if num_sets(ambiguity_set) % num_actions != 0
        throw(ArgumentError("The number of sets in the ambiguity set must be a multiple of the number of actions."))
    end

    source_dims = (num_sets(ambiguity_set) ÷ num_actions,)
    action_vars = (num_actions,)
    marginal = Marginal(ambiguity_set, source_dims, action_vars)

    return IntervalMarkovDecisionProcess(marginal, initial_states)
end

"""
    IntervalMarkovDecisionProcess(ps::Vector{<:IntervalAmbiguitySets}, initial_states::InitialStates = AllStates())

A convenience constructor for a [`FactoredRobustMarkovDecisionProcess`](@ref) representing an interval Markov decision process
from a vector of [`IntervalAmbiguitySets`](@ref) objects, one for each state and with the same number of actions in each. 

### Example
```jldoctest
using IntervalMDP

prob = IntervalAmbiguitySets(;
    lower = [
        0    1/2  1/10 1/5  0 0
        1/10 3/10 1/5  3/10 0 0
        1/5  1/10 3/10 2/5  1 1
    ],
    upper = [
        1/2  7/10 3/5 2/5 0 0
        3/5  1/2  1/2 2/5 0 0
        7/10 3/10 2/5 2/5 1 1
    ],
)

num_actions = 2
initial_states = [1]
mdp = IntervalMarkovDecisionProcess(prob, num_actions, initial_states)

# output

FactoredRobustMarkovDecisionProcess
├─ 1 state variables with cardinality: (3,)
├─ 1 action variables with cardinality: (2,)
├─ Initial states: [1]
├─ Transition marginals:
│  └─ Marginal 1:
│     ├─ Conditional variables: states = (1,), actions = (1,)
│     └─ Ambiguity set type: Interval (dense, Matrix{Float64})
└─Inferred properties
   ├─Model type: Interval MDP
   ├─Number of states: 3
   ├─Number of actions: 2
   ├─Default model checking algorithm: Robust Value Iteration
   └─Default Bellman operator algorithm: O-Maximization
```

"""
function IntervalMarkovDecisionProcess(
    ps::Vector{<:IntervalAmbiguitySets{R, MR}},
    initial_states::InitialStates = AllStates(),
) where {R, MR <: AbstractMatrix{R}}
    marginal = interval_prob_hcat(ps)
    return IntervalMarkovDecisionProcess(marginal, initial_states)
end

function interval_prob_hcat(
    ps::Vector{<:IntervalAmbiguitySets{R, MR}},
) where {R, MR <: AbstractMatrix{R}}
    if length(ps) == 0
        throw(ArgumentError("Cannot concatenate an empty vector of IntervalAmbiguitySets."))
    end

    num_actions = num_sets(ps[1])
    for (i, p) in enumerate(ps)
        if num_sets(p) != num_actions
            throw(DimensionMismatch("All IntervalAmbiguitySets must have the same number of sets (actions). Expected $num_actions, was $(num_sets(p)) at index $i."))
        end
    end

    l = mapreduce(p -> p.lower, hcat, ps)
    g = mapreduce(p -> p.gap, hcat, ps)

    ambiguity_set = IntervalAmbiguitySets(l, g)

    source_dims = (length(ps),)
    action_vars = (num_actions,)
    marginal = Marginal(ambiguity_set, source_dims, action_vars)

    return marginal
end