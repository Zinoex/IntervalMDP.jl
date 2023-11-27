
"""
    IntervalMarkovProcess

An abstract type for interval Markov processes including [`IntervalMarkovChain`](@ref) and [`IntervalMarkovDecisionProcess`](@ref).
"""
abstract type IntervalMarkovProcess end

"""
    IntervalMarkovChain{P <: IntervalProbabilities, T <: Integer}

A type representing Interval Markov Chains (IMC), which are Markov Chains with uncertainty in the form of intervals on
the transition probabilities.

Formally, let ``(S, s_0, \\bar{P}, \\underbar{P})`` be an interval Markov chain, where ``S`` is the set of states, ``s_0`` is the initial state,
and ``\\bar{P} : \\mathbb{R}^{|S| \\times |S|}`` and ``\\underbar{P} : \\mathbb{R}^{|S| \\times |S|}`` are the upper and lower bound transition probability matrices prespectively.
Then the `IntervalMarkovChain` type is defined as follows: indices `1:num_states` are the states in ``S``,
`transition_prob` represents ``\\bar{P}`` and ``\\underbar{P}``, and `initial_state` is the initial state ``s_0``.

### Fields
- `transition_prob::P`: interval on transition probabilities.
- `initial_state::T`: initial state.
- `num_states::T`: number of states.

### Examples

```jldoctest
prob = IntervalProbabilities(;
    lower = [
        0.0 0.5 0.0
        0.1 0.3 0.0
        0.2 0.1 1.0
    ],
    upper = [
        0.5 0.7 0.0
        0.6 0.5 0.0
        0.7 0.3 1.0
    ],
)

mc = IntervalMarkovChain(prob, 1)
```

"""
struct IntervalMarkovChain{P <: IntervalProbabilities, T <: Integer} <:
       IntervalMarkovProcess
    transition_prob::P
    initial_state::T
    num_states::T
end

function IntervalMarkovChain(
    transition_prob::P,
    initial_state::T,
) where {P <: IntervalProbabilities, T <: Integer}
    num_states = checksize_imc!(transition_prob)
    num_states = T(num_states)

    return IntervalMarkovChain(transition_prob, initial_state, num_states)
end

function checksize_imc!(p::IntervalProbabilities)
    num_states = num_source(p)
    if num_target(p) != num_states
        throw(
            DimensionMismatch(
                "The number of transition probabilities in the matrix is not equal to the number of states in the problem",
            ),
        )
    end

    return num_states
end

"""
    transition_prob(s::IntervalMarkovChain)

Return the interval on transition probabilities.
"""
transition_prob(s::IntervalMarkovChain) = s.transition_prob

"""
    initial_state(s::IntervalMarkovChain)

Return the initial state.
"""
initial_state(s::IntervalMarkovChain) = s.initial_state

"""
    num_states(s::IntervalMarkovChain)

Return the number of states.
"""
num_states(s::IntervalMarkovChain) = s.num_states

"""
    IntervalMarkovDecisionProcess{
        P <: IntervalProbabilities,
        T <: Integer,
        VT <: AbstractVector{T},
        VA <: AbstractVector,
    }

A type representing Interval Markov Decision Processes (IMDP), which are Markov Decision Processes with uncertainty in the form of intervals on
the transition probabilities.

Formally, let ``(S, s_0, A, \\bar{P}, \\underbar{P})`` be an interval Markov decision processes, where ``S`` is the set of states, ``s_0`` is the initial state,
``A`` is the set of actions, and ``\\bar{P} : A \\to \\mathbb{R}^{|S| \\times |S|}`` and ``\\underbar{P} : A \\to \\mathbb{R}^{|S| \\times |S|}`` are functions
representing the upper and lower bound transition probability matrices prespectively for each action. Then the ```IntervalMarkovDecisionProcess``` type is
defined as follows: indices `1:num_states` are the states in ``S``, `transition_prob` represents ``\\bar{P}`` and ``\\underbar{P}``,
`action_vals` contains the actions available in each state, and `initial_state` is the initial state ``s_0``.

### Fields
- `transition_prob::P`: interval on transition probabilities where columns represent source/action pairs and rows represent target states.
- `stateptr::VT`: pointer to the start of each source state in `transition_prob` (i.e. `transition_prob[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition
    probability matrix for source state `j`) in the style of colptr for sparse matrices in CSC format.
- `action_vals::VA`: actions available in each state. Can be any eltype.
- `initial_state::T`: initial state.
- `num_states::T`: number of states.

### Examples

```jldoctest
transition_probs = IntervalProbabilities(;
    lower = [
        0.0 0.5 0.1 0.2 0.0
        0.1 0.3 0.2 0.3 0.0
        0.2 0.1 0.3 0.4 1.0
    ],
    upper = [
        0.5 0.7 0.6 0.6 0.0
        0.6 0.5 0.5 0.5 0.0
        0.7 0.3 0.4 0.4 1.0
    ],
)

stateptr = [1, 3, 5, 6]
actions = ["a1", "a2", "a1", "a2", "sinking"]
initial_state = 1

mdp = IntervalMarkovDecisionProcess(transition_probs, stateptr, actions, initial_state)
```

There is also a constructor for `IntervalMarkovDecisionProcess` where the transition probabilities are given as a list of 
mappings from actions to transition probabilities for each source state.

```jldoctest
prob1 = IntervalProbabilities(;
    lower = [
        0.0 0.5
        0.1 0.3
        0.2 0.1
    ],
    upper = [
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ],
)

prob2 = IntervalProbabilities(;
    lower = [
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ],
    upper = [
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ],
)

prob3 = IntervalProbabilities(;
    lower = [0.0; 0.0; 1.0],
    upper = [0.0; 0.0; 1.0]
)

transition_probs = [["a1", "a2"] => prob1, ["a1", "a2"] => prob2, ["sinking"] => prob3]
initial_state = 1

mdp = IntervalMarkovDecisionProcess(transition_probs, initial_state)
```

"""
struct IntervalMarkovDecisionProcess{
    P <: IntervalProbabilities,
    T <: Integer,
    VT <: AbstractVector{T},
    VA <: AbstractVector,
} <: IntervalMarkovProcess
    transition_prob::P
    stateptr::VT
    action_vals::VA
    initial_state::T
    num_states::T
end

function IntervalMarkovDecisionProcess(
    transition_prob::P,
    stateptr::VT,
    action_vals::VA,
    initial_state::T,
) where {
    P <: IntervalProbabilities,
    T <: Integer,
    VT <: AbstractVector{T},
    VA <: AbstractVector,
}
    num_states = checksize_imdp!(transition_prob, stateptr)
    num_states = T(num_states)

    return IntervalMarkovDecisionProcess(
        transition_prob,
        stateptr,
        action_vals,
        initial_state,
        num_states,
    )
end

function IntervalMarkovDecisionProcess(
    transition_probs::Vector{P},
    action_vals::VA,
    initial_state::T,
) where {P <: IntervalProbabilities, T <: Integer, VA <: AbstractVector}
    transition_prob, stateptr = interval_prob_hcat(T, transition_probs)

    return IntervalMarkovDecisionProcess(
        transition_prob,
        stateptr,
        action_vals,
        initial_state,
    )
end

function IntervalMarkovDecisionProcess(
    transition_probs::Vector{Pair{VA, P}},
    initial_state::T,
) where {P <: IntervalProbabilities, T <: Integer, VA <: AbstractVector}
    action_vals = mapreduce(first, vcat, transition_probs)
    transition_probs = map(x -> x[2], transition_probs)

    return IntervalMarkovDecisionProcess(transition_probs, action_vals, initial_state)
end

function checksize_imdp!(p::IntervalProbabilities, stateptr)
    num_states = length(stateptr) - 1

    num_actions_per_state = diff(stateptr)
    @assert all(num_actions_per_state .> 0) "The number of actions per state must be positive"

    if num_target(p) != num_states
        throw(
            DimensionMismatch(
                "The number of transition probabilities in the matrix is not equal to the number of states in the problem",
            ),
        )
    end

    return num_states
end

"""
    transition_prob(s::IntervalMarkovDecisionProcess)

Return the interval on transition probabilities.
"""
transition_prob(s::IntervalMarkovDecisionProcess) = s.transition_prob

"""
    actions(s::IntervalMarkovDecisionProcess)

Return a vector of actions (choices in PRISM terminology).
"""
actions(s::IntervalMarkovDecisionProcess) = s.action_vals

"""
    num_choices(s::IntervalMarkovDecisionProcess)

Return the sum of the number of actions available in each state ``\\sum_{j} \\mathrm{num_actions}(s_j)``.
"""
num_choices(s::IntervalMarkovDecisionProcess) = length(actions(s))

"""
    initial_state(s::IntervalMarkovDecisionProcess)

Return the initial state.
"""
initial_state(s::IntervalMarkovDecisionProcess) = s.initial_state

"""
    num_states(s::IntervalMarkovDecisionProcess)

Return the number of states.
"""
num_states(s::IntervalMarkovDecisionProcess) = s.num_states

stateptr(s::IntervalMarkovDecisionProcess) = s.stateptr
