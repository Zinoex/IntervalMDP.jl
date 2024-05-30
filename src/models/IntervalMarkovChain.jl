"""
    IntervalMarkovChain{P <: IntervalProbabilities, T <: Integer, VT <: AbstractVector{T}}

A type representing Interval Markov Chains (IMC), which are Markov Chains with uncertainty in the form of intervals on
the transition probabilities.

Formally, let ``(S, S_0, \\bar{P}, \\underbar{P})`` be an interval Markov chain, where ``S`` is the set of states, ``S_0 \\subset S`` is a set of initial states,
and ``\\bar{P} : \\mathbb{R}^{|S| \\times |S|}`` and ``\\underbar{P} : \\mathbb{R}^{|S| \\times |S|}`` are the upper and lower bound transition probability matrices prespectively.
Then the `IntervalMarkovChain` type is defined as follows: indices `1:num_states` are the states in ``S``,
`transition_prob` represents ``\\bar{P}`` and ``\\underbar{P}``, and `initial_states` is the set of initial state ``S_0``.
If no initial states are specified, then the initial states are assumed to be all states in ``S``.

### Fields
- `transition_prob::P`: interval on transition probabilities.
- `initial_states::VT`: initial states.
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

mc = IntervalMarkovChain(prob)
# or
initial_states = [1, 2, 3]
mc = IntervalMarkovChain(prob, initial_states)
```

"""
struct IntervalMarkovChain{
    P <: IntervalProbabilities,
    T <: Integer,
    VT <: AbstractVector{T},
} <: IntervalMarkovProcess{P}
    transition_prob::P
    initial_states::VT
    num_states::T
end

function IntervalMarkovChain(
    transition_prob::P,
    initial_states::VT,
) where {P <: IntervalProbabilities, T <: Integer, VT <: AbstractVector{T}}
    num_states = checksize_imc!(transition_prob)
    num_states = T(num_states)

    return IntervalMarkovChain(transition_prob, initial_states, num_states)
end

function IntervalMarkovChain(transition_prob::P) where {P <: IntervalProbabilities}
    return IntervalMarkovChain(
        transition_prob,
        all_initial_states(num_source(transition_prob)),
    )
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
    initial_states(s::IntervalMarkovChain)

Return the initial states.
"""
initial_states(s::IntervalMarkovChain) = s.initial_states

"""
    num_states(s::IntervalMarkovChain)

Return the number of states.
"""
num_states(s::IntervalMarkovChain) = s.num_states
