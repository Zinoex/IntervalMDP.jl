
"""
    IntervalMarkovChain{P <: IntervalProbabilities, VI <: Union{AllStates, AbstractVector}}

A type representing (stationary) Interval Markov Chains (IMC), which are Markov Chains with uncertainty in the form of intervals on
the transition probabilities. The stationarity assumption is that the transition probabilities are time-invariant.

Formally, let ``(S, S_0, \\bar{P}, \\underbar{P})`` be an stationary interval Markov chain, where ``S`` is the set of states, ``S_0 \\subset S`` is a set of initial states,
and ``\\bar{P} : \\mathbb{R}^{|S| \\times |S|}`` and ``\\underbar{P} : \\mathbb{R}^{|S| \\times |S|}`` are the upper and lower bound transition probability matrices prespectively.
Then the `IntervalMarkovChain` type is defined as follows: indices `1:num_states` are the states in ``S``,
`transition_prob` represents ``\\bar{P}`` and ``\\underbar{P}``, and `initial_states` is the set of initial state ``S_0``.
If no initial states are specified, then the initial states are assumed to be all states in ``S``.

### Fields
- `transition_prob::P`: interval on transition probabilities.
- `initial_states::VI`: initial states.
- `num_states::Int32`: number of states.

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
    VI <: InitialStates,
} <: StationaryIntervalMarkovProcess{P}
    transition_prob::P
    initial_states::VI
    num_states::Int32
end

function IntervalMarkovChain(
    transition_prob::P,
    initial_states::VI,
) where {P <: IntervalProbabilities, VI <: InitialStates}
    num_states = checksize_imc!(transition_prob)

    return IntervalMarkovChain(transition_prob, initial_states, num_states)
end

function IntervalMarkovChain(transition_prob::P) where {P <: IntervalProbabilities}
    return IntervalMarkovChain(
        transition_prob,
        AllStates(),
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

    # Store in Int32 since we don't expect to have more than 2^31 states
    return Int32(num_states)
end