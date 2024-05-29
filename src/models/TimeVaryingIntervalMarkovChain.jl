"""
    TimeVaryingIntervalMarkovChain{P <: IntervalProbabilities, T <: Integer, VT <: AbstractVector{T}}

A type representing Time-varying Interval Markov Chains (IMC), which are Markov Chains with uncertainty in the form of intervals on
the transition probabilities. The time variablity must be finite.

Formally, let ``(S, S_0, \\{ \\bar{P}_t \\}_{t \\in T}, \\{ \\underbar{P}_t \\}_{t \\in T})`` be a time-varying interval Markov chain, where ``S`` is the set of states, ``S_0 \\subset S`` is a set of initial states,
and ``\\bar{P}_t : \\mathbb{R}^{|S| \\times |S|}`` and ``\\underbar{P}_t : \\mathbb{R}^{|S| \\times |S|}`` are the upper and lower bound transition probability matrices for time step ``t \\in T`` prespectively.
Then the `TimeVaryingIntervalMarkovChain` type is defined as follows: indices `1:num_states` are the states in ``S``,
`transition_prob` represents ``\\bar{P}_t`` and ``\\underbar{P}_t`` at time ``t``, and `initial_states` is the set of initial state ``S_0``.
If no initial states are specified, then the initial states are assumed to be all states in ``S``.

Note that for time-varying models, model checking is only enabled for finite time properties of equal length.

### Fields
- `transition_probs::Vector{P}`: interval on transition probabilities.
- `initial_states::VT`: initial states.
- `num_states::T`: number of states.

### Examples

```jldoctest
prob1 = IntervalProbabilities(;
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

prob2 = IntervalProbabilities(;
    lower = [
        0.2 0.1 0.0
        0.1 0.3 0.0
        0.0 0.5 1.0
    ],
    upper = [
        0.7 0.3 0.0
        0.6 0.5 0.0
        0.5 0.7 1.0
    ],
)

mc = TimeVaryingIntervalMarkovChain([prob1, prob2])
# or
initial_states = [1, 2, 3]
mc = TimeVaryingIntervalMarkovChain([prob1, prob2], initial_states)
```

"""
struct TimeVaryingIntervalMarkovChain{
    P <: IntervalProbabilities,
    T <: Integer,
    VT <: AbstractVector{T},
} <: TimeVaryingIntervalMarkovProcess{P}
    transition_probs::Vector{P}
    initial_states::VT
    num_states::T
end

function TimeVaryingIntervalMarkovChain(
    transition_probs::Vector{P},
    initial_states::VT,
) where {P <: IntervalProbabilities, T <: Integer, VT <: AbstractVector{T}}
    @assert !isempty(transition_probs) "The vector of transition probabilities must not be empty"

    num_states = checksize_imc!(first(transition_probs))

    for transition_prob in transition_probs
        num_states_t = checksize_imc!(transition_prob)
        @assert num_states_t == num_states "The number of states must be the same for all time steps"
    end

    num_states = T(num_states)

    return TimeVaryingIntervalMarkovChain(transition_probs, initial_states, num_states)
end

function TimeVaryingIntervalMarkovChain(
    transition_probs::Vector{P},
) where {P <: IntervalProbabilities}
    return TimeVaryingIntervalMarkovChain(
        transition_probs,
        all_initial_states(num_source(first(transition_probs))),
    )
end
