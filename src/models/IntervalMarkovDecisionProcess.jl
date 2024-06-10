"""
    IntervalMarkovDecisionProcess{
        P <: IntervalProbabilities,
        VT <: AbstractVector{Int32},
        VI <: Union{AllStates, AbstractVector}
    }

A type representing (stationary) Interval Markov Decision Processes (IMDP), which are Markov Decision Processes with uncertainty in the form of intervals on
the transition probabilities.

Formally, let ``(S, S_0, A, \\bar{P}, \\underbar{P})`` be an interval Markov decision processes, where ``S`` is the set of states, ``S_0 \\subset S`` is the set of initial states,
``A`` is the set of actions, and ``\\bar{P} : A \\to \\mathbb{R}^{|S| \\times |S|}`` and ``\\underbar{P} : A \\to \\mathbb{R}^{|S| \\times |S|}`` are functions
representing the upper and lower bound transition probability matrices prespectively for each action. Then the ```IntervalMarkovDecisionProcess``` type is
defined as follows: indices `1:num_states` are the states in ``S``, `transition_prob` represents ``\\bar{P}`` and ``\\underbar{P}``, actions are 
implicitly defined by `stateptr` (e.g. if `stateptr[3] == 4` and `stateptr[4] == 7` then the actions available to state 3 are `[4, 5, 6]`), 
and `initial_states` is the set of initial states ``S_0``. If no initial states are specified, then the initial states are assumed to be all states in ``S``.

### Fields
- `transition_prob::P`: interval on transition probabilities where columns represent source/action pairs and rows represent target states.
- `stateptr::VT`: pointer to the start of each source state in `transition_prob` (i.e. `transition_prob[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition
    probability matrix for source state `j`) in the style of colptr for sparse matrices in CSC format.
- `initial_states::VI`: initial states.
- `num_states::Int32`: number of states.

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
initial_states = [1]

mdp = IntervalMarkovDecisionProcess(transition_probs, stateptr, initial_states)
```

There is also a constructor for `IntervalMarkovDecisionProcess` where the transition probabilities are given as a list of 
transition probabilities for each source state.

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

transition_probs = [prob1, prob2, prob3]
initial_states = [1]

mdp = IntervalMarkovDecisionProcess(transition_probs, initial_states)
```

"""
struct IntervalMarkovDecisionProcess{
    P <: IntervalProbabilities,
    VT <: AbstractVector{Int32},
    VI <: InitialStates,
} <: StationaryIntervalMarkovProcess{P}
    transition_prob::P
    stateptr::VT
    initial_states::VI
    num_states::Int32
end

function IntervalMarkovDecisionProcess(
    transition_prob::IntervalProbabilities,
    stateptr::AbstractVector{Int32},
    initial_states::InitialStates = AllStates(),
)
    num_states = checksize_imdp!(transition_prob, stateptr)

    return IntervalMarkovDecisionProcess(
        transition_prob,
        stateptr,
        initial_states,
        num_states,
    )
end

function IntervalMarkovDecisionProcess(
    transition_probs::Vector{<:IntervalProbabilities},
    initial_states::InitialStates = AllStates(),
)
    transition_prob, stateptr = interval_prob_hcat(transition_probs)

    return IntervalMarkovDecisionProcess(transition_prob, stateptr, initial_states)
end

function checksize_imdp!(p::IntervalProbabilities, stateptr::AbstractVector{Int32})
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

    return Int32(num_states)
end

"""
    stateptr(mdp::IntervalMarkovDecisionProcess)

Return the state pointer of the Interval Markov Decision Process. The state pointer is a vector of integers where the `i`-th element
is the index of the first element of the `i`-th state in the transition probability matrix. 
I.e. `transition_prob[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition probability matrix for source state `j`.
"""
stateptr(mdp::IntervalMarkovDecisionProcess) = mdp.stateptr

max_actions(mdp::IntervalMarkovDecisionProcess) = stateptr(mdp) |> diff |> maximum

"""
    tomarkovchain(mdp::IntervalMarkovDecisionProcess, strategy::AbstractVector)

Extract an Interval Markov Chain (IMC) from an Interval Markov Decision Process under a stationary strategy. The returned type remains
an IntervalMarkovDecisionProcess with only one action per state. The extracted IMC is stationary.
"""
function tomarkovchain(mdp::IntervalMarkovDecisionProcess, strategy::AbstractVector)
    probs = transition_prob(mdp)
    new_probs = probs[strategy]

    istates = initial_states(mdp)
    sptr = UnitRange{Int32}(1, num_states(mdp) + 1)

    return IntervalMarkovDecisionProcess(new_probs, sptr, istates)
end

"""
    tomarkovchain(mdp::IntervalMarkovDecisionProcess, strategy::AbstractVector{<:AbstractVector})

Extract an Interval Markov Chain (IMC) from an Interval Markov Decision Process under a time-varying strategy. The returned type is
a TimeVaryingIntervalMarkovDecisionProcess with only one action per state per time-step. The extracted IMC is time-varying.
"""
function tomarkovchain(
    mdp::IntervalMarkovDecisionProcess,
    strategy::AbstractVector{<:AbstractVector},
)
    probs = transition_prob(mdp)
    new_probs = Vector{typeof(probs)}(undef, length(strategy))

    for (t, strategy_step) in enumerate(strategy)
        new_probs[t] = probs[strategy_step]
    end

    sptr = UnitRange{Int32}(1, num_states(mdp) + 1)
    istates = initial_states(mdp)

    return TimeVaryingIntervalMarkovDecisionProcess(new_probs, sptr, istates)
end
