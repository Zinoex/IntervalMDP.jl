"""
    TimeVaryingIntervalMarkovDecisionProcess{P <: IntervalProbabilities, VT <: AbstractVector{Int32}, VI <: Union{AllStates, AbstractVector}}

A type representing Time-varying Interval Markov Decision Processes (IMDP), which are Markov Decision Processes with uncertainty in the form of intervals on
the transition probabilities varying over time. The time variablity must be finite.

Formally, let ``(S, S_0, A, \\{ \\bar{P}_t \\}_{t \\in T}, \\{ \\underbar{P}_t \\}_{t \\in T})`` be a time-varying interval Markov chain, where ``S`` is the set of states, 
``S_0 \\subset S`` is a set of initial states, ``A`` is the set of actions, and ``\\bar{P}_t : A \\to \\mathbb{R}^{|S| \\times |S|}`` and ``\\underbar{P}_t : A \\to \\mathbb{R}^{|S| \\times |S|}``
are the upper and lower bound transition probability matrices for time step ``t \\in T`` prespectively. Then the `TimeVaryingIntervalMarkovDecisionProcess` type is defined as follows:
indices `1:num_states` are the states in ``S``, `transition_probs[t]` represents ``\\bar{P}_t`` and ``\\underbar{P}_t`` at time ``t``, actions are 
implicitly defined by `stateptr` (e.g. if `stateptr[3] == 4` and `stateptr[4] == 7` then the actions available to state 3 are `[4, 5, 6]`), and `initial_states` is the set of initial state ``S_0``.
If no initial states are specified, then the initial states are assumed to be all states in ``S``.

Note that for time-varying models, model checking is only enabled for finite time properties of equal length.

### Fields
- `transition_probs::Vector{P}`: interval on transition probabilities.
- `stateptr::VT`: pointer to the start of each source state in `transition_prob` (i.e. `transition_prob[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition
    probability matrix for source state `j`) in the style of colptr for sparse matrices in CSC format.
- `initial_states::VI`: initial states.
- `num_states::Int32`: number of states.

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

mc = TimeVaryingIntervalMarkovDecisionProcess([prob1, prob2])
# or
initial_states = [1, 2, 3]
mc = TimeVaryingIntervalMarkovDecisionProcess([prob1, prob2], initial_states)
```

"""
struct TimeVaryingIntervalMarkovDecisionProcess{
    P <: IntervalProbabilities,
    VT <: AbstractVector{Int32},
    VI <: InitialStates,
} <: TimeVaryingIntervalMarkovProcess
    transition_probs::Vector{P}
    stateptr::VT
    initial_states::VI
    num_states::Int32
end

function TimeVaryingIntervalMarkovDecisionProcess(
    transition_probs::Vector{<:IntervalProbabilities},
    stateptr::AbstractVector{Int32},
    initial_states::InitialStates = AllStates(),
)
    @assert !isempty(transition_probs) "The vector of transition probabilities must not be empty"

    num_states = checksize_imdp!(first(transition_probs), stateptr)

    for transition_prob in transition_probs
        num_states_t = checksize_imdp!(transition_prob, stateptr)
        @assert num_states_t == num_states "The number of states must be the same for all time steps"
    end

    return TimeVaryingIntervalMarkovDecisionProcess(
        transition_probs,
        stateptr,
        initial_states,
        num_states,
    )
end

"""
    TimeVaryingIntervalMarkovChain(transition_probs::Vector{<:IntervalProbabilities}, initial_states::InitialStates = AllStates())

Construct an Interval Markov Chain from a sequence of square matrix pairs of interval transition probabilities. The initial states are optional and if not specified,
all states are assumed to be initial states. The number of states is inferred from the size of the transition probability matrix.

The returned type is an `TimeVaryingIntervalMarkovDecisionProcess` with only one action per state (i.e. `stateptr[j + 1] - stateptr[j] == 1` for all `j`).
This is done to unify the interface for value iteration.
"""
function TimeVaryingIntervalMarkovChain(
    transition_probs::Vector{<:IntervalProbabilities},
    initial_states::InitialStates = AllStates(),
)
    stateptr = UnitRange{Int32}(1, num_source(first(transition_probs)) + 1)
    return TimeVaryingIntervalMarkovDecisionProcess(
        transition_probs,
        stateptr,
        initial_states,
    )
end

"""
    stateptr(mdp::TimeVaryingIntervalMarkovDecisionProcess)

Return the state pointer of the Time-Varying Interval Markov Decision Process. The state pointer is a vector of integers where the `i`-th element
is the index of the first element of the `i`-th state in the transition probability matrix. 
I.e. `transition_prob[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition probability matrix for source state `j`.
"""
stateptr(mdp::TimeVaryingIntervalMarkovDecisionProcess) = mdp.stateptr

max_actions(mdp::TimeVaryingIntervalMarkovDecisionProcess) = maxdiff(stateptr(mdp))
transition_matrix_type(mp::TimeVaryingIntervalMarkovDecisionProcess) =
    typeof(gap(first(transition_probs(mp))))

"""
    tomarkovchain(mdp::TimeVaryingIntervalMarkovDecisionProcess, strategy::AbstractVector{<:AbstractVector})

Extract an Interval Markov Chain (IMC) from a Time-Varying Interval Markov Decision Process under a time-varying strategy. The length of 
`strategy` must be equal to the time length of the time-varying model. The returned type is a `TimeVaryingIntervalMarkovDecisionProcess`
with only one action per state per time-step. The extracted IMC is time-varying.
"""
function tomarkovchain(
    mdp::TimeVaryingIntervalMarkovDecisionProcess,
    strategy::AbstractVector{<:AbstractVector},
)
    if length(strategy) != time_length(mdp)
        throw(
            ArgumentError(
                "The length of the strategy ($(length(strategy))) must be equal to the time length of the time-varying model ($(time_length(mdp)))",
            ),
        )
    end

    new_probs = Vector{typeof(probs)}(undef, length(strategy))

    for (t, strategy_step) in enumerate(strategy)
        new_probs[t] = transition_prob(mdp, t)[strategy_step]
    end

    istates = initial_states(mdp)

    return TimeVaryingIntervalMarkovChain(new_probs, istates)
end
