"""
    TimeVaryingDeterministicMarkovDecisionProcess{
        P <: Transitions,
        VT <: AbstractVector{Int32},
        VI <: Union{AllStates, AbstractVector}
    }

A type representing (time-varying) Interval Markov Decision Processes (IMDP), where the lower transition probability bounds are all zeros
and the upper bounds are ones for targets where source/action pairs can transport the system to.

Formally, let ``(S, S_0, A, \\{ \\bar{P}_t \\}_{t \\in T}, \\{ \\underbar{P}_t \\}_{t \\in T})`` be a time-varying interval Markov decision processes, where ``S`` is the set of states, ``S_0 \\subset S`` is the set of initial states,
``A`` is the set of actions, and ``\\bar{P}_t : A \\to \\mathbb{R}^{|S| \\times |S|}`` and ``\\underbar{P} : A \\to \\mathbb{R}^{|S| \\times |S|}`` are functions
representing the upper and lower bound transition probability matrices prespectively for each action. Then the ```DeterministicMarkovDecisionProcess``` type is
defined as follows: indices `1:num_states` are the states in ``S``, `transition_prob` represents the ones in ``\\bar{P}_t`` and ``\\underbar{P}_t = 0``  at time ``t`, actions are 
implicitly defined by `stateptr` (e.g. if `stateptr[3] == 4` and `stateptr[4] == 7` then the actions available to state 3 are `[4, 5, 6]`), 
and `initial_states` is the set of initial states ``S_0``. If no initial states are specified, then the initial states are assumed to be all states in ``S``.

### Fields
- `transition_prob::Vector{P}`: a vector of a CSC-based format where columns represent source/action pairs and rows represent target states and non-zero entries represent a transition probability of 1.
- `stateptr::VT`: pointer to the start of each source state in `transition_prob` (i.e. `transition_prob[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition matrix for source state `j`) in the style of colptr for sparse matrices in CSC format.
- `initial_states::VI`: initial states.
- `num_states::Int32`: number of states.

### Examples
# TODO: Update example
```jldoctest
transitions = Transitions(
    [1, 3, 5, 6],
    [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    (6, 6),
    5,
)

stateptr = [1, 3, 5, 6]
initial_states = [1]

mdp = TimeVaryingDeterministicMarkovDecisionProcess(transitions, stateptr, initial_states)
```
"""
struct TimeVaryingDeterministicMarkovDecisionProcess{
    P <: Transitions,
    VT <: AbstractVector{Int32},
    VI <: InitialStates,
} <: TimeVaryingIntervalMarkovProcess
    transition_prob::P
    stateptr::VT
    initial_states::VI
    num_states::Int32
end

function TimeVaryingDeterministicMarkovDecisionProcess(
    transition_probs::Vector{Transitions},
    stateptr::AbstractVector{Int32},
    initial_states::InitialStates = AllStates(),
)
    @assert !isempty(transition_probs) "The vector of transition probabilities must not be empty"

    num_states = checksize_imdp!(first(transition_probs), stateptr)

    for transition_prob in transition_probs
        num_states_t = checksize_imdp!(transition_prob, stateptr)
        @assert num_states_t == num_states "The number of states must be the same for all time steps"
    end

    return TimeVaryingDeterministicMarkovDecisionProcess(
        transition_prob,
        stateptr,
        initial_states,
        num_states,
    )
end

# TODO: Document
function TimeVaryingDeterministicMarkovChain(
    transition_prob::Vector{<:Transitions},
    initial_states::InitialStates = AllStates(),
)
    stateptr = UnitRange{Int32}(1, num_source(transition_prob) + 1)
    return TimeVaryingDeterministicMarkovDecisionProcess(transition_prob, stateptr, initial_states)
end

"""
    stateptr(mdp::TimeVaryingDeterministicMarkovDecisionProcess)

Return the state pointer of the Time-Varying Deterministic Markov Decision Process. The state pointer is a vector of integers where the `i`-th element
is the index of the first element of the `i`-th state in the transition probability matrix. 
I.e. `transitions[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition probability matrix for source state `j`.
"""
stateptr(mdp::TimeVaryingDeterministicMarkovDecisionProcess) = mdp.stateptr

max_actions(mdp::TimeVaryingDeterministicMarkovDecisionProcess) = maxdiff(stateptr(mdp))
transition_matrix_type(mp::TimeVaryingDeterministicMarkovDecisionProcess) = typeof(first(transition_probs(mp)))

"""
    tomarkovchain(mdp::TimeVaryingDeterministicMarkovDecisionProcess, strategy::AbstractVector{<:AbstractVector})

Extract a Deterministic Markov Chain from an Deterministic Markov Decision Process under a time-varying strategy. The returned type is
a TimeVaryingDeterministicMarkovDecisionProcess with only one action per state per time-step. The extracted IMC is time-varying.
"""
function tomarkovchain(
    mdp::TimeVaryingDeterministicMarkovDecisionProcess,
    strategy::AbstractVector{<:AbstractVector},
)
    if length(strategy) != time_length(mdp)
        throw(ArgumentError("The length of the strategy ($(length(strategy))) must be equal to the time length of the time-varying model ($(time_length(mdp)))"))
    end

    new_probs = Vector{typeof(probs)}(undef, length(strategy))

    for (t, strategy_step) in enumerate(strategy)
        new_probs[t] = transition_prob(mdp, t)[strategy_step]
    end

    istates = initial_states(mdp)

    return TimeVaryingDeterministicMarkovChain(new_probs, istates)
end
