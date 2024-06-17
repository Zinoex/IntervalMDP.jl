"""
    DeterministicMarkovDecisionProcess{
        P <: Transitions,
        VT <: AbstractVector{Int32},
        VI <: Union{AllStates, AbstractVector}
    }

A type representing (stationary) Interval Markov Decision Processes (IMDP), where the lower transition probability bounds are all zeros
and the upper bounds are ones for targets where source/action pairs can transport the system to.

Formally, let ``(S, S_0, A, \\bar{P}, \\underbar{P})`` be an interval Markov decision processes, where ``S`` is the set of states, ``S_0 \\subset S`` is the set of initial states,
``A`` is the set of actions, and ``\\bar{P} : A \\to \\mathbb{R}^{|S| \\times |S|}`` and ``\\underbar{P} : A \\to \\mathbb{R}^{|S| \\times |S|}`` are functions
representing the upper and lower bound transition probability matrices prespectively for each action. Then the ```DeterministicMarkovDecisionProcess``` type is
defined as follows: indices `1:num_states` are the states in ``S``, `transition_prob` represents the ones in ``\\bar{P}`` and ``\\underbar{P} = 0``, actions are 
implicitly defined by `stateptr` (e.g. if `stateptr[3] == 4` and `stateptr[4] == 7` then the actions available to state 3 are `[4, 5, 6]`), 
and `initial_states` is the set of initial states ``S_0``. If no initial states are specified, then the initial states are assumed to be all states in ``S``.

### Fields
- `transition_prob::P`: a CSC-based format where columns represent source/action pairs and rows represent target states and non-zero entries represent a transition probability of 1.
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

mdp = DeterministicMarkovDecisionProcess(transitions, stateptr, initial_states)
```

There is also a constructor for `DeterministicMarkovDecisionProcess` where the transition probabilities are given as a list of 
transition probabilities for each source state.

# TODO: Update example
```jldoctest

transitions
initial_states = [1]

mdp = DeterministicMarkovDecisionProcess(transitions, stateptr, initial_states)
```

"""
struct DeterministicMarkovDecisionProcess{
    P <: Transitions,
    VT <: AbstractVector{Int32},
    VI <: InitialStates,
} <: StationaryIntervalMarkovProcess
    transition_prob::P
    stateptr::VT
    initial_states::VI
    num_states::Int32
end

function DeterministicMarkovDecisionProcess(
    transition_prob::Transitions,
    stateptr::AbstractVector{Int32},
    initial_states::InitialStates = AllStates(),
)
    num_states = checksize_imdp!(transition_prob, stateptr)

    return DeterministicMarkovDecisionProcess(
        transition_prob,
        stateptr,
        initial_states,
        num_states,
    )
end

function DeterministicMarkovDecisionProcess(
    transition_probs::Vector{<:Transitions},
    initial_states::InitialStates = AllStates(),
)
    transition_prob, stateptr = _transition_hcat(transition_probs...)

    return DeterministicMarkovDecisionProcess(transition_prob, stateptr, initial_states)
end

# TODO: Document
function DeterministicMarkovChain(
    transition_prob::Transitions,
    initial_states::InitialStates = AllStates(),
)
    stateptr = UnitRange{Int32}(1, num_source(transition_prob) + 1)
    return DeterministicMarkovDecisionProcess(transition_prob, stateptr, initial_states)
end

function checksize_imdp!(p::Transitions, stateptr::AbstractVector{Int32})
    num_states = length(stateptr) - 1

    min_actions = mindiff(stateptr)
    @assert all(min_actions > 0) "The number of actions per state must be positive"

    if num_target(p) != num_states
        throw(
            DimensionMismatch(
                "The number of transitions in the matrix is not equal to the number of states in the problem",
            ),
        )
    end

    return Int32(num_states)
end

"""
    stateptr(mdp::DeterministicMarkovDecisionProcess)

Return the state pointer of the Deterministic Markov Decision Process. The state pointer is a vector of integers where the `i`-th element
is the index of the first element of the `i`-th state in the transition probability matrix. 
I.e. `transitions[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition probability matrix for source state `j`.
"""
stateptr(mdp::DeterministicMarkovDecisionProcess) = mdp.stateptr

max_actions(mdp::DeterministicMarkovDecisionProcess) = maxdiff(stateptr(mdp))
transition_matrix_type(mp::DeterministicMarkovDecisionProcess) = typeof(transition_prob(mp))

"""
    tomarkovchain(mdp::DeterministicMarkovDecisionProcess, strategy::AbstractVector)

Extract a Deterministic Markov Chain from an Deterministic Markov Decision Process under a stationary strategy. The returned type remains
an DeterministicMarkovDecisionProcess with only one action per state. The extracted IMC is stationary.
"""
function tomarkovchain(mdp::DeterministicMarkovDecisionProcess, strategy::AbstractVector)
    probs = transition_prob(mdp)
    new_probs = probs[strategy]

    istates = initial_states(mdp)

    return DeterministicMarkovChain(new_probs, istates)
end

"""
    tomarkovchain(mdp::DeterministicMarkovDecisionProcess, strategy::AbstractVector{<:AbstractVector})

Extract a Deterministic Markov Chain from an Deterministic Markov Decision Process under a time-varying strategy. The returned type is
a TimeVaryingDeterministicMarkovDecisionProcess with only one action per state per time-step. The extracted IMC is time-varying.
"""
function tomarkovchain(
    mdp::DeterministicMarkovDecisionProcess,
    strategy::AbstractVector{<:AbstractVector},
)
    probs = transition_prob(mdp)
    new_probs = Vector{typeof(probs)}(undef, length(strategy))

    for (t, strategy_step) in enumerate(strategy)
        new_probs[t] = probs[strategy_step]
    end

    istates = initial_states(mdp)

    return TimeVaryingDeterministicMarkovChain(new_probs, istates)
end
