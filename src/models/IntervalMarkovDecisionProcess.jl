"""
    IntervalMarkovDecisionProcess{
        P <: IntervalProbabilities,
        VT <: AbstractVector{Int32},
        VI <: Union{AllStates, AbstractVector}
    }

A type representing (stationary) Interval Markov Decision Processes (IMDP), which are Markov Decision Processes with uncertainty in the form of intervals on
the transition probabilities.

Formally, let ``(S, S_0, A, \\Gamma)`` be an interval Markov decision process, where 
- ``S`` is the set of states,
- ``S_0 \\subseteq S`` is the set of initial states,
- ``A`` is the set of actions, and
- ``\\Gamma = \\{\\Gamma_{s,a}\\}_{s \\in S, a \\in A}`` is a set of interval ambiguity sets on the transition probabilities, for each source-action pair.

Then the ```IntervalMarkovDecisionProcess``` type is defined as follows: indices `1:num_states` are the states in ``S``, 
`transition_prob` represents ``\\Gamma``, actions are implicitly defined by `stateptr` (e.g. if `stateptr[3] == 4` and `stateptr[4] == 7` then
the actions available to state 3 are `[1, 2, 3]`), and `initial_states` is the set of initial states ``S_0``. If no initial states are specified,
then the initial states are assumed to be all states in ``S`` represented by `AllStates`. See [`IntervalProbabilities`](@ref) and [Theory](@ref) for more information
on the structure of the transition probability ambiguity sets.

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
} <: IntervalMarkovProcess
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
    num_states = checksize_imdp(transition_prob, stateptr)

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

"""
    IntervalMarkovChain(transition_prob::IntervalProbabilities, initial_states::InitialStates = AllStates())

Construct an Interval Markov Chain from a square matrix pair of interval transition probabilities. The initial states are optional and if not specified,
all states are assumed to be initial states. The number of states is inferred from the size of the transition probability matrix.

The returned type is an `IntervalMarkovDecisionProcess` with only one action per state (i.e. `stateptr[j + 1] - stateptr[j] == 1` for all `j`).
This is done to unify the interface for value iteration.
"""
function IntervalMarkovChain(
    transition_prob::IntervalProbabilities,
    initial_states::InitialStates = AllStates(),
)
    stateptr = UnitRange{Int32}(1, num_source(transition_prob) + 1)
    return IntervalMarkovDecisionProcess(transition_prob, stateptr, initial_states)
end

function checksize_imdp(p::IntervalProbabilities, stateptr::AbstractVector{Int32})
    num_states = length(stateptr) - 1

    min_actions = mindiff(stateptr)
    if any(min_actions <= 0)
        throw(ArgumentError("The number of actions per state must be positive."))
    end

    if num_states > num_target(p)
        throw(
            DimensionMismatch(
                "The number of target states ($(num_target(p))) is less than the number of states in the stateptr $(num_states).",
            ),
        )
    end

    if stateptr[end] - 1 != num_source(p)
        throw(
            DimensionMismatch(
                "The number of source states ($(num_source(p))) must be equal to the number of states in the stateptr $(num_states).",
            ),
        )
    end

    return Int32(num_target(p))
end

"""
    stateptr(mdp::IntervalMarkovDecisionProcess)

Return the state pointer of the Interval Markov Decision Process. The state pointer is a vector of integers where the `i`-th element
is the index of the first element of the `i`-th state in the transition probability matrix. 
I.e. `transition_prob[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition probability matrix for source state `j`.
"""
stateptr(mdp::IntervalMarkovDecisionProcess) = mdp.stateptr

max_actions(mdp::IntervalMarkovDecisionProcess) = maxdiff(stateptr(mdp))
Base.ndims(::IntervalMarkovDecisionProcess) = one(Int32)
product_num_states(mp::IntervalMarkovDecisionProcess) = (num_states(mp),)
source_shape(mp::IntervalMarkovDecisionProcess) = (length(stateptr(mp)) - 1,)
