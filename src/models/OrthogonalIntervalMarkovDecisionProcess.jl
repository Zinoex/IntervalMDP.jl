"""
    OrthogonaIntervalMarkovDecisionProcess{
        P <: IntervalProbabilities,
        VT <: AbstractVector{Int32},
        VI <: Union{AllStates, AbstractVector}
    }

A type representing (stationary) Orthogona Interval Markov Decision Processes (OIMDP), which are IMDPs where the transition 
probabilities for each state can be represented as the product of the transition probabilities of individual processes.

# TODO: Update theory section

Formally, let ``(S, S_0, A, \\bar{P}, \\underbar{P})`` be an interval Markov decision processes, where ``S`` is the set of states, ``S_0 \\subset S`` is the set of initial states,
``A`` is the set of actions, and ``\\bar{P} : A \\to \\mathbb{R}^{|S| \\times |S|}`` and ``\\underbar{P} : A \\to \\mathbb{R}^{|S| \\times |S|}`` are functions
representing the upper and lower bound transition probability matrices prespectively for each action. Then the ```IntervalMarkovDecisionProcess``` type is
defined as follows: indices `1:num_states` are the states in ``S``, `transition_prob` represents ``\\bar{P}`` and ``\\underbar{P}``, actions are 
implicitly defined by `stateptr` (e.g. if `stateptr[3] == 4` and `stateptr[4] == 7` then the actions available to state 3 are `[4, 5, 6]`), 
and `initial_states` is the set of initial states ``S_0``. If no initial states are specified, then the initial states are assumed to be all states in ``S``.

### Fields
# TODO: Update fields

- `transition_prob::P`: interval on transition probabilities where columns represent source/action pairs and rows represent target states.
- `stateptr::VT`: pointer to the start of each source state in `transition_prob` (i.e. `transition_prob[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition
    probability matrix for source state `j`) in the style of colptr for sparse matrices in CSC format.
- `initial_states::VI`: initial states.
- `num_states::Int32`: number of states.

### Examples

# TODO: Update examples

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
struct OrthogonalIntervalMarkovDecisionProcess{
    P <: OrthogonalIntervalProbabilities,
    VT <: AbstractVector{Int32},
    VI <: InitialStates,
} <: StationaryIntervalMarkovProcess
    transition_prob::P
    stateptr::VT
    initial_states::VI
    num_states::Int32
end

function OrthogonalIntervalMarkovDecisionProcess(
    transition_prob::OrthogonalIntervalProbabilities,
    stateptr::AbstractVector{Int32},
    initial_states::InitialStates = AllStates(),
)
    num_states = checksize_imdp!(transition_prob, stateptr)

    return OrthogonalIntervalMarkovDecisionProcess(
        transition_prob,
        stateptr,
        initial_states,
        num_states,
    )
end

function OrthogonalIntervalMarkovDecisionProcess(
    transition_probs::Vector{<:OrthogonalIntervalProbabilities},
    initial_states::InitialStates = AllStates(),
)
    # TODO: Fix
    transition_prob, stateptr = interval_prob_hcat(transition_probs)

    return OrthogonalIntervalMarkovDecisionProcess(transition_prob, stateptr, initial_states)
end

"""
    OrthogonalIntervalMarkovChain(transition_prob::OrthogonalIntervalProbabilities, initial_states::InitialStates = AllStates())

# TODO: Update docstring 

Construct a Orthogonal Interval Markov Chain from a square matrix pair of interval transition probabilities. The initial states are optional and if not specified,
all states are assumed to be initial states. The number of states is inferred from the size of the transition probability matrix.

The returned type is an `IntervalMarkovDecisionProcess` with only one action per state (i.e. `stateptr[j + 1] - stateptr[j] == 1` for all `j`).
This is done to unify the interface for value iteration.
"""
function OrthogonalIntervalMarkovChain(
    transition_prob::IntervalProbabilities,
    initial_states::InitialStates = AllStates(),
)
    stateptr = UnitRange{Int32}(1, num_source(transition_prob) + 1)
    return OrthogonalIntervalMarkovDecisionProcess(transition_prob, stateptr, initial_states)
end

function checksize_imdp!(p::OrthogonalIntervalProbabilities, stateptr::AbstractVector{Int32})
    num_states = length(stateptr) - 1

    min_actions = mindiff(stateptr)
    if any(min_actions <= 0)
        throw(ArgumentError("The number of actions per state must be positive."))
    end

    if prod(num_target, p.probs) != num_states
        throw(
            DimensionMismatch(
                "The number of target states ($(prod(num_target, p.probs)) = $(map(num_target, p.probs))) is not equal to the number of states in the problem $(num_states).",
            ),
        )
    end

    return Int32(num_states)
end

"""
    stateptr(mdp::OrthogonalIntervalMarkovDecisionProcess)

Return the state pointer of the Interval Markov Decision Process. The state pointer is a vector of integers where the `i`-th element
is the index of the first element of the `i`-th state in the transition probability matrix. 
I.e. `transition_prob[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition probability matrix for source state `j`.
"""
stateptr(mdp::OrthogonalIntervalMarkovDecisionProcess) = mdp.stateptr

max_actions(mdp::OrthogonalIntervalMarkovDecisionProcess) = maxdiff(stateptr(mdp))
Base.ndims(::OrthogonalIntervalMarkovDecisionProcess{N}) where {N} = Int32(N)
product_num_states(mp::OrthogonalIntervalMarkovDecisionProcess) = num_target(transition_prob(mp))
transition_matrix_type(mp::OrthogonalIntervalMarkovDecisionProcess) = typeof(gap(first(transition_prob(mp).probs)))

"""
    tomarkovchain(mdp::OrthogonalIntervalMarkovDecisionProcess, strategy::AbstractVector)

Extract a Product Interval Markov Chain (IMC) from an Interval Markov Decision Process under a stationary strategy. The returned type remains
an `IntervalMarkovDecisionProcess` with only one action per state. The extracted IMC is stationary.
"""
function tomarkovchain(mdp::OrthogonalIntervalMarkovDecisionProcess, strategy::AbstractVector)
    # TODO: Fix

    probs = transition_prob(mdp)
    new_probs = probs[strategy]

    istates = initial_states(mdp)

    return IntervalMarkovChain(new_probs, istates)
end

"""
    tomarkovchain(mdp::IntervalMarkovDecisionProcess, strategy::AbstractVector{<:AbstractVector})

Extract a Product Interval Markov Chain (IMC) from an Interval Markov Decision Process under a time-varying strategy. The returned type is
a `TimeVaryingIntervalMarkovDecisionProcess` with only one action per state per time-step. The extracted IMC is time-varying.
"""
function tomarkovchain(
    mdp::OrthogonalIntervalMarkovDecisionProcess,
    strategy::AbstractVector{<:AbstractVector},
)
    # TODO: Fix

    probs = transition_prob(mdp)
    new_probs = Vector{typeof(probs)}(undef, length(strategy))

    for (t, strategy_step) in enumerate(strategy)
        new_probs[t] = probs[strategy_step]
    end

    istates = initial_states(mdp)

    return TimeVaryingIntervalMarkovChain(new_probs, istates)
end
