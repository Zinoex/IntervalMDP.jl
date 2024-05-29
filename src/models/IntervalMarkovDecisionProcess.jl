"""
    IntervalMarkovDecisionProcess{
        P <: IntervalProbabilities,
        T <: Integer,
        VT <: AbstractVector{T},
        VI <: AbstractVector{T},
        VA <: AbstractVector,
    }

A type representing (stationary) Interval Markov Decision Processes (IMDP), which are Markov Decision Processes with uncertainty in the form of intervals on
the transition probabilities.

Formally, let ``(S, S_0, A, \\bar{P}, \\underbar{P})`` be an interval Markov decision processes, where ``S`` is the set of states, ``S_0 \\subset S`` is the set of initial states,
``A`` is the set of actions, and ``\\bar{P} : A \\to \\mathbb{R}^{|S| \\times |S|}`` and ``\\underbar{P} : A \\to \\mathbb{R}^{|S| \\times |S|}`` are functions
representing the upper and lower bound transition probability matrices prespectively for each action. Then the ```IntervalMarkovDecisionProcess``` type is
defined as follows: indices `1:num_states` are the states in ``S``, `transition_prob` represents ``\\bar{P}`` and ``\\underbar{P}``,
`action_vals` contains the actions available in each state, and `initial_states` is the set of initial states ``S_0``.
If no initial states are specified, then the initial states are assumed to be all states in ``S``.

### Fields
- `transition_prob::P`: interval on transition probabilities where columns represent source/action pairs and rows represent target states.
- `stateptr::VT`: pointer to the start of each source state in `transition_prob` (i.e. `transition_prob[:, stateptr[j]:stateptr[j + 1] - 1]` is the transition
    probability matrix for source state `j`) in the style of colptr for sparse matrices in CSC format.
- `action_vals::VA`: actions available in each state. Can be any eltype.
- `initial_states::VI`: initial states.
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
initial_states = [1]

mdp = IntervalMarkovDecisionProcess(transition_probs, stateptr, actions, initial_states)
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
initial_states = [1]

mdp = IntervalMarkovDecisionProcess(transition_probs, initial_states)
```

"""
struct IntervalMarkovDecisionProcess{
    P <: IntervalProbabilities,
    T <: Integer,
    VT <: AbstractVector{T},
    VI <: AbstractVector{T},
    VA <: AbstractVector,
} <: StationaryIntervalMarkovProcess{P}
    transition_prob::P
    stateptr::VT
    action_vals::VA
    initial_states::VI
    num_states::T
end

function IntervalMarkovDecisionProcess(
    transition_prob::P,
    stateptr::VT,
    action_vals::VA,
    initial_states,
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
        initial_states,
        num_states,
    )
end

function IntervalMarkovDecisionProcess(
    transition_prob::P,
    stateptr::VT,
    action_vals::VA,
) where {P <: IntervalProbabilities, VT <: AbstractVector{<:Integer}, VA <: AbstractVector}
    return IntervalMarkovDecisionProcess(
        transition_prob,
        stateptr,
        action_vals,
        all_initial_states(length(stateptr) - 1),
    )
end

function IntervalMarkovDecisionProcess(
    transition_probs::Vector{P},
    actions::VA,
    initial_states::VI,
) where {
    P <: IntervalProbabilities,
    VA <: AbstractVector,
    T <: Integer,
    VI <: AbstractVector{T},
}
    transition_prob, stateptr = interval_prob_hcat(T, transition_probs)

    return IntervalMarkovDecisionProcess(transition_prob, stateptr, actions, initial_states)
end

function IntervalMarkovDecisionProcess(
    transition_probs::Vector{P},
    actions::VA,
) where {P <: IntervalProbabilities, VA <: AbstractVector}
    return IntervalMarkovDecisionProcess(
        transition_probs,
        actions,
        all_initial_states(length(transition_probs)),
    )
end

function IntervalMarkovDecisionProcess(
    transition_probs::Vector{Pair{VA, P}},
    initial_states::VI,
) where {
    P <: IntervalProbabilities,
    VA <: AbstractVector,
    T <: Integer,
    VI <: AbstractVector{T},
}
    action_vals = mapreduce(first, vcat, transition_probs)
    transition_probs = map(x -> x[2], transition_probs)

    return IntervalMarkovDecisionProcess(transition_probs, action_vals, initial_states)
end

function IntervalMarkovDecisionProcess(
    transition_probs::Vector{Pair{VA, P}},
) where {P <: IntervalProbabilities, VA <: AbstractVector}
    return IntervalMarkovDecisionProcess(
        transition_probs,
        all_initial_states(length(transition_probs)),
    )
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
    actions(mdp::IntervalMarkovDecisionProcess)

Return a vector of actions (choices in PRISM terminology).
"""
actions(mdp::IntervalMarkovDecisionProcess) = mdp.action_vals

"""
    num_choices(mdp::IntervalMarkovDecisionProcess)

Return the sum of the number of actions available in each state ``\\sum_{j} \\mathrm{num\\_actions}(s_j)``.
"""
num_choices(mdp::IntervalMarkovDecisionProcess) = length(actions(mdp))

stateptr(mdp::IntervalMarkovDecisionProcess) = mdp.stateptr

"""
    tomarkovchain(mdp::IntervalMarkovDecisionProcess)

Convert an Interval Markov Decision Process to an Interval Markov Chain, provided that
each state of the IMDP only has one action. The IMC is stationary.
"""
function tomarkovchain(mdp::IntervalMarkovDecisionProcess)
    sptr = stateptr(mdp)
    num_choices_per_state = diff(sptr)

    if any(num_choices_per_state .> 1)
        throw(
            ArgumentError(
                "The number of actions per state must be 1 or a strategy must be provided.",
            ),
        )
    end

    probs = transition_prob(mdp)
    istates = initial_states(mdp)

    return IntervalMarkovChain(probs, istates)
end

"""
    tomarkovchain(mdp::IntervalMarkovDecisionProcess, strategy::AbstractVector)

Extract an Interval Markov Chain from an Interval Markov Decision Process under a stationary strategy. The extracted IMC is stationary.
"""
function tomarkovchain(mdp::IntervalMarkovDecisionProcess, strategy::AbstractVector)
    sptr = stateptr(mdp)

    probs = transition_prob(mdp)

    strategy_idxes = eltype(sptr)[]
    for i in 1:num_states(mdp)
        state_actions = actions(mdp)[sptr[i]:(sptr[i + 1] - 1)]
        if !(strategy[i] in state_actions)
            throw(
                ArgumentError(
                    "The strategy must be a valid action for each state. Was $(strategy[i]) for state $i, available actions are $state_actions.",
                ),
            )
        end

        for j in sptr[i]:(sptr[i + 1] - 1)
            if actions(mdp)[j] == strategy[i]
                push!(strategy_idxes, j)
                break
            end
        end
    end

    new_probs = probs[strategy_idxes]

    istates = initial_states(mdp)

    return IntervalMarkovChain(new_probs, istates)
end

"""
    tomarkovchain(mdp::IntervalMarkovDecisionProcess, strategy::AbstractVector{<:AbstractVector})

Extract an Interval Markov Chain from an Interval Markov Decision Process under a time-varying strategy. The extracted IMC is time-varying.
"""
function tomarkovchain(
    mdp::IntervalMarkovDecisionProcess,
    strategy::AbstractVector{<:AbstractVector},
)
    sptr = stateptr(mdp)

    probs = transition_prob(mdp)
    new_probs = Vector{typeof(probs)}(undef, length(strategy))

    for (t, strategy_step) in enumerate(strategy)
        strategy_idxes = eltype(sptr)[]
        for i in 1:num_states(mdp)
            state_actions = actions(mdp)[sptr[i]:(sptr[i + 1] - 1)]
            if !(strategy_step[i] in state_actions)
                throw(
                    ArgumentError(
                        "The strategy must be a valid action for each state. Was $(strategy_step[i]) for state $i at time $t, available actions are $state_actions.",
                    ),
                )
            end

            for j in sptr[i]:(sptr[i + 1] - 1)
                if actions(mdp)[j] == strategy_step[i]
                    push!(strategy_idxes, j)
                    break
                end
            end
        end

        new_probs[t] = probs[strategy_idxes]
    end

    istates = initial_states(mdp)

    return TimeVaryingIntervalMarkovChain(new_probs, istates)
end