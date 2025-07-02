"""
    MixtureIntervalMarkovDecisionProcess{
        P <: IntervalProbabilities,
        VT <: AbstractVector{Int32},
        VI <: Union{AllStates, AbstractVector}
    }

A type representing (stationary) Mixture Interval Markov Decision Processes (OIMDP), which are IMDPs where the transition 
probabilities for each state can be represented as the product of the transition probabilities of individual processes.

Formally, let ``(S, S_0, A, \\Gamma, \\Gamma_\\alpha)`` be an interval Markov decision processes, where
- ``S = S_1 \\times \\cdots \\times S_n`` is the set of joint states with ``S_i`` the set of states for the `i`-th marginal,
- ``S_0 \\subseteq S`` is the set of initial states,
- ``A`` is the set of actions,
- ``\\Gamma = \\{\\Gamma_{s,a}\\}_{s \\in S, a \\in A}`` is a set of interval ambiguity sets on the transition probabilities, 
  for each source-action pair, with ``\\Gamma_{s,a} = \\bigotimes_{i=1}^n \\Gamma_{s,a}^i`` and ``\\Gamma_{s,a}^i`` is a marginal interval ambiguity sets
  on the ``i``-th marginal, and
- ``\\Gamma^\\alpha = \\{\\Gamma^\\alpha_{s,a}\\}_{s \\in S, a \\in A}`` is the interval ambiguity set for the mixture.
    
Then the ```MixtureIntervalMarkovDecisionProcess``` type is defined as follows: indices `1:num_states` are the states in ``S`` and 
`transition_prob` represents ``\\Gamma`` and ``\\Gamma^\\alpha``. Actions are implicitly defined by `stateptr` (e.g. if `source_dims` in `transition_prob`
is `(2, 3, 2)`, and `stateptr[3] == 4` and `stateptr[4] == 7` then the actions available to state `CartesianIndex(1, 2, 1)` are `[1, 2, 3]`), and `initial_states`
is the set of initial states ``S_0``. If no initial states are specified, then the initial states are assumed to be all states in ``S``
represented by `AllStates`. See [`MixtureIntervalProbabilities`](@ref) and [Theory](@ref) for more information on the structure
of the transition probability ambiguity sets.

### Fields
- `transition_prob::P`: ambiguity set on transition probabilities (see [`MixtureIntervalProbabilities`](@ref) for the structure).
- `stateptr::VT`: pointer to the start of each source state in `transition_prob` (i.e. `transition_prob[k][l][:, stateptr[j]:stateptr[j + 1] - 1]` is the transition
    probability matrix for source state `j` for each model `k` and axis `l`) in the style of colptr for sparse matrices in CSC format.
- `initial_states::VI`: initial states.
- `num_states::Int32`: number of states.

### Examples
The following example is a simple mixture of two `OrthogonalIntervalProbabilities` with one dimension and the same source/action pairs.
The first state has two actions and the second state has one action. The weighting ambiguity set is also specified for the same three source-action pairs.

```jldoctest
prob1 = OrthogonalIntervalProbabilities(
    (
        IntervalProbabilities(;
            lower = [
                0.0 0.5 0.1
                0.1 0.3 0.2
            ],
            upper = [
                0.5 0.7 0.6
                0.7 0.4 0.8
            ],
        ),
    ),
    (Int32(2),),
)
prob2 = OrthogonalIntervalProbabilities(
    (
        IntervalProbabilities(;
            lower = [
                0.1 0.4 0.2
                0.3 0.0 0.1
            ],
            upper = [
                0.4 0.6 0.5
                0.7 0.5 0.7
            ],
        ),
    ),
    (Int32(2),),
)
weighting_probs = IntervalProbabilities(; lower = [
    0.3 0.5 0.4
    0.4 0.3 0.2
], upper = [
    0.8 0.7 0.7
    0.7 0.5 0.4
])
mixture_prob = MixtureIntervalProbabilities((prob1, prob2), weighting_probs)

stateptr = Int32[1, 3, 4]
mdp = MixtureIntervalMarkovDecisionProcess(mixture_prob, stateptr)
```
"""
struct MixtureIntervalMarkovDecisionProcess{
    P <: MixtureIntervalProbabilities,
    VT <: AbstractVector{Int32},
    VI <: InitialStates,
} <: IntervalMarkovProcess
    transition_prob::P
    stateptr::VT
    initial_states::VI
    num_states::Int32
end

function MixtureIntervalMarkovDecisionProcess(
    transition_prob::MixtureIntervalProbabilities,
    stateptr::AbstractVector{Int32},
    initial_states::InitialStates = AllStates(),
)
    num_states = checksize_imdp(transition_prob, stateptr)

    return MixtureIntervalMarkovDecisionProcess(
        transition_prob,
        stateptr,
        initial_states,
        num_states,
    )
end

function MixtureIntervalMarkovDecisionProcess(
    transition_probs::Vector{<:MixtureIntervalProbabilities},
    initial_states::InitialStates = AllStates(),
)
    # TODO: Fix
    transition_prob, stateptr = interval_prob_hcat(transition_probs)

    return MixtureIntervalMarkovDecisionProcess(transition_prob, stateptr, initial_states)
end

"""
    MixtureIntervalMarkovChain(transition_prob::MixtureIntervalProbabilities, initial_states::InitialStates = AllStates())

Construct a Mixture Interval Markov Chain from mixture interval transition probabilities. The initial states are optional and if not specified,
all states are assumed to be initial states. The number of states is inferred from the size of the transition probability matrix.

The returned type is an `MixtureIntervalMarkovDecisionProcess` with only one action per state (i.e. `stateptr[j + 1] - stateptr[j] == 1` for all `j`).
This is done to unify the interface for value iteration.
"""
function MixtureIntervalMarkovChain(
    transition_prob::MixtureIntervalProbabilities,
    initial_states::InitialStates = AllStates(),
)
    stateptr = UnitRange{Int32}(1, num_source(transition_prob) + 1)
    return MixtureIntervalMarkovDecisionProcess(transition_prob, stateptr, initial_states)
end

function checksize_imdp(p::MixtureIntervalProbabilities, stateptr::AbstractVector{Int32})
    num_states = length(stateptr) - 1

    min_actions = mindiff(stateptr)
    if any(min_actions <= 0)
        throw(ArgumentError("The number of actions per state must be positive."))
    end

    if num_states > prod(num_target, first(p))
        throw(
            DimensionMismatch(
                "The number of target states ($(prod(num_target, first(p))) = $(map(num_target, first(p)))) is less than the number of states in the problem $(num_states).",
            ),
        )
    end
    # TODO:: Check that source_dims match stateptr

    return Int32(prod(num_target, first(p)))
end

"""
    stateptr(mdp::MixtureIntervalMarkovDecisionProcess)

Return the state pointer of the Interval Markov Decision Process. The state pointer is a vector of integers where the `i`-th element
is the index of the first element of the `i`-th state in the transition probability matrix. 
I.e. `mixture_probs(transition_prob)[k][l][:, stateptr[j]:stateptr[j + 1] - 1]` is the independent transition probability matrix for (flattened) source state `j`
for axis `l` and model `k`, and `mixture_probs(transition_prob)[:, stateptr[j]:stateptr[j + 1] - 1]` is the weighting matrix for `j`.
"""
stateptr(mdp::MixtureIntervalMarkovDecisionProcess) = mdp.stateptr

max_actions(mdp::MixtureIntervalMarkovDecisionProcess) = maxdiff(stateptr(mdp))
Base.ndims(::MixtureIntervalMarkovDecisionProcess{N}) where {N} = Int32(N)
product_num_states(mp::MixtureIntervalMarkovDecisionProcess) =
    num_target(transition_prob(mp))
source_shape(mp::MixtureIntervalMarkovDecisionProcess) = source_shape(transition_prob(mp))
