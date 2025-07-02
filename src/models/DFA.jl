"""
    DFA{
        T <: TransitionFunction,  
        VT <: AbstractVector{Int32},
        DA <: AbstractDict{String, Int32}
    }

A type representing Deterministic Finite Automaton (DFA) which are finite automata with deterministic transitions.

Formally, let ``(Q, 2^{AP}, \\delta, q_0, Q_{ac})`` be an DFA, where 
- ``Q`` is the set of states,
- ``Q_{ac} \\subseteq Q`` is the set of accepting states,
- ``Q_0`` is the initial state,
- ``2^{AP}`` is the power set of automic propositions, and
- ``\\delta : |Q| \\times |2^{AP}| => |Q|`` is the deterministic transition function, for each state-input pair.

Then the `DFA` type is defined as follows: indices `1:num_states` are the states in ``Q``, 
`transition` represents ``\\delta``, the set ``2^{AP}`` is , and `initial_states` is the set of initial states ``q_0``. 
See [`TransitionFunction`](@ref) for more information on the structure of the transition function.

### Fields
- `transition::T`: transition function.
- `initial_state::Int32`: initial states.
- `accepting_states::VT`: vector of accepting states
- `labelmap::DA`: mapping from label to index.

TODO: Add explicit sink states for non-accepting self-looping states since we do not need to iterate for these.
TODO: Detection of non-accepting end components. They can be replaced by a single state.

"""
struct DFA{T <: TransitionFunction, DA <: AbstractDict{String, Int32}} <:
       DeterministicAutomaton
    transition::T # delta : |Q| x |2^{AP}| => |Q|   
    initial_state::Int32 # q_0
    labelmap::DA
end

function DFA(
    transition::TransitionFunction,
    initial_state::Int32,
    atomic_propositions::AbstractVector{String},
)
    labelmap = atomicpropositions2labels(atomic_propositions)
    checkdfa(transition, initial_state, labelmap)

    return DFA(transition, initial_state, labelmap)
end

"""
    Given vector of atomic_propositions ``AP``, compute power set ``2^{AP}`` 
    Returns the alphabet (powerset) and corresponding index as Dictionary
"""
function atomicpropositions2labels(atomic_propositions::AbstractVector{String})
    labels = [""] # already add empty set

    for atomic_proposition in atomic_propositions
        append!(labels, string.(labels, atomic_proposition))
    end

    return label2index(labels)
end

"""
    Given vector of labels ``2^{AP}``, maps its index for lookup
    Returns dictionary, assume same indices used in Transition function.
"""
function label2index(labels::AbstractVector{String})
    idxs = eachindex(labels)

    label2idx = Dict{String, Int32}(zip(labels, idxs))
    return label2idx
end

function checkdfa(transition::TransitionFunction, initial_state::Int32, labelmap)
    num_states = size(transition, 2)

    # Check size of transition function
    if num_labels(transition) != length(labelmap)
        throw(
            DimensionMismatch(
                "The labels in the transition function ($(num_labels(transition))) are not equal to the number of labels of the DFA ($(length(labelmap))).",
            ),
        )
    end

    # Check z_0 in Z
    if !(1 <= initial_state <= num_states)
        throw(ArgumentError("Initial state not in the set of states."))
    end

    return num_states
end

"""
    transition(dfa::DFA)

Return the transition object of the Deterministic Finite Automaton. 
"""
transition(dfa::DFA) = dfa.transition

"""
    num_states(dfa::DFA)

Return the number of states ``|Q|`` of the Deterministic Finite Automaton.
"""
num_states(dfa::DFA) = num_states(transition(dfa))

"""
    num_labels(dfa::DFA)
Return the number of labels (DFA inputs) in the Deterministic Finite Automaton.
"""
num_labels(dfa::DFA) = num_labels(transition(dfa))

"""
    size(dfa::DFA)

Return ``|Q|`` and ``|2^{AP}|`` of the Deterministic Finite Automaton. 
"""
Base.size(dfa::DFA) = (num_states(dfa), num_labels(dfa))

"""
    labelmap(dfa::DFA)

Return the label index mapping ``2^{AP} \\to \\mathbb{N}`` of the Deterministic Finite Automaton. 
"""
labelmap(dfa::DFA) = dfa.labelmap

"""
    initial_state(dfa::DFA)

Return the initial state of the Deterministic Finite Automaton. 
"""
initial_state(dfa::DFA) = dfa.initial_state

"""
    accepting_states(dfa::DFA)

Return the accepting states of the Deterministic Finite Automaton. 
"""
accepting_states(dfa::DFA) = dfa.accepting_states

"""
    getindex(dfa::DFA, q, w)

Return the the next state for source state ``q`` and input ``w`` of the Deterministic Finite Automaton. 
"""
Base.getindex(dfa::DFA, q, w) = dfa.transition[q, w]

"""
    getindex(dfa::DFA, q, w::String)

Return the the next state for source state ``q`` and `String` input ``w`` (mapping it to the appropriate index) of the Deterministic Finite Automaton. 
"""
Base.getindex(dfa::DFA, q, w::String) = dfa[q, dfa.labelmap[w]]

Base.iterate(dfa::DFA, state::Int32 = one(Int32)) =
    state > num_states(dfa) ? nothing : (state, state + one(Int32))
