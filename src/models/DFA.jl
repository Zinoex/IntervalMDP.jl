"""
    DFA{
        T <: TransitionFunction,  
        VT <: AbstractVector{Int32},
        DA <: AbstractDict{String, Int32}
    }

A type representing Deterministic Finite Automaton (DFA) which are finite automata with deterministic transitions.

Formally, let ``(Z, 2^{AP}, \\tau, z_0, Z_{ac})`` be an DFA, where 
- ``Z`` is the set of states,
- ``Z_{ac} \\subseteq Z`` is the set of accepting states,
- ``z_0`` is the initial state,
- ``2^{AP}`` is the power set of automic propositions, and
- ``\\tau : |Z| \\times |2^{AP}| => |Z|`` is the deterministic transition function, for each state-input pair.

Then the ```DFA``` type is defined as follows: indices `1:num_states` are the states in ``Z``, 
`transition` represents ``\\tau``, inputs are implicitly defined by indices enumerating the alphabet, and `initial_states` is the set of initial states ``z_0``. 
See [TransitionFunction](@ref) for more information on the structure of the transition function.

### Fields
- `transition::T`: transition function where columns represent inputs and rows source states.
- `initial_state::Int32`: initial states.
- `accepting_states::VT`: vector of accepting states
- `alphabetptr::DA`: mapping from input to index.

"""

struct DFA{
    T <: TransitionFunction,
    VT <: AbstractVector{Int32},
    DA <: AbstractDict{String, Int32},
} <: DeterministicAutomaton
    transition::T # : |Z| x |2^{AP}| => |Z|   
    initial_state::Int32 #z0
    accepting_states::VT #Z_ac
    alphabetptr::DA
    num_states::Int32
    num_alphabet::Int32
end

function DFA(
    transition::TransitionFunction,
    initial_state::Int32,
    accepting_states::AbstractVector{Int32},
    alphabet::AbstractVector{String},
)
    checkdfa!(transition, initial_state, accepting_states, alphabet)

    alphabetptr, num_alphabet = alphabet2index(alphabet)

    num_states = getsize_dfa!(transition)

    return DFA(
        transition,
        initial_state,
        accepting_states,
        alphabetptr,
        Int32(num_states),
        Int32(num_alphabet),
    )
end

"""
    Given vector of letters L, compute power set 2^L 
    Returns the alphabet (powerset) and corresponding index as Dictionary
"""
function letters2alphabet(letters::AbstractVector{String})
    alphabet = [""] #already add empty set

    for letter in letters
        append!(alphabet, string.(alphabet, letter))
    end

    return alphabet2index(alphabet)
end

"""
    Given vector of alphabet 2^L, maps its index for lookup
    Returns dictionary, assume same indices used in Transition function.
"""
function alphabet2index(alphabet::AbstractVector{String})
    N = length(alphabet)
    idxs = collect(1:N)

    alphabet_idx = Dict{String, Int32}(zip(alphabet, idxs))
    return alphabet_idx, N
end

"""
Check given dfa valid
"""
function checkdfa!(
    transition::TransitionFunction,
    initial_state::Int32,
    accepting_states::AbstractVector{Int32},
    alphabet::AbstractVector{String},
)
    # check z0 and Z_ac in Z, check size(alphabet) == size(transition dim)
    # @assert size(transition.transition, 1) == length(alphabet) "size of alphabet match"
    # @assert all(accepting_state .>= 1) "all accepting states exists"
    # @assert all(accepting_state .<= size(transition.transition, 2)) "all accepting states exists"
    # @assert initial_state .>= 1 "initial state exists"
    # @assert initial_state .<= size(transition.transition, 2) "initial state exists"

    if size(transition.transition, 1) != length(alphabet)
        throw(
            DimensionMismatch(
                "The size of alphabet ($(length(alphabet))) is not equal to the size of the transition column $(size(transition.transition, 1))",
            ),
        )
    end

    if !all(accepting_states .>= 1)
        throw(ArgumentError("Next state index cannot be zero or negative"))
    end

    if !all(accepting_states .<= size(transition.transition, 2))
        throw(
            ArgumentError("Next state index cannot be larger than total number of states"),
        )
    end

    if length(accepting_states) > size(transition.transition, 2)
        throw(ArgumentError("Invalid Accepting States"))
    end

    if initial_state < 1
        throw(ArgumentError("Initial state index cannot be zero or negative"))
    end

    if initial_state > size(transition.transition, 2)
        throw(
            ArgumentError(
                "Initial state index cannot be larger than total number of states",
            ),
        )
    end
end

function getsize_dfa!(transition)
    return size(transition.transition, 2)
end

"""
    transition(dfa::DFA)

Return the transition object of the Deterministic Finite Automaton. 
"""
transition(dfa::DFA) = dfa.transition

"""
    size(dfa::DFA)

Return ``|Z|`` and ``|2^{AP}|`` of the Deterministic Finite Automaton. 
"""
Base.size(dfa::DFA) = (dfa.num_states, dfa.num_alphabet)

"""
    alphabetptr(dfa::DFA)

Return the alphabet (``2^{AP}``) index mapping of the Deterministic Finite Automaton. 
"""
alphabetptr(dfa::DFA) = dfa.alphabetptr

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
    getindex(dfa::DFA, z::Int, w::Int)

Return the the next state for source state ``z`` and int input ``w`` of the Deterministic Finite Automaton. 
"""
Base.getindex(dfa::DFA, z::Int, w::Int) = dfa.transition[z, w]

"""
    getindex(dfa::DFA, z::Int, w::String)

Return the the next state for source state ``z`` and string input ``w`` of the Deterministic Finite Automaton. 
"""
Base.getindex(dfa::DFA, z::Int, w::String) = dfa[z, Int(dfa.alphabetptr[w])]
