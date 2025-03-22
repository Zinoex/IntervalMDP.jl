"""
    DFA{
        T <: TransitionFunction,  
        VT <: AbstractVector{Int32},
        DA <: AbstractDict{String, Int32}
    }

A type representing Deterministic Finite Automata (DFA) which are finite automata with deterministic transitions.

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
    DA <: AbstractDict{String, Int32}
} <: DeterministicAutomata
    transition::T # : |Z| x |2^{AP}| => |Z|   
    initial_state::Int32 #z0
    accepting_states::VT #Z_ac
    alphabetptr::DA
end

                              

function DFA(
    transition::TransitionFunction,
    initial_state::Int32,
    accepting_states::AbstractVector{Int32}
    alphabet::AbstractVector{String},
)
    checkdfa!(transition, initial_state, accepting_state, alphabet) 

    alphabetptr = alphabet2index(alphabet)

    return DFA(
        transition,
        initial_states,
        accepting_state,
        alphabetptr
    )
end


"""
    Given vector of letters L, compute power set 2^L 
    Returns the alphabet (powerset) and corresponding index as Dictionary
"""
function letters2alphabet(
    letters::AbstractVector{String},
)   

    alphabet = [""]; #already add empty set

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
    N = length(alphabet);
    idxs = collect(1:N);

    alphabet_idx = Dict{String, Int32}(zip(alphabet, idxs)); 
    return alphabet_idx
end


"""
Check given dfa valid
"""
function checkdfa!(transition::TransitionFunction,
                    initial_state::Int32,
                    accepting_state::AbstractVector{Int32}
                    alphabet::AbstractVector{String},
                    )
    # check z0 and Z_ac in Z, check size(alphabet) == size(transition dim)
    @assert size(transition.transition, 1) == length(alphabet) "size of alphabet match"
    @assert all(accepting_state .>= 1) "all accepting states exists"
    @assert all(accepting_state .<= size(transition.transition, 2)) "all accepting states exists"
    @assert initial_state .>= 1 "initial state exists"
    @assert initial_state .<= size(transition.transition, 2) "initial state exists"
end

#TODO add accessor methods 