

struct DFA{
    T <: TransitionFunction,  
    VT <: AbstractVector{Int32},
    DA <: AbstractDict{String, Int32}
} <: DeterministicAutomata
    transition::T # Z x 2^Y => Z   
    initial_state::Int32 #z0
    accepting_state::VT #Z_ac
    alphabetptr::DA
end



function DFA(
    transition::TransitionFunction,
    initial_state::Int32,
    accepting_state::AbstractVector{Int32}
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
    N = length(letters);
    alphabet = [""]; #already add empty set

    for letter in letters
        alphabet = vcat(alphabet, [string(word, letter) for word in alphabet])
    end 

    idxs = collect(1:2^N); # powerset cardinality 2^n 
    
    alphabet_idx = Dict{String, Int32}(zip(alphabet, idxs)); 
    return alphabet_idx
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