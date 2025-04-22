"""
    struct ProductIntervalMarkovDecisionProcessDFA{
        D <: DFA,
        M <: IntervalMarkovDecisionProcess,
        L <: AbstractLabelling,
    }

A type representing the product between Interval Markov Decision Processes (IMDP) and Deterministic Finite Automata (DFA). 

Formally, let ``(S^{\\prime}, A, \\Gamma^{\\prime}, S^{\\prime}_{ac}, S^{\\prime}_{0}, L)`` be an product interval Markov decision process DFA, where 
- ``S^{\\prime} = S \\times Z`` is the set of product states q = (s, z)``,
- ``S^{\\prime}_{0} = S_0 \\times z_0 \\subset S^{\\prime}`` is the set of initial product states ``q = (s, z_0)``,
- ``S^{\\prime}_{ac} = S \\times Z_{ac} \\subseteq S^{\\prime}`` is the set of accepting product states,
- ``A`` is the set of actions,
- ``\\Gamma^{\\prime} = \\{\\Gamma^{\\prime}_{q,a}\\}_{q \\in S^{\\prime}, a \\in A}`` is a set of interval ambiguity sets on the transition probabilities, for each product source-action pair, and
- ``L : S => 2^{AP}`` is the labelling function that maps a state in the IMDP to an input to the DFA.

Then the ```ProductIntervalMarkovDecisionProcessDFA``` type is defined as follows: DFA part of the product as a DFA object, IMDP part of the product as a IMDP object and a Labelling function to specify the relationshup between the DFA and IMDP.

See [IntervalMarkovDecisionProcess](@ref) and [DFA](@ref) for more information on the structure, definition, and usage of the DFA and IMDP.

### Fields
- `imdp::M`: contains details for the IMDP
- `dfa::D`: contains details for the DFA
- `labelling_func::L`: the labelling function from IMDP states to DFA actions
"""

struct ProductIntervalMarkovDecisionProcessDFA{
    M <: IntervalMarkovDecisionProcess,
    D <: DFA,
    L <: AbstractLabelling,
} <: ProductIntervalMarkovProcess
    imdp::M
    dfa::D
    labelling_func::L

    function ProductIntervalMarkovDecisionProcessDFA(
        imdp::M,
        dfa::D,
        labelling_func::L,
    ) where {M <: IntervalMarkovDecisionProcess, D <: DFA, L <: AbstractLabelling}
        checklabelling!(transition, initial_state, accepting_state, alphabet)

        return new{M, D, L}(imdp, dfa, labelling_func)
    end
end

"""
Check given imdp, dfa, labelling combination is valid
"""
function checklabelling!(
    imdp::IntervalMarkovDecisionProcess,
    dfa::DFA,
    labelling_func::AbstractLabelling,
)

    # check labelling states (input) match IMDP states
    if labelling_func.num_inputs == imdp.num_states
        throw(
            DimensionMismatch(
                "The number of IMDP states ($(imdp.num_states)) is not equal to number of mapped states  $(labelling_func.num_inputs) in the labelling function.",
            ),
        )
    end

    # check state labels (output) match DFA alphabet
    if labelling_func.num_outputs <= dfa.num_alphabet # not all actions needed to be mapped so can be less but certainly not more
        throw(
            DimensionMismatch(
                "The number of DFA inputs ($(dfa.num_alphabet)) is not equal to number of mapped states  $(labelling_func.num_outputs) in the labelling function.",
            ),
        )
    end

    # check all S in Labelfunc
    # check all 2^{AP} in Label func
end

"""
    imdp(md::ProductIntervalMarkovDecisionProcessDFA)

Return the interval markov decision process of the product 
"""
imdp(md::ProductIntervalMarkovDecisionProcessDFA) = md.imdp

"""
    automaton(md::ProductIntervalMarkovDecisionProcessDFA)

Return the deterministic finite automaton of the product 
"""
automaton(md::ProductIntervalMarkovDecisionProcessDFA) = md.dfa

"""
    labelling_function(md::ProductIntervalMarkovDecisionProcessDFA)

Return the labelling function of the product 
"""
labelling_function(md::ProductIntervalMarkovDecisionProcessDFA) = md.labelling_func
