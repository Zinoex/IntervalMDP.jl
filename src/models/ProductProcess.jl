"""
    struct ProductProcess{
        M <: IntervalMarkovProcess,
        D <: DeterministicAutomaton,
        L <: AbstractLabelling,
    }

A type representing the product between interval Markov processes (e.g. [`IntervalMarkovDecisionProcess`](@ref) or [`OrthogonalIntervalMarkovDecisionProcess`](@ref))
and an automaton (typically a deterministic finite automaton [`DFA`](@ref)). 

Formally, given an interval Markov process ``M = (S, A, \\Gamma, S_{0})``, a labelling function ``L : S \\to 2^{AP}``, and a DFA ``D = (Q, 2^{AP}, \\delta, q_{0}, Q_{ac})``,
then a product process is a tuple ``M_{prod} = (Z, A, \\Gamma^{prod}, Z_{ac}, Z_{0})`` where 
- ``Z = S \\times Q`` is the set of product states q = (s, z)``,
- ``Q_{0} = S_0 \\times \\{q_0\\} \\subset Z`` is the set of initial product states ``z_0 = (s_0, q_0)``,
- ``Z_{ac} = S \\times Q_{ac} \\subseteq Z`` is the set of accepting product states,
- ``A`` is the set of actions, and
- ``\\Gamma^{prod} = \\{\\Gamma^{prod}_{z,a}\\}_{z \\in Z, a \\in A}`` where ``\\Gamma^{prod}_{z,a} = \\{ \\gamma_{z,a} : \\gamma_{z,a}((t, z')) = \\gamma_{s,a}(t)\\delta_{q,L(s)}(z') \\}``
is a set of ambiguity sets on the product transition probabilities, for each product source-action pair.

See [`IntervalMarkovDecisionProcess`](@ref) and [`DFA`](@ref) for more information on the structure, definition, and usage of the DFA and IMDP.

### Fields
- `imdp::M`: contains details for the interval Markov process.
- `dfa::D`: contains details for the DFA
- `labelling_func::L`: the labelling function from IMDP states to DFA actions
"""
struct ProductProcess{
    M <: IntervalMarkovProcess,
    D <: DeterministicAutomaton,
    L <: AbstractLabelling,
} <: StochasticProcess
    imdp::M
    dfa::D
    labelling_func::L

    function ProductProcess(
        imdp::M,
        dfa::D,
        labelling_func::L,
    ) where {
        M <: IntervalMarkovProcess,
        D <: DeterministicAutomaton,
        L <: AbstractLabelling,
    }
        checkproduct(imdp, dfa, labelling_func)

        return new{M, D, L}(imdp, dfa, labelling_func)
    end
end

function checkproduct(
    imdp::IntervalMarkovProcess,
    dfa::DeterministicAutomaton,
    labelling_func::AbstractLabelling,
)

    # check labelling states (input) match IMDP states
    if size(labelling_func) != source_shape(imdp)
        throw(
            DimensionMismatch(
                "The number of IMDP states $(source_shape(imdp)) is not equal to number of mapped states  $(size(labelling_func)) in the labelling function.",
            ),
        )
    end

    # check state labels (output) match DFA alphabet
    if num_labels(labelling_func) > num_labels(dfa) # not all actions needed to be mapped so can be less but certainly not more
        throw(
            DimensionMismatch(
                "The number of DFA inputs ($(num_labels(dfa))) is less than to number of labels $(num_labels(labelling_func)) in the labelling function.",
            ),
        )
    end
end

"""
    markov_process(proc::ProductIntervalMarkovDecisionProcessDFA)

Return the interval markov decision process of the product 
"""
markov_process(proc::ProductProcess) = proc.imdp

"""
    automaton(proc::ProductIntervalMarkovDecisionProcessDFA)

Return the deterministic finite automaton of the product 
"""
automaton(proc::ProductProcess) = proc.dfa

"""
    labelling_function(proc::ProductProcess)

Return the labelling function of the product 
"""
labelling_function(proc::ProductProcess) = proc.labelling_func

product_num_states(proc::ProductProcess) =
    (product_num_states(markov_process(proc))..., num_states(automaton(proc)))
source_shape(proc::ProductProcess) =
    (source_shape(markov_process(proc))..., num_states(automaton(proc)))
