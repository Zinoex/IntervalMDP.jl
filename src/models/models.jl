include("IntervalMarkovProcess.jl")
export IntervalMarkovProcess, AllStates
export transition_prob, num_states, initial_states, stateptr, tomarkovchain

include("IntervalMarkovDecisionProcess.jl")
export IntervalMarkovDecisionProcess, IntervalMarkovChain

include("OrthogonalIntervalMarkovDecisionProcess.jl")
export OrthogonalIntervalMarkovDecisionProcess, OrthogonalIntervalMarkovChain

include("MixtureIntervalMarkovDecisionProcess.jl")
export MixtureIntervalMarkovDecisionProcess, MixtureIntervalMarkovChain

include("DeterministicAutomaton.jl")

include("DFA.jl")
export DFA,
    letters2alphabet,
    alphabet2index,
    transition,
    alphabetptr,
    initial_state,
    accepting_states,
    getindex

include("ProductIntervalMarkovProcess.jl")

include("ProductIntervalMarkovDecisionProcess.jl")
export ProductIntervalMarkovDecisionProcessDFA, imdp, automaton, labelling_function
