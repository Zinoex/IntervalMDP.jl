include("IntervalMarkovProcess.jl")
export IntervalMarkovProcess, AllStates
export transition_prob, num_states, initial_states, stateptr, tomarkovchain

include("IntervalMarkovDecisionProcess.jl")
export IntervalMarkovDecisionProcess, IntervalMarkovChain

include("OrthogonalIntervalMarkovDecisionProcess.jl")
export OrthogonalIntervalMarkovDecisionProcess, OrthogonalIntervalMarkovChain

include("MixtureIntervalMarkovDecisionProcess.jl")
export MixtureIntervalMarkovDecisionProcess, MixtureIntervalMarkovChain