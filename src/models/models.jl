abstract type StochasticProcess end

include("IntervalMarkovProcess.jl")
export IntervalMarkovProcess, AllStates
export num_states, num_actions, initial_states

include("FactoredRobustMarkovDecisionProcess.jl")
export FactoredRobustMarkovDecisionProcess, state_variables, action_variables, marginals

# Convenience model constructors - they all return a FactoredRobustMarkovDecisionProcess
include("IntervalMarkovChain.jl")
include("IntervalMarkovDecisionProcess.jl")
export IntervalMarkovChain, IntervalMarkovDecisionProcess

include("DeterministicAutomaton.jl")

include("DFA.jl")
export DFA, transition, labelmap, initial_state, accepting_states

include("ProductProcess.jl")
export ProductProcess, markov_process, automaton, labelling_function
