abstract type StochasticProcess end

include("IntervalMarkovProcess.jl")
export IntervalMarkovProcess, AllStates
export num_states, num_actions, initial_states

include("FactoredRobustMarkovDecisionProcess.jl")
const FactoredRMDP = FactoredRobustMarkovDecisionProcess
export FactoredRobustMarkovDecisionProcess, state_variables, action_variables

include("DeterministicAutomaton.jl")

include("DFA.jl")
export DFA, transition, labelmap, initial_state, accepting_states

include("ProductProcess.jl")
export ProductProcess, markov_process, automaton, labelling_function
