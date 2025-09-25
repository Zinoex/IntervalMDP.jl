abstract type BellmanAlgorithm end
struct OMaximization <: BellmanAlgorithm end
Base.@kwdef struct LPMcCormickRelaxation{O} <: BellmanAlgorithm
    lp_solver::O = HiGHS.Optimizer
end
struct VertexEnumeration <: BellmanAlgorithm end

default_bellman_algorithm(pp::ProductProcess) = default_bellman_algorithm(markov_process(pp))
default_bellman_algorithm(mdp::FactoredRMDP) = default_bellman_algorithm(mdp, modeltype(mdp))
default_bellman_algorithm(::FactoredRMDP, ::IsIMDP) = OMaximization()
default_bellman_algorithm(::FactoredRMDP, ::IsFIMDP) = LPMcCormickRelaxation()
default_bellman_algorithm(::IntervalAmbiguitySets) = OMaximization()

function showbellmanalg(io::IO, prefix, ::IsIMDP,::OMaximization)
    println(io, prefix, "└─", styled"Default Bellman operator algorithm: {green:O-Maximization}")
end

function showbellmanalg(io::IO, prefix, ::IsFIMDP,::OMaximization)
    println(io, prefix, "└─", styled"Default Bellman operator algorithm: {green:Recursive O-Maximization}")
end

function showbellmanalg(io::IO, prefix, ::IsFIMDP, ::LPMcCormickRelaxation)
    println(io, prefix, "└─", styled"Default Bellman operator algorithm: {green:Binary tree LP McCormick Relaxation}")
end

function showbellmanalg(io::IO, prefix, ::IsFIMDP, ::VertexEnumeration)
    println(io, prefix, "└─", styled"Default Bellman operator algorithm: {green:Vertex Enumeration}")
end

function showbellmanalg(io::IO, prefix, _, ::BellmanAlgorithm)
    println(io, prefix, "└─", styled"Default Bellman operator algorithm: {green:None}")
end

abstract type ModelCheckingAlgorithm end

##########################
# Robust Value Iteration #
##########################
"""
    RobustValueIteration

A robust value iteration algorithm for solving interval Markov decision processes (IMDPs) with interval ambiguity sets.
This algorithm is designed to handle both finite and infinite time specifications, optimizing for either the maximum or
minimum expected value based on the given specification.
"""
struct RobustValueIteration{B <: BellmanAlgorithm} <: ModelCheckingAlgorithm
    bellman_alg::B
end
bellman_algorithm(alg::RobustValueIteration) = alg.bellman_alg

############################
# Interval Value Iteration #
############################

# TODO: Provide implementation for this algorithm. When provided, consider changing the default algorithm.
struct IntervalValueIteration <: ModelCheckingAlgorithm end

# TODO: Consider topological value iteration as an alternative algorithm (infinite time only).

##### Default algorithm for solving Interval MDP problems
default_algorithm(problem::AbstractIntervalMDPProblem) = default_algorithm(system(problem))
default_algorithm(system::StochasticProcess) = RobustValueIteration(default_bellman_algorithm(system))

solve(problem::AbstractIntervalMDPProblem; kwargs...) =
    solve(problem, default_algorithm(problem); kwargs...)


function showmcalgorithm(io::IO, prefix, ::RobustValueIteration)
    println(io, prefix,"├─", styled"Default model checking algorithm: {green:Robust Value Iteration}")
end

function showmcalgorithm(io::IO, prefix, ::ModelCheckingAlgorithm)
    println(io, prefix,"├─", styled"Default model checking algorithm: {green:None}")
end