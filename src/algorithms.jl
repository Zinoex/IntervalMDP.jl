abstract type BellmanAlgorithm end
struct OMaximization <: BellmanAlgorithm end
Base.@kwdef struct LPMcCormickRelaxation{O} <: BellmanAlgorithm
    lp_optimizer::O = HiGHS.Optimizer
end

default_bellman_algorithm(::IMDP) = OMaximization()
default_bellman_algorithm(::IntervalAmbiguitySets) = OMaximization()
default_bellman_algorithm(::FactoredRMDP{N, M, <:NTuple{N, <:Marginal{<:PolytopicAmbiguitySets}}}) where {N, M} = LPMcCormickRelaxation()
default_bellman_algorithm(pp::ProductProcess) = default_bellman_algorithm(markov_process(pp))

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
default_algorithm(problem::AbstractIntervalMDPProblem) = RobustValueIteration(default_bellman_algorithm(system(problem)))

solve(problem::AbstractIntervalMDPProblem; kwargs...) =
    solve(problem, default_algorithm(problem); kwargs...)
