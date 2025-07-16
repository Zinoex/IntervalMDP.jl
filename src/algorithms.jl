abstract type AbstractIntervalMDPAlgorithm end

##########################
# Robust Value Iteration #
##########################
"""
    RobustValueIteration

A robust value iteration algorithm for solving interval Markov decision processes (IMDPs) with interval ambiguity sets.
This algorithm is designed to handle both finite and infinite time specifications, optimizing for either the maximum or
minimum expected value based on the given specification.
"""
struct RobustValueIteration <: AbstractIntervalMDPAlgorithm end

############################
# Interval Value Iteration #
############################

# TODO: Provide implementation for this algorithm. When provided, consider changing the default algorithm.
struct IntervalValueIteration <: AbstractIntervalMDPAlgorithm end

# TODO: Consider topological value iteration as an alternative algorithm (infinite time only).

##### Default algorithm for solving Interval MDP problems
default_algorithm(::AbstractIntervalMDPProblem) = RobustValueIteration()

solve(problem::AbstractIntervalMDPProblem; kwargs...) =
    solve(problem, default_algorithm(problem); kwargs...)
