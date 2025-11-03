using Revise, BenchmarkTools, ProgressMeter
using Random, StatsBase
using IntervalMDP, IntervalMDP.Data, SparseArrays, CUDA, Adapt

path = joinpath(@__DIR__, "multiObj_robotIMDP.txt")
mdp, terminal_states = read_bmdp_tool_file(path)

marginal = marginals(mdp)[1]
amb = IntervalAmbiguitySets(
    Array(ambiguity_sets(marginal).lower),
    Array(ambiguity_sets(marginal).gap),
)

mdp = IntervalMarkovDecisionProcess(amb, num_actions(mdp), initial_states(mdp))
prop = FiniteTimeReachability(terminal_states, 100)
spec = Specification(prop, Pessimistic, Maximize)
prob = VerificationProblem(mdp, spec)
