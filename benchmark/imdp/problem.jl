using Revise, BenchmarkTools, ProgressMeter
using Random, StatsBase
using IntervalMDP, IntervalMDP.Data, SparseArrays, CUDA, Adapt

path = joinpath(@__DIR__, "multiObj_robotIMDP.txt")
mdp, terminal_states = read_bmdp_tool_file(path)
prop = FiniteTimeReachability(terminal_states, 100)
spec = Specification(prop, Pessimistic, Maximize)
prob = VerificationProblem(mdp, spec)
