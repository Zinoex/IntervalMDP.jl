using Revise, BenchmarkTools, ProgressMeter
using Random, StatsBase
using IMDP, IMDP.Data, SparseArrays, CUDA, Adapt

path = joinpath(@__DIR__, "multiObj_robotIMDP.txt")
mdp, terminal_states = read_bmdp_tool_file(path)
prob = Problem(mdp, InfiniteTimeReachability(terminal_states, num_states(mdp), 1e-6))
