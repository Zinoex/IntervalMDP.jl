using Revise, BenchmarkTools
using IntervalMDP, CUDA

V_conv, _, u = solve(prob)

display(@benchmark solve(prob))
