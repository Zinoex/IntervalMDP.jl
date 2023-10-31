using Revise, BenchmarkTools
using IMDP, CUDA

V_conv, _, u = interval_value_iteration(prob; maximize = true, upper_bound = false)
display(@benchmark interval_value_iteration(prob; maximize = true, upper_bound = false))