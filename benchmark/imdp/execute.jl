using Revise, BenchmarkTools
using IntervalMDP, CUDA

cu_prob = IntervalMDP.cu(prob)

function test()
    CUDA.@sync solve(cu_prob)
end

test()  # Warm-up
display(@benchmark test())
