using Revise, BenchmarkTools
using IntervalMDP, CUDA

display(@benchmark bellman!($ordering, $Vres, $V, $prob; max = true) seconds = 60)

if CUDA.functional()
    display(
        @benchmark CUDA.@sync(
            bellman!($cuda_ordering, $cuda_Vres, $cuda_V, $cuda_prob; max = true)
        ) seconds = 60
    )
end
