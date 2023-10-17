using Revise, BenchmarkTools
using IMDP, CUDA

display(@benchmark ominmax!($ordering, $p, $prob, $V; max = true) seconds = 60)

if CUDA.functional()
    display(
        @benchmark ominmax!($cuda_ordering, $cuda_p, $cuda_prob, $cuda_V; max = true) seconds =
            60
    )
end
