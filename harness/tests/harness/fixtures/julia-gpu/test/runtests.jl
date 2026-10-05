# GPU fixture test entry point (layout mirrors IntervalMDP.jl: CUDA is a weak
# dependency plus a test-target extra).
#
# Markers parsed by tools/harness/gpu_check.py (exit code alone NEVER yields PASS):
#   HARNESS_GPU_UNAVAILABLE: <reason>   CUDA loaded but no functional device -> exit 3
#   HARNESS_GPU_TESTS_RAN: ...          printed only after the CUDA testset ran and passed
using Test, JuliaGpu

@testset "JuliaGpu CPU path" begin
    @test bellman_step([1.0], ones(1, 1), [1.0]) == [2.0]
end

if get(ENV, "FIXTURE_TEST_GPU", "false") == "true"
    using CUDA  # a load failure here is a setup error (FAIL), not "no GPU"
    if !(CUDA.functional() && length(CUDA.devices()) > 0)
        println("HARNESS_GPU_UNAVAILABLE: CUDA.functional()=", CUDA.functional(), " (no usable CUDA device)")
        exit(3)
    end
    @testset "JuliaGpu CUDA path" begin
        r = CUDA.CuArray([1.0, 2.0])
        P = CUDA.CuArray([1.0 0.0; 0.0 1.0])
        v = CUDA.CuArray([3.0, 4.0])
        out = bellman_step(r, P, v)
        @test out isa CUDA.CuArray
        @test Array(out) == [4.0, 6.0]
        @test sum(CUDA.ones(4)) == 4.0
    end
    println("HARNESS_GPU_TESTS_RAN: device=", CUDA.name(CUDA.device()))
end
