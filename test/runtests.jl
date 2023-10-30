using Test
using Random, StatsBase
using IMDP, SparseArrays, CUDA

@testset "IMDP.jl" begin
    test_files = ["ominmax.jl", "partial.jl", "ivi.jl"]
    for f in test_files
        @testset "$f" begin
            include(f)
        end
    end
end

@testset "IMDPSpareExt.jl" include("sparse/sparse.jl")

if CUDA.functional()
    @info "Running tests with CUDA"
    @testset "IMDPCudaExt.jl" include("cuda/cuda.jl")
end
