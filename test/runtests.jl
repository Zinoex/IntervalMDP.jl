using Test
using Random, StatsBase
using IntervalMDP, SparseArrays, CUDA

@testset "IntervalMDP.jl" begin
    test_files =
        ["ominmax.jl", "partial.jl", "vi.jl", "imdp.jl", "synthesis.jl", "specification.jl"]
    for f in test_files
        @testset "$f" begin
            include(f)
        end
    end
end

@testset "sparse" include("sparse/sparse.jl")
@testset "data" include("data/data.jl")

if CUDA.functional()
    @info "Running tests with CUDA"
    @testset "IMDPCudaExt.jl" include("cuda/cuda.jl")
end
