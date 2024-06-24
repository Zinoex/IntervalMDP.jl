using Test
using Random, StatsBase
using IntervalMDP, SparseArrays, CUDA

@testset verbose = true "IntervalMDP.jl" begin
    @testset verbose = true "base" include("base/base.jl")
    @testset verbose = true "sparse" include("sparse/sparse.jl")
    @testset verbose = true "parallel_product" include("parallel_product/parallel.jl")
    @testset verbose = true "data" include("data/data.jl")

    if CUDA.functional()
        @info "Running tests with CUDA"
        @testset verbose = true "cuda" include("cuda/cuda.jl")
    end
end
