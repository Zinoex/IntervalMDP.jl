using Test

@testset verbose = true "IntervalMDP.jl" begin
    @testset verbose = true "base" include("base/base.jl")
    @testset verbose = true "sparse" include("sparse/sparse.jl")
    @testset verbose = true "data" include("data/data.jl")
    @testset verbose = true "cuda" include("cuda/cuda.jl")
end
