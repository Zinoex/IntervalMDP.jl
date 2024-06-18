
@testset "cuda/adapt" begin
    adaptor = IntervalMDP.CuModelAdaptor{Float64}

    @test IntervalMDP.valtype(adaptor) == Float64
end

test_files = [
    # "dense/bellman.jl",
    # "dense/vi.jl",
    # "dense/imdp.jl",
    # "dense/synthesis.jl",
    # "sparse/bellman.jl",
    # "sparse/vi.jl",
    # "sparse/imdp.jl",
    # "sparse/synthesis.jl",
    "parallel/bellman.jl",
    # "parallel/vi.jl",
    # "parallel/synthesis.jl",
]
for f in test_files
    @testset "cuda/$f" include(f)
end
