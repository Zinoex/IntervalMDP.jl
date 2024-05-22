
@testset "cuda/adapt" begin
    adaptor = IntervalMDP.CuModelAdaptor{Float64, Int32}

    @test IntervalMDP.valtype(adaptor) == Float64
    @test IntervalMDP.indtype(adaptor) == Int32
end

test_files = [
    "sparse/bellman.jl",
    "sparse/vi.jl",
    "sparse/imdp.jl",
    "sparse/synthesis.jl",
    "dense/bellman.jl",
    "dense/vi.jl",
    "dense/imdp.jl",
    "dense/synthesis.jl",
]
for f in test_files
    @testset "cuda/$f" begin
        include(f)
    end
end
