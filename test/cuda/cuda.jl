
@testset "cuda/adapt" begin
    adaptor = IntervalMDP.CuModelAdaptor{Float64, Int32}

    @test IntervalMDP.valtype(adaptor) == Float64
    @test IntervalMDP.indtype(adaptor) == Int32
end

test_files = ["ominmax.jl", "partial.jl", "vi.jl", "imdp.jl", "synthesis.jl"]
for f in test_files
    @testset "cuda/$f" begin
        include(f)
    end
end
