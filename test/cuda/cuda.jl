
@testset "cuda/adapt" begin
    adaptor = IMDP.CuModelAdaptor{Float64, Int32}

    @test valtype(adaptor) == Float64
    @test indtype(adaptor) == Int32
end


test_files = ["ominmax.jl", "partial.jl", "vi.jl", "imdp.jl"]
for f in test_files
    @testset "cuda/$f" begin
        include(f)
    end
end
