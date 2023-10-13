using IMDP
using Test

@testset "IMDP.jl" begin
    test_files = ["ominmax.jl"]
    for f in test_files
        @testset "$f" begin
            include(f)
        end
    end
end