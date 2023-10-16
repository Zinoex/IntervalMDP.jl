using IMDP, SparseArrays, CUDA
using Test

@testset "IMDP.jl" begin
    test_files = ["ominmax.jl", "partial.jl", "ivi.jl"]
    for f in test_files
        @testset "$f" begin
            include(f)
        end
    end
end

@testset "IMDPSpareExt.jl" begin
    test_files = ["sparse.jl"]
    for f in test_files
        @testset "$f" begin
            include(f)
        end
    end
end

@testset "IMDPCudaExt.jl" begin
    if CUDA.functional()
        @info "Running tests with CUDA"
        test_files = ["cuda.jl"]
        for f in test_files
            @testset "$f" begin
                include(f)
            end
        end
    end
end