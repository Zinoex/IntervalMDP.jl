
test_files = ["ominmax.jl", "partial.jl", "ivi.jl"]
for f in test_files
    @testset "cuda/$f" begin
        include(f)
    end
end