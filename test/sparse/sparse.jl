
test_files = ["ominmax.jl", "partial.jl", "ivi.jl"]
for f in test_files
    @testset "sparse/$f" begin
        include(f)
    end
end