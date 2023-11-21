
test_files = ["ominmax.jl", "partial.jl", "vi.jl"]
for f in test_files
    @testset "cuda/$f" begin
        include(f)
    end
end
