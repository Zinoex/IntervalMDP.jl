
test_files = ["bellman.jl", "vi.jl", "imdp.jl", "synthesis.jl"]
for f in test_files
    @testset "sparse/$f" begin
        include(f)
    end
end
