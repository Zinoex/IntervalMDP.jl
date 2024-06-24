
test_files = ["bellman.jl", "vi.jl", "synthesis.jl"]
for f in test_files
    @testset "parallel/$f" include(f)
end
