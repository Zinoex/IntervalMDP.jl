
test_files = ["vi.jl", "synthesis.jl"]
for f in test_files
    @testset "sequential/$f" include(f)
end
