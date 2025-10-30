
test_files = ["probabilities.jl", "bellman.jl", "vi.jl", "imdp.jl", "synthesis.jl", "factored.jl"]

for f in test_files
    @testset "sparse/$f" include(f)
end
