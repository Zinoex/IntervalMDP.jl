
test_files = [
    "probabilities.jl",
    "bellman.jl",
    "vi.jl",
    "imdp.jl",
    "synthesis.jl",
    "specification.jl",
    "factored.jl",
    # "mixture.jl",
    "labelling.jl",
    "dfa.jl",
    "product.jl",
]
for f in test_files
    @testset "base/$f" include(f)
end
