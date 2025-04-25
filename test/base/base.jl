
test_files = [
    "bellman.jl",
    "vi.jl",
    "imdp.jl",
    "synthesis.jl",
    "specification.jl",
    "orthogonal.jl",
    "mixture.jl",
    "transitionfunc.jl",
    "dfa.jl",
    "labelling.jl",
    "prodimdp.jl",
]
for f in test_files
    @testset "base/$f" include(f)
end
