
test_files = ["bellman.jl", "vi.jl", "imdp.jl", "synthesis.jl", "specification.jl", "product.jl"]
for f in test_files
    @testset "base/$f" include(f)
end
