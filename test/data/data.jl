using IntervalMDP.Data

test_files = ["bmdp_tool.jl", "prism.jl", "intervalmdp.jl"]
for f in test_files
    @testset "data/$f" include(f)
end
