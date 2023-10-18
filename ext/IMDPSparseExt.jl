module IMDPSparseExt
using IMDP, SparseArrays

include("sparse/interval_probabilities.jl")
include("sparse/ordering.jl")
include("sparse/probability_assignment.jl")

end
