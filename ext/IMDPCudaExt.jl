module IMDPCudaExt

using IMDP, CUDA, CUDA.CUSPARSE, Adapt, SparseArrays

Adapt.@adapt_structure MatrixIntervalProbabilities

include("cuda/ordering.jl")
include("cuda/probability_assignment.jl")

end