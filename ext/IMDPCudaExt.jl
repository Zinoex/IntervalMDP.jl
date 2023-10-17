module IMDPCudaExt

import LLVM
using LLVM.Interop: assume

using CUDA, CUDA.CUSPARSE, Adapt, SparseArrays

using IMDP

Adapt.@adapt_structure MatrixIntervalProbabilities

include("cuda/array.jl")
include("cuda/ordering.jl")
include("cuda/probability_assignment.jl")

end
