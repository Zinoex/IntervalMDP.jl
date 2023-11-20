module IMDPCudaExt

import LLVM
using LLVM.Interop: assume

using CUDA, CUDA.CUSPARSE, Adapt, SparseArrays

using IMDP

Adapt.@adapt_structure IntervalProbabilities

function Adapt.adapt_structure(::Type{<:CuArray}, mdp::IntervalMarkovChain)
    return IntervalMarkovChain(
        adapt(CuArray{Float64}, transition_prob(mdp)),
        initial_state(mdp),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(::Type{<:CuArray}, mdp::IntervalMarkovDecisionProcess)
    return IntervalMarkovDecisionProcess(
        adapt(CuArray{Float64}, transition_prob(mdp)),
        adapt(CuArray{Int32}, IMDP.stateptr(mdp)),
        actions(mdp),
        initial_state(mdp),
        num_states(mdp),
    )
end

include("cuda/array.jl")
include("cuda/ordering.jl")
include("cuda/probability_assignment.jl")
include("cuda/value_iteration.jl")
include("cuda/interval_probabilities.jl")

end
