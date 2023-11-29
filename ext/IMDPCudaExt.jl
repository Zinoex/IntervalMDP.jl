module IMDPCudaExt

import LLVM
using LLVM.Interop: assume

using CUDA, CUDA.CUSPARSE, Adapt, SparseArrays

using IMDP, LinearAlgebra

Adapt.@adapt_structure IntervalProbabilities

# Opinionated conversion to GPU with Float64 values and Int32 indices
IMDP.cu(model) = adapt(IMDP.CuModelAdaptor{Float64, Int32}, model)

function Adapt.adapt_structure(T::Type{<:IMDP.CuModelAdaptor}, mc::IntervalMarkovChain)
    return IntervalMarkovChain(
        adapt(T, transition_prob(mc)),
        initial_state(mc),
        num_states(mc),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IMDP.CuModelAdaptor},
    mdp::IntervalMarkovDecisionProcess,
)
    return IntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(CuArray{IMDP.indtype(T)}, IMDP.stateptr(mdp)),
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
include("cuda/specification.jl")

end
