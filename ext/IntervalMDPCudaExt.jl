module IntervalMDPCudaExt

import LLVM
using LLVM.Interop: assume

using CUDA, CUDA.CUSPARSE, Adapt, SparseArrays

using IntervalMDP, LinearAlgebra

Adapt.@adapt_structure IntervalProbabilities

# Opinionated conversion to GPU with Float64 values and Int32 indices
IntervalMDP.cu(model) = adapt(IntervalMDP.CuModelAdaptor{Float64, Int32}, model)

function Adapt.adapt_structure(T::Type{<:IntervalMDP.CuModelAdaptor}, mc::IntervalMarkovChain)
    return IntervalMarkovChain(
        adapt(T, transition_prob(mc)),
        adapt(CuArray{IntervalMDP.indtype(T)}, initial_states(mc)),
        num_states(mc),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mdp::IntervalMarkovDecisionProcess,
)
    return IntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(CuArray{IntervalMDP.indtype(T)}, IntervalMDP.stateptr(mdp)),
        actions(mdp),
        adapt(CuArray{IntervalMDP.indtype(T)}, initial_states(mdp)),
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
