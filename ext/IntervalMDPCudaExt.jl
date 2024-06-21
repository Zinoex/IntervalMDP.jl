module IntervalMDPCudaExt

import LLVM
using LLVM.Interop: assume

using CUDA, CUDA.CUSPARSE, Adapt, SparseArrays
using GPUArrays: AbstractGPUArray, AbstractGPUVector, AbstractGPUMatrix

using IntervalMDP, LinearAlgebra

Adapt.@adapt_structure IntervalProbabilities

# Opinionated conversion to GPU with Float64 values and Int32 indices
IntervalMDP.cu(model) = adapt(IntervalMDP.CuModelAdaptor{Float64}, model)

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mdp::IntervalMarkovDecisionProcess,
)
    return IntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(CuArray{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(CuArray{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mdp::TimeVaryingIntervalMarkovDecisionProcess,
)
    return TimeVaryingIntervalMarkovDecisionProcess(
        adapt.(T, IntervalMDP.transition_probs(mdp)),
        adapt(CuArray{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(CuArray{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(T::Type{<:IntervalMDP.CuModelAdaptor}, mdp::ParallelProduct)
    return ParallelProduct(
        convert(
            Vector{IntervalMarkovProcess},
            adapt.(T, IntervalMDP.orthogonal_processes(mdp)),
        ),
        adapt(CuArray{Int32}, initial_states(mdp)),
        num_states(mdp),
        IntervalMDP.subdims(mdp),
    )
end

Adapt.adapt_structure(T::Type{<:IntervalMDP.CuModelAdaptor}, is::AllStates) = is

include("cuda/utils.jl")
include("cuda/array.jl")
include("cuda/sorting.jl")
include("cuda/workspace.jl")
include("cuda/strategy.jl")
include("cuda/bellman/dense.jl")
include("cuda/bellman/sparse.jl")
include("cuda/value_iteration.jl")
include("cuda/interval_probabilities.jl")
include("cuda/specification.jl")

end
