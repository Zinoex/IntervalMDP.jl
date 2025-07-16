module IntervalMDPCudaExt

import LLVM
using LLVM.Interop: assume

using CUDA, CUDA.CUSPARSE, Adapt, SparseArrays
using GPUArrays: AbstractGPUArray, AbstractGPUVector, AbstractGPUMatrix

using IntervalMDP, LinearAlgebra

Adapt.@adapt_structure IntervalProbabilities
Adapt.@adapt_structure OrthogonalIntervalProbabilities
Adapt.@adapt_structure StationaryStrategy
Adapt.@adapt_structure TimeVaryingStrategy

# Opinionated conversion to GPU with Float64 values and Int32 indices
IntervalMDP.cu(model) = adapt(IntervalMDP.CuModelAdaptor{Float64}, model)
IntervalMDP.cpu(model) = adapt(IntervalMDP.CpuModelAdaptor{Float64}, model)

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
    T::Type{<:IntervalMDP.CpuModelAdaptor},
    mdp::IntervalMarkovDecisionProcess,
)
    return IntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(Array{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(Array{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mdp::OrthogonalIntervalMarkovDecisionProcess,
)
    return OrthogonalIntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(CuArray{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(CuArray{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CpuModelAdaptor},
    mdp::OrthogonalIntervalMarkovDecisionProcess,
)
    return OrthogonalIntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(Array{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(Array{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mdp::MixtureIntervalMarkovDecisionProcess,
)
    return MixtureIntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(CuArray{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(CuArray{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CpuModelAdaptor},
    mdp::MixtureIntervalMarkovDecisionProcess,
)
    return MixtureIntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(Array{Int32}, IntervalMDP.stateptr(mdp)),
        adapt(Array{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

Adapt.adapt_structure(T::Type{<:IntervalMDP.CuModelAdaptor}, is::AllStates) = is
Adapt.adapt_structure(T::Type{<:IntervalMDP.CpuModelAdaptor}, is::AllStates) = is

function IntervalMDP.checkdevice(::AbstractGPUArray, ::AbstractGPUMatrix)
    # Both arguments are on the GPU.
    return nothing
end

function IntervalMDP.checkdevice(::AbstractGPUArray, ::AbstractCuSparseMatrix)
    # Both arguments are on the GPU.
    return nothing
end

function IntervalMDP.checkdevice(b::AbstractArray, A::AbstractGPUMatrix)
    # The first argument is on the CPU (technically in RAM) and the second is on the GPU.
    @assert false "The reward vector is a CPU array ($(typeof(b))) and the transition matrix is on the GPU ($(typeof(A)))."
end

function IntervalMDP.checkdevice(b::AbstractArray, A::AbstractCuSparseMatrix)
    # The first argument is on the CPU (technically in RAM) and the second is on the GPU.
    @assert false "The reward vector is a CPU array ($(typeof(b))) and the transition matrix is on the GPU ($(typeof(A)))."
end

function IntervalMDP.checkdevice(b::AbstractGPUArray, A::AbstractMatrix)
    # The first argument is on the GPU and the second is on the CPU (technically in RAM).
    @assert false "The reward vector is on the GPU ($(typeof(b))) and the transition matrix is a CPU matrix ($(typeof(A)))."
end

IntervalMDP.arrayfactory(
    ::MR,
    T,
    num_states,
) where {R, MR <: Union{CuSparseMatrixCSC{R}, CuArray{R}}} = CUDA.zeros(T, num_states)

include("cuda/utils.jl")
include("cuda/array.jl")
include("cuda/sorting.jl")
include("cuda/workspace.jl")
include("cuda/strategy.jl")
include("cuda/bellman/dense.jl")
include("cuda/bellman/sparse.jl")
include("cuda/interval_probabilities.jl")
include("cuda/specification.jl")

end
