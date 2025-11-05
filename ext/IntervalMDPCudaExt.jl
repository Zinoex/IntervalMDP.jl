module IntervalMDPCudaExt

import LLVM
using LLVM.Interop: assume

using CUDA, CUDA.CUSPARSE, Adapt, SparseArrays
using GPUArrays: AbstractGPUArray, AbstractGPUVector, AbstractGPUMatrix

using IntervalMDP, LinearAlgebra

# Opinionated conversion to GPU with preserved value types and Int32 indices
IntervalMDP.cu(obj) = adapt(IntervalMDP.CuModelAdaptor{IntervalMDP.valuetype(obj)}, obj)
IntervalMDP.cpu(obj) = adapt(IntervalMDP.CpuModelAdaptor{IntervalMDP.valuetype(obj)}, obj)

Adapt.@adapt_structure Marginal
Adapt.@adapt_structure StationaryStrategy
Adapt.adapt_structure(to, strategy::TimeVaryingStrategy) =
    TimeVaryingStrategy([adapt(to, s) for s in strategy.strategy])

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mdp::IntervalMDP.FactoredRMDP,
)
    return IntervalMDP.FactoredRMDP(
        state_values(mdp),
        action_values(mdp),
        IntervalMDP.source_shape(mdp),
        adapt(T, marginals(mdp)),
        adapt(CuArray{eltype(initial_states(mdp))}, initial_states(mdp)),
        Val(false), # check = false
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CpuModelAdaptor},
    mdp::IntervalMDP.FactoredRMDP,
)
    return IntervalMDP.FactoredRMDP(
        state_values(mdp),
        action_values(mdp),
        IntervalMDP.source_shape(mdp),
        adapt(T, marginals(mdp)),
        adapt(Array{eltype(initial_states(mdp))}, initial_states(mdp)),
        Val(false), # check = false
    )
end

function Adapt.adapt_structure(
    to,
    mdp::IntervalMDP.FactoredRMDP,
)
    return IntervalMDP.FactoredRMDP(
        state_values(mdp),
        action_values(mdp),
        IntervalMDP.source_shape(mdp),
        adapt(to, marginals(mdp)),
        adapt(to, initial_states(mdp)),
        Val(false), # check = false
    )
end

function Adapt.adapt_structure(to, as::IntervalAmbiguitySets)
    return IntervalAmbiguitySets(
        adapt(to, as.lower),
        adapt(to, as.gap),
        Val(false), # check = false
    )
end

Adapt.@adapt_structure IntervalMDP.AllStates

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
    sizes,
) where {R, MR <: Union{CuSparseMatrixCSC{R}, CuArray{R}}} = CuArray{T}(undef, sizes)

include("cuda/utils.jl")
include("cuda/array.jl")
include("cuda/sorting.jl")
include("cuda/workspace.jl")
include("cuda/strategy.jl")
include("cuda/bellman/dense.jl")
include("cuda/bellman/sparse.jl")
include("cuda/bellman/factored.jl")
include("cuda/probabilities.jl")
include("cuda/specification.jl")

end
