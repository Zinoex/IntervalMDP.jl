module IntervalMDPCudaExt

import LLVM
using LLVM.Interop: assume

using CUDA, CUDA.CUSPARSE, Adapt, SparseArrays

using IntervalMDP, LinearAlgebra

Adapt.@adapt_structure IntervalProbabilities

# Opinionated conversion to GPU with Float64 values and Int32 indices
IntervalMDP.cu(model) = adapt(IntervalMDP.CuModelAdaptor{Float64}, model)

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mc::IntervalMarkovChain,
)
    return IntervalMarkovChain(
        adapt(T, transition_prob(mc)),
        adapt(CuArray{Int32}, initial_states(mc)),
        num_states(mc),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mdp::IntervalMarkovDecisionProcess,
)
    return IntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(CuArray{Int32}, IntervalMDP.stateptr(mdp)),
        actions(mdp),
        adapt(CuArray{Int32}, initial_states(mdp)),
        num_states(mdp),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    policy_cache::IntervalMDP.TimeVaryingPolicyCache,
)
    return IntervalMDP.TimeVaryingPolicyCache(
        adapt(CuArray{Int32}, policy_cache.cur_policy),
        adapt.(CuArray{Int32}, policy_cache.policy),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    policy_cache::IntervalMDP.StationaryPolicyCache,
)
    return IntervalMDP.StationaryPolicyCache(adapt(CuArray{Int32}, policy_cache.policy))
end

include("cuda/utils.jl")
include("cuda/array.jl")
include("cuda/ordering.jl")
include("cuda/value_assignment.jl")
include("cuda/value_iteration.jl")
include("cuda/interval_probabilities.jl")
include("cuda/specification.jl")

end
