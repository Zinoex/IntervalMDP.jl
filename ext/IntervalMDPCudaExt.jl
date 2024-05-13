module IntervalMDPCudaExt

import LLVM
using LLVM.Interop: assume

using CUDA, CUDA.CUSPARSE, Adapt, SparseArrays

using IntervalMDP, LinearAlgebra

Adapt.@adapt_structure IntervalProbabilities

# Opinionated conversion to GPU with Float64 values and Int32 indices
IntervalMDP.cu(model) = adapt(IntervalMDP.CuModelAdaptor{Float64, Int32}, model)

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mc::IntervalMarkovChain,
)
    Itype = IntervalMDP.indtype(T)
    return IntervalMarkovChain(
        adapt(T, transition_prob(mc)),
        adapt(CuArray{Itype}, initial_states(mc)),
        Itype(num_states(mc)),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    mdp::IntervalMarkovDecisionProcess,
)
    Itype = IntervalMDP.indtype(T)
    return IntervalMarkovDecisionProcess(
        adapt(T, transition_prob(mdp)),
        adapt(CuArray{Itype}, IntervalMDP.stateptr(mdp)),
        actions(mdp),
        adapt(CuArray{Itype}, initial_states(mdp)),
        Itype(num_states(mdp)),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    policy_cache::IntervalMDP.TimeVaryingPolicyCache,
)
    Itype = IntervalMDP.indtype(T)
    return IntervalMDP.TimeVaryingPolicyCache(
        adapt(CuArray{Itype}, policy_cache.cur_policy),
        adapt.(CuArray{Itype}, policy_cache.policy),
    )
end

function Adapt.adapt_structure(
    T::Type{<:IntervalMDP.CuModelAdaptor},
    policy_cache::IntervalMDP.StationaryPolicyCache,
)
    Itype = IntervalMDP.indtype(T)
    return IntervalMDP.StationaryPolicyCache(adapt(CuArray{Itype}, policy_cache.cur_policy))
end

include("cuda/utils.jl")
include("cuda/array.jl")
include("cuda/ordering.jl")
include("cuda/probability_assignment.jl")
include("cuda/value_iteration.jl")
include("cuda/interval_probabilities.jl")
include("cuda/specification.jl")

end
