abstract type AbstractPolicyCache end

struct NoPolicyCache <: AbstractPolicyCache end

struct TimeVaryingPolicyCache{T, VT <: AbstractVector{T}} <: AbstractPolicyCache
    cur_policy::VT
    policy::Vector{VT}
end

function TimeVaryingPolicyCache(cur_policy::VT) where {VT}
    return TimeVaryingPolicyCache(cur_policy, Vector{VT}())
end

function TimeVaryingPolicyCache(num_states, T::Type{<:Integer})
    cur_policy = zeros(T, num_states)
    return TimeVaryingPolicyCache(cur_policy)
end

function create_policy_cache(
    problem::Problem{M},
    time_varying::Val{true},
) where {I, M <: IntervalMarkovDecisionProcess{<:IntervalProbabilities, I}}
    return TimeVaryingPolicyCache(num_states(system(problem)), I)
end

function policy_indices_to_actions(problem::Problem{<:IntervalMarkovDecisionProcess}, policy_cache::TimeVaryingPolicyCache)
    mdp = system(problem)
    act = actions(mdp)
    return [act[Vector(indices)] for indices in policy_cache.policy]
end

struct StationaryPolicyCache{T, VT <: AbstractVector{T}} <: AbstractPolicyCache
    policy::VT
end

function StationaryPolicyCache(num_states, T::Type{<:Integer})
    cur_policy = zeros(T, num_states)
    return StationaryPolicyCache(cur_policy)
end

function create_policy_cache(
    problem::Problem{M},
    time_varying::Val{false},
) where {I, M <: IntervalMarkovDecisionProcess{<:IntervalProbabilities, I}}
    return StationaryPolicyCache(num_states(system(problem)), I)
end

function policy_indices_to_actions(problem::Problem{<:IntervalMarkovDecisionProcess}, policy_cache::StationaryPolicyCache)
    mdp = system(problem)
    act = actions(mdp)
    return act[Vector(policy_cache.policy)]
end