abstract type AbstractPolicyCache end

struct NoPolicyCache <: AbstractPolicyCache end

struct TimeVaryingPolicyCache{VT <: AbstractVector{Int32}} <: AbstractPolicyCache
    cur_policy::VT
    policy::Vector{VT}
end

function TimeVaryingPolicyCache(cur_policy::VT) where {VT}
    return TimeVaryingPolicyCache(cur_policy, Vector{VT}())
end

function TimeVaryingPolicyCache(num_states::Integer)
    cur_policy = zeros(Int32, num_states)
    return TimeVaryingPolicyCache(cur_policy)
end

function create_policy_cache(
    problem::Problem{<:IntervalMarkovDecisionProcess},
    time_varying::Val{true},
)
    return TimeVaryingPolicyCache(num_states(system(problem)))
end

function policy_indices_to_actions(
    problem::Problem{<:IntervalMarkovDecisionProcess},
    policy_cache::TimeVaryingPolicyCache,
)
    mdp = system(problem)
    act = actions(mdp)
    return [act[Vector(indices)] for indices in policy_cache.policy]
end

struct StationaryPolicyCache{VT <: AbstractVector{Int32}} <: AbstractPolicyCache
    policy::VT
end

function StationaryPolicyCache(num_states::Integer)
    cur_policy = zeros(Int32, num_states)
    return StationaryPolicyCache(cur_policy)
end

function create_policy_cache(
    problem::Problem{<:IntervalMarkovDecisionProcess},
    time_varying::Val{false},
)
    return StationaryPolicyCache(num_states(system(problem)))
end

function policy_indices_to_actions(
    problem::Problem{<:IntervalMarkovDecisionProcess},
    policy_cache::StationaryPolicyCache,
)
    mdp = system(problem)
    act = actions(mdp)
    return act[Vector(policy_cache.policy)]
end
