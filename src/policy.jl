# Abstract type
abstract type AbstractPolicyCache end

max_actions(cache::AbstractPolicyCache) = maximum(diff(cache.stateptr))

struct StateIterator
    stateptr::Vector{Int32}
end
Base.eltype(::StateIterator) = Tuple{Int32, Int32}
Base.length(si::StateIterator) = length(si.stateptr) - 1
Base.iterate(si::StateIterator, state = 1) = state == length(si) + 1 ? nothing : ((state, (si.stateptr[state], si.stateptr[state + 1])), state + 1)
Base.firstindex(si::StateIterator) = 1
Base.lastindex(si::StateIterator) = length(si)
Base.getindex(si::StateIterator, i) = (i, (si.stateptr[i], si.stateptr[i + 1]))

iterate_states(cache::AbstractPolicyCache) = StateIterator(cache.stateptr)

# Policy cache for not storing policies - useful for dispatching
struct NoPolicyCache{VT <: AbstractVector{Int32}} <: AbstractPolicyCache 
    stateptr::VT
end

NoPolicyCache(prob::IntervalProbabilities) = NoPolicyCache(stateptr(prob))
NoPolicyCache(mp::IntervalMarkovProcess) = NoPolicyCache(stateptr(mp))

function extract_policy!(
    ::NoPolicyCache,
    values,
    V,
    j,
    maximize,
)
    return maximize ? maximum(values) : minimum(values)
end
postprocess_policy_cache!(::NoPolicyCache) = nothing

# Policy cache for storing time-varying policies
struct TimeVaryingPolicyCache{VT <: AbstractVector{Int32}} <: AbstractPolicyCache
    stateptr::VT
    cur_policy::VT
    policy::Vector{VT}
end

function TimeVaryingPolicyCache(stateptr, cur_policy::VT) where {VT}
    return TimeVaryingPolicyCache(stateptr, cur_policy, Vector{VT}())
end

function create_policy_cache(
    mdp::M,
    time_varying::Val{true},
) where {R, P <: IntervalProbabilities{R, <:AbstractVector{R}}, M <: IntervalMarkovDecisionProcess{P}}
    cur_policy = zeros(Int32, num_states(mdp))
    return TimeVaryingPolicyCache(stateptr(mdp), cur_policy)
end

function policy_indices_to_actions(
    problem::Problem{<:IntervalMarkovDecisionProcess},
    policy_cache::TimeVaryingPolicyCache,
)
    mdp = system(problem)
    act = actions(mdp)
    return [act[Vector(indices)] for indices in policy_cache.policy]
end

function extract_policy!(
    policy_cache::TimeVaryingPolicyCache,
    values,
    V,
    j,
    maximize,
)
    gt = maximize ? (>) : (<)

    @inbounds s₁ = policy_cache.stateptr[j]

    opt_val = values[1]
    opt_index = s₁

    for (i, v) in Iterators.drop(enumerate(values), 1)
        if gt(v, opt_val)
            opt_val = v
            opt_index = s₁ + i - 1
        end
    end

    @inbounds policy_cache.cur_policy[j] = opt_index
    return opt_val
end
function postprocess_policy_cache!(policy_cache::TimeVaryingPolicyCache)
    push!(policy_cache.policy, copy(policy_cache.cur_policy))
end

# Policy cache for storing stationary policies
struct StationaryPolicyCache{VT <: AbstractVector{Int32}} <: AbstractPolicyCache
    stateptr::VT
    policy::VT
end

function create_policy_cache(
    mdp::M,
    time_varying::Val{false},
) where {R, P <: IntervalProbabilities{R, <:AbstractVector{R}}, M <: IntervalMarkovDecisionProcess{P}}
    cur_policy = zeros(Int32, num_states(mdp))
    return StationaryPolicyCache(stateptr(mdp), cur_policy)
end

function policy_indices_to_actions(
    problem::Problem{<:IntervalMarkovDecisionProcess},
    policy_cache::StationaryPolicyCache,
)
    mdp = system(problem)
    act = actions(mdp)
    return act[Vector(policy_cache.policy)]
end

function extract_policy!(
    policy_cache::StationaryPolicyCache,
    values,
    V,
    j,
    maximize,
)
    gt = maximize ? (>) : (<)

    @inbounds s₁ = policy_cache.stateptr[j]

    if iszero(policy_cache.policy[j])
        opt_val = values[1]
        opt_index = s₁
    else
        opt_val = V[j]
        opt_index = policy_cache.policy[j]
    end

    for (i, v) in Iterators.drop(enumerate(values), iszero(policy_cache.policy[j]) ? 1 : 0)
        if gt(v, opt_val)
            opt_val = v
            opt_index = s₁ + i - 1
        end
    end

    @inbounds policy_cache.policy[j] = opt_index
    return opt_val
end
postprocess_policy_cache!(::StationaryPolicyCache) = nothing
