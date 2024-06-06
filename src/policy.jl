# Abstract type
abstract type AbstractPolicyCache end

# Policy cache for not storing policies - useful for dispatching
struct NoPolicyCache <: AbstractPolicyCache end

function extract_policy!(
    ::NoPolicyCache,
    values,
    V,
    j,
    other_indices,
    s₁,
    maximize,
)
    return maximize ? maximum(values) : minimum(values)
end
postprocess_policy_cache!(::NoPolicyCache) = nothing

# Policy cache for storing time-varying policies
struct TimeVaryingPolicyCache{A <: AbstractArray{Int32}} <: AbstractPolicyCache
    cur_policy::A
    policy::Vector{A}
end

function TimeVaryingPolicyCache(cur_policy::A) where {A}
    return TimeVaryingPolicyCache(cur_policy, Vector{A}())
end

function create_policy_cache(
    mp::M,
    time_varying::Val{true},
) where {R, P <: IntervalProbabilities{R, <:AbstractVector{R}}, M <: SimpleIntervalMarkovProcess{P}}
    cur_policy = zeros(Int32, num_states(mp))
    return TimeVaryingPolicyCache(cur_policy)
end

function cachetopolicy(
    policy_cache::TimeVaryingPolicyCache,
)
    return [Vector(indices) for indices in reverse(policy_cache.policy)]
end

function extract_policy!(
    policy_cache::TimeVaryingPolicyCache,
    values,
    V,
    j,
    other_indices,
    s₁,
    maximize,
)
    gt = maximize ? (>) : (<)

    opt_val = values[1]
    opt_index = s₁

    for (i, v) in Iterators.drop(enumerate(values), 1)
        if gt(v, opt_val)
            opt_val = v
            opt_index = s₁ + i - 1
        end
    end

    @inbounds policy_cache.cur_policy[j, other_indices...] = opt_index
    return opt_val
end
function postprocess_policy_cache!(policy_cache::TimeVaryingPolicyCache)
    push!(policy_cache.policy, copy(policy_cache.cur_policy))
end

# Policy cache for storing stationary policies
struct StationaryPolicyCache{A <: AbstractArray{Int32}} <: AbstractPolicyCache
    policy::A
end

function create_policy_cache(
    mp::M,
    time_varying::Val{false},
) where {R, P <: IntervalProbabilities{R, <:AbstractVector{R}}, M <: SimpleIntervalMarkovProcess{P}}
    cur_policy = zeros(Int32, num_states(mp))
    return StationaryPolicyCache(cur_policy)
end

function cachetopolicy(
    policy_cache::StationaryPolicyCache,
)
    return Vector(policy_cache.policy)
end

function extract_policy!(
    policy_cache::StationaryPolicyCache,
    values,
    V,
    j,
    other_indices,
    s₁,
    maximize,
)
    gt = maximize ? (>) : (<)

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

    @inbounds policy_cache.policy[j, other_indices...] = opt_index
    return opt_val
end
postprocess_policy_cache!(::StationaryPolicyCache) = nothing
