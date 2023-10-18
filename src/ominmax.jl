function ominmax(prob, V; max = true)
    ordering = construct_ordering(gap(prob))
    return ominmax!(ordering, prob, V; max = max)
end

function ominmax!(ordering::AbstractStateOrdering, prob, V; max = true)
    p = deepcopy(gap(prob))
    return ominmax!(ordering, p, prob, V; max = max)
end

function ominmax!(ordering::AbstractStateOrdering, p, prob, V; max = true)
    sort_states!(ordering, V; max = max)
    probability_assignment!(p, prob, ordering)

    return p
end

function partial_ominmax(prob, V, indices; max = true)
    ordering = construct_ordering(gap(prob))
    return partial_ominmax!(ordering, prob, V, indices; max = max)
end

function partial_ominmax!(ordering::AbstractStateOrdering, prob, V, indices; max = true)
    p = deepcopy(gap(prob))
    return partial_ominmax!(ordering, p, prob, V, indices; max = max)
end

function partial_ominmax!(ordering::AbstractStateOrdering, p, prob, V, indices; max = true)
    sort_states!(ordering, V; max = max)
    probability_assignment!(p, prob, ordering, indices)

    return p
end

# Vector of vectors
function probability_assignment!(
    p::VVR,
    prob::Vector{<:StateIntervalProbabilities{R}},
    ordering::AbstractStateOrdering,
) where {R, VVR <: AbstractVector{<:AbstractVector{R}}}
    return probability_assignment!(p, prob, ordering, eachindex(p))
end

function probability_assignment!(
    p::VVR,
    prob::Vector{<:StateIntervalProbabilities{R}},
    ordering::AbstractStateOrdering,
    indices,
) where {R, VVR <: AbstractVector{<:AbstractVector{R}}}
    Threads.@threads for j in indices
        probⱼ = prob[j]

        @inbounds copyto!(p[j], lower(probⱼ))

        gⱼ = gap(probⱼ)
        lⱼ = sum_lower(probⱼ)

        add_gap!(p[j], gⱼ, lⱼ, perm(ordering, j))
    end

    return p
end

# Matrix
function probability_assignment!(
    p::MR,
    prob::MatrixIntervalProbabilities{R},
    ordering::AbstractStateOrdering,
) where {R, MR <: AbstractMatrix{R}}
    return probability_assignment!(p, prob, ordering, axes(p, 2))
end

function probability_assignment!(
    p::MR,
    prob::MatrixIntervalProbabilities{R},
    ordering::AbstractStateOrdering,
    indices,
) where {R, MR <: AbstractMatrix{R}}
    @inbounds copyto!(p, lower(prob))

    Threads.@threads for j in indices
        pⱼ = view(p, :, j)
        gⱼ = view(gap(prob), :, j)
        lⱼ = sum_lower(prob)[j]

        add_gap!(pⱼ, gⱼ, lⱼ, perm(ordering, j))
    end

    return p
end

# Shared
function add_gap!(
    p::VR,
    gap::VR,
    sum_lower::R,
    perm,
) where {R, VR <: AbstractVector{R}}
    remaining = 1.0 - sum_lower

    for i in perm
        @inbounds p[i] += gap[i]
        @inbounds remaining -= gap[i]
        if remaining < 0.0
            @inbounds p[i] += remaining
            remaining = 0.0
            break
        end
    end

    return p
end
