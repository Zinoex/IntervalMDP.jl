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
    for j in indices
        probability_assignment_from!(p[j], prob[j], perm(ordering, j))
    end
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
    for j in indices
        pⱼ = view(p, :, j)
        probⱼ = StateIntervalProbabilities(
            view(lower(prob), :, j),
            view(gap(prob), :, j),
            sum_lower(prob)[j],
        )
        probability_assignment_from!(pⱼ, probⱼ, perm(ordering, j))
    end
end

# Shared
function probability_assignment_from!(
    p::VR,
    prob::StateIntervalProbabilities{R},
    perm,
) where {R, VR <: AbstractVector{R}}
    @inbounds copyto!(p, lower(prob))

    remaining = 1.0 - sum_lower(prob)
    g = gap(prob)

    for i in perm
        @inbounds p[i] += g[i]
        @inbounds remaining -= g[i]
        if remaining < 0.0
            @inbounds p[i] += remaining
            remaining = 0.0
            break
        end
    end

    return p
end
