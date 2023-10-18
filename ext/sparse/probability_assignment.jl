function IMDP.probability_assignment!(
    p::VVR,
    prob::Vector{<:StateIntervalProbabilities{R}},
    ordering::SparseOrdering,
    indices,
) where {R, VVR <: AbstractVector{<:SparseVector{R}}} 
    Threads.@threads for j in indices
        probⱼ = prob[j]

        @inbounds copyto!(p[j], lower(probⱼ))

        gⱼ = gap(probⱼ)
        lⱼ = sum_lower(probⱼ)

        IMDP.add_gap!(p[j], gⱼ, lⱼ, perm(ordering, j))
    end

    return p
end

function IMDP.probability_assignment!(
    p::MR,
    prob::MatrixIntervalProbabilities{R},
    ordering::SparseOrdering,
    indices,
) where {R, MR <: AbstractSparseMatrix{R}}
    @inbounds copyto!(p, lower(prob))

    Threads.@threads for j in indices
        pⱼ = view(p, :, j)
        gⱼ = view(gap(prob), :, j)
        lⱼ = sum_lower(prob)[j]

        IMDP.add_gap!(pⱼ, gⱼ, lⱼ, perm(ordering, j))
    end

    return p
end
