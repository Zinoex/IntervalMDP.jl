function IMDP.probability_assignment!(
    p::VVR,
    prob::Vector{<:StateIntervalProbabilities{R}},
    ordering::SparseOrdering,
    indices,
) where {R, VVR <: AbstractVector{<:AbstractSparseVector{R}}}
    Threads.@threads for j in indices
        probⱼ = prob[j]

        @inbounds copyto!(p[j], lower(probⱼ))

        gⱼ = gap(probⱼ)
        lⱼ = sum_lower(probⱼ)

        IMDP.add_gap!(nonzeros(p[j]), nonzeros(gⱼ), lⱼ, perm(ordering, j))
    end

    return p
end

function IMDP.probability_assignment!(
    p::MR,
    prob::MatrixIntervalProbabilities{R},
    ordering::SparseOrdering,
    indices,
) where {R, MR <: SparseArrays.AbstractSparseMatrixCSC{R}}
    @inbounds copyto!(p, lower(prob))
    g = gap(prob)

    Threads.@threads for j in indices
        # p and g must share nonzero structure.
        pⱼ = view(nonzeros(p), p.colptr[j]:p.colptr[j + 1] - 1)
        gⱼ = view(nonzeros(g), p.colptr[j]:p.colptr[j + 1] - 1)
        lⱼ = sum_lower(prob)[j]

        IMDP.add_gap!(pⱼ, gⱼ, lⱼ, perm(ordering, j))
    end

    return p
end
