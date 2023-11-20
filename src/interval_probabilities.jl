struct MatrixIntervalProbabilities{R, VR <: AbstractVector{R}, MR <: AbstractMatrix{R}}
    lower::MR
    gap::MR

    sum_lower::VR
end

function MatrixIntervalProbabilities(lower::MR, gap::MR) where {R, MR <: AbstractMatrix{R}}
    sum_lower = vec(sum(lower; dims = 1))

    max_lower_bound = maximum(sum_lower)
    @assert max_lower_bound <= 1 "The joint lower bound transition probability per column (max is $max_lower_bound) should be less than or equal to 1."

    sum_upper = vec(sum(lower + gap; dims = 1))

    max_upper_bound = minimum(sum_upper)
    @assert max_upper_bound >= 1 "The joint upper bound transition probability per column (min is $max_upper_bound) should be greater than or equal to 1."

    return MatrixIntervalProbabilities(lower, gap, sum_lower)
end

# Keyword constructor
function MatrixIntervalProbabilities(; lower::MR, upper::MR) where {MR <: AbstractMatrix}
    lower, gap = compute_gap(lower, upper)
    return MatrixIntervalProbabilities(lower, gap)
end

function compute_gap(lower::MR, upper::MR) where {MR <: AbstractMatrix}
    gap = upper - lower
    return lower, gap
end

function compute_gap(
    lower::MR,
    upper::MR,
) where {MR <: SparseArrays.AbstractSparseMatrixCSC}
    I, J, _ = findnz(upper)
    gap_nonzeros = map((i, j) -> upper[i, j] - lower[i, j], I, J)
    lower_nonzeros = map((i, j) -> lower[i, j], I, J)

    gap = SparseArrays.FixedSparseCSC(
        size(upper)...,
        upper.colptr,
        upper.rowval,
        gap_nonzeros,
    )
    lower = SparseArrays.FixedSparseCSC(
        size(upper)...,
        upper.colptr,
        upper.rowval,
        lower_nonzeros,
    )
    return lower, gap
end

gap(s::MatrixIntervalProbabilities) = s.gap
lower(s::MatrixIntervalProbabilities) = s.lower
sum_lower(s::MatrixIntervalProbabilities) = s.sum_lower
num_src(s::MatrixIntervalProbabilities) = size(gap(s), 2)

function interval_prob_hcat(
    T,
    transition_probs::Vector{<:MatrixIntervalProbabilities{R, VR, MR}},
) where {R, VR, MR <: AbstractMatrix{R}}
    l = mapreduce(lower, hcat, transition_probs)
    g = mapreduce(gap, hcat, transition_probs)

    sl = mapreduce(sum_lower, vcat, transition_probs)

    lengths = map(num_src, transition_probs)
    stateptr = T[1; cumsum(lengths) .+ 1]

    return MatrixIntervalProbabilities(l, g, sl), stateptr
end

function interval_prob_hcat(
    T,
    transition_probs::Vector{<:MatrixIntervalProbabilities{R, VR, MR}},
) where {R, VR, MR <: SparseArrays.AbstractSparseMatrixCSC{R}}
    l = map(lower, transition_probs)
    l = hcat(l...)

    g = map(gap, transition_probs)
    g = hcat(g...)

    sl = mapreduce(sum_lower, vcat, transition_probs)

    lengths = map(num_src, transition_probs)
    stateptr = T[1; cumsum(lengths) .+ 1]

    return MatrixIntervalProbabilities(l, g, sl), stateptr
end
