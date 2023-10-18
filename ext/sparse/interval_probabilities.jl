
function IMDP.compute_gap(lower::VR, upper::VR) where {VR <: AbstractSparseVector}
    indices = SparseArrays.nonzeroinds(upper)
    gap_nonzeros = map(i -> upper[i] - lower[i], indices)
    lower_nonzeros = map(i -> lower[i], indices)

    gap = SparseArrays.FixedSparseVector(length(lower), indices, gap_nonzeros)
    lower = SparseArrays.FixedSparseVector(length(lower), indices, lower_nonzeros)
    return lower, gap
end

function IMDP.compute_gap(lower::MR, upper::MR) where {MR <: AbstractSparseMatrix}
    gap = upper - lower
    return lower, gap
end