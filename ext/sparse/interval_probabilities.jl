
function IMDP.compute_gap(lower::VR, upper::VR) where {VR <: AbstractSparseVector}
    indices = SparseArrays.nonzeroinds(upper)
    gap_nonzeros = map(i -> upper[i] - lower[i], indices)
    lower_nonzeros = map(i -> lower[i], indices)

    gap = SparseArrays.FixedSparseVector(length(lower), indices, gap_nonzeros)
    lower = SparseArrays.FixedSparseVector(length(lower), indices, lower_nonzeros)
    return lower, gap
end

function IMDP.compute_gap(lower::MR, upper::MR) where {MR <: SparseArrays.AbstractSparseMatrixCSC}
    I, J, _ = findnz(upper)
    gap_nonzeros = map((i, j) -> upper[i, j] - lower[i, j], I, J)
    lower_nonzeros = map((i, j) -> lower[i, j], I, J)

    gap = SparseArrays.FixedSparseCSC(size(upper)..., upper.colptr, upper.rowval, gap_nonzeros)
    lower = SparseArrays.FixedSparseCSC(size(upper)..., upper.colptr, upper.rowval, lower_nonzeros)
    return lower, gap
end