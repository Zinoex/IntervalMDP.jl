
function IntervalMDP.compute_gap(
    lower::M,
    upper::M,
) where {Tv, M <: CuSparseMatrixCSC{Tv}}
    # FIXME: This is an ugly, non-robust hack.
    upper = SparseMatrixCSC(upper)
    lower = SparseMatrixCSC(lower)
    lower, gap = IntervalMDP.compute_gap(lower, upper)

    return adapt(IntervalMDP.CuModelAdaptor{Tv}, lower), adapt(IntervalMDP.CuModelAdaptor{Tv}, gap)
end

IntervalMDP.support(p::IntervalMDP.IntervalAmbiguitySet{R, <:SubArray{R, 1, <:CuSparseDeviceMatrixCSC}}) where {R} = rowvals(p.gap)
IntervalMDP.supportsize(p::IntervalMDP.IntervalAmbiguitySet{R, <:SubArray{R, 1, <:CuSparseDeviceMatrixCSC}}) where {R} = nnz(p.gap)

IntervalMDP.support(p::IntervalMDP.IntervalAmbiguitySet{R, <:SubArray{R, 1, <:CuDeviceMatrix}}) where {R} = eachindex(p.gap)
IntervalMDP.supportsize(p::IntervalMDP.IntervalAmbiguitySet{R, <:SubArray{R, 1, <:CuDeviceMatrix}}) where {R} = length(p.gap)
