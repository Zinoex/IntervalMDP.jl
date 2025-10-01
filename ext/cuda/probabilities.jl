
function IntervalMDP.compute_gap(lower::M, upper::M) where {Tv, M <: CuSparseMatrixCSC{Tv}}
    # FIXME: This is an ugly, non-robust hack.
    upper = SparseMatrixCSC(upper)
    lower = SparseMatrixCSC(lower)
    lower, gap = IntervalMDP.compute_gap(lower, upper)

    return adapt(IntervalMDP.CuModelAdaptor{Tv}, lower),
    adapt(IntervalMDP.CuModelAdaptor{Tv}, gap)
end

const CuSparseDeviceColumnView{Tv, Ti} = SubArray{
    Tv,
    1,
    <:CuSparseDeviceMatrixCSC{Tv, Ti},
    Tuple{Base.Slice{Base.OneTo{Int}}, Int},
}
IntervalMDP.support(
    p::IntervalMDP.IntervalAmbiguitySet{R, <:CuSparseDeviceColumnView{R}},
) where {R} = rowvals(p.gap)
IntervalMDP.supportsize(
    p::IntervalMDP.IntervalAmbiguitySet{R, <:CuSparseDeviceColumnView{R}},
) where {R} = nnz(p.gap)

IntervalMDP.maxsupportsize(
    p::IntervalMDP.IntervalAmbiguitySets{R, <:CuSparseMatrixCSC{R}},
) where {R} = maxdiff(SparseArrays.getcolptr(p.gap))
