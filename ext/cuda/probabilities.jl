
function IntervalMDP.compute_gap(lower::M, upper::M) where {Tv, M <: CuSparseMatrixCSC{Tv}}
    # FIXME: This is an ugly, non-robust hack.
    upper = SparseMatrixCSC(upper)
    lower = SparseMatrixCSC(lower)
    lower, gap = IntervalMDP.compute_gap(lower, upper)

    return adapt(IntervalMDP.CuModelAdaptor{Tv}, lower),
    adapt(IntervalMDP.CuModelAdaptor{Tv}, gap)
end

const CuDeviceColumnView{Tv} = SubArray{Tv, 1, <:CuDeviceMatrix{Tv}, Tuple{Base.Slice{Base.OneTo{Int}}, Int32}, true}
IntervalMDP.support(
    p::IntervalMDP.IntervalAmbiguitySet{R, <:CuDeviceColumnView{R}},
) where {R} = eachindex(p.gap)
IntervalMDP.support(
    ::IntervalMDP.IntervalAmbiguitySet{R, <:CuDeviceColumnView{R}}, s
) where {R} = s
IntervalMDP.supportsize(
    p::IntervalMDP.IntervalAmbiguitySet{R, <:CuDeviceColumnView{R}},
) where {R} = unsafe_trunc(Int32, length(p.gap))

const CuSparseDeviceColumnView{Tv, Ti} = SubArray{
    Tv,
    1,
    <:CuSparseDeviceMatrixCSC{Tv, Ti},
    Tuple{Base.Slice{Base.OneTo{Int}}, Int32},
    false,
}
Base.@propagate_inbounds IntervalMDP.support(
    p::IntervalMDP.IntervalAmbiguitySet{R, <:CuSparseDeviceColumnView{R}},
) where {R} = rowvals(p.gap)
Base.@propagate_inbounds IntervalMDP.support(
    p::IntervalMDP.IntervalAmbiguitySet{R, <:CuSparseDeviceColumnView{R}}, s
) where {R} = support(p)[s]
Base.@propagate_inbounds IntervalMDP.supportsize(
    p::IntervalMDP.IntervalAmbiguitySet{R, <:CuSparseDeviceColumnView{R}},
) where {R} = unsafe_trunc(Int32, nnz(p.gap))

IntervalMDP.maxsupportsize(
    p::IntervalMDP.IntervalAmbiguitySets{R, <:CuSparseMatrixCSC{R}},
) where {R} = maxdiff(SparseArrays.getcolptr(p.gap))
