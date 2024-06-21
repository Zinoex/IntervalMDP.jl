
function IntervalMDP.compute_gap(
    lower::M,
    upper::M,
) where {Tv, Ti, M <: CuSparseMatrixCSC{Tv, Ti}}
    # lower = CuSparseMatrixCOO(lower)

    # FIXME: This is an ugly, non-robust hack.
    upper = SparseMatrixCSC(upper)
    lower = SparseMatrixCSC(lower)
    lower, gap = IntervalMDP.compute_gap(lower, upper)
    return adapt(IntervalMDP.CuModelAdaptor{Tv}, lower),
    adapt(IntervalMDP.CuModelAdaptor{Tv}, gap)
end

function IntervalMDP.interval_prob_hcat(
    transition_probs::Vector{
        <:IntervalProbabilities{Tv, <:AbstractVector{Tv}, <:CuSparseMatrixCSC{Tv, Ti}},
    },
) where {Tv, Ti}
    num_dest = size(lower(first(transition_probs)), 1)

    @assert all(x -> size(lower(x), 1) == num_dest, transition_probs) "The dimensions of all matrices must be the same"
    @assert all(x -> size(gap(x), 1) == num_dest, transition_probs) "The dimensions of all matrices must be the same"

    num_col = mapreduce(x -> size(lower(x), 2), +, transition_probs)
    dims = (num_dest, num_col)

    l = map(lower, transition_probs)

    l_colptr = CUDA.zeros(Ti, num_col + 1)
    nnz_sofar = 0
    nX_sofar = 0
    @inbounds for i in eachindex(l)
        li = l[i]
        nX = size(li, 2)
        l_colptr[(1:(nX + 1)) .+ nX_sofar] = li.colPtr .+ nnz_sofar
        nnz_sofar += nnz(li)
        nX_sofar += nX
    end

    l_rowval = mapreduce(lower -> lower.rowVal, vcat, l)
    l_nzval = mapreduce(lower -> lower.nzVal, vcat, l)
    l = CuSparseMatrixCSC(l_colptr, l_rowval, l_nzval, dims)

    g = map(gap, transition_probs)

    g_colptr = CUDA.zeros(Ti, num_col + 1)
    nnz_sofar = 0
    nX_sofar = 0
    @inbounds for i in eachindex(g)
        gi = g[i]
        nX = size(gi, 2)
        g_colptr[(1:(nX + 1)) .+ nX_sofar] = gi.colPtr .+ nnz_sofar
        nnz_sofar += nnz(gi)
        nX_sofar += nX
    end

    g_rowval = mapreduce(lower -> lower.rowVal, vcat, g)
    g_nzval = mapreduce(lower -> lower.nzVal, vcat, g)
    g = CuSparseMatrixCSC(g_colptr, g_rowval, g_nzval, dims)

    sl = mapreduce(sum_lower, vcat, transition_probs)

    lengths = map(num_source, transition_probs)
    stateptr = CuVector{Ti}([1; cumsum(lengths) .+ 1])

    return IntervalProbabilities(l, g, sl), stateptr
end
