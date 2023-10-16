
function IMDP.probability_assignment!(
    p::MR,
    prob::MatrixIntervalProbabilities{R},
    ordering::SparseCudaOrdering,
    indices,
) where {R, MR <: AbstractCuSparseMatrix{R}}
    launch_fixed_index_copýto!(p, lower(prob))
    launch_add_gap_scalar_kernel!(p, prob, ordering, indices)

    return p
end

function launch_fixed_index_copýto!(p::MR, lower::MR) where {R, T, MR <: AbstractCuSparseMatrix{R, T}}
    n = size(p, 2)

    threads = 1024
    blocks = ceil(Int64, n / threads)

    @cuda blocks=blocks threads=threads fixed_index_copyto_kernel!(p, lower)

    return p
end

function fixed_index_copyto_kernel!(p, lower)
    T = eltype(p.colPtr)
    j = (blockIdx().x - T(1)) * blockDim().x + threadIdx().x

    if j <= size(p, 2)
        p_colptr = p.colPtr
        p_nzinds = p.rowVal
        p_nzs = p.nzVal

        lower_colptr = lower.colPtr
        lower_nzinds = lower.rowVal
        lower_nzs = lower.nzVal

        R = eltype(lower_nzs)

        nrow = lower_colptr[j + T(1)] - lower_colptr[j]

        p_curind = p_colptr[j]
        p_endind = p_colptr[j + T(1)]
        lower_startind = lower_colptr[j]

        for i in T(0):nrow - T(1)
            while p_nzinds[p_curind] < lower_nzinds[lower_startind + i]
                p_nzs[p_curind] = R(0.0)
                p_curind += T(1)
            end

            p_nzs[p_curind] = lower_nzs[lower_startind + i]
            p_curind += T(1)
        end

        while p_curind < p_endind
            p_nzs[p_curind] = R(0.0)
            p_curind += T(1)
        end
    end

    return nothing
end

function launch_add_gap_scalar_kernel!(
    p::MR,
    prob::MatrixIntervalProbabilities{R},
    ordering::SparseCudaOrdering,
    indices,
) where {R, MR <: AbstractCuSparseMatrix{R}}
    n = size(p, 2)

    threads = 1024
    blocks = ceil(Int64, n / threads)

    @cuda blocks=blocks threads=threads add_gap_scalar_kernel!(p, gap(prob), sum_lower(prob), ordering, indices)

    return p
end

function add_gap_scalar_kernel!(
    p,
    gap,
    sum_lower,
    ordering::SparseCudaOrdering{T},
    indices
) where {T}

    k = (blockIdx().x - T(1)) * blockDim().x + threadIdx().x

    if k <= length(indices)
        j = indices[k]
        p_start, p_end = p.colPtr[j], p.colPtr[j + T(1)]  # p and gap have the same sparsity pattern
        p_nzinds = p.rowVal
        p_nzs = p.nzVal
        g_nzs = gap.nzVal

        subset_colptr = ordering.subsets.colPtr
        s_nzs = ordering.subsets.nzVal

        subset_nrow = subset_colptr[j + T(1)] - subset_colptr[j]

        remaining = eltype(p)(1) - sum_lower[j]

        for s in T(0):subset_nrow - T(1)
            # i = s_nzs[subset_colptr[j] + s]
            # t = bisect_indices(p_nzinds, p_start, p_end - T(1), i)

            # gₜ = g_nzs[t]

            # p_nzs[t] += gₜ
            # remaining -= gₜ

            # if remaining < 0.0
            #     p_nzs[t] += remaining
            #     remaining = 0.0
            #     break
            # end
        end
    end

    return nothing
end

function bisect_indices(inds::AbstractVector, lo::T, hi::T, i) where T<:Integer
    hi = hi + T(1)
    len = hi - lo
    @inbounds while len != 0
        half_len = len >>> 0x01
        m = lo + half_len
        if v[m] < i
            lo = m + 1
            len -= half_len + 1
        else
            hi = m
            len = half_len
        end
    end
    return lo
end
