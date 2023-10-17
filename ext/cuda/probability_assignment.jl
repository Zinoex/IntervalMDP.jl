
function IMDP.probability_assignment!(
    p::CuSparseMatrixCSC{R, T},
    prob::MatrixIntervalProbabilities{R},
    ordering::CuSparseOrdering{T},
    indices,
) where {R, T}
    launch_fixed_index_copýto!(p, lower(prob))
    launch_add_gap_scalar_kernel!(p, prob, ordering, indices)
    # launch_add_gap_vector_kernel!(p, prob, ordering, indices)

    return p
end

function launch_fixed_index_copýto!(p::MR, lower::MR) where {MR <: CuSparseMatrixCSC}
    n = size(p, 2)

    threads = 1024
    blocks = ceil(Int64, n / threads)

    @cuda blocks = blocks threads = threads fixed_index_copyto_kernel!(p, lower)

    return p
end

function fixed_index_copyto_kernel!(
    p::CuSparseDeviceMatrixCSC{R, T, A},
    lower::CuSparseDeviceMatrixCSC{R, T, A},
) where {R, T, A}
    j = (blockIdx().x - T(1)) * blockDim().x + threadIdx().x

    if j <= size(p, 2)
        p_colptr = p.colPtr
        p_nzinds = p.rowVal
        p_nzs = p.nzVal

        lower_colptr = lower.colPtr
        lower_nzinds = lower.rowVal
        lower_nzs = lower.nzVal

        nrow = lower_colptr[j + T(1)] - lower_colptr[j]

        p_curind = p_colptr[j]
        p_endind = p_colptr[j + T(1)]
        lower_startind = lower_colptr[j]

        for i in T(0):(nrow - T(1))
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
    p::CuSparseMatrixCSC{R, T},
    prob::MatrixIntervalProbabilities{R},
    ordering::CuSparseOrdering{T},
    indices,
) where {R, T}
    n = size(p, 2)

    threads = 256
    blocks = ceil(Int64, n / threads)

    @cuda blocks = blocks threads = threads add_gap_scalar_kernel!(
        p,
        gap(prob),
        sum_lower(prob),
        ordering,
        indices,
    )

    return p
end

function add_gap_scalar_kernel!(
    p::CuSparseDeviceMatrixCSC{R, T, A},
    gap::CuSparseDeviceMatrixCSC{R, T, A},
    sum_lower::CuDeviceVector{R, A},
    ordering::CuSparseDeviceOrdering{T, A},
    indices,
) where {R, T, A}
    k = (blockIdx().x - T(1)) * blockDim().x + threadIdx().x

    if k <= length(indices)
        j = T(indices[k])
        # p and gap have the same sparsity pattern
        p_nzinds = p.rowVal
        p_nzs = p.nzVal
        g_nzs = gap.nzVal

        subset = ordering.subsets[j]

        remaining = R(1) - sum_lower[j]

        for s in T(1):T(length(subset))
            i = subset[s]
            t = bisect_indices(p_nzinds, p.colPtr[j], p.colPtr[j + T(1)], i)

            gₜ = g_nzs[t]

            p_nzs[t] = p_nzs[t] + gₜ
            remaining -= gₜ

            if remaining < R(0.0)
                p_nzs[t] += remaining
                remaining = R(0.0)
                break
            end
        end
    end

    return nothing
end

function launch_add_gap_vector_kernel!(
    p::CuSparseMatrixCSC{R, T},
    prob::MatrixIntervalProbabilities{R},
    ordering::CuSparseOrdering{T},
    indices,
) where {R, T}
    n = size(p, 2) * 32

    threads = 256
    blocks = ceil(Int64, n / threads)

    @cuda blocks = blocks threads = threads add_gap_vector_kernel!(
        p,
        gap(prob),
        sum_lower(prob),
        ordering,
        indices,
    )

    return p
end

function add_gap_vector_kernel!(
    p::CuSparseDeviceMatrixCSC{R, T, A},
    gap::CuSparseDeviceMatrixCSC{R, T, A},
    sum_lower::CuDeviceVector{R, A},
    ordering::CuSparseDeviceOrdering{T, A},
    indices,
) where {R, T, A}
    assume(warpsize() == 32)

    thread_id = (blockIdx().x - T(1)) * blockDim().x + threadIdx().x
    warp_id, lane = fldmod1(thread_id, 32)

    if warp_id <= length(indices)
        j = T(indices[warp_id])
        # p and gap have the same sparsity pattern
        p_nzinds = p.rowVal
        p_nzs = p.nzVal
        g_nzs = gap.nzVal

        subset = ordering.subsets[j]
        remaining = R(1) - sum_lower[j]

        s = T(1)
        while s <= length(subset)
            # Find index of the permutation, and lookup the corresponding gap
            sₗ = s + lane - T(1)
            if sₗ <= length(subset)
                i = subset[sₗ]
                t = bisect_indices(p_nzinds, p.colPtr[j], p.colPtr[j + T(1)], i)

                g = g_nzs[t]
                cum_gap = g
            else
                # 0 gap is a neural element
                cum_gap = T(0)
            end

            # Cummulatively sum the gap with a tree reduction
            delta = 1
            while delta < 32
                gₙ = shfl_up_sync(0xffffffff, cum_gap, delta)
                if lane > delta
                    cum_gap += gₙ
                end

                delta *= 2
            end

            # Update the remaining probability
            remaining -= gₜ
            if sₗ <= length(subset)
                remaining += g
            end
            remaining = ifelse(remaining < R(0.0), R(0.0), remaining)

            # Update the probability
            if sₗ <= length(subset)
                sub = min(g, remaining)
                p_nzs[t] += sub
                remaining -= sub
            end

            # Update the remaining probability from the last thread in the warp
            remaining = shfl_sync(0xffffffff, remaining, 31)

            # Early exit if the remaining probability is zero
            if remaining <= R(0.0)
                break
            end

            s += T(32)
        end
    end

    return nothing
end

function bisect_indices(inds::AbstractVector, lo::T, hi::T, i) where {T <: Integer}
    len = hi - lo
    @inbounds while len != 0
        half_len = len >>> 0x01
        m = lo + half_len
        if inds[m] < i
            lo = m + 1
            len -= half_len + 1
        else
            hi = m
            len = half_len
        end
    end
    return lo
end
