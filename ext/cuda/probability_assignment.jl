
function IntervalMDP.probability_assignment!(
    p::CuSparseMatrixCSC{Tv, Ti},
    prob::IntervalProbabilities{Tv},
    ordering::CuSparseOrdering{Ti},
    indices,
) where {Tv, Ti}
    copyto!(nonzeros(p), nonzeros(lower(prob)))

    # This will only convert/copy the indices if necessary.
    indices = adapt(CuArray{Ti}, indices)

    # add_gap_scalar!(p, prob, ordering, indices)
    add_gap_vector!(p, prob, ordering, indices)

    return p
end

function add_gap_scalar!(
    p::CuSparseMatrixCSC{Tv, Ti},
    prob::IntervalProbabilities{Tv},
    ordering::CuSparseOrdering{Ti},
    indices,
) where {Tv, Ti}
    n = size(p, 2)

    threads = min(256, n)
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
    p::CuSparseDeviceMatrixCSC{Tv, Ti, A},
    gap::CuSparseDeviceMatrixCSC{Tv, Ti, A},
    sum_lower::CuDeviceVector{Tv, A},
    ordering::CuSparseDeviceOrdering{Ti, A},
    indices,
) where {Tv, Ti, A}
    k = (blockIdx().x - Ti(1)) * blockDim().x + threadIdx().x

    if k <= length(indices)
        j = Ti(indices[k])
        # p and gap have the same sparsity pattern
        p_nzs = p.nzVal
        g_nzs = gap.nzVal

        subset = ordering.subsets[j]

        remaining = one(Tv) - sum_lower[j]

        for s in one(Ti):Ti(length(subset))
            t = subset.offset + subset[s] - one(Ti)

            gₜ = g_nzs[t]

            p_nzs[t] += min(gₜ, remaining)
            remaining -= gₜ

            if remaining <= zero(Tv)
                remaining = zero(Tv)
                break
            end
        end
    end

    return nothing
end

function add_gap_vector!(
    p::CuSparseMatrixCSC{Tv, Ti},
    prob::IntervalProbabilities{Tv},
    ordering::CuSparseOrdering{Ti},
    indices,
) where {Tv, Ti}
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
    p::CuSparseDeviceMatrixCSC{Tv, Ti, A},
    gap::CuSparseDeviceMatrixCSC{Tv, Ti, A},
    sum_lower::CuDeviceVector{Tv, A},
    ordering::CuSparseDeviceOrdering{Ti, A},
    indices,
) where {Tv, Ti, A}
    assume(warpsize() == 32)

    thread_id = (blockIdx().x - one(Ti)) * blockDim().x + threadIdx().x
    warp_id, lane = fldmod1(thread_id, 32)

    if warp_id <= length(indices)
        j = Ti(indices[warp_id])
        # p and gap have the same sparsity pattern
        p_nzs = p.nzVal
        g_nzs = gap.nzVal

        subset = ordering.subsets[j]
        subset_length = length(subset)
        remaining = one(Tv) - sum_lower[j]

        s = one(Ti)
        while s <= subset_length
            # Find index of the permutation, and lookup the corresponding gap
            sₗ = s + lane - one(Ti)
            if sₗ <= subset_length
                t = subset.offset + subset[sₗ] - one(Ti)

                g = g_nzs[t]
                cum_gap = g
            else
                # 0 gap is a neural element
                cum_gap = zero(Tv)
            end

            # Cummulatively sum the gap with a tree reduction
            delta = Ti(1)
            while delta < Ti(32)
                gₙ = shfl_up_sync(0xffffffff, cum_gap, delta)
                if lane > delta
                    cum_gap += gₙ
                end

                delta *= Ti(2)
            end

            # Update the remaining probability
            remaining -= cum_gap
            if sₗ <= subset_length
                remaining += g
            end
            remaining = max(zero(Tv), remaining)

            # Update the probability
            if sₗ <= subset_length
                sub = min(g, remaining)
                p_nzs[t] += sub
                remaining -= sub
            end

            # Update the remaining probability from the last thread in the warp
            remaining = shfl_sync(0xffffffff, remaining, Ti(32))

            # Early exit if the remaining probability is zero
            if remaining <= zero(Tv)
                break
            end

            s += Ti(32)
        end
    end

    return nothing
end
