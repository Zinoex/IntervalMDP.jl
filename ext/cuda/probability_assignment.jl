
function probability_assignment!(
    p::CuSparseMatrixCSC{Tv, Ti},
    prob::IntervalProbabilities{Tv},
    ordering::CuSparseOrdering{Ti},
    indices,
) where {Tv, Ti}
    copyto!(nonzeros(p), nonzeros(lower(prob)))

    # This will only convert/copy the indices if necessary.
    indices = adapt(CuArray{Ti}, indices)

    add_gap_vector!(p, prob, ordering, indices)

    return p
end

function add_gap_vector!(
    p::CuSparseMatrixCSC{Tv, Ti},
    prob::IntervalProbabilities{Tv},
    ordering::CuSparseOrdering{Ti},
    indices,
) where {Tv, Ti}
    kernel = @cuda launch = false add_gap_vector_kernel!(
        p,
        gap(prob),
        sum_lower(prob),
        ordering,
        indices,
    )

    config = launch_configuration(kernel.fun)
    threads = prevwarp(device(), config.threads)

    states_per_block = threads ÷ 32
    blocks = min(65535, ceil(Int64, size(p, 2) / states_per_block))

    kernel(
        p,
        gap(prob),
        sum_lower(prob),
        ordering,
        indices;
        blocks = blocks,
        threads = threads,
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
    wid, lane = fldmod1(thread_id, warpsize())

    while wid <= length(indices)
        @inbounds j = Ti(indices[wid])

        # p and gap have the same sparsity pattern
        p_nzs = p.nzVal
        g_nzs = gap.nzVal

        @inbounds subset = ordering.subsets[j]
        subset_length = length(subset)
        @inbounds remaining = one(Tv) - sum_lower[j]

        s = lane
        while s <= subset_length
            # Find index of the permutation, and lookup the corresponding gap
            g = if s <= subset_length
                @inbounds t = subset.offset + subset[s] - one(Ti)

                @inbounds g_nzs[t]
            else
                # 0 gap is a neural element
                zero(Tv)
            end

            # Cummulatively sum the gap with a tree reduction
            cum_gap = cumsum_warp(g, lane)

            # Update the remaining probability
            remaining -= cum_gap
            if s <= subset_length
                remaining += g
            end
            remaining = max(zero(Tv), remaining)

            # Update the probability
            if s <= subset_length
                sub = min(g, remaining)
                @inbounds p_nzs[t] += sub
                remaining -= sub
            end

            # Update the remaining probability from the last thread in the warp
            remaining = shfl_sync(0xffffffff, remaining, warpsize())

            # Early exit if the remaining probability is zero
            if remaining <= zero(Tv)
                break
            end

            s += warpsize()
        end

        thread_id += gridDim().x * blockDim().x
        wid, lane = fldmod1(thread_id, warpsize())
    end

    return nothing
end

@inline function cumsum_warp(val, lane)
    assume(warpsize() == 32)
    offset = 0x00000001
    while offset < warpsize()
        up_val = shfl_up_sync(0xffffffff, val, offset)
        if lane > offset
            val += up_val
        end
        offset <<= 1
    end

    return val
end
