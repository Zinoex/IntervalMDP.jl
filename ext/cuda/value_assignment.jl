
function IntervalMDP.value_assignment!(
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    ordering::CuSparseOrdering{Ti},
    indices,
) where {Tv, Ti}
    l = lower(prob)

    Vres .= Transpose(Transpose(V) * l)

    # This will only convert/copy the indices if necessary.
    indices = adapt(CuArray{Ti}, indices)

    add_gap_mul_V!(Vres, V, prob, ordering, indices)

    return Vres
end

function add_gap_mul_V!(
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    ordering::CuSparseOrdering{Ti},
    indices,
) where {Tv, Ti}
    kernel = @cuda launch = false add_gap_mul_V_kernel!(
        Vres,
        V,
        gap(prob),
        sum_lower(prob),
        ordering,
        indices,
    )

    config = launch_configuration(kernel.fun)
    threads = prevwarp(device(), config.threads)

    states_per_block = threads ÷ 32
    blocks = min(65535, ceil(Int64, num_source(prob) / states_per_block))

    kernel(
        Vres,
        V,
        gap(prob),
        sum_lower(prob),
        ordering,
        indices;
        blocks = blocks,
        threads = threads,
    )

    return Vres
end

function add_gap_mul_V_kernel!(
    Vres,
    V,
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
        g_nzs = gap.nzVal
        g_inds = gap.rowVal

        @inbounds subset = ordering.subsets[j]
        subset_length = length(subset)
        @inbounds remaining = one(Tv) - sum_lower[j]
        gap_value = zero(Tv)

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
            remaining += g

            # Update the probability
            if s <= subset_length
                @inbounds t = subset.offset + subset[s] - one(Ti)
                sub = clamp(remaining, zero(Tv), g)
                @inbounds gap_value += sub * V[g_inds[t]]
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

        gap_value = CUDA.reduce_warp(+, gap_value)

        if lane == 1
            Vres[j] += gap_value
        end
        sync_warp()

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
