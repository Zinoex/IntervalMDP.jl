
function IntervalMDP.value_assignment!(
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    ordering::O,
    indices,
) where {Tv, Ti, O <: CuOrdering{Ti}}
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
    kernel = @cuda launch = false add_gap_mul_V_sparse_kernel!(
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

function add_gap_mul_V_sparse_kernel!(
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

        g_nzs = gap.nzVal
        g_inds = gap.rowVal

        @inbounds subset = ordering.subsets[j]
        warp_aligned_length = kernel_nextwarp(length(subset))
        @inbounds remaining = one(Tv) - sum_lower[j]
        gap_value = zero(Tv)

        s = lane
        while s <= warp_aligned_length
            # Find index of the permutation, and lookup the corresponding gap
            g = if s <= length(subset)
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
            if s <= length(subset)
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

function add_gap_mul_V!(
    Vres,
    V,
    prob::IntervalProbabilities{Tv},
    ordering::CuDenseOrdering{Ti},
    indices,
) where {Tv, Ti}
    kernel = @cuda launch = false add_gap_mul_V_dense_kernel!(
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

function add_gap_mul_V_dense_kernel!(
    Vres,
    V,
    gap::CuDeviceMatrix{Tv, A},
    sum_lower::CuDeviceVector{Tv, A},
    ordering::CuDenseOrdering{Ti},
    indices,
) where {Tv, Ti, A}
    assume(warpsize() == 32)

    thread_id = (blockIdx().x - one(Ti)) * blockDim().x + threadIdx().x
    wid, lane = fldmod1(thread_id, warpsize())

    while wid <= length(indices)
        @inbounds j = Ti(indices[wid])

        @inbounds gapⱼ = @view gap[:, j]
        warp_aligned_length = kernel_nextwarp(length(gapⱼ))
        @inbounds remaining = one(Tv) - sum_lower[j]
        gap_value = zero(Tv)

        s = lane
        while s <= warp_aligned_length
            # Find index of the permutation, and lookup the corresponding gap
            g = if s <= length(gapⱼ)
                @inbounds gapⱼ[ordering.perm[s]]
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
            if s <= length(gapⱼ)
                sub = clamp(remaining, zero(Tv), g)
                @inbounds gap_value += sub * V[ordering.perm[s]]
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