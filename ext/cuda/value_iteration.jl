
function IntervalMDP.construct_value_function(
    ::MR,
    num_states,
) where {R, MR <: CuSparseMatrixCSC{R}}
    V = CUDA.zeros(R, num_states)
    return V
end

function IntervalMDP.step_imdp!(
    value_function::IntervalMDP.IMDPValueFunction,
    policy_cache::IntervalMDP.NoPolicyCache,
    ordering,
    p,
    prob::IntervalProbabilities{R, VR, MR},
    stateptr;
    maximize,
    upper_bound,
) where {R, VR <: AbstractVector{R}, MR <: CuSparseMatrixCSC{R}}
    ominmax!(ordering, p, prob, value_function.prev; max = upper_bound)

    value_function.action_values .= Transpose(value_function.prev_transpose * p)

    V_per_state =
        CuVectorOfVector(stateptr, value_function.action_values, maximum(diff(stateptr)))

    kernel = @cuda launch=false reduce_vov_kernel!(
        maximize ? max : min,
        maximize ? typemin(R) : typemax(R),
        value_function.cur,
        V_per_state,
    )

    config = launch_configuration(kernel.fun)

    blocks = num_target(prob)

    max_threads = prevwarp(device(), config.threads)
    wanted_threads = nextwarp(device(), maxlength(V_per_state))
    threads = min(wanted_threads, max_threads)

    kernel(
        maximize ? max : min,
        maximize ? typemin(R) : typemax(R),
        value_function.cur,
        V_per_state;
        threads=threads,
        blocks=blocks
    )

    return value_function, policy_cache
end

function reduce_vov_kernel!(
    op,
    neutral,
    res::CuDeviceVector{Tv, A},
    vov::CuDeviceVectorOfVector{Tv, Ti, A},
) where {Tv, Ti, A}
    assume(warpsize() == 32)
    shared = CuStaticSharedArray(Tv, 32)

    wid, lane = fldmod1(threadIdx().x, warpsize())

    j = blockIdx().x
    while j <= length(vov)
        # Tree reduce
        @inbounds subset = vov[j]
        num_iter = nextpow(Ti(32), length(subset))

        if lane == 1
            @inbounds shared[wid] = neutral
        end

        # Reduce within each warp
        i = lane
        while i < num_iter
            val = if i <= length(subset)
                subset[i]
            else
                neutral
            end

            val = CUDA.reduce_warp(op, val)

            if lane == 1
                @inbounds shared[wid] = op(shared[wid], val)
            end

            i += blockDim().x
        end

        # Wait for all partial reductions
        sync_threads()

        # read from shared memory only if that warp existed
        val = if threadIdx().x <= fld1(blockDim().x, warpsize())
            @inbounds shared[lane]
        else
            neutral
        end

        # final reduce within first warp
        if wid == 1
            val = CUDA.reduce_warp(op, val)
        
            if lane == 1
                @inbounds res[j] = val
            end
        end

        j += gridDim().x
    end
end
