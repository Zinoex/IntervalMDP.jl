
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

    threads = prevwarp(device(), config.threads)

    states_per_block = threads ÷ 32
    blocks = min(65535, ceil(Int64, length(V_per_state) / states_per_block))

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

    thread_id = (blockIdx().x - one(Ti)) * blockDim().x + threadIdx().x
    wid, lane = fldmod1(thread_id, warpsize())

    while wid <= length(vov)
        # Tree reduce
        @inbounds subset = vov[wid]
        bound = kernel_nextwarp(length(subset))

        val = neutral

        # Reduce within each warp
        i = lane
        while i <= bound
            val = op(val, if i <= length(subset)
                @inbounds subset[i]
            else
                neutral
            end)

            val = CUDA.reduce_warp(op, val)

            i += blockDim().x
        end

        if lane == 1
            @inbounds res[wid] = val
        end

        thread_id += gridDim().x * blockDim().x
        wid, lane = fldmod1(thread_id, warpsize())
    end
end