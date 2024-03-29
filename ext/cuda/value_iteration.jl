
function IntervalMDP.construct_value_function(
    ::MR,
    num_states,
) where {R, MR <: CuSparseMatrixCSC{R}}
    V = CUDA.zeros(R, num_states)
    return V
end

function IntervalMDP.step_imdp!(
    ordering,
    p,
    prob::IntervalProbabilities{R, VR, MR},
    stateptr,
    value_function::IntervalMDP.IMDPValueFunction;
    maximize,
    upper_bound,
    discount = 1.0,
) where {R, VR <: AbstractVector{R}, MR <: CuSparseMatrixCSC{R}}
    ominmax!(
        ordering,
        p,
        prob,
        value_function.prev;
        max = upper_bound,
    )

    value_function.action_values .= Transpose(value_function.prev_transpose * p)
    rmul!(value_function.action_values, discount)

    V_per_state =
        CuVectorOfVector(stateptr, value_function.action_values, maximum(diff(stateptr)))

    blocks = num_target(prob)
    threads = 32
    @cuda blocks = blocks threads = threads extremum_vov_kernel!(
        value_function.cur,
        V_per_state,
        discount,
        maximize,
    )

    return value_function
end

function extremum_vov_kernel!(
    V::CuDeviceVector{Tv, A},
    V_per_state::CuDeviceVectorOfVector{Tv, Ti, A},
    discount,
    maximize,
) where {Tv, Ti, A}
    j = blockIdx().x
    while j <= length(V_per_state)
        subset = V_per_state[j]

        # Tree reduce to find the maximum/minimum
        lane = threadIdx().x
        i = lane
        while i < nextpow(Ti(32), length(subset))
            if i <= length(subset)
                val = subset[i]
            else
                val = maximize ? zero(Tv) : one(Tv)
            end

            delta = Ti(16)
            while delta > zero(Ti)
                up = shfl_down_sync(0xffffffff, val, delta)
                val = maximize ? max(val, up) : min(val, up)

                delta ÷= Ti(2)
            end

            # A bit of shared memory could reduce the number of global memory accesses
            if lane == 1
                if i == 1
                    V[j] = val
                else
                    V[j] = maximize ? max(V[j], val) : min(V[j], val)
                end
            end

            i += blockDim().x
        end

        V[j] *= discount
        j += gridDim().x
    end
end
