
function IMDP.construct_value_function(
    ::MR,
    num_states,
) where {R, MR <: CuSparseMatrixCSC{R}}
    V = CUDA.zeros(R, num_states)
    return V
end

function IMDP.construct_nonterminal(
    mc::IntervalMarkovChain{<:IntervalProbabilities{R, VR, MR}},
    terminal::AbstractVector{Ti},
) where {R, VR <: AbstractVector{R}, MR <: CuSparseMatrixCSC{R}, Ti}
    nonterminal = setdiff(collect(Ti(1):Ti(num_states(mc))), terminal)
    nonterminal = adapt(CuArray{Ti}, nonterminal)

    return nonterminal
end

function IMDP.construct_nonterminal(
    mdp::IntervalMarkovDecisionProcess{<:IntervalProbabilities{R, VR, MR}},
    terminal::AbstractVector{Ti},
) where {R, VR <: AbstractVector{R}, MR <: CuSparseMatrixCSC{R}, Ti}
    sptr = Vector(IMDP.stateptr(mdp))

    nonterminal = convert.(Ti, setdiff(collect(1:num_states(mdp)), terminal))
    nonterminal_actions = mapreduce(i -> sptr[i]:(sptr[i + 1] - 1), vcat, nonterminal)

    nonterminal = adapt(CuArray{Ti}, nonterminal)
    nonterminal_actions = adapt(CuArray{Ti}, nonterminal_actions)

    return nonterminal, nonterminal_actions
end

function IMDP.step_imc!(
    ordering,
    p,
    prob::IntervalProbabilities{R, VR, MR},
    value_function::IMDP.IMCValueFunction;
    upper_bound,
    discount,
) where {R, VR <: AbstractVector{R}, MR <: CuSparseMatrixCSC{R}}
    indices = value_function.nonterminal_indices
    partial_ominmax!(ordering, p, prob, value_function.prev, indices; max = upper_bound)

    # For CUDA, we have to create a result array as reshape gives a copy.
    res = transpose(value_function.prev) * p
    res .*= discount
    value_function.cur[indices] .= view(res, 1, indices)

    return value_function
end

function IMDP.step_imdp!(
    ordering,
    p,
    prob::IntervalProbabilities{R, VR, MR},
    stateptr,
    maxactions,
    value_function::IMDP.IMDPValueFunction;
    maximize,
    upper_bound,
    discount,
) where {R, VR <: AbstractVector{R}, MR <: CuSparseMatrixCSC{R}}
    partial_ominmax!(
        ordering,
        p,
        prob,
        value_function.prev,
        value_function.nonterminal_actions;
        max = upper_bound,
    )

    res = transpose(value_function.prev) * p
    res .*= discount

    V_per_state = CuVectorOfVector(
        stateptr,
        view(res, 1, value_function.nonterminal_actions),
        maxactions,
    )

    blocks = length(state_indices)
    threads = 32
    @cuda blocks = blocks threads = threads extremum_vov_kernel!(
        value_function.cur,
        V_per_state,
        state_indices,
        discount,
        maximize,
    )

    return value_function
end

function extremum_vov_kernel!(
    V::CuDeviceVector{Tv, A},
    V_per_state,
    state_indices::CuDeviceVector{Ti, A},
    discount,
    maximize,
) where {Tv, Ti, A}
    j = blockIdx().x
    while j <= length(state_indices)
        k = state_indices[j]
        subset = V_per_state[k]

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
                    V[k] = val
                else
                    V[k] = maximize ? max(V[k], val) : min(V[k], val)
                end
            end

            i += blockDim().x
        end

        V[k] *= discount
        j += gridDim().x
    end
end
