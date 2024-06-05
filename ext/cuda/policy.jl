function IntervalMDP.create_policy_cache(
    mp::M,
    time_varying::Val{true},
) where {R, P <: IntervalProbabilities{R, CuVector{R}}, M <: IntervalMarkovDecisionProcess{P}}
    cur_policy = CUDA.zeros(Int32, num_states(mp))
    return IntervalMDP.TimeVaryingPolicyCache(IntervalMDP.stateptr(mp), cur_policy)
end

function IntervalMDP.create_policy_cache(
    mp::M,
    time_varying::Val{false},
) where {R, P <: IntervalProbabilities{R, CuVector{R}}, M <: IntervalMarkovDecisionProcess{P}}
    cur_policy = CUDA.zeros(Int32, num_states(mp))
    return IntervalMDP.StationaryPolicyCache(IntervalMDP.stateptr(mp), cur_policy)
end