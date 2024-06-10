function IntervalMDP.construct_strategy_cache(mp::M, ::IntervalMDP.TimeVaryingStrategyConfig) where {R, P <: IntervalProbabilities{R, CuVector{R}}, M <: SimpleIntervalMarkovProcess{P}}
    cur_strategy = CUDA.zeros(Int32, num_states(mp))
    return IntervalMDP.TimeVaryingStrategyCache(IntervalMDP.stateptr(mp), cur_strategy)
end

function IntervalMDP.construct_strategy_cache(mp::M, ::IntervalMDP.StationaryStrategyConfig) where {R, P <: IntervalProbabilities{R, CuVector{R}}, M <: IntervalMarkovDecisionProcess{P}}
    cur_strategy = CUDA.zeros(Int32, num_states(mp))
    return IntervalMDP.StationaryStrategyCache(IntervalMDP.stateptr(mp), cur_strategy)
end