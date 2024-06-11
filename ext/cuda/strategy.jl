# function IntervalMDP.construct_strategy_cache(mp::M, ::TimeVaryingStrategyConfig) where {M <: SimpleIntervalMarkovProcess}
#     cur_strategy = CUDA.zeros(Int32, num_states(mp))
#     return IntervalMDP.TimeVaryingStrategyCache(stateptr(mp), cur_strategy)
# end

# function IntervalMDP.construct_strategy_cache(mp::M, ::StationaryStrategyConfig) where {M <: SimpleIntervalMarkovProcess}
#     cur_strategy = CUDA.zeros(Int32, num_states(mp))
#     return IntervalMDP.StationaryStrategyCache(stateptr(mp), cur_strategy)
# end