IntervalMDP.ValueFunction(
    ::MR,
    num_states,
) where {R, MR <: Union{CuSparseMatrixCSC{R}, CuMatrix{R}}} =
    IntervalMDP.ValueFunction(CUDA.zeros(R, num_states))
