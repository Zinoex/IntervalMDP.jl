
function IntervalMDP.construct_value_function(
    ::Type{MR},
    num_states,
) where {R, MR <: Union{CuSparseMatrixCSC{R}, CuMatrix{R}}}
    V = CUDA.zeros(R, num_states)
    return V
end
