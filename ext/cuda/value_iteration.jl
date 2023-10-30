
function IMDP.construct_value_function(p::MR, num_states) where {R, MR <: CuSparseMatrixCSC{R}}
    V = CUDA.zeros(R, num_states)
    return V
end
