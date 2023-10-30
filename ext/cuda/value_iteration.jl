

function IMDP.construct_value_function(p::MR) where {R, MR <: CuSparseMatrixCSC{R}}
    V = CUDA.zeros(R, size(p, 1))
    return V
end