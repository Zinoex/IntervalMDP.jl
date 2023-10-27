

function IMDP.construct_value_function(p::MatrixIntervalProbabilities{R, CuVector{R}, <:CuSparseMatrixCSC{R}}) where {R}
    V = CUDA.zeros(R, size(p, 1))
    return V
end