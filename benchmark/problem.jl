using Revise, BenchmarkTools, Random, ProgressMeter
using IMDP, SparseArrays, CUDA

rng = MersenneTwister(55392)

n = 40000
nnz_per_column = 200
prob_split = 1 / nnz_per_column

q = collect(1:n)
rand_val_lower = rand(rng, Float64, nnz_per_column * n) .* prob_split
rand_val_upper = rand(rng, Float64, nnz_per_column * n) .* prob_split .+ prob_split

row_vals = Vector{Int32}(undef, nnz_per_column * n)
col_ptrs = Int32[1; collect(1:n) .* nnz_per_column .+ 1]

@showprogress for j in 1:n
    # TODO: Find a faster way to select a small subset
    rand_index = shuffle!(rng, q)[1:nnz_per_column]
    sort!(rand_index)

    row_vals[((j - 1) * nnz_per_column + 1):(j * nnz_per_column)] .= rand_index
end

lower = SparseMatrixCSC{Float64, Int32}(n, n, col_ptrs, row_vals, rand_val_lower)
upper = SparseMatrixCSC{Float64, Int32}(n, n, col_ptrs, row_vals, rand_val_upper)

prob = MatrixIntervalProbabilities(; lower = lower, upper = upper)
V = rand(Float64, n)

p = deepcopy(gap(prob))
ordering = construct_ordering(gap(prob))

if CUDA.functional()
    cuda_prob = cu(prob)
    cuda_V = cu(V)
    cuda_p = deepcopy(gap(cuda_prob))
    cuda_ordering = construct_ordering(gap(cuda_prob))
end
