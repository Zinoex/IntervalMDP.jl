using Revise, BenchmarkTools, ProgressMeter
using Random, StatsBase
using IntervalMDP, SparseArrays, CUDA, Adapt

rng = MersenneTwister(55392)

n = 40000
nnz_per_column = 200
prob_split = 1 / nnz_per_column

q = collect(1:n)
rand_val_lower = rand(rng, Float64, nnz_per_column * n) .* prob_split
rand_val_upper = rand(rng, Float64, nnz_per_column * n) .* prob_split .+ prob_split
rand_index = collect(1:nnz_per_column)

row_vals = Vector{Int32}(undef, nnz_per_column * n)
col_ptrs = Int32[1; collect(1:n) .* nnz_per_column .+ 1]

@showprogress for j in 1:n
    StatsBase.seqsample_a!(1:n, rand_index)  # Select nnz_per_column elements from 1:n
    sort!(rand_index)

    row_vals[((j - 1) * nnz_per_column + 1):(j * nnz_per_column)] .= rand_index
end

lower = SparseMatrixCSC{Float64, Int32}(n, n, col_ptrs, row_vals, rand_val_lower)
upper = SparseMatrixCSC{Float64, Int32}(n, n, col_ptrs, row_vals, rand_val_upper)

prob = IntervalProbabilities(; lower = lower, upper = upper)
V = rand(Float64, n)
Vres = copy(V)
ordering = construct_ordering(gap(prob))

if CUDA.functional()
    cuda_prob = adapt(CuArray{Float64}, prob)
    cuda_V = adapt(CuArray{Float64}, V)
    cuda_Vres = copy(cuda_V)
    cuda_ordering = construct_ordering(gap(cuda_prob))
end
