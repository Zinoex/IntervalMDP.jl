#### Maximization

# Matrix
prob = IMDP.cu(
    IntervalProbabilities(;
        lower = sparse_hcat(
            SparseVector(Int32(15), Int32[4, 10], [0.1, 0.2]),
            SparseVector(Int32(15), Int32[5, 6, 7], [0.5, 0.3, 0.1]),
        ),
        upper = sparse_hcat(
            SparseVector(Int32(15), Int32[1, 4, 10], [0.5, 0.6, 0.7]),
            SparseVector(Int32(15), Int32[5, 6, 7], [0.7, 0.5, 0.3]),
        ),
    ),
)

V = IMDP.cu(collect(1.0:15.0))

p = ominmax(prob, V; max = true)
p = SparseMatrixCSC(p)
@test p[:, 1] ≈ SparseVector(15, [4, 10], [0.3, 0.7])
@test p[:, 2] ≈ SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.2])

# Matrix - to gpu first
prob = IntervalProbabilities(;
    lower = IMDP.cu(
        sparse_hcat(
            SparseVector(Int32(15), Int32[4, 10], [0.1, 0.2]),
            SparseVector(Int32(15), Int32[5, 6, 7], [0.5, 0.3, 0.1]),
        ),
    ),
    upper = IMDP.cu(
        sparse_hcat(
            SparseVector(Int32(15), Int32[1, 4, 10], [0.5, 0.6, 0.7]),
            SparseVector(Int32(15), Int32[5, 6, 7], [0.7, 0.5, 0.3]),
        ),
    ),
)

V = IMDP.cu(collect(1.0:15.0))

p = ominmax(prob, V; max = true)
p = SparseMatrixCSC(p)
@test p[:, 1] ≈ SparseVector(15, [4, 10], [0.3, 0.7])
@test p[:, 2] ≈ SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.2])

#### Minimization

# Matrix
prob = IMDP.cu(
    IntervalProbabilities(;
        lower = sparse_hcat(
            SparseVector(Int32(15), Int32[4, 10], [0.1, 0.2]),
            SparseVector(Int32(15), Int32[5, 6, 7], [0.5, 0.3, 0.1]),
        ),
        upper = sparse_hcat(
            SparseVector(Int32(15), Int32[1, 4, 10], [0.5, 0.6, 0.7]),
            SparseVector(Int32(15), Int32[5, 6, 7], [0.7, 0.5, 0.3]),
        ),
    ),
)

V = IMDP.cu(collect(1.0:15.0))

p = ominmax(prob, V; max = false)
p = SparseMatrixCSC(p)
@test p[:, 1] ≈ SparseVector(15, [1, 4, 10], [0.5, 0.3, 0.2])
@test p[:, 2] ≈ SparseVector(15, [5, 6, 7], [0.6, 0.3, 0.1])

#### Large matrices

function sample_sparse_interval_probabilities(n, m, nnz_per_column)
    prob_split = 1 / nnz_per_column

    rand_val_lower = rand(rng, Float64, nnz_per_column * m) .* prob_split
    rand_val_upper = rand(rng, Float64, nnz_per_column * m) .* prob_split .+ prob_split
    rand_index = collect(1:nnz_per_column)

    row_vals = Vector{Int32}(undef, nnz_per_column * m)
    col_ptrs = Int32[1; collect(1:m) .* nnz_per_column .+ 1]

    for j in 1:m
        StatsBase.seqsample_a!(1:n, rand_index)  # Select nnz_per_column elements from 1:n
        sort!(rand_index)

        row_vals[((j - 1) * nnz_per_column + 1):(j * nnz_per_column)] .= rand_index
    end

    lower = SparseMatrixCSC{Float64, Int32}(n, m, col_ptrs, row_vals, rand_val_lower)
    upper = SparseMatrixCSC{Float64, Int32}(n, m, col_ptrs, row_vals, rand_val_upper)

    prob = IntervalProbabilities(; lower = lower, upper = upper)
    V = rand(Float64, n)

    cuda_prob = IMDP.cu(prob)
    cuda_V = IMDP.cu(V)

    return prob, V, cuda_prob, cuda_V
end

# Many columns
rng = MersenneTwister(55392)

n = 100000
m = 100000  # It has to be greater than 2^16 to exceed maximum grid size
nnz_per_column = 10
prob, V, cuda_prob, cuda_V = sample_sparse_interval_probabilities(n, m, nnz_per_column)

p_cpu = ominmax(prob, V; max = false)
p_gpu = SparseMatrixCSC(ominmax(cuda_prob, cuda_V; max = false))

for j in 1:m
    # It needs to be done this way because the matrices are too large
    # to assess if printed to the log
    if !(p_cpu[:, j] ≈ p_gpu[:, j])
        println("Column $j")
        @test p_cpu[:, j] ≈ p_gpu[:, j]
        break
    end
end

# Many non-zeros
rng = MersenneTwister(55392)

n = 100000
m = 10
nnz_per_column = 1500   # It has to be greater than 1024 to exceed maximum block size
prob, V, cuda_prob, cuda_V = sample_sparse_interval_probabilities(n, m, nnz_per_column)

p_cpu = ominmax(prob, V; max = false)
p_gpu = SparseMatrixCSC(ominmax(cuda_prob, cuda_V; max = false))

for j in 1:m
    # It needs to be done this way because the matrices are too large
    # to assess if printed to the log
    if !(p_cpu[:, j] ≈ p_gpu[:, j])
        @test p_cpu[:, j] ≈ p_gpu[:, j]
        break
    end
end
