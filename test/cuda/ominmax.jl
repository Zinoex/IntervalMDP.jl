# Maximization

# Matrix
prob = cu(
    MatrixIntervalProbabilities(;
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

V = cu(collect(1.0:15.0))

p = ominmax(prob, V; max = true)
p = SparseMatrixCSC(p)
@test p[:, 1] ≈ SparseVector(15, [4, 10], [0.3, 0.7])
@test p[:, 2] ≈ SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.2])

# Matrix - to gpu first
prob = MatrixIntervalProbabilities(;
    lower = cu(
        sparse_hcat(
            SparseVector(Int32(15), Int32[4, 10], [0.1, 0.2]),
            SparseVector(Int32(15), Int32[5, 6, 7], [0.5, 0.3, 0.1]),
        ),
    ),
    upper = cu(
        sparse_hcat(
            SparseVector(Int32(15), Int32[1, 4, 10], [0.5, 0.6, 0.7]),
            SparseVector(Int32(15), Int32[5, 6, 7], [0.7, 0.5, 0.3]),
        ),
    ),
)

V = cu(collect(1.0:15.0))

p = ominmax(prob, V; max = true)
p = SparseMatrixCSC(p)
@test p[:, 1] ≈ SparseVector(15, [4, 10], [0.3, 0.7])
@test p[:, 2] ≈ SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.2])

# Minimization

# Matrix
prob = cu(
    MatrixIntervalProbabilities(;
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

V = cu(collect(1.0:15.0))

p = ominmax(prob, V; max = false)
p = SparseMatrixCSC(p)
@test p[:, 1] ≈ SparseVector(15, [1, 4, 10], [0.5, 0.3, 0.2])
@test p[:, 2] ≈ SparseVector(15, [5, 6, 7], [0.6, 0.3, 0.1])