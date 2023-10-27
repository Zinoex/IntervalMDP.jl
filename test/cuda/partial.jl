# Matrix
prob = cu(
    MatrixIntervalProbabilities(;
        lower = sparse_hcat(
            SparseVector(15, [4, 10], [0.1, 0.2]),
            SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
        ),
        upper = sparse_hcat(
            SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
            SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
        ),
    ),
)

V = cu(collect(1.0:15.0))

p = partial_ominmax(prob, V, [2]; max = true)
p = SparseMatrixCSC(p)
@test p[:, 2] ≈ SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.2])
