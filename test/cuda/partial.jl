prob = IntervalMDP.cu(
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

V = IntervalMDP.cu(collect(1.0:15.0))

p = partial_ominmax(prob, V, Int32[2]; max = true)
p = SparseMatrixCSC(p)
@test p[:, 2] ≈ SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.2])
