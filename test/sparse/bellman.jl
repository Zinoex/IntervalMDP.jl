#### Maximization
prob = IntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(15, [4, 10], [0.1, 0.2]),
        SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
    ),
    upper = sparse_hcat(
        SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
        SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
    ),
)

V = collect(1.0:15.0)

Vres = bellman(V, prob; max = true)
@test Vres ≈ [0.3 * 4 + 0.7 * 10, 0.5 * 5 + 0.3 * 6 + 0.2 * 7]

#### Minimization
prob = IntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(15, [4, 10], [0.1, 0.2]),
        SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
    ),
    upper = sparse_hcat(
        SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
        SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
    ),
)

V = collect(1.0:15.0)

Vres = bellman(V, prob; max = false)
@test Vres ≈ [0.5 * 1 + 0.3 * 4 + 0.2 * 10, 0.6 * 5 + 0.3 * 6 + 0.1 * 7]