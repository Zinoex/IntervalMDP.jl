prob = MatrixIntervalProbabilities(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
)

V = [1, 2, 3]

p = partial_ominmax(prob, V, [2]; max = true)
@test p[:, 2] ≈ [0.5, 0.3, 0.2]
