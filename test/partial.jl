
# Vector of vectors
prob1 = StateIntervalProbabilities(;
    lower = [0.0, 0.1, 0.2],
    upper = [0.5, 0.6, 0.7],
)
prob2 = StateIntervalProbabilities(;
    lower = [0.5, 0.3, 0.1],
    upper = [0.7, 0.5, 0.3],
)
prob = [prob1, prob2]

V = [1, 2, 3]

p = partial_ominmax(prob, V, [2]; max = true)
@test p[2] ≈ [0.5, 0.3, 0.2]

# Matrix
prob = MatrixIntervalProbabilities(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3]
)

V = [1, 2, 3]

p = partial_ominmax(prob, V, [2]; max = true)
@test p[:, 2] ≈ [0.5, 0.3, 0.2]