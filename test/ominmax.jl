

prob1 = StateIntervalProbabilities(;
    lower = [0.0, 0.1, 0.2],
    upper = [0.5, 0.6, 0.7],
)
prob2 = StateIntervalProbabilities(;
    lower = [0.5, 0.3, 0.1],
    upper = [0.7, 0.5, 0.3],
)
V = [1, 2, 3]

p = ominmax([prob1, prob2], V; max = true)
@test p[1] ≈ [0.0, 0.3, 0.7]
@test p[2] ≈ [0.5, 0.3, 0.2]