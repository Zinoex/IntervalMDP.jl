#### Maximization
prob = IntervalProbabilities(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
)

V = [1, 2, 3]

p = ominmax(prob, V; max = true)
@test p[:, 1] ≈ [0.0, 0.3, 0.7]
@test p[:, 2] ≈ [0.5, 0.3, 0.2]

#### Minimization
prob = IntervalProbabilities(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
)

V = [1, 2, 3]

p = ominmax(prob, V; max = false)
@test p[:, 1] ≈ [0.5, 0.3, 0.2]
@test p[:, 2] ≈ [0.6, 0.3, 0.1]
