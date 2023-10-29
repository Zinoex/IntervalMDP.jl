
# Vector of vectors
prob1 = StateIntervalProbabilities(;
    lower = SparseVector(3, [2, 3], [0.1, 0.2]),
    upper = SparseVector(3, [1, 2, 3], [0.5, 0.6, 0.7]),
)
prob2 = StateIntervalProbabilities(;
    lower = SparseVector(3, [1, 2, 3], [0.5, 0.3, 0.1]),
    upper = SparseVector(3, [1, 2, 3], [0.7, 0.5, 0.3]),
)
prob3 = StateIntervalProbabilities(;
    lower = SparseVector(3, [3], [1.0]),
    upper = SparseVector(3, [3], [1.0]),
)
prob = [prob1, prob2, prob3]

V = collect(1.0:3.0)

V_fixed_it, k, last_dV =
    interval_value_iteration(prob, V, [3], FixedIterationsCriteria(10); max = true)
@test k == 10

V_conv, k, last_dV =
    interval_value_iteration(prob, V, [3], CovergenceCriteria(1e-6); max = true)
@test maximum(last_dV) <= 1e-6

# Matrix
prob = MatrixIntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(3, [2, 3], [0.1, 0.2]),
        SparseVector(3, [1, 2, 3], [0.5, 0.3, 0.1]),
        SparseVector(3, [3], [1.0]),
    ),
    upper = sparse_hcat(
        SparseVector(3, [1, 2, 3], [0.5, 0.6, 0.7]),
        SparseVector(3, [1, 2, 3], [0.7, 0.5, 0.3]),
        SparseVector(3, [3], [1.0]),
    ),
)

V = collect(1.0:3.0)

V_fixed_it, k, last_dV =
    interval_value_iteration(prob, V, [3], FixedIterationsCriteria(10); max = true)
@test k == 10

V_conv, k, last_dV =
    interval_value_iteration(prob, V, [3], CovergenceCriteria(1e-6); max = true)
@test maximum(last_dV) <= 1e-6