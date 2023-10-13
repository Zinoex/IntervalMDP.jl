# Vector of vectors
prob1 = StateIntervalProbabilities(; lower = [0.0, 0.1, 0.2], upper = [0.5, 0.6, 0.7])
prob2 = StateIntervalProbabilities(; lower = [0.5, 0.3, 0.1], upper = [0.7, 0.5, 0.3])
prob = [prob1, prob2]

V = [0.0, 0.0, 1.0]

V_fixed_it, k, last_step =
    interval_value_iteration(prob, V, [3], FixedIterationsCriteria(10); max = true)
@test k == 10

V_conv, k, last_step =
    interval_value_iteration(prob, V, [3], CovergenceCriteria(1e-6); max = true)
@test maximum(last_step) <= 1e-6

# Matrix
prob = MatrixIntervalProbabilities(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
)

V = [0.0, 0.0, 1.0]

V_fixed_it, k, last_step =
    interval_value_iteration(prob, V, [3], FixedIterationsCriteria(10); max = true)
@test k == 10

V_conv, k, last_step =
    interval_value_iteration(prob, V, [3], CovergenceCriteria(1e-6); max = true)
@test maximum(last_step) <= 1e-6