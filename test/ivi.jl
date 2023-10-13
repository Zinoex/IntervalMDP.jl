
prob = MatrixIntervalProbabilities(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3]
)

V = [0.0, 0.0, 1.0]

V_fixed_it, k, last_step = interval_value_iteration(prob, V, [3], FixedIterationsCriteria(10); max = true)
V_conv, k, last_step = interval_value_iteration(prob, V, [3], CovergenceCriteria(1e-6); max = true)