#### Maximization
prob = IntervalProbabilities(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
)

V = [1.0, 2.0, 3.0]

Vres = bellman(V, prob; max = true)
@test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

Vres = similar(Vres)
bellman!(Vres, V, prob; max = true)
@test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

ws = construct_workspace(prob)
Vres = similar(Vres)
bellman!(ws, Vres, V, prob; max = true)
@test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

ws = IntervalMDP.DenseWorkspace(gap(prob))
Vres = similar(Vres)
bellman!(ws, Vres, V, prob; max = true)
@test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

ws = IntervalMDP.ThreadedDenseWorkspace(gap(prob))
Vres = similar(Vres)
bellman!(ws, Vres, V, prob; max = true)
@test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

#### Minimization
prob = IntervalProbabilities(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
)

V = [1.0, 2.0, 3.0]

Vres = bellman(V, prob; max = false)
@test Vres ≈ [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

Vres = similar(Vres)
bellman!(Vres, V, prob; max = false)
@test Vres ≈ [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

ws = construct_workspace(prob)
Vres = similar(Vres)
bellman!(ws, Vres, V, prob; max = false)
@test Vres ≈ [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

ws = IntervalMDP.DenseWorkspace(gap(prob))
Vres = similar(Vres)
bellman!(ws, Vres, V, prob; max = false)
@test Vres ≈ [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

ws = IntervalMDP.ThreadedDenseWorkspace(gap(prob))
Vres = similar(Vres)
bellman!(ws, Vres, V, prob; max = false)
@test Vres ≈ [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]
