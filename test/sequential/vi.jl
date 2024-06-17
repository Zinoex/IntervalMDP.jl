# Dense MDP
prob1 = IntervalProbabilities(;
    lower = [
        0.0 0.5
        0.1 0.3
        0.2 0.1
    ],
    upper = [
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ],
)

prob2 = IntervalProbabilities(;
    lower = [
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ],
    upper = [
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ],
)

prob3 = IntervalProbabilities(; lower = [
    0.0
    0.0
    1.0
][:, :], upper = [
    0.0
    0.0
    1.0
][:, :])

transition_probs = [prob1, prob2, prob3]
dense_mdp = IntervalMarkovDecisionProcess(transition_probs)

# Sparse MDP
prob1 = IntervalProbabilities(;
    lower = sparse([
        0.0 0.5
        0.1 0.3
        0.2 0.1
    ]),
    upper = sparse([
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ]),
)

prob2 = IntervalProbabilities(;
    lower = sparse([
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ]),
    upper = sparse([
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ]),
)

prob3 = IntervalProbabilities(;
    lower = sparse([
        0.0
        0.0
        1.0
    ][:, :]),
    upper = sparse([
        0.0
        0.0
        1.0
    ][:, :]),
)

transition_probs = [prob1, prob2, prob3]
sparse_mdp = IntervalMarkovDecisionProcess(transition_probs)

# Dense/Sparse Parallel Product
product_mdp = ParallelProduct([dense_mdp, sparse_mdp])

# Deterministic MDP
prob1 = transition_hcat(9, [1, 5])
prob2 = transition_hcat(9, [2, 8])
prob3 = transition_hcat(9, [3, 9])
prob4 = transition_hcat(9, [3, 4, 6])
prob5 = transition_hcat(9, [1, 8])
prob6 = transition_hcat(9, [4, 5])
prob7 = transition_hcat(9, [6, 7])
prob8 = transition_hcat(9, [1, 2])
prob9 = transition_hcat(9, [9])

transition_probs = [prob1, prob2, prob3, prob4, prob5, prob6, prob7, prob8, prob9]
deterministic_mdp = DeterministicMarkovDecisionProcess(transition_probs)
wrapped_mdp = MultiDim(deterministic_mdp, (3, 3))

# Sequential - e.g. x(k + 1) = f(x(k), u(k)) + w(k) where w(k) has diagonal covariance
# except that w(k) has actions? (w(k) = u(k) + v(k) where v(k) is the noise?)
sequential_mdp = Sequential([wrapped_mdp, product_mdp])

# Finite time reachability
prop = FiniteTimeReachability([(3, 3)], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(sequential_mdp, spec)
V_fixed_it, k, _ = value_iteration(problem)
@test k == 10
