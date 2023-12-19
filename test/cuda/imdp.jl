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

transition_probs = [["a1", "a2"] => prob1, ["a1", "a2"] => prob2, ["sinking"] => prob3]
initial_state = Int32(1)

mdp = IMDP.cu(IntervalMarkovDecisionProcess(transition_probs, initial_state))

# Finite time reachability
prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
V_fixed_it1, k, _ = value_iteration(problem)
@test k == 10

spec = Specification(prop, Optimistic, Maximize)
problem = Problem(mdp, spec)
V_fixed_it2, k, _ = value_iteration(problem)
@test k == 10
@test all(V_fixed_it1 .<= V_fixed_it2)

spec = Specification(prop, Pessimistic, Minimize)
problem = Problem(mdp, spec)
V_fixed_it1, k, _ = value_iteration(problem)
@test k == 10

spec = Specification(prop, Optimistic, Minimize)
problem = Problem(mdp, spec)
V_fixed_it2, k, _ = value_iteration(problem)
@test k == 10
@test all(V_fixed_it1 .<= V_fixed_it2)

# Infinite time reachability
prop = InfiniteTimeReachability([3], 1e-6)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
V_conv, _, u = value_iteration(problem)
@test maximum(u) <= 1e-6

# Finite time reach avoid
prop = FiniteTimeReachAvoid([3], [2], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
V_fixed_it1, k, _ = value_iteration(problem)
@test k == 10

spec = Specification(prop, Optimistic, Maximize)
problem = Problem(mdp, spec)
V_fixed_it2, k, _ = value_iteration(problem)
@test k == 10
@test all(V_fixed_it1 .<= V_fixed_it2)

spec = Specification(prop, Pessimistic, Minimize)
problem = Problem(mdp, spec)
V_fixed_it1, k, _ = value_iteration(problem)
@test k == 10

spec = Specification(prop, Optimistic, Minimize)
problem = Problem(mdp, spec)
V_fixed_it2, k, _ = value_iteration(problem)
@test k == 10
@test all(V_fixed_it1 .<= V_fixed_it2)

# Infinite time reach avoid
prop = InfiniteTimeReachAvoid([3], [2], 1e-6)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
V_conv, _, u = value_iteration(problem)
@test maximum(u) <= 1e-6

# Finite time reward
prop = FiniteTimeReward(IMDP.cu([2.0, 1.0, 0.0]), 0.9, 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
V_fixed_it1, k, _ = value_iteration(problem)
@test k == 10

spec = Specification(prop, Optimistic, Maximize)
problem = Problem(mdp, spec)
V_fixed_it2, k, _ = value_iteration(problem)
@test k == 10
@test all(V_fixed_it1 .<= V_fixed_it2)

spec = Specification(prop, Pessimistic, Minimize)
problem = Problem(mdp, spec)
V_fixed_it1, k, _ = value_iteration(problem)
@test k == 10

spec = Specification(prop, Optimistic, Minimize)
problem = Problem(mdp, spec)
V_fixed_it2, k, _ = value_iteration(problem)
@test k == 10
@test all(V_fixed_it1 .<= V_fixed_it2)
