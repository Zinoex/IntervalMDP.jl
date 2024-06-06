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
istates = [Int32(1)]

mdp = IntervalMarkovDecisionProcess(transition_probs, istates)
@test initial_states(mdp) == istates

mdp = IntervalMarkovDecisionProcess(transition_probs)

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
prop = FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10)
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

# Infinite time reward
# problem = Problem(mdp, InfiniteTimeReward([2.0, 1.0, 0.0], 0.9, 1e-6))
# V_conv, _, u = value_iteration(problem; maximize = true, upper_bound = false)
# @test maximum(u) <= 1e-6

# Test extraction of IMC from IMDP if the IMDP only has a single action per state
prob1 = IntervalProbabilities(; lower = [
    0.0
    0.1
    0.2
][:, :], upper = [
    0.5
    0.6
    0.7
][:, :])

prob2 = IntervalProbabilities(; lower = [
    0.1
    0.2
    0.3
][:, :], upper = [
    0.6
    0.5
    0.4
][:, :])

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
istates = [Int32(1)]

mdp = IntervalMarkovDecisionProcess(transition_probs, istates)
mc = tomarkovchain(mdp)
@test initial_states(mdp) == initial_states(mc)
@test num_states(mdp) == num_states(mc)
@test transition_prob(mc) == transition_prob(mdp)
