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

transition_probs = [["a1", "a2"] => prob1, ["a1", "a2"] => prob2, ["sinking"] => prob3]
istates = [Int32(1)]

mdp = IntervalMarkovDecisionProcess(transition_probs, istates)

# Finite time reachability
prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
policy, V, k, res = control_synthesis(problem)

@test length(policy) == 10
for row in policy
    @test row == ["a1", "a2", "sinking"]
end

# Extract IMC from IMDP and policy from control_synthesis
mc = tomarkovchain(mdp, policy)
prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)

@test num_states(mc) == num_states(mdp)
@test time_length(mc) == 10

# Check if the value iteration for the extracted IMC is the same as the value iteration for the original IMDP
V_mc, k, res = value_iteration(problem)
@test V ≈ V_mc

# Finite time reward
prop = FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
policy, V, k, res = control_synthesis(problem)
@test length(policy) == 10

for row in policy
    @test row == ["a2", "a2", "sinking"]
end

prop = InfiniteTimeReachability([3], 1e-6)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
policy, V, k, res = control_synthesis(problem)
@test policy == ["a1", "a2", "sinking"]

# Extract IMC from IMDP and policy from control_synthesis
mc = tomarkovchain(mdp, policy)
prop = InfiniteTimeReachability([3], 1e-6)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)

@test num_states(mc) == num_states(mdp)

# Check if the value iteration for the extracted IMC is the same as the value iteration for the original IMDP
V_mc, k, res = value_iteration(problem)
@test V ≈ V_mc
