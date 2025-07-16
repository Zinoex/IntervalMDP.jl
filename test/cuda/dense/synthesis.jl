using Revise, Test
using IntervalMDP, SparseArrays, CUDA

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
mdp = IntervalMDP.cu(mdp)

# Finite time reachability
prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = ControlSynthesisProblem(mdp, spec)
policy, V, k, res = solve(problem)

@test policy isa TimeVaryingStrategy
@test time_length(policy) == 10
for k in 1:time_length(policy)
    @test Vector(policy[k]) == [1, 2, 1]
end

# Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
problem = VerificationProblem(mdp, spec, policy)
V_mc, k, res = solve(problem)
@test V ≈ V_mc

# Finite time reward
prop = FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10)
prop = IntervalMDP.cu(prop)
spec = Specification(prop, Pessimistic, Maximize)
problem = ControlSynthesisProblem(mdp, spec)

policy, V, k, res = solve(problem)

@test time_length(policy) == 10
for k in 1:time_length(policy)
    @test Vector(policy[k]) == [2, 2, 1]
end

# Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
problem = VerificationProblem(mdp, spec, policy)
V_mc, k, res = solve(problem)
@test V ≈ V_mc

# Infinite time reachability
prop = InfiniteTimeReachability([3], 1e-6)
spec = Specification(prop, Pessimistic, Maximize)
problem = ControlSynthesisProblem(mdp, spec)
policy, V, k, res = solve(problem)

@test policy isa StationaryStrategy
@test IntervalMDP.cpu(policy)[1] == [1, 2, 1]

# Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
problem = VerificationProblem(mdp, spec, policy)
V_mc, k, res = solve(problem)
@test V ≈ V_mc

# Finite time safety
prop = FiniteTimeSafety([3], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = ControlSynthesisProblem(mdp, spec)
policy, V, k, res = solve(problem)

@test all(V .>= 0.0)
@test CUDA.@allowscalar(V[3]) ≈ 0.0

@test policy isa TimeVaryingStrategy
@test time_length(policy) == 10
for k in 1:(time_length(policy) - 1)
    @test Vector(policy[k]) == [2, 2, 1]
end

# The last time step (aka. the first value iteration step) has a different strategy.
@test Vector(policy[time_length(policy)]) == [2, 1, 1]
