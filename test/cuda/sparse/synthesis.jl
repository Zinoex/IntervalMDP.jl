using Revise, Test
using IntervalMDP, SparseArrays, CUDA

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
istates = [Int32(1)]

mdp = IntervalMarkovDecisionProcess(transition_probs, istates)

prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
problem = IntervalMDP.cu(problem)

policy, V, k, res = control_synthesis(problem)

@test time_length(policy) == 10
for k in 1:time_length(policy)
    @test Vector(policy[k]) == [1, 2, 1]
end

prop = FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
problem = IntervalMDP.cu(problem)

policy, V, k, res = control_synthesis(problem)

@test time_length(policy) == 10
for k in 1:time_length(policy)
    @test Vector(policy[k]) == [2, 2, 1]
end

prop = InfiniteTimeReachability([3], 1e-6)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
problem = IntervalMDP.cu(problem)

policy, V, k, res = control_synthesis(problem)
@test Vector(policy[1]) == [1, 2, 1]
