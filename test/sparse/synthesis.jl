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

mdp = IntervalMarkovDecisionProcess(transition_probs, initial_state)

problem = Problem(mdp, FiniteTimeReachability([3], 10))
policy = control_synthesis(problem; maximize = true)

@test length(policy) == 10
for row in policy
    @test row == ["a1", "a2", "sinking"]
end

problem = Problem(mdp, FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10))
policy = control_synthesis(problem; maximize = true)
@test length(policy) == 10

for row in policy
    @test row == ["a2", "a2", "sinking"]
end

problem = Problem(mdp, InfiniteTimeReachability([3], 1e-6))
policy = control_synthesis(problem; maximize = true)
@test policy == ["a1", "a2", "sinking"]