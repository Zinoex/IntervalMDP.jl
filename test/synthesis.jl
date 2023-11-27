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
initial_state = Int32(1)

mdp = IntervalMarkovDecisionProcess(transition_probs, initial_state)



problem = Problem(mdp, InfiniteTimeReachability([3], 1e-6))
policy = control_synthesis(problem; maximize = true)
@test policy == ["a1", "a2", "sinking"]