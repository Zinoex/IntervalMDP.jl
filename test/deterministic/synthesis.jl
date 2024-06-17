prob1 = transition_hcat(3, [2, 3], [1, 2])
prob2 = transition_hcat(3, [1, 2], [3])
prob3 = transition_hcat(3, [3])

transition_probs = [prob1, prob2, prob3]
istates = [Int32(1)]

mdp = DeterministicMarkovDecisionProcess(transition_probs, istates)

prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
policy, V, k, res = control_synthesis(problem)

@test length(policy) == 10
for row in policy
    @test row == [1, 4, 5]
end

prop = FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
policy, V, k, res = control_synthesis(problem)
@test length(policy) == 10

for row in policy
    @test row == [2, 3, 5]
end

prop = InfiniteTimeReachability([3], 1e-6)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(mdp, spec)
policy, V, k, res = control_synthesis(problem)
@test policy == [1, 4, 5]
