using Revise, Test
using IntervalMDP, SparseArrays

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

# Finite time reachability
@testset "finite time reachability" begin
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
end

# Infinite time reachability
@testset "infinite time reachability" begin
    prop = InfiniteTimeReachability([3], 1e-6)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = Problem(mdp, spec)
    V_conv, _, u = value_iteration(problem)
    @test maximum(u) <= 1e-6
end

# Finite time reach avoid
@testset "finite time reach/avoid" begin
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
end

# Infinite time reach avoid
@testset "infinite time reach/avoid" begin
    prop = InfiniteTimeReachAvoid([3], [2], 1e-6)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = Problem(mdp, spec)
    V_conv, _, u = value_iteration(problem)
    @test maximum(u) <= 1e-6
end

# Finite time reward
@testset "finite time reward" begin
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
end
