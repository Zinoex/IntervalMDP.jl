using Revise, Test
using IntervalMDP

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
@test initial_states(mdp) == istates

mdp = IntervalMarkovDecisionProcess(transition_probs)

@testset "explicit sink state" begin
    transition_prob, _ = IntervalMDP.interval_prob_hcat(transition_probs)
    @test_throws DimensionMismatch IntervalMarkovChain(transition_prob)

    # Finite time reachability
    @testset "finite time reachability" begin
        prop = FiniteTimeReachability([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= 0.0)

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= 0.0)

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
    end

    # Infinite time reachability
    @testset "infinite time reachability" begin
        prop = InfiniteTimeReachability([3], 1e-6)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv, _, u = solve(problem)
        @test maximum(u) <= 1e-6
        @test all(V_conv .>= 0.0)
    end

    # Finite time reach avoid
    @testset "finite time reach/avoid" begin
        prop = FiniteTimeReachAvoid([3], [2], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= 0.0)

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= 0.0)

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
    end

    # Infinite time reach avoid
    @testset "infinite time reach/avoid" begin
        prop = InfiniteTimeReachAvoid([3], [2], 1e-6)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv, _, u = solve(problem)
        @test maximum(u) <= 1e-6
        @test all(V_conv .>= 0.0)
    end

    # Finite time reward
    @testset "finite time reward" begin
        prop = FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= 0.0)

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it1, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .>= 0.0)

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_fixed_it2, k, _ = solve(problem)
        @test k == 10
        @test all(V_fixed_it1 .<= V_fixed_it2)
    end

    # Infinite time reward
    @testset "infinite time reward" begin
        prop = InfiniteTimeReward([2.0, 1.0, 0.0], 0.9, 1e-6)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv, _, u = solve(problem)
        @test maximum(u) <= 1e-6
        @test all(V_conv .>= 0.0)
    end

    # Expected exit time
    @testset "expected exit time" begin
        prop = ExpectedExitTime([3], 1e-6)

        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv1, _, u = solve(problem)
        @test maximum(u) <= 1e-6
        @test all(V_conv1 .>= 0.0)
        @test V_conv1[3] == 0.0

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv2, _, u = solve(problem)
        @test maximum(u) <= 1e-6
        @test all(V_conv1 .<= V_conv2)
        @test V_conv2[3] == 0.0

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_conv1, _, u = solve(problem)
        @test maximum(u) <= 1e-6
        @test all(V_conv1 .>= 0.0)
        @test V_conv1[3] == 0.0

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V_conv2, _, u = solve(problem)
        @test maximum(u) <= 1e-6
        @test all(V_conv1 .<= V_conv2)
        @test V_conv2[3] == 0.0
    end
end

@testset "implicit sink state" begin
    transition_probs = [prob1, prob2]
    implicit_mdp = IntervalMarkovDecisionProcess(transition_probs)

    # Finite time reachability
    @testset "finite time reachability" begin
        prop = FiniteTimeReachability([3], 10)

        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end

    # Infinite time reachability
    @testset "infinite time reachability" begin
        prop = InfiniteTimeReachability([3], 1e-6)
        spec = Specification(prop, Pessimistic, Maximize)

        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end

    # Finite time reach avoid
    @testset "finite time reach/avoid" begin
        prop = FiniteTimeReachAvoid([3], [2], 10)

        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end

    # Infinite time reach avoid
    @testset "infinite time reach/avoid" begin
        prop = InfiniteTimeReachAvoid([3], [2], 1e-6)
        spec = Specification(prop, Pessimistic, Maximize)

        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end

    # Finite time reward
    @testset "finite time reward" begin
        prop = FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10)

        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Pessimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit

        spec = Specification(prop, Optimistic, Minimize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end

    # Infinite time reward
    @testset "infinite time reward" begin
        prop = InfiniteTimeReward([2.0, 1.0, 0.0], 0.9, 1e-6)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end

    # Expected exit time
    @testset "expected exit time" begin
        prop = ExpectedExitTime([3], 1e-6)
        spec = Specification(prop, Pessimistic, Maximize)

        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test V ≈ V_implicit
        @test k == k_implicit
        @test res ≈ res_implicit
    end
end
