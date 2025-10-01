using Revise, Test
using IntervalMDP, CUDA

@testset for N in [Float32, Float64]
    prob1 = IntervalAmbiguitySets(;
        lower = sparse(N[
            0 1//2
            1//10 3//10
            1//5 1//10
        ]),
        upper = sparse(N[
            1//2 7//10
            3//5 1//2
            7//10 3//10
        ]),
    )

    prob2 = IntervalAmbiguitySets(;
        lower = sparse(N[
            1//10 1//5
            1//5 3//10
            3//10 2//5
        ]),
        upper = sparse(N[
            3//5 3//5
            1//2 1//2
            2//5 2//5
        ]),
    )

    prob3 = IntervalAmbiguitySets(;
        lower = sparse(N[
            0 0
            0 0
            1 1
        ]),
        upper = sparse(N[
            0 0
            0 0
            1 1
        ]),
    )

    transition_probs = [prob1, prob2, prob3]
    istates = [1]

    mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs, istates))
    @test IntervalMDP.cpu(initial_states(mdp)) == istates

    mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

    @testset "bellman" begin
        V = IntervalMDP.cu(N[1, 2, 3])
        Vres = IntervalMDP.bellman(V, mdp; upper_bound = false, maximize = true)
        Vres = IntervalMDP.cpu(Vres)  # Convert to CPU for testing
        @test Vres ≈ N[
            (1 // 2) * 1 + (3 // 10) * 2 + (1 // 5) * 3,
            (3 // 10) * 1 + (3 // 10) * 2 + (2 // 5) * 3,
            1 * 3,
        ]

        Vres = IntervalMDP.cu(similar(Vres))
        IntervalMDP.bellman!(Vres, V, mdp; upper_bound = false, maximize = true)
        Vres = IntervalMDP.cpu(Vres)  # Convert to CPU for testing
        @test Vres ≈ N[
            (1 // 2) * 1 + (3 // 10) * 2 + (1 // 5) * 3,
            (3 // 10) * 1 + (3 // 10) * 2 + (2 // 5) * 3,
            1 * 3,
        ]
    end

    @testset "explicit sink state" begin
        transition_prob = IntervalMDP.interval_prob_hcat(transition_probs)
        @test_throws DimensionMismatch IntervalMarkovChain(transition_prob)

        # Finite time reachability
        @testset "finite time reachability" begin
            prop = FiniteTimeReachability([3], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10
            @test all(V_fixed_it1 .>= N(0))
            @test V_fixed_it1[3] == N(1)

            spec = Specification(prop, Optimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)
            @test V_fixed_it2[3] == N(1)

            spec = Specification(prop, Pessimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10
            @test all(V_fixed_it1 .>= N(0))
            @test V_fixed_it1[3] == N(1)

            spec = Specification(prop, Optimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)
            @test V_fixed_it2[3] == N(1)
        end

        # Infinite time reachability
        @testset "infinite time reachability" begin
            prop = InfiniteTimeReachability([3], N(1//1_000_000))
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_conv, _, u = solve(problem)
            V_conv = IntervalMDP.cpu(V_conv)
            @test maximum(u) <= N(1//1_000_000)
            @test all(V_conv .>= N(0))
            @test V_conv[3] == N(1)
        end

        # Exact time reachability
        @testset "exact time reachability" begin
            prop = ExactTimeReachability([3], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10
            @test all(V_fixed_it1 .>= N(0))
            @test all(V_fixed_it1 .<= N(1))

            spec = Specification(prop, Optimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)

            spec = Specification(prop, Pessimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10
            @test all(V_fixed_it1 .>= N(0))
            @test all(V_fixed_it1 .<= N(1))

            spec = Specification(prop, Optimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)

            # Compare exact time to finite time
            prop = ExactTimeReachability([3], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10

            prop = FiniteTimeReachability([3], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)
        end

        # Finite time reach avoid
        @testset "finite time reach/avoid" begin
            prop = FiniteTimeReachAvoid([3], [2], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10
            @test all(V_fixed_it1 .>= N(0))
            @test all(V_fixed_it1 .<= N(1))
            @test V_fixed_it1[3] == N(1)
            @test V_fixed_it1[2] == N(0)

            spec = Specification(prop, Optimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)
            @test V_fixed_it2[3] == N(1)
            @test V_fixed_it2[2] == N(0)

            spec = Specification(prop, Pessimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10
            @test all(V_fixed_it1 .>= N(0))
            @test all(V_fixed_it1 .<= N(1))
            @test V_fixed_it1[3] == N(1)
            @test V_fixed_it1[2] == N(0)

            spec = Specification(prop, Optimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)
            @test V_fixed_it2[3] == N(1)
            @test V_fixed_it2[2] == N(0)
        end

        # Infinite time reach avoid
        @testset "infinite time reach/avoid" begin
            prop = InfiniteTimeReachAvoid([3], [2], N(1//1_000_000))
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_conv, _, u = solve(problem)
            V_conv = IntervalMDP.cpu(V_conv)
            @test maximum(u) <= N(1//1_000_000)
            @test all(V_conv .>= N(0))
            @test all(V_conv .<= N(1))
            @test V_conv[3] == N(1)
            @test V_conv[2] == N(0)
        end

        # Exact time reach avoid
        @testset "exact time reach/avoid" begin
            prop = ExactTimeReachAvoid([3], [2], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10
            @test all(V_fixed_it1 .>= N(0))
            @test all(V_fixed_it1 .<= N(1))
            @test V_fixed_it1[2] == N(0)

            spec = Specification(prop, Optimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)
            @test V_fixed_it2[2] == N(0)

            spec = Specification(prop, Pessimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10
            @test all(V_fixed_it1 .>= N(0))
            @test V_fixed_it1[2] == N(0)

            spec = Specification(prop, Optimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)
            @test V_fixed_it2[2] == N(0)

            # Compare exact time to finite time
            prop = ExactTimeReachAvoid([3], [2], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10

            prop = FiniteTimeReachAvoid([3], [2], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)
        end

        # Finite time reward
        @testset "finite time reward" begin
            prop = IntervalMDP.cu(FiniteTimeReward(N[2, 1, 0], N(9//10), 10))
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10
            @test all(V_fixed_it1 .>= N(0))

            spec = Specification(prop, Optimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)

            spec = Specification(prop, Pessimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it1, k, _ = solve(problem)
            V_fixed_it1 = IntervalMDP.cpu(V_fixed_it1)
            @test k == 10
            @test all(V_fixed_it1 .>= N(0))

            spec = Specification(prop, Optimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_fixed_it2, k, _ = solve(problem)
            V_fixed_it2 = IntervalMDP.cpu(V_fixed_it2)
            @test k == 10
            @test all(V_fixed_it1 .<= V_fixed_it2)
        end

        # Infinite time reward
        @testset "infinite time reward" begin
            prop = IntervalMDP.cu(InfiniteTimeReward(N[2, 1, 0], N(9//10), N(1//1_000_000)))
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_conv, _, u = solve(problem)
            V_conv = IntervalMDP.cpu(V_conv)
            @test maximum(u) <= N(1//1_000_000)
            @test all(V_conv .>= N(0))
        end

        # Expected exit time
        @testset "expected exit time" begin
            prop = ExpectedExitTime([3], N(1//1_000_000))

            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_conv1, _, u = solve(problem)
            V_conv1 = IntervalMDP.cpu(V_conv1)
            @test maximum(u) <= N(1//1_000_000)
            @test all(V_conv1 .>= N(0))
            @test V_conv1[3] == N(0)

            spec = Specification(prop, Optimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V_conv2, _, u = solve(problem)
            V_conv2 = IntervalMDP.cpu(V_conv2)
            @test maximum(u) <= N(1//1_000_000)
            @test all(V_conv1 .<= V_conv2)
            @test V_conv2[3] == N(0)

            spec = Specification(prop, Pessimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_conv1, _, u = solve(problem)
            V_conv1 = IntervalMDP.cpu(V_conv1)
            @test maximum(u) <= N(1//1_000_000)
            @test all(V_conv1 .>= N(0))
            @test V_conv1[3] == N(0)

            spec = Specification(prop, Optimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V_conv2, _, u = solve(problem)
            V_conv2 = IntervalMDP.cpu(V_conv2)
            @test maximum(u) <= N(1//1_000_000)
            @test all(V_conv1 .<= V_conv2)
            @test V_conv2[3] == N(0)
        end
    end

    @testset "implicit sink state" begin
        transition_probs = [prob1, prob2]
        implicit_mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

        # Finite time reachability
        @testset "finite time reachability" begin
            prop = FiniteTimeReachability([3], 10)

            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Optimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Pessimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Optimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
        end

        # Infinite time reachability
        @testset "infinite time reachability" begin
            prop = InfiniteTimeReachability([3], N(1//1_000_000))
            spec = Specification(prop, Pessimistic, Maximize)

            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
        end

        # Exact time reachability
        @testset "exact time reachability" begin
            prop = ExactTimeReachability([3], 10)

            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Optimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Pessimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Optimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
        end

        # Finite time reach avoid
        @testset "finite time reach/avoid" begin
            prop = FiniteTimeReachAvoid([3], [2], 10)

            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Optimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Pessimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Optimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
        end

        # Infinite time reach avoid
        @testset "infinite time reach/avoid" begin
            prop = InfiniteTimeReachAvoid([3], [2], N(1//1_000_000))
            spec = Specification(prop, Pessimistic, Maximize)

            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
        end

        # Exact time reach avoid
        @testset "exact time reach/avoid" begin
            prop = ExactTimeReachAvoid([3], [2], 10)

            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Optimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Pessimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Optimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
        end

        # Finite time reward
        @testset "finite time reward" begin
            prop = IntervalMDP.cu(FiniteTimeReward(N[2, 1, 0], N(9//10), 10))

            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Optimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Pessimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5

            spec = Specification(prop, Optimistic, Minimize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test k == k_implicit
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
        end

        # Infinite time reward
        @testset "infinite time reward" begin
            prop = IntervalMDP.cu(InfiniteTimeReward(N[2, 1, 0], N(9//10), N(1//1_000_000)))
            spec = Specification(prop, Pessimistic, Maximize)
            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
        end

        # Expected exit time
        @testset "expected exit time" begin
            prop = ExpectedExitTime([3], N(1//1_000_000))
            spec = Specification(prop, Pessimistic, Maximize)

            problem = VerificationProblem(mdp, spec)
            V, k, res = solve(problem)

            problem_implicit = VerificationProblem(implicit_mdp, spec)
            V_implicit, k_implicit, res_implicit = solve(problem_implicit)

            @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
            @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
        end
    end
end
