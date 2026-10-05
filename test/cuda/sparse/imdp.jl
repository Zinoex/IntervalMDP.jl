@testmodule CudaSparseImdpModels begin
    using IntervalMDP, CUDA, SparseArrays

    function build(N)
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

        return (; prob1, prob2, prob3, transition_probs)
    end
end

@testitem "cuda/sparse/imdp: construction" setup = [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)
        istates = [1]

        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs, istates))
        @test IntervalMDP.cpu(initial_states(mdp)) == istates
    end
end

@testitem "cuda/sparse/imdp: bellman" setup = [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)

        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

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
end

@testitem "cuda/sparse/imdp: explicit sink — IntervalMarkovChain dimension mismatch" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)

        transition_prob = IntervalMDP.interval_prob_hcat(transition_probs)
        @test_throws DimensionMismatch IntervalMarkovChain(transition_prob)
    end
end

@testitem "cuda/sparse/imdp: explicit sink — finite time reachability" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

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
end

@testitem "cuda/sparse/imdp: explicit sink — infinite time reachability" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

        prop = InfiniteTimeReachability([3], N(1//1_000_000))
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv, _, u = solve(problem)
        V_conv = IntervalMDP.cpu(V_conv)
        @test maximum(u) <= N(1//1_000_000)
        @test all(V_conv .>= N(0))
        @test V_conv[3] == N(1)
    end
end

@testitem "cuda/sparse/imdp: explicit sink — exact time reachability" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

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
end

@testitem "cuda/sparse/imdp: explicit sink — finite time reach/avoid" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

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
end

@testitem "cuda/sparse/imdp: explicit sink — infinite time reach/avoid" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

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
end

@testitem "cuda/sparse/imdp: explicit sink — exact time reach/avoid" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

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
end

@testitem "cuda/sparse/imdp: explicit sink — finite time reward" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

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
end

@testitem "cuda/sparse/imdp: explicit sink — infinite time reward" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

        prop = IntervalMDP.cu(InfiniteTimeReward(N[2, 1, 0], N(9//10), N(1//1_000_000)))
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V_conv, _, u = solve(problem)
        V_conv = IntervalMDP.cpu(V_conv)
        @test maximum(u) <= N(1//1_000_000)
        @test all(V_conv .>= N(0))
    end
end

@testitem "cuda/sparse/imdp: explicit sink — expected exit time" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))

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

@testitem "cuda/sparse/imdp: implicit sink — finite time reachability" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; prob1, prob2, transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))
        implicit_mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess([prob1, prob2]))

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
end

@testitem "cuda/sparse/imdp: implicit sink — infinite time reachability" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; prob1, prob2, transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))
        implicit_mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess([prob1, prob2]))

        prop = InfiniteTimeReachability([3], N(1//1_000_000))
        spec = Specification(prop, Pessimistic, Maximize)

        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
        @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
    end
end

@testitem "cuda/sparse/imdp: implicit sink — exact time reachability" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; prob1, prob2, transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))
        implicit_mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess([prob1, prob2]))

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
end

@testitem "cuda/sparse/imdp: implicit sink — finite time reach/avoid" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; prob1, prob2, transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))
        implicit_mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess([prob1, prob2]))

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
end

@testitem "cuda/sparse/imdp: implicit sink — infinite time reach/avoid" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; prob1, prob2, transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))
        implicit_mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess([prob1, prob2]))

        prop = InfiniteTimeReachAvoid([3], [2], N(1//1_000_000))
        spec = Specification(prop, Pessimistic, Maximize)

        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
        @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
    end
end

@testitem "cuda/sparse/imdp: implicit sink — exact time reach/avoid" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; prob1, prob2, transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))
        implicit_mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess([prob1, prob2]))

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
end

@testitem "cuda/sparse/imdp: implicit sink — finite time reward" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; prob1, prob2, transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))
        implicit_mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess([prob1, prob2]))

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
end

@testitem "cuda/sparse/imdp: implicit sink — infinite time reward" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; prob1, prob2, transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))
        implicit_mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess([prob1, prob2]))

        prop = IntervalMDP.cu(InfiniteTimeReward(N[2, 1, 0], N(9//10), N(1//1_000_000)))
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        V, k, res = solve(problem)

        problem_implicit = VerificationProblem(implicit_mdp, spec)
        V_implicit, k_implicit, res_implicit = solve(problem_implicit)

        @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_implicit) atol=1e-5
        @test IntervalMDP.cpu(res) ≈ IntervalMDP.cpu(res_implicit) atol=1e-5
    end
end

@testitem "cuda/sparse/imdp: implicit sink — expected exit time" setup =
    [CudaSparseImdpModels] tags = [:cuda] begin
    using CUDA
    using SparseArrays

    @testset for N in [Float32, Float64]
        (; prob1, prob2, transition_probs) = CudaSparseImdpModels.build(N)
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs))
        implicit_mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess([prob1, prob2]))

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
