@testmodule CudaDenseSynthesisModels begin
    using IntervalMDP, CUDA

    function build()
        prob1 = IntervalAmbiguitySets(;
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

        prob2 = IntervalAmbiguitySets(;
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

        prob3 = IntervalAmbiguitySets(;
            lower = [
                0.0 0.0
                0.0 0.0
                1.0 1.0
            ],
            upper = [
                0.0 0.0
                0.0 0.0
                1.0 1.0
            ],
        )

        transition_probs = [prob1, prob2, prob3]
        istates = [Int32(1)]

        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs, istates))

        return (; prob1, prob2, prob3, transition_probs, mdp)
    end
end

@testitem "cuda/dense/synthesis: finite time reachability" setup =
    [CudaDenseSynthesisModels] tags = [:cuda] begin
    using CUDA

    (; mdp) = CudaDenseSynthesisModels.build()

    prop = FiniteTimeReachability([3], 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    sol = solve(problem)
    policy, V, k, res = sol

    @test strategy(sol) == policy
    @test value_function(sol) == V
    @test num_iterations(sol) == k
    @test residual(sol) == res

    policy = IntervalMDP.cpu(policy)  # Convert to CPU for testing

    @test policy isa TimeVaryingStrategy
    @test time_length(policy) == 10
    for k in 1:time_length(policy)
        @test policy[k] == [(1,), (2,), (1,)]
    end

    # Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
    problem = VerificationProblem(mdp, spec, IntervalMDP.cu(policy))
    V_mc, k, res = solve(problem)
    @test V ≈ V_mc
end

@testitem "cuda/dense/synthesis: finite time reward" setup = [CudaDenseSynthesisModels] tags =
    [:cuda] begin
    using CUDA

    (; mdp) = CudaDenseSynthesisModels.build()

    prop = IntervalMDP.cu(FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10))
    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    policy, V, k, res = solve(problem)

    policy = IntervalMDP.cpu(policy)  # Convert to CPU for testing

    @test policy isa TimeVaryingStrategy
    @test time_length(policy) == 10
    for k in 1:time_length(policy)
        @test policy[k] == [(2,), (2,), (1,)]
    end

    # Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
    problem = VerificationProblem(mdp, spec, IntervalMDP.cu(policy))
    V_mc, k, res = solve(problem)
    @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_mc) atol=1e-5
end

@testitem "cuda/dense/synthesis: infinite time reachability" setup =
    [CudaDenseSynthesisModels] tags = [:cuda] begin
    using CUDA

    (; mdp) = CudaDenseSynthesisModels.build()

    prop = InfiniteTimeReachability([3], 1e-6)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    policy, V, k, res = solve(problem)

    policy = IntervalMDP.cpu(policy)  # Convert to CPU for testing

    @test policy isa StationaryStrategy
    @test policy[1] == [(1,), (2,), (1,)]

    # Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
    problem = VerificationProblem(mdp, spec, IntervalMDP.cu(policy))
    V_mc, k, res = solve(problem)
    @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_mc) atol=1e-5
end

@testitem "cuda/dense/synthesis: finite time safety" setup = [CudaDenseSynthesisModels] tags =
    [:cuda] begin
    using CUDA

    (; mdp) = CudaDenseSynthesisModels.build()

    prop = FiniteTimeSafety([3], 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    policy, V, k, res = solve(problem)

    policy = IntervalMDP.cpu(policy)  # Convert to CPU for testing
    V = IntervalMDP.cpu(V)  # Convert to CPU for testing

    @test all(V .>= 0.0)
    @test V[3] ≈ 0.0

    @test policy isa TimeVaryingStrategy
    @test time_length(policy) == 10
    for k in 1:(time_length(policy) - 1)
        @test policy[k] == [(2,), (2,), (1,)]
    end

    # The last time step (aka. the first value iteration step) has a different strategy.
    @test policy[time_length(policy)] == [(2,), (1,), (1,)]
end

@testitem "cuda/dense/synthesis: implicit sink state" setup = [CudaDenseSynthesisModels] tags =
    [:cuda] begin
    using CUDA

    (; prob1, prob2) = CudaDenseSynthesisModels.build()

    transition_probs = [prob1, prob2]
    mdp = IntervalMarkovDecisionProcess(transition_probs)

    # Finite time reachability
    prop = FiniteTimeReachability([3], 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    policy, V, k, res = solve(problem)

    policy = IntervalMDP.cpu(policy)  # Convert to CPU for testing

    @test policy isa TimeVaryingStrategy
    @test time_length(policy) == 10
    for k in 1:time_length(policy)
        @test policy[k] == [(1,), (2,)]
    end
end
