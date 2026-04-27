@testitem "implicit sink state" tags =
    [:cuda, :implicit_sink_state] begin
    using IntervalMDP, CUDA, SparseArrays
    if CUDA.functional()
        prob1 = IntervalAmbiguitySets(;
            lower = sparse([0.0 0.5; 0.1 0.3; 0.2 0.1]),
            upper = sparse([0.5 0.7; 0.6 0.5; 0.7 0.3]),
        )
        prob2 = IntervalAmbiguitySets(;
            lower = sparse([0.1 0.2; 0.2 0.3; 0.3 0.4]),
            upper = sparse([0.6 0.6; 0.5 0.5; 0.4 0.4]),
        )
        prob3 = IntervalAmbiguitySets(;
            lower = sparse([0.0 0.0; 0.0 0.0; 1.0 1.0]),
            upper = sparse([0.0 0.0; 0.0 0.0; 1.0 1.0]),
        )
        transition_probs = [prob1, prob2, prob3]
        istates = [Int32(1)]
        mdp = IntervalMDP.cu(IntervalMarkovDecisionProcess(transition_probs, istates))
        prop = FiniteTimeReachability([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = ControlSynthesisProblem(mdp, spec)
        sol = solve(problem)
        (policy, V, k, res) = sol
        @test strategy(sol) == policy
        @test value_function(sol) == V
        @test num_iterations(sol) == k
        @test residual(sol) == res
        policy = IntervalMDP.cpu(policy)
        @test policy isa TimeVaryingStrategy
        @test time_length(policy) == 10
        for k in 1:time_length(policy)
            @test policy[k] == [(1,), (2,), (1,)]
        end
        problem = VerificationProblem(mdp, spec, IntervalMDP.cu(policy))
        (V_mc, k, res) = solve(problem)
        @test V ≈ V_mc
        prop = IntervalMDP.cu(FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10))
        spec = Specification(prop, Pessimistic, Maximize)
        problem = ControlSynthesisProblem(mdp, spec)
        (policy, V, k, res) = solve(problem)
        policy = IntervalMDP.cpu(policy)
        @test policy isa TimeVaryingStrategy
        @test time_length(policy) == 10
        for k in 1:time_length(policy)
            @test policy[k] == [(2,), (2,), (1,)]
        end
        problem = VerificationProblem(mdp, spec, IntervalMDP.cu(policy))
        (V_mc, k, res) = solve(problem)
        @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_mc) atol = 1.0e-5
        prop = InfiniteTimeReachability([3], 1.0e-6)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = ControlSynthesisProblem(mdp, spec)
        (policy, V, k, res) = solve(problem)
        policy = IntervalMDP.cpu(policy)
        @test policy isa StationaryStrategy
        @test policy[1] == [(1,), (2,), (1,)]
        problem = VerificationProblem(mdp, spec, IntervalMDP.cu(policy))
        (V_mc, k, res) = solve(problem)
        @test IntervalMDP.cpu(V) ≈ IntervalMDP.cpu(V_mc) atol = 1.0e-5
        prop = FiniteTimeSafety([3], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = ControlSynthesisProblem(mdp, spec)
        (policy, V, k, res) = solve(problem)
        policy = IntervalMDP.cpu(policy)
        V = IntervalMDP.cpu(V)
        @test all(V .>= 0.0)
        @test V[3] ≈ 0.0
        @test policy isa TimeVaryingStrategy
        @test time_length(policy) == 10
        for k in 1:(time_length(policy) - 1)
            @test policy[k] == [(2,), (2,), (1,)]
        end
        @test policy[time_length(policy)] == [(2,), (1,), (1,)]
        @testset "implicit sink state" begin
            transition_probs = [prob1, prob2]
            mdp = IntervalMarkovDecisionProcess(transition_probs)
            prop = FiniteTimeReachability([3], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            problem = ControlSynthesisProblem(mdp, spec)
            (policy, V, k, res) = solve(problem)
            policy = IntervalMDP.cpu(policy)
            @test policy isa TimeVaryingStrategy
            @test time_length(policy) == 10
            for k in 1:time_length(policy)
                @test policy[k] == [(1,), (2,)]
            end
        end
    end
end
