using Revise, Test
using IntervalMDP

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

mdp = IntervalMarkovDecisionProcess([prob1, prob2, prob3], [Int32(1)])
ivi_alg = IntervalValueIteration(default_bellman_algorithm(mdp))
rvi_alg = RobustValueIteration(default_bellman_algorithm(mdp))

@testset "finite-time reach-avoid verification" begin
    prop = FiniteTimeReachAvoid([3], [2], 10)

    @testset "pessimistic-maximize" begin
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)

        sol_ivi = solve(problem, ivi_alg)
        sol_rvi = solve(problem, rvi_alg)

        V_lower = value_function(sol_ivi)
        gap = residual(sol_ivi)
        V_upper = sol_ivi.additional_data

        # IVI must agree with RVI on the satisfaction-mode bound.
        @test V_lower ≈ value_function(sol_rvi)
        @test num_iterations(sol_ivi) == 10

        # IVI's two bounds bracket the value function and the gap is correct.
        @test all(V_lower .<= V_upper)
        @test all(0.0 .<= V_lower .<= 1.0)
        @test all(0.0 .<= V_upper .<= 1.0)
        @test gap ≈ V_upper .- V_lower

        # Reach/avoid states are pinned.
        @test V_lower[3] ≈ 1.0
        @test V_upper[3] ≈ 1.0
        @test V_lower[2] ≈ 0.0
        @test V_upper[2] ≈ 0.0
    end

    @testset "optimistic-maximize" begin
        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)

        sol_ivi = solve(problem, ivi_alg)
        sol_rvi = solve(problem, rvi_alg)

        V_upper = value_function(sol_ivi)
        V_lower = sol_ivi.additional_data
        V_rvi = value_function(sol_rvi)

        # At finite horizon the two bounds bracket the RVI value (V_upper
        # over-approximates from above, V_lower under-approximates from
        # below; both converge to V_rvi as horizon grows).
        @test all(V_lower .<= V_rvi .+ 1e-12)
        @test all(V_rvi .<= V_upper .+ 1e-12)
    end
end

@testset "infinite-time reach-avoid verification" begin
    tol = 1e-6
    prop = InfiniteTimeReachAvoid([3], [2], tol)

    @testset "pessimistic-maximize" begin
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)

        sol_ivi = solve(problem, ivi_alg)
        V_lower = value_function(sol_ivi)
        gap = residual(sol_ivi)
        V_upper = sol_ivi.additional_data

        # Termination: gap on the initial set is below tol.
        for s in initial_states(mdp)
            @test gap[CartesianIndex(s)] < tol
        end

        # IVI's pessimistic lower bound matches RVI to within the tolerance.
        sol_rvi = solve(problem, rvi_alg)
        @test all(isapprox.(V_lower, value_function(sol_rvi); atol = tol))
        @test all(V_lower .<= V_upper)
    end

    @testset "optimistic-maximize" begin
        spec = Specification(prop, Optimistic, Maximize)
        problem = VerificationProblem(mdp, spec)

        sol_ivi = solve(problem, ivi_alg)
        V_upper = value_function(sol_ivi)
        gap = residual(sol_ivi)

        for s in initial_states(mdp)
            @test gap[CartesianIndex(s)] < tol
        end

        sol_rvi = solve(problem, rvi_alg)
        @test all(isapprox.(V_upper, value_function(sol_rvi); atol = tol))
    end
end

@testset "reach-avoid control synthesis" begin
    @testset "finite-time" begin
        prop = FiniteTimeReachAvoid([3], [2], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = ControlSynthesisProblem(mdp, spec)

        sol = solve(problem, ivi_alg)
        policy = strategy(sol)
        V_lower = value_function(sol)

        @test policy isa TimeVaryingStrategy
        @test time_length(policy) == 10

        # Re-verifying with the synthesized policy must reproduce V_lower.
        verify = VerificationProblem(mdp, spec, policy)
        V_mc, _, _ = solve(verify, rvi_alg)
        @test V_mc ≈ V_lower
    end

    @testset "infinite-time" begin
        tol = 1e-6
        prop = InfiniteTimeReachAvoid([3], [2], tol)
        spec = Specification(prop, Pessimistic, Maximize)
        problem = ControlSynthesisProblem(mdp, spec)

        sol = solve(problem, ivi_alg)
        policy = strategy(sol)
        V_lower = value_function(sol)
        gap = residual(sol)

        @test policy isa StationaryStrategy

        for s in initial_states(mdp)
            @test gap[CartesianIndex(s)] < tol
        end

        # The synthesized policy, re-applied via RVI, must produce a value at
        # least as good as V_lower (within tolerance).
        verify_prop = InfiniteTimeReachAvoid([3], [2], tol)
        verify_spec = Specification(verify_prop, Pessimistic, Maximize)
        verify = VerificationProblem(mdp, verify_spec, policy)
        V_mc, _, _ = solve(verify, rvi_alg)
        @test all(V_mc .>= V_lower .- tol)
    end
end

@testset "rejects non-reach-avoid properties" begin
    # Plain reachability without an avoid set is not supported.
    bad_props = (
        FiniteTimeReachability([3], 10),
        InfiniteTimeReachability([3], 1e-6),
        FiniteTimeSafety([3], 10),
        FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10),
    )
    for prop in bad_props
        spec = Specification(prop, Pessimistic, Maximize)
        problem = VerificationProblem(mdp, spec)
        @test_throws ArgumentError solve(problem, ivi_alg)
    end
end
