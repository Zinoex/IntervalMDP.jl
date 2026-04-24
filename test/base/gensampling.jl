using Revise, Test
using IntervalMDP

# Phase 1 parity test: `GeneralizedSamplingbasedRobustDynamicProgramming`
# with the default `AllSampling()` sampling strategy should produce the same
# value function as `RobustValueIteration`, since both route through the
# state-value `bellman_update!` path over the full (a, s) product.

@testset "IMDP verification parity (Pessimistic, Maximize)" for N in [
    Float32,
    Float64,
    Rational{BigInt},
]
    prob = IntervalAmbiguitySets(;
        lower = N[
            0 1//2 0
            1//10 3//10 0
            1//5 1//10 1
        ],
        upper = N[
            1//2 7//10 0
            3//5 1//2 0
            7//10 3//10 1
        ],
    )

    prob2 = IntervalAmbiguitySets(;
        lower = N[
            1//10 1//5 0
            1//5 1//5 0
            3//10 2//5 1
        ],
        upper = N[
            1//2 1//2 0
            1//2 2//5 0
            2//5 2//5 1
        ],
    )

    transition_probs = [prob, prob2, prob2]
    mdp = IntervalMarkovDecisionProcess(transition_probs, [1])

    rvi = RobustValueIteration(default_bellman_algorithm(mdp))
    gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))

    # Finite-time reachability
    prop = FiniteTimeReachability([3], 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)
    V_rvi, k_rvi, _ = solve(problem, rvi)
    V_gsdp, k_gsdp, _ = solve(problem, gsdp)
    @test k_rvi == k_gsdp
    @test V_rvi == V_gsdp

    # Infinite-time reachability (stationary strategy cache path on both sides
    # because neither is a ControlSynthesisProblem).
    prop = InfiniteTimeReachability([3], N(1//1_000_000))
    spec = Specification(prop, Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)
    V_rvi, k_rvi, _ = solve(problem, rvi)
    V_gsdp, k_gsdp, _ = solve(problem, gsdp)
    @test k_rvi == k_gsdp
    @test V_rvi == V_gsdp

    # Finite-time safety
    prop = FiniteTimeSafety([3], 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)
    V_rvi, k_rvi, _ = solve(problem, rvi)
    V_gsdp, k_gsdp, _ = solve(problem, gsdp)
    @test k_rvi == k_gsdp
    @test V_rvi == V_gsdp

    # Finite-time reach-avoid
    prop = FiniteTimeReachAvoid([3], [2], 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)
    V_rvi, k_rvi, _ = solve(problem, rvi)
    V_gsdp, k_gsdp, _ = solve(problem, gsdp)
    @test k_rvi == k_gsdp
    @test V_rvi == V_gsdp

    # Finite-time reward
    prop = FiniteTimeReward(N[2, 1, 0], N(9//10), 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = VerificationProblem(mdp, spec)
    V_rvi, k_rvi, _ = solve(problem, rvi)
    V_gsdp, k_gsdp, _ = solve(problem, gsdp)
    @test k_rvi == k_gsdp
    @test V_rvi == V_gsdp
end

@testset "IMDP control synthesis parity (Pessimistic, Maximize)" for N in [Float32, Float64]
    prob = IntervalAmbiguitySets(;
        lower = N[
            0 1//2 0
            1//10 3//10 0
            1//5 1//10 1
        ],
        upper = N[
            1//2 7//10 0
            3//5 1//2 0
            7//10 3//10 1
        ],
    )

    prob2 = IntervalAmbiguitySets(;
        lower = N[
            1//10 1//5 0
            1//5 1//5 0
            3//10 2//5 1
        ],
        upper = N[
            1//2 1//2 0
            1//2 2//5 0
            2//5 2//5 1
        ],
    )

    mdp = IntervalMarkovDecisionProcess([prob, prob2, prob2], [1])

    rvi = RobustValueIteration(default_bellman_algorithm(mdp))
    gsdp = GeneralizedSamplingbasedRobustDynamicProgramming(default_bellman_algorithm(mdp))

    # Infinite-horizon reachability synthesis produces a stationary strategy
    # on both sides — parity expected on V (and strategy, since Pessimistic
    # Maximize with the same tie-breaking is deterministic given the shared
    # strategy-cache construction).
    prop = InfiniteTimeReachability([3], N(1//1_000_000))
    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    sol_rvi = solve(problem, rvi)
    sol_gsdp = solve(problem, gsdp)
    @test num_iterations(sol_rvi) == num_iterations(sol_gsdp)
    @test value_function(sol_rvi) == value_function(sol_gsdp)
end
