@testmodule BaseSynthesisModels begin
    using IntervalMDP

    const prob1 = IntervalAmbiguitySets(;
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

    const prob2 = IntervalAmbiguitySets(;
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

    const prob3 = IntervalAmbiguitySets(;
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

    const transition_probs = [prob1, prob2, prob3]
    const istates = [Int32(1)]

    const mdp = IntervalMarkovDecisionProcess(transition_probs, istates)
end

@testitem "base/synthesis: finite time reachability" setup = [BaseSynthesisModels] begin
    mdp = BaseSynthesisModels.mdp

    # Finite time reachability
    prop = FiniteTimeReachability([3], 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    sol = solve(problem)
    policy, V, k, res = sol

    @test strategy(sol) == policy
    @test value_function(sol) == V
    @test num_iterations(sol) == k
    @test residual(sol) == res

    @test policy isa TimeVaryingStrategy
    @test time_length(policy) == 10
    for k in 1:time_length(policy)
        @test policy[k] == [(1,), (2,), (1,)]
    end

    # Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
    problem = VerificationProblem(mdp, spec, policy)
    V_mc, k, res = solve(problem)
    @test V ≈ V_mc
end

@testitem "base/synthesis: finite time reward" setup = [BaseSynthesisModels] begin
    mdp = BaseSynthesisModels.mdp

    # Finite time reward
    prop = FiniteTimeReward([2.0, 1.0, 0.0], 0.9, 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    policy, V, k, res = solve(problem)

    @test policy isa TimeVaryingStrategy
    @test time_length(policy) == 10
    for k in 1:time_length(policy)
        @test policy[k] == [(2,), (2,), (1,)]
    end

    # Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
    problem = VerificationProblem(mdp, spec, policy)
    V_mc, k, res = solve(problem)
    @test V ≈ V_mc
end

@testitem "base/synthesis: infinite time reachability" setup = [BaseSynthesisModels] begin
    mdp = BaseSynthesisModels.mdp

    # Infinite time reachability
    prop = InfiniteTimeReachability([3], 1e-6)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    policy, V, k, res = solve(problem)

    @test policy isa StationaryStrategy
    @test policy[1] == [(1,), (2,), (1,)]

    # Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
    problem = VerificationProblem(mdp, spec, policy)
    V_mc, k, res = solve(problem)
    @test V ≈ V_mc atol=1e-6
end

@testitem "base/synthesis: finite time safety" setup = [BaseSynthesisModels] begin
    mdp = BaseSynthesisModels.mdp

    # Finite time safety
    prop = FiniteTimeSafety([3], 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    policy, V, k, res = solve(problem)

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

@testitem "base/synthesis: implicit sink state" setup = [BaseSynthesisModels] begin
    prob1 = BaseSynthesisModels.prob1
    prob2 = BaseSynthesisModels.prob2

    transition_probs = [prob1, prob2]
    mdp = IntervalMarkovDecisionProcess(transition_probs)

    # Finite time reachability
    prop = FiniteTimeReachability([3], 10)
    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    policy, V, k, res = solve(problem)

    @test policy isa TimeVaryingStrategy
    @test time_length(policy) == 10
    for k in 1:time_length(policy)
        @test policy[k] == [(1,), (2,)]
    end
end

# Regression tests for Finding B-1 (issue #119, Lean Finding F3): the stationary strategy
# cache compared the state index instead of the cached action with the available actions,
# so for states with index > number of actions the previous action was dropped on ties and
# the returned stationary strategy could be suboptimal.
@testmodule StationaryCacheModels begin
    using IntervalMDP

    # Deterministic IMDP: `targets[s][a]` is the successor of state `s` under action `a`.
    function deterministic_mdp(targets)
        ns = length(targets)
        probs = map(targets) do tgts
            P = zeros(ns, length(tgts))
            for (a, t) in enumerate(tgts)
                P[t, a] = 1.0
            end
            return IntervalAmbiguitySets(; lower = P, upper = P)
        end
        return IntervalMarkovDecisionProcess(probs)
    end

    # Synthesise a stationary strategy and evaluate it with a verification problem.
    function synthesize_and_evaluate(mdp, goal, satisfaction, strategy_mode)
        prop = InfiniteTimeReachability([goal], 1e-6)
        spec = Specification(prop, satisfaction, strategy_mode)
        policy, V, _, _ = solve(ControlSynthesisProblem(mdp, spec))
        V_policy, _, _ = solve(VerificationProblem(mdp, spec, policy))
        return policy, V, V_policy
    end
end

@testitem "base/synthesis: stationary strategy cache, B-1 reproduction (6 states, 4 actions)" setup =
    [StationaryCacheModels] begin
    # State 1 is the goal; states 2, 4, 5 are absorbing; states 3 and 6 are identical:
    # action 2 goes to the goal, actions 1, 3, 4 self-loop.
    mdp = StationaryCacheModels.deterministic_mdp([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 1, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5, 5],
        [6, 1, 6, 6],
    ])

    for sat in (Pessimistic, Optimistic)
        policy, V, V_policy =
            StationaryCacheModels.synthesize_and_evaluate(mdp, 1, sat, Maximize)

        @test policy isa StationaryStrategy
        @test V ≈ [1.0, 0.0, 1.0, 0.0, 0.0, 1.0] atol = 1e-6
        @test policy[1][3] == (Int32(2),)
        @test policy[1][6] == (Int32(2),)
        # The synthesised stationary strategy is optimal: its value equals the reported value.
        @test V_policy ≈ V atol = 1e-6
    end
end

@testitem "base/synthesis: stationary strategy cache, F3 minimal case (3 states, 2 actions)" setup =
    [StationaryCacheModels] begin
    # State 1 is the goal, state 2 is absorbing, state 3 (index > number of actions):
    # action 1 self-loops, action 2 goes to the goal.
    mdp = StationaryCacheModels.deterministic_mdp([[1, 1], [2, 2], [3, 1]])

    for sat in (Pessimistic, Optimistic)
        policy, V, V_policy =
            StationaryCacheModels.synthesize_and_evaluate(mdp, 1, sat, Maximize)

        @test V ≈ [1.0, 0.0, 1.0] atol = 1e-6
        @test policy[1][3] == (Int32(2),)
        @test V_policy ≈ V atol = 1e-6
    end
end

@testitem "base/synthesis: stationary strategy cache, minimize sanity" setup =
    [StationaryCacheModels] begin
    # Mirrored model: in state 3, action 1 goes to the goal and action 2 self-loops.
    mdp = StationaryCacheModels.deterministic_mdp([[1, 1], [2, 2], [1, 3]])

    for sat in (Pessimistic, Optimistic)
        policy, V, V_policy =
            StationaryCacheModels.synthesize_and_evaluate(mdp, 1, sat, Minimize)

        @test V ≈ [1.0, 0.0, 0.0] atol = 1e-6
        @test policy[1][3] == (Int32(2),)
        @test V_policy ≈ V atol = 1e-6
    end
end
