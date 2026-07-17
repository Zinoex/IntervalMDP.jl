@testmodule BaseBellmanModels begin
    using IntervalMDP

    function build(N)
        prob = IntervalAmbiguitySets(;
            lower = N[0 1//2; 1//10 3//10; 2//10 1//10],
            upper = N[5//10 7//10; 6//10 5//10; 7//10 3//10],
        )

        V = N[1, 2, 3]

        states = FullUpdateSequence(prob)

        return (; prob, V, states)
    end
end

@testitem "base/bellman: maximization" setup = [BaseBellmanModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; prob, V, states) = BaseBellmanModels.build(N)

        #### Maximization
        ws = IntervalMDP.construct_workspace(prob)
        strategy_cache = IntervalMDP.construct_strategy_cache(prob)
        Vres = zeros(N, 2)
        IntervalMDP._bellman_helper!(
            ws, strategy_cache, Vres, V, prob, states;
            upper_bound = true,
        )
        @test Vres ≈ N[27 // 10, 17 // 10] # [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

        ws = IntervalMDP.DenseIntervalOMaxWorkspace(prob, 1)
        strategy_cache = IntervalMDP.construct_strategy_cache(prob)
        Vres = similar(Vres)
        IntervalMDP._bellman_helper!(
            ws, strategy_cache, Vres, V, prob, states;
            upper_bound = true,
        )
        @test Vres ≈ N[27 // 10, 17 // 10]

        ws = IntervalMDP.ThreadedDenseIntervalOMaxWorkspace(prob, 1)
        strategy_cache = IntervalMDP.construct_strategy_cache(prob)
        Vres = similar(Vres)
        IntervalMDP._bellman_helper!(
            ws, strategy_cache, Vres, V, prob, states;
            upper_bound = true,
        )
        @test Vres ≈ N[27 // 10, 17 // 10]
    end
end

@testitem "base/bellman: minimization" setup = [BaseBellmanModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        (; prob, V, states) = BaseBellmanModels.build(N)

        #### Minimization
        ws = IntervalMDP.construct_workspace(prob)
        strategy_cache = IntervalMDP.construct_strategy_cache(prob)
        Vres = zeros(N, 2)
        IntervalMDP._bellman_helper!(
            ws, strategy_cache, Vres, V, prob, states;
            upper_bound = false,
        )
        @test Vres ≈ N[17 // 10, 15 // 10]  # [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

        ws = IntervalMDP.DenseIntervalOMaxWorkspace(prob, 1)
        strategy_cache = IntervalMDP.construct_strategy_cache(prob)
        Vres = similar(Vres)
        IntervalMDP._bellman_helper!(
            ws, strategy_cache, Vres, V, prob, states;
            upper_bound = false,
        )
        @test Vres ≈ N[17 // 10, 15 // 10]

        ws = IntervalMDP.ThreadedDenseIntervalOMaxWorkspace(prob, 1)
        strategy_cache = IntervalMDP.construct_strategy_cache(prob)
        Vres = similar(Vres)
        IntervalMDP._bellman_helper!(
            ws, strategy_cache, Vres, V, prob, states;
            upper_bound = false,
        )
        @test Vres ≈ N[17 // 10, 15 // 10]
    end
end

@testitem "base/bellman: FullUpdateSequence == default" setup = [BaseBellmanModels] begin
    @testset for N in [Float32, Float64, Rational{BigInt}]
        #### Default `states` matches explicit FullUpdateSequence (public bellman! API on an IMDP)
        # 3 source states × 1 action, 3 targets (square IMDP so bellman! works)
        square = IntervalAmbiguitySets(;
            lower = N[0 1//2 0; 1//10 3//10 0; 1//5 1//10 1],
            upper = N[1//2 7//10 0; 3//5 1//2 0; 7//10 3//10 1],
        )
        mdp = IntervalMarkovDecisionProcess(square, 1, [Int32(1)])
        Vin = N[1, 2, 3]

        Vdefault = zeros(N, 3)
        IntervalMDP.bellman!(Vdefault, Vin, mdp; upper_bound = true)

        Vexplicit = zeros(N, 3)
        IntervalMDP.bellman!(
            Vexplicit,
            Vin,
            mdp,
            IntervalMDP.default_bellman_algorithm(mdp),
            FullUpdateSequence(mdp);
            upper_bound = true,
        )
        @test Vdefault == Vexplicit

        # Same result with a hand-constructed CartesianIndices wrapper
        Vhand = zeros(N, 3)
        IntervalMDP.bellman!(
            Vhand,
            Vin,
            mdp,
            IntervalMDP.default_bellman_algorithm(mdp),
            FullUpdateSequence(CartesianIndices(IntervalMDP.source_shape(mdp)));
            upper_bound = true,
        )
        @test Vdefault == Vhand
    end
end
