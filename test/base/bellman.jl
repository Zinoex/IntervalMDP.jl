using Revise, Test
using IntervalMDP

for N in [Float32, Float64]
    @testset "N = $N" begin
        prob = IntervalProbabilities(;
            lower = N[0.0 0.5; 0.1 0.3; 0.2 0.1],
            upper = N[0.5 0.7; 0.6 0.5; 0.7 0.3],
        )

        V = N[1.0, 2.0, 3.0]

        #### Maximization
        @testset "maximization" begin
            ws = construct_workspace(prob)
            strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
            Vres = zeros(N, 2)
            IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = true)
            @test Vres ≈ N[0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

            ws = IntervalMDP.DenseWorkspace(gap(prob), 1)
            strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = true)
            @test Vres ≈ N[0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

            ws = IntervalMDP.ThreadedDenseWorkspace(gap(prob), 1)
            strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = true)
            @test Vres ≈ N[0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]
        end

        #### Minimization
        @testset "minimization" begin
            ws = construct_workspace(prob)
            strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
            Vres = zeros(N, 2)
            IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = false)
            @test Vres ≈ N[0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

            ws = IntervalMDP.DenseWorkspace(gap(prob), 1)
            strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = false)
            @test Vres ≈ N[0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

            ws = IntervalMDP.ThreadedDenseWorkspace(gap(prob), 1)
            strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = false)
            @test Vres ≈ N[0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]
        end
    end
end