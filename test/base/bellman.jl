using Revise, Test
using IntervalMDP

for N in [Float32, Float64, Rational{BigInt}]
    @testset "N = $N" begin
        prob = IntervalProbabilities(;
            lower = N[0 1//2; 1//10 3//10; 2//10 1//10],
            upper = N[5//10 7//10; 6//10 5//10; 7//10 3//10],
        )

        V = N[1, 2, 3]

        #### Maximization
        @testset "maximization" begin
            ws = IntervalMDP.construct_workspace(prob)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = zeros(N, 2)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = true,
            )
            @test Vres ≈ N[27 // 10, 17 // 10] # [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

            ws = IntervalMDP.DenseWorkspace(gap(prob), 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = true,
            )
            @test Vres ≈ N[27 // 10, 17 // 10]

            ws = IntervalMDP.ThreadedDenseWorkspace(gap(prob), 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = true,
            )
            @test Vres ≈ N[27 // 10, 17 // 10]
        end

        #### Minimization
        @testset "minimization" begin
            ws = IntervalMDP.construct_workspace(prob)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = zeros(N, 2)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = false,
            )
            @test Vres ≈ N[17 // 10, 15 // 10]  # [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

            ws = IntervalMDP.DenseWorkspace(gap(prob), 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = false,
            )
            @test Vres ≈ N[17 // 10, 15 // 10]

            ws = IntervalMDP.ThreadedDenseWorkspace(gap(prob), 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = false,
            )
            @test Vres ≈ N[17 // 10, 15 // 10]
        end
    end
end
