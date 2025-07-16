using Revise, Test
using IntervalMDP, SparseArrays

for N in [Float32, Float64, Rational{BigInt}]
    @testset "N = $N" begin
        prob = IntervalProbabilities(;
            lower = sparse_hcat(
                SparseVector(15, [4, 10], N[1 // 10, 2 // 10]),
                SparseVector(15, [5, 6, 7], N[5 // 10, 3 // 10, 1 // 10]),
            ),
            upper = sparse_hcat(
                SparseVector(15, [1, 4, 10], N[5 // 10, 6 // 10, 7 // 10]),
                SparseVector(15, [5, 6, 7], N[7 // 10, 5 // 10, 3 // 10]),
            ),
        )

        V = collect(1.0:15.0)

        #### Maximization
        @testset "maximization" begin
            ws = IntervalMDP.construct_workspace(prob)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = zeros(Float64, 2)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = true,
            )
            @test Vres ≈ N[82 // 10, 57 // 10]  # [0.3 * 4 + 0.7 * 10, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

            ws = IntervalMDP.SparseWorkspace(gap(prob), 1)
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
            @test Vres ≈ N[82 // 10, 57 // 10]

            ws = IntervalMDP.ThreadedSparseWorkspace(gap(prob), 1)
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
            @test Vres ≈ N[82 // 10, 57 // 10]
        end

        #### Minimization
        @testset "minimization" begin
            ws = IntervalMDP.construct_workspace(prob)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = zeros(Float64, 2)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = false,
            )
            @test Vres ≈ N[37 // 10, 55 // 10]  # [0.5 * 1 + 0.3 * 4 + 0.2 * 10, 0.6 * 5 + 0.3 * 6 + 0.1 * 7]

            ws = IntervalMDP.SparseWorkspace(gap(prob), 1)
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
            @test Vres ≈ N[37 // 10, 55 // 10]

            ws = IntervalMDP.ThreadedSparseWorkspace(gap(prob), 1)
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
            @test Vres ≈ N[37 // 10, 55 // 10]
        end
    end
end
