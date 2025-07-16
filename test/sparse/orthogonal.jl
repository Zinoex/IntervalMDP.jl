using Revise, Test
using IntervalMDP, SparseArrays
using Random: MersenneTwister

for N in [Float32, Float64, Rational{BigInt}]
    @testset "N = $N" begin
        @testset "bellman 1d" begin
            prob = OrthogonalIntervalProbabilities(
                (
                    IntervalProbabilities(;
                        lower = sparse(N[0 5//10; 1//10 3//10; 2//10 1//10]),
                        upper = sparse(N[5//10 7//10; 6//10 5//10; 7//10 3//10]),
                    ),
                ),
                (Int32(2),),
            )

            V = N[1, 2, 3]

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
        end

        @testset "bellman 3d" begin
            lower1 = N[
                1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6 1//10 1//30 1//10 1//15 1//10 1//15 4//15 4//15 1//3
                1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10 1//5 1//5 1//10 1//30 4//15 1//10 1//5 1//6 7//30
                4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10 3//10 1//10 1//15 1//30 2//15 1//6 1//5 1//10 4//15
            ]
            lower2 = N[
                1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15 1//6 7//30 1//15 2//15 1//10 1//3 7//30 1//30 7//30
                3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15 3//10 0 3//10 1//6 3//10 1//5 0 7//30 2//15
                3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15 1//15 1//3 2//15 1//15 1//5 1//5 1//15 7//30 1//15
            ]
            lower3 = N[
                4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15 7//30 4//15 1//10 1//3 1//5 7//30 1//30 1//5 7//30
                2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15 1//5 4//15 1//10 1//10 1//3 7//30 3//10 1//3 3//10
                1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30 0 1//30 1//15 2//15 1//6 7//30 4//15 4//15 7//30
            ]

            upper1 = N[
                7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30 17//30 13//30 2//5 2//5 2//3 2//5 17//30 2//5 19//30
                8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30 8//15 7//15 7//15 13//30 8//15 2//5 8//15 17//30 3//5
                11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15 2//5 2//5 11//30 17//30 17//30 1//2 2//5 19//30 13//30
            ]
            upper2 = N[
                2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30 13//30 17//30 17//30 13//30 11//30 19//30 8//15 2//5 8//15
                1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3 1//2 8//15 8//15 8//15 8//15 2//5 3//5 2//3 13//30
                17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3 1//3 2//3 17//30 8//15 17//30 3//5 2//5 19//30 11//30
            ]
            upper3 = N[
                3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30 2//3 17//30 3//5 7//15 19//30 1//2 3//5 1//3 19//30
                3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30 2//3 17//30 8//15 13//30 13//30 3//5 1//2 8//15 8//15
                3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30 0 2//5 11//30 19//30 19//30 2//5 1//2 7//15 7//15
            ]

            prob = OrthogonalIntervalProbabilities(
                (
                    IntervalProbabilities(; lower = lower1, upper = upper1),
                    IntervalProbabilities(; lower = lower2, upper = upper2),
                    IntervalProbabilities(; lower = lower3, upper = upper3),
                ),
                (Int32(3), Int32(3), Int32(3)),
            )

            sparse_prob = OrthogonalIntervalProbabilities(
                (
                    IntervalProbabilities(; lower = sparse(lower1), upper = sparse(upper1)),
                    IntervalProbabilities(; lower = sparse(lower2), upper = sparse(upper2)),
                    IntervalProbabilities(; lower = sparse(lower3), upper = sparse(upper3)),
                ),
                (Int32(3), Int32(3), Int32(3)),
            )

            V = N[
                23,
                27,
                16,
                6,
                26,
                17,
                12,
                9,
                8,
                22,
                1,
                21,
                11,
                24,
                4,
                10,
                13,
                19,
                3,
                14,
                25,
                20,
                18,
                7,
                5,
                15,
                2,
            ]
            V = reshape(V, 3, 3, 3)

            #### Maximization
            @testset "maximization" begin
                ws = IntervalMDP.construct_workspace(prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(prob)
                Vres_dense = zeros(N, 3, 3, 3)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres_dense,
                    V,
                    prob,
                    stateptr(prob);
                    upper_bound = true,
                )

                ws = IntervalMDP.construct_workspace(sparse_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(sparse_prob)
                Vres = similar(Vres_dense)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    sparse_prob,
                    stateptr(prob);
                    upper_bound = true,
                )
                @test Vres ≈ Vres_dense

                ws = IntervalMDP.SparseOrthogonalWorkspace(sparse_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(sparse_prob)
                Vres = similar(Vres_dense)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    sparse_prob,
                    stateptr(prob);
                    upper_bound = true,
                )
                @test Vres ≈ Vres_dense

                ws = IntervalMDP.ThreadedSparseOrthogonalWorkspace(sparse_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(sparse_prob)
                Vres = similar(Vres_dense)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    sparse_prob,
                    stateptr(prob);
                    upper_bound = true,
                )
                @test Vres ≈ Vres_dense
            end

            #### Minimization
            @testset "minimization" begin
                ws = IntervalMDP.construct_workspace(prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(prob)
                Vres_dense = zeros(N, 3, 3, 3)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres_dense,
                    V,
                    prob,
                    stateptr(prob);
                    upper_bound = false,
                )

                ws = IntervalMDP.construct_workspace(sparse_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(sparse_prob)
                Vres = similar(Vres_dense)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    sparse_prob,
                    stateptr(prob);
                    upper_bound = false,
                )
                @test Vres ≈ Vres_dense

                ws = IntervalMDP.SparseOrthogonalWorkspace(sparse_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(sparse_prob)
                Vres = similar(Vres_dense)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    sparse_prob,
                    stateptr(prob);
                    upper_bound = false,
                )
                @test Vres ≈ Vres_dense

                ws = IntervalMDP.ThreadedSparseOrthogonalWorkspace(sparse_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(sparse_prob)
                Vres = similar(Vres_dense)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    sparse_prob,
                    stateptr(prob);
                    upper_bound = false,
                )
                @test Vres ≈ Vres_dense
            end
        end
    end
end
