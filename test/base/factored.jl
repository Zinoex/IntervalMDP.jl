using Revise, Test
using IntervalMDP
using Random: MersenneTwister

for N in [Float32, Float64]
    @testset "N = $N" begin
        @testset "bellman 1d" begin
            ambiguity_sets = IntervalAmbiguitySets(;
                lower = N[
                    0     5//10 2//10
                    1//10 3//10 3//10
                    2//10 1//10 5//10
                ],
                upper = N[
                    5//10 7//10 3//10
                    6//10 5//10 4//10
                    7//10 3//10 5//10
                ],
            )
            imc = IntervalMarkovChain(ambiguity_sets)

            V = N[1, 2, 3]

            @testset "maximization" begin
                Vexpected = N[27 // 10, 17 // 10, 23 // 10]

                ws = IntervalMDP.construct_workspace(imc, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = zeros(N, 3)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    imc;
                    upper_bound = true,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(imc, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    imc;
                    upper_bound = true,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(imc, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    imc;
                    upper_bound = true,
                )
                @test Vres ≈ Vexpected
            end

            @testset "minimization" begin
                Vexpected = N[17 // 10, 15 // 10, 23 // 10]

                ws = IntervalMDP.construct_workspace(imc, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = zeros(N, 3)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    imc;
                    upper_bound = false,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(imc, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    imc;
                    upper_bound = false,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(imc, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    imc;
                    upper_bound = false,
                )
                @test Vres ≈ Vexpected
            end
        end

        @testset "bellman 2d" begin
            state_indices = (1, 2)
            action_indices = (1,)
            state_vars = (2, 3)
            action_vars = (1,)

            marginal1 = Marginal(IntervalAmbiguitySets(;
                lower = N[
                    1//15  7//30  1//15  13//30  4//15  1//6
                    2//5   7//30  1//30  11//30  2//15  1//10
                ],
                upper = N[
                    17//30   7//10  2//3   4//5  7//10   2//3
                    9//10  13//15  9//10  5//6  4//5   14//15
                ]
            ), state_indices, action_indices, state_vars, action_vars)

            marginal2 = Marginal(IntervalAmbiguitySets(;
                lower = N[
                    1//30  1//3   1//6   1//15  2//5   2//15
                    4//15  1//4   1//6   1//30  2//15  1//30
                    2//15  7//30  1//10  7//30  7//15  1//5
                ],
                upper = N[
                    2//3   7//15   4//5   11//30  19//30   1//2
                    23//30  4//5   23//30   3//5    7//10   8//15
                    7//15  4//5   23//30   7//10   7//15  23//30
                ]
            ), state_indices, action_indices, state_vars, action_vars)

            mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, (marginal1, marginal2))

            V = N[
                 3 13 18
                12 16  8
            ]

            #### Maximization
            @testset "maximization" begin
                V_vertex = N[
                    14.346666666666664  14.263333333333334 11.133333333333336
                    12.341111111111111  13.74333333333333  13.444444444444443
                ]

                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first = zeros(N, 2, 3)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres_first,
                    V,
                    mdp;
                    upper_bound = true,
                )

                epsilon = N == Float32 ? 1e-5 : 1e-8
                @test all(Vres_first .>= 0.0)
                @test all(Vres_first .<= maximum(V))
                @test all(Vres_first .+ epsilon .>= V_vertex)

                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mdp;
                    upper_bound = true,
                )
                @test Vres ≈ Vres_first

                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mdp;
                    upper_bound = true,
                )
                @test Vres ≈ Vres_first
            end

            #### Minimization
            @testset "minimization" begin
                V_vertex = N[
                    9.775555555555554 8.200000000000001 10.844444444444443
                    10.33             10.86             10.027777777777777
                ]

                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first = zeros(N, 2, 3)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres_first,
                    V,
                    mdp;
                    upper_bound = false,
                )

                epsilon = N == Float32 ? 1e-5 : 1e-8
                @test all(Vres_first .>= 0.0)
                @test all(Vres_first .<= maximum(V))
                @test all(Vres_first .- epsilon .<= V_vertex)

                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mdp;
                    upper_bound = false,
                )
                @test Vres ≈ Vres_first

                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mdp;
                    upper_bound = false,
                )
                @test Vres ≈ Vres_first
            end
        end

        # @testset "bellman 2d partial dependence" begin
        #     state_indices = (1, 2)
        #     action_indices = (1,)
        #     state_vars = (2, 3)
        #     action_vars = (1,)

        #     marginal1 = Marginal(IntervalAmbiguitySets(;
        #         lower = N[
        #             1//15  7//30  1//15  13//30  4//15  1//6
        #             2//5   7//30  1//30  11//30  2//15  1//10
        #         ],
        #         upper = N[
        #             17//30   7//10  2//3   4//5  7//10   2//3
        #             9//10  13//15  9//10  5//6  4//5   14//15
        #         ]
        #     ), state_indices, action_indices, state_vars, action_vars)

        #     marginal2 = Marginal(IntervalAmbiguitySets(;
        #         lower = N[
        #             1//30  1//3   1//6
        #             4//15  1//4   1//6
        #             2//15  7//30  1//10
        #         ],
        #         upper = N[
        #             2//3   7//15   4//5
        #             23//30  4//5   23//30
        #             7//15  4//5   23//30
        #         ]
        #     ), (2,), action_indices, (3,), action_vars)

        #     mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, (marginal1, marginal2))

        #     V = N[
        #          3 13 18
        #         12 16  8
        #     ]

        #     #### Maximization
        #     @testset "maximization" begin
        #         V_vertex = N[
        #             14.346666666666664  14.263333333333334 11.133333333333336
        #             12.341111111111111  13.74333333333333  13.444444444444443
        #         ]

        #         ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
        #         strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
        #         Vres_first = zeros(N, 2, 3)
        #         IntervalMDP.bellman!(
        #             ws,
        #             strategy_cache,
        #             Vres_first,
        #             V,
        #             mdp;
        #             upper_bound = true,
        #         )

        #         epsilon = N == Float32 ? 1e-5 : 1e-8
        #         @test all(Vres_first .>= 0.0)
        #         @test all(Vres_first .<= maximum(V))
        #         @test all(Vres_first .+ epsilon .>= V_vertex)

        #         ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
        #         strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
        #         Vres = similar(Vres_first)
        #         IntervalMDP.bellman!(
        #             ws,
        #             strategy_cache,
        #             Vres,
        #             V,
        #             mdp;
        #             upper_bound = true,
        #         )
        #         @test Vres ≈ Vres_first

        #         ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
        #         strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
        #         Vres = similar(Vres_first)
        #         IntervalMDP.bellman!(
        #             ws,
        #             strategy_cache,
        #             Vres,
        #             V,
        #             mdp;
        #             upper_bound = true,
        #         )
        #         @test Vres ≈ Vres_first
        #     end

        #     #### Minimization
        #     @testset "minimization" begin
        #         V_vertex = N[
        #             9.775555555555554 8.200000000000001 10.844444444444443
        #             10.33             10.86             10.027777777777777
        #         ]

        #         ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
        #         strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
        #         Vres_first = zeros(N, 2, 3)
        #         IntervalMDP._bellman_helper!(
        #             ws,
        #             strategy_cache,
        #             Vres_first,
        #             V,
        #             mdp;
        #             upper_bound = false,
        #         )

        #         epsilon = N == Float32 ? 1e-5 : 1e-8
        #         @test all(Vres_first .>= 0.0)
        #         @test all(Vres_first .<= maximum(V))
        #         @test all(Vres_first .- epsilon .<= V_vertex)

        #         ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
        #         strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
        #         Vres = similar(Vres_first)
        #         IntervalMDP.bellman!(
        #             ws,
        #             strategy_cache,
        #             Vres,
        #             V,
        #             mdp;
        #             upper_bound = false,
        #         )
        #         @test Vres ≈ Vres_first

        #         ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
        #         strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
        #         Vres = similar(Vres_first)
        #         IntervalMDP.bellman!(
        #             ws,
        #             strategy_cache,
        #             Vres,
        #             V,
        #             mdp;
        #             upper_bound = false,
        #         )
        #         @test Vres ≈ Vres_first
        #     end
        # end

        @testset "bellman 3d" begin
            state_indices = (1, 2, 3)
            action_indices = (1,)
            state_vars = (3, 3, 3)
            action_vars = (1,)
            marginal1 = Marginal(IntervalAmbiguitySets(;
                lower = N[
                    1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6 1//10 1//30 1//10 1//15 1//10 1//15 4//15 4//15 1//3
                    1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10 1//5 1//5 1//10 1//30 4//15 1//10 1//5 1//6 7//30
                    4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10 3//10 1//10 1//15 1//30 2//15 1//6 1//5 1//10 4//15
                ],
                upper = N[
                    7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30 17//30 13//30 2//5 2//5 2//3 2//5 17//30 2//5 19//30
                    8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30 8//15 7//15 7//15 13//30 8//15 2//5 8//15 17//30 3//5
                    11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15 2//5 2//5 11//30 17//30 17//30 1//2 2//5 19//30 13//30
                ]
            ), state_indices, action_indices, state_vars, action_vars)

            marginal2 = Marginal(IntervalAmbiguitySets(;
                lower = N[
                    1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15 1//6 7//30 1//15 2//15 1//10 1//3 7//30 1//30 7//30
                    3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15 3//10 0 3//10 1//6 3//10 1//5 0 7//30 2//15
                    3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15 1//15 1//3 2//15 1//15 1//5 1//5 1//15 7//30 1//15
                ],
                upper = N[
                    2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30 13//30 17//30 17//30 13//30 11//30 19//30 8//15 2//5 8//15
                    1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3 1//2 8//15 8//15 8//15 8//15 2//5 3//5 2//3 13//30
                    17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3 1//3 2//3 17//30 8//15 17//30 3//5 2//5 19//30 11//30
                ]
            ), state_indices, action_indices, state_vars, action_vars)
            
            marginal3 = Marginal(IntervalAmbiguitySets(;
                lower = N[
                    4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15 7//30 4//15 1//10 1//3 1//5 7//30 1//30 1//5 7//30
                    2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15 1//5 4//15 1//10 1//10 1//3 7//30 3//10 1//3 3//10
                    1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30 0 1//30 1//15 2//15 1//6 7//30 4//15 4//15 7//30
                ],
                upper = N[
                    3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30 2//3 17//30 3//5 7//15 19//30 1//2 3//5 1//3 19//30
                    3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30 2//3 17//30 8//15 13//30 13//30 3//5 1//2 8//15 8//15
                    3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30 0 2//5 11//30 19//30 19//30 2//5 1//2 7//15 7//15
                ]
            ), state_indices, action_indices, state_vars, action_vars)

            mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, (marginal1, marginal2, marginal3))

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
                V_vertex = N[
                    16.19533333333333,
                    15.225999999999996,
                    17.999333333333325,
                    15.795888888888884,
                    17.75407407407408,
                    14.759111111111114,
                    16.94551851851852,
                    15.592148148148148,
                    15.816333333333333,
                    15.059555555555557,
                    16.611333333333334,
                    16.774814814814814,
                    18.133333333333333,
                    17.964999999999996,
                    17.491666666666664,
                    15.506666666666668,
                    16.986962962962956,
                    14.952518518518515,
                    18.215555555555554,
                    16.101592592592596,
                    17.483888888888895,
                    17.05688888888889,
                    16.394444444444442,
                    16.340666666666667,
                    16.880444444444446,
                    16.045185185185186,
                    16.494074074074074,
                ]
                V_vertex = reshape(V_vertex, (3, 3, 3))

                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first = zeros(N, 3, 3, 3)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres_first,
                    V,
                    mdp;
                    upper_bound = true,
                )

                epsilon = N == Float32 ? 1e-5 : 1e-8
                @test all(Vres_first .>= 0.0)
                @test all(Vres_first .<= maximum(V))
                @test all(Vres_first .+ epsilon .>= V_vertex)

                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mdp;
                    upper_bound = true,
                )
                @test Vres ≈ Vres_first

                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mdp;
                    upper_bound = true,
                )
                @test Vres ≈ Vres_first
            end

            #### Minimization
            @testset "minimization" begin
                V_vertex = N[
                    10.856370370370371,
                    11.341333333333333,
                    12.446333333333333,
                    11.282999999999996,
                    12.351592592592592,
                    12.094370370370369,
                    9.957037037037036,
                    9.98859259259259,
                    12.769888888888888,
                    12.433333333333332,
                    12.526444444444442,
                    11.579851851851851,
                    13.808888888888886,
                    11.394074074074076,
                    13.171555555555557,
                    12.11111111111111,
                    12.080148148148147,
                    11.569777777777778,
                    13.288888888888888,
                    11.581629629629631,
                    11.545259259259257,
                    10.001851851851853,
                    11.602074074074073,
                    12.530444444444445,
                    12.070666666666666,
                    10.503851851851852,
                    12.760740740740742,
                ]
                V_vertex = reshape(V_vertex, (3, 3, 3))

                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first = zeros(N, 3, 3, 3)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres_first,
                    V,
                    mdp;
                    upper_bound = false,
                )

                epsilon = N == Float32 ? 1e-5 : 1e-8
                @test all(Vres_first .>= 0.0)
                @test all(Vres_first .<= maximum(V))
                @test all(Vres_first .- epsilon .<= V_vertex)

                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mdp;
                    upper_bound = false,
                )
                @test Vres ≈ Vres_first

                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first)
                IntervalMDP.bellman!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mdp;
                    upper_bound = false,
                )
                @test Vres ≈ Vres_first
            end
        end

        @testset "implicit sink state" begin
            @testset "first dimension" begin
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = (3, 3, 3)
                source_dims = (2, 3, 3)
                action_vars = (1,)

                # Explicit
                marginal1 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//15 3//10 0 1//15 3//10 0 1//30 1//3 0 7//30 4//15 0 1//6 1//5 0 1//10 1//5 0 0 7//30 0 7//30 1//5 0 2//15 1//6 0
                        1//5 4//15 0 1//10 1//5 0 3//10 3//10 0 1//10 1//15 0 3//10 3//10 0 7//30 1//5 0 1//10 1//5 0 1//5 1//30 0 1//5 3//10 0
                        4//15 1//30 1 1//5 1//5 1 7//30 4//15 1 2//15 7//30 1 1//5 1//3 1 2//15 1//6 1 1//6 1//3 1 4//15 3//10 1 1//30 3//10 1
                    ],
                    upper = N[
                        7//15 17//30 0 13//30 3//5 0 17//30 17//30 0 17//30 13//30 0 3//5 2//3 0 11//30 7//15 0 0 1//2 0 17//30 13//30 0 7//15 13//30 0
                        8//15 1//2 0 3//5 7//15 0 8//15 17//30 0 2//3 17//30 0 11//30 7//15 0 19//30 19//30 0 13//15 1//2 0 17//30 13//30 0 3//5 11//30 0
                        11//30 1//3 1 2//5 8//15 1 7//15 3//5 1 2//3 17//30 1 2//3 8//15 1 2//15 3//5 1 2//3 3//5 1 17//30 2//3 1 7//15 8//15 1
                    ]
                ), state_indices, action_indices, state_vars, action_vars)

                marginal2 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 1 3//10 0 0 1//6 1//15 0 1//15 1//6 1 1//6 1//30 0 1//10 1//10 0 1//3 2//15 1 3//10 4//15 0 2//15 2//15 0
                        3//10 1//5 0 3//10 2//15 1 0 1//30 0 0 1//15 0 1//30 7//30 1 1//30 1//15 0 7//30 1//15 0 1//6 1//30 1 1//10 1//15 0
                        3//10 4//15 0 1//10 3//10 0 2//15 1//3 1 3//10 1//10 0 1//6 3//10 0 7//30 1//6 1 1//15 1//15 0 1//10 1//5 0 1//5 4//15 1
                    ],
                    upper = N[
                        2//5 17//30 1 3//5 11//30 0 3//5 7//15 0 19//30 2//5 1 3//5 2//3 0 2//3 8//15 0 8//15 19//30 1 8//15 8//15 0 13//30 13//30 0
                        1//3 13//30 0 11//30 2//5 1 2//3 2//3 0 0 13//30 0 1//2 17//30 1 17//30 1//3 0 2//5 1//3 0 13//30 11//30 1 8//15 1//3 0
                        17//30 3//5 0 8//15 1//2 0 7//15 1//2 1 2//3 17//30 0 11//30 2//5 0 1//2 7//15 1 2//5 17//30 0 11//30 2//5 0 11//30 2//3 1
                    ]
                ), state_indices, action_indices, state_vars, action_vars)

                marginal3 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 1 3//10 3//10 1 4//15 7//30 1 1//5 4//15 0 7//30 1//6 0 1//5 0 0 1//15 1//30 0 3//10 1//3 0 2//15 1//15 0
                        2//15 4//15 0 1//10 1//30 0 7//30 2//15 0 1//15 1//30 1 3//10 1//3 1 1//5 1//10 1 2//15 1//30 0 2//15 4//15 0 0 4//15 0
                        1//5 1//3 0 3//10 1//10 0 1//15 1//10 0 1//30 1//5 0 2//15 7//30 0 1//3 2//15 0 1//10 1//6 1 3//10 1//5 1 7//30 1//30 1
                    ],
                    upper = N[
                        3//5 17//30 1 1//2 3//5 1 19//30 2//5 1 8//15 1//3 0 11//30 2//5 0 17//30 13//30 0 2//5 3//5 0 3//5 11//30 0 1//2 11//30 0
                        3//5 2//3 0 13//30 19//30 0 1//3 2//5 0 17//30 7//15 1 11//30 3//5 1 19//30 7//15 1 2//5 8//15 0 17//30 11//30 0 19//30 13//30 0
                        3//5 2//3 0 1//2 1//2 0 2//3 7//15 0 3//5 3//5 0 1//2 1//3 0 2//5 8//15 0 2//5 11//30 1 1//3 8//15 1 7//15 13//30 1
                    ]
                ), state_indices, action_indices, state_vars, action_vars)

                mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, (marginal1, marginal2, marginal3))

                # Implicit
                marginal1 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6
                        1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10
                        4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10
                    ],
                    upper = N[
                        7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30
                        8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30
                        11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15
                    ]
                ), state_indices, action_indices, source_dims, action_vars)

                marginal2 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15
                        3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15
                        3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15
                    ],
                    upper = N[
                        2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30
                        1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3
                        17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3
                    ]
                ), state_indices, action_indices, source_dims, action_vars)
                
                marginal3 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15
                        2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15
                        1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30
                    ],
                    upper = N[
                        3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30
                        3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30
                        3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30
                    ]
                ), state_indices, action_indices, source_dims, action_vars)

                implicit_mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, source_dims, (marginal1, marginal2, marginal3))

                prop = FiniteTimeSafety([(3, i, j) for i in 1:3 for j in 1:3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                implicit_prob = VerificationProblem(implicit_mdp, spec)

                V, k, res = solve(prob)
                V_implicit, k_implicit, res_implicit = solve(implicit_prob)

                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end

            @testset "second dimension" begin
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = (3, 3, 3)
                source_dims = (3, 2, 3)
                action_vars = (1,)

                # Explicit
                marginal1 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//15 3//10 1//15 3//10 1//30 1//3 1 0 0 7//30 4//15 1//6 1//5 1//10 1//5 1 0 0 0 7//30 7//30 1//5 2//15 1//6 1 0 0
                        1//5 4//15 1//10 1//5 3//10 3//10 0 1 0 1//10 1//15 3//10 3//10 7//30 1//5 0 1 0 1//10 1//5 1//5 1//30 1//5 3//10 0 1 0
                        4//15 1//30 1//5 1//5 7//30 4//15 0 0 1 2//15 7//30 1//5 1//3 2//15 1//6 0 0 1 1//6 1//3 4//15 3//10 1//30 3//10 0 0 1
                    ],
                    upper = N[
                        7//15 17//30 13//30 3//5 17//30 17//30 1 0 0 17//30 13//30 3//5 2//3 11//30 7//15 1 0 0 0 1//2 17//30 13//30 7//15 13//30 1 0 0
                        8//15 1//2 3//5 7//15 8//15 17//30 0 1 0 2//3 17//30 11//30 7//15 19//30 19//30 0 1 0 13//15 1//2 17//30 13//30 3//5 11//30 0 1 0
                        11//30 1//3 2//5 8//15 7//15 3//5 0 0 1 2//3 17//30 2//3 8//15 2//15 3//5 0 0 1 2//3 3//5 17//30 2//3 7//15 8//15 0 0 1
                    ]
                ), state_indices, action_indices, state_vars, action_vars)
                
                marginal2 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 3//10 0 1//6 1//15 0 0 0 1//15 1//6 1//6 1//30 1//10 1//10 0 0 0 1//3 2//15 3//10 4//15 2//15 2//15 0 0 0
                        3//10 1//5 3//10 2//15 0 1//30 0 0 0 0 1//15 1//30 7//30 1//30 1//15 0 0 0 7//30 1//15 1//6 1//30 1//10 1//15 0 0 0
                        3//10 4//15 1//10 3//10 2//15 1//3 1 1 1 3//10 1//10 1//6 3//10 7//30 1//6 1 1 1 1//15 1//15 1//10 1//5 1//5 4//15 1 1 1
                    ],
                    upper = N[
                        2//5 17//30 3//5 11//30 3//5 7//15 0 0 0 19//30 2//5 3//5 2//3 2//3 8//15 0 0 0 8//15 19//30 8//15 8//15 13//30 13//30 0 0 0
                        1//3 13//30 11//30 2//5 2//3 2//3 0 0 0 0 13//30 1//2 17//30 17//30 1//3 0 0 0 2//5 1//3 13//30 11//30 8//15 1//3 0 0 0
                        17//30 3//5 8//15 1//2 7//15 1//2 1 1 1 2//3 17//30 11//30 2//5 1//2 7//15 1 1 1 2//5 17//30 11//30 2//5 11//30 2//3 1 1 1
                    ]
                ), state_indices, action_indices, state_vars, action_vars)
                
                marginal3 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1 1 1 1//5 4//15 7//30 1//6 1//5 0 0 0 0 1//15 1//30 3//10 1//3 2//15 1//15 0 0 0
                        2//15 4//15 1//10 1//30 7//30 2//15 0 0 0 1//15 1//30 3//10 1//3 1//5 1//10 1 1 1 2//15 1//30 2//15 4//15 0 4//15 0 0 0
                        1//5 1//3 3//10 1//10 1//15 1//10 0 0 0 1//30 1//5 2//15 7//30 1//3 2//15 0 0 0 1//10 1//6 3//10 1//5 7//30 1//30 1 1 1
                    ],
                    upper = N[
                        3//5 17//30 1//2 3//5 19//30 2//5 1 1 1 8//15 1//3 11//30 2//5 17//30 13//30 0 0 0 2//5 3//5 3//5 11//30 1//2 11//30 0 0 0
                        3//5 2//3 13//30 19//30 1//3 2//5 0 0 0 17//30 7//15 11//30 3//5 19//30 7//15 1 1 1 2//5 8//15 17//30 11//30 19//30 13//30 0 0 0
                        3//5 2//3 1//2 1//2 2//3 7//15 0 0 0 3//5 3//5 1//2 1//3 2//5 8//15 0 0 0 2//5 11//30 1//3 8//15 7//15 13//30 1 1 1
                    ]
                ), state_indices, action_indices, state_vars, action_vars)

                mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, (marginal1, marginal2, marginal3))

                # Implicit
                marginal1 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6
                        1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10
                        4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10
                    ],
                    upper = N[
                        7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30
                        8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30
                        11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15
                    ]   
                ), state_indices, action_indices, source_dims, action_vars)

                marginal2 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15
                        3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15
                        3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15
                    ],
                    upper = N[
                        2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30
                        1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3
                        17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3
                    ]
                ), state_indices, action_indices, source_dims, action_vars)

                marginal3 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15
                        2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15
                        1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30
                    ],
                    upper = N[
                        3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30
                        3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30
                        3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30
                    ]
                ), state_indices, action_indices, source_dims, action_vars)

                implicit_mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, source_dims, (marginal1, marginal2, marginal3))

                prop = FiniteTimeSafety([(i, 3, j) for i in 1:3 for j in 1:3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                implicit_prob = VerificationProblem(implicit_mdp, spec)

                V, k, res = solve(prob)
                V_implicit, k_implicit, res_implicit = solve(implicit_prob)

                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end

            @testset "last dimension" begin
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = (3, 3, 3)
                source_dims = (3, 3, 2)
                action_vars = (1,)

                # Explicit
                marginal1 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6 1 0 0 1 0 0 1 0 0
                        1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10 0 1 0 0 1 0 0 1 0
                        4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10 0 0 1 0 0 1 0 0 1
                    ],
                    upper = N[
                        7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30 1 0 0 1 0 0 1 0 0
                        8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30 0 1 0 0 1 0 0 1 0
                        11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15 0 0 1 0 0 1 0 0 1
                    ]
                ), state_indices, action_indices, state_vars, action_vars)

                marginal2 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15 1 1 1 0 0 0 0 0 0
                        3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15 0 0 0 1 1 1 0 0 0
                        3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15 0 0 0 0 0 0 1 1 1
                    ],
                    upper = N[
                        2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30 1 1 1 0 0 0 0 0 0
                        1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3 0 0 0 1 1 1 0 0 0
                        17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3 0 0 0 0 0 0 1 1 1
                    ]
                ), state_indices, action_indices, state_vars, action_vars)

                marginal3 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15 0 0 0 0 0 0 0 0 0
                        2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15 0 0 0 0 0 0 0 0 0
                        1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30 1 1 1 1 1 1 1 1 1
                    ],
                    upper = N[
                        3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30 0 0 0 0 0 0 0 0 0
                        3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30 0 0 0 0 0 0 0 0 0
                        3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30 1 1 1 1 1 1 1 1 1
                    ]
                ), state_indices, action_indices, state_vars, action_vars)

                mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, (marginal1, marginal2, marginal3))

                # Implicit
                marginal1 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6
                        1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10
                        4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10
                    ],
                    upper = N[
                        7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30
                        8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30
                        11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15
                    ]
                ), state_indices, action_indices, source_dims, action_vars)

                marginal2 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15
                        3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15
                        3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15
                    ],
                    upper = N[
                        2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30
                        1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3
                        17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3
                    ]
                ), state_indices, action_indices, source_dims, action_vars)

                marginal3 = Marginal(IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15
                        2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15
                        1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30
                    ],
                    upper = N[
                        3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30
                        3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30
                        3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30
                    ]
                ), state_indices, action_indices, source_dims, action_vars)

                implicit_mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, source_dims, (marginal1, marginal2, marginal3))

                prop = FiniteTimeSafety([(i, j, 3) for i in 1:3 for j in 1:3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                implicit_prob = VerificationProblem(implicit_mdp, spec)

                V, k, res = solve(prob)
                V_implicit, k_implicit, res_implicit = solve(implicit_prob)

                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end
        end
    end
end

for N in [Float32, Float64]
    @testset "N = $N" begin
        # 4-D abstraction
        @testset "4D abstraction" begin
            rng = MersenneTwister(995)

            prob_lower = [rand(rng, N, 3, 81) ./ N(3) for _ in 1:4]
            prob_upper = [(rand(rng, N, 3, 81) .+ N(1)) ./ N(3) for _ in 1:4]

            ambiguity_sets = ntuple(
                i -> IntervalAmbiguitySets(;
                    lower = prob_lower[i],
                    upper = prob_upper[i],
                ),
                4,
            )

            marginals = ntuple(
                i -> Marginal(ambiguity_sets[i], (1, 2, 3, 4), (1,), (3, 3, 3, 3), (1,)),
                4,
            )

            mdp = FactoredRobustMarkovDecisionProcess((3, 3, 3, 3), (1,), marginals)

            prop = FiniteTimeReachability([(3, 3, 3, 3)], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            prob = VerificationProblem(mdp, spec)

            V_ortho, it_ortho, res_ortho = solve(prob)

            @test V_ortho[3, 3, 3, 3] ≈ one(N)
            @test all(V_ortho .>= zero(N))
            @test all(V_ortho .<= one(N))

            # Test against the naive construction
            prob_lower_simple = zeros(N, 81, 81)
            prob_upper_simple = zeros(N, 81, 81)

            lin = LinearIndices((3, 3, 3, 3))
            act_idx = CartesianIndex(1)
            for I in CartesianIndices((3, 3, 3, 3))
                for J in CartesianIndices((3, 3, 3, 3))
                    marginal_ambiguity_sets = map(marginal -> marginal[act_idx, I], marginals)

                    prob_lower_simple[lin[J], lin[I]] = prod(
                        lower(marginal_ambiguity_sets[i], J[i]) for i in 1:4
                    )

                    prob_upper_simple[lin[J], lin[I]] = prod(
                        upper(marginal_ambiguity_sets[i], J[i]) for i in 1:4
                    )
                end
            end

            ambiguity_set = IntervalAmbiguitySets(;
                lower = prob_lower_simple,
                upper = prob_upper_simple,
            )

            imc = IntervalMarkovChain(ambiguity_set)

            prop = FiniteTimeReachability([81], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            prob = VerificationProblem(imc, spec)

            V_direct, it_direct, res_direct = solve(prob)
            @test V_direct[81] ≈ one(N)
            @test all(V_ortho .≥ reshape(V_direct, 3, 3, 3, 3))
        end

        @testset "synthesis" begin
            rng = MersenneTwister(3286)

            num_states_per_axis = 3
            num_axis = 3
            num_states = num_states_per_axis^num_axis
            num_actions = 2
            num_choices = num_states * num_actions
            state_indices = (1, 2, 3)
            action_indices = (1,)
            state_vars = ntuple(_ -> num_states_per_axis, num_axis)
            action_vars = (num_actions,)

            prob_lower = [
                rand(rng, N, num_states_per_axis, num_choices) ./ num_states_per_axis
                for _ in 1:num_axis
            ]
            prob_upper = [
                (rand(rng, N, num_states_per_axis, num_choices) .+ N(1)) ./
                num_states_per_axis for _ in 1:num_axis
            ]

            ambiguity_sets = ntuple(
                i -> IntervalAmbiguitySets(;
                    lower = prob_lower[i],
                    upper = prob_upper[i],
                ),
                num_axis,
            )

            marginals = ntuple(
                i -> Marginal(ambiguity_sets[i], state_indices, action_indices, state_vars, action_vars),
                num_axis,
            )

            mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)

            prop = FiniteTimeReachability(
                [(num_states_per_axis, num_states_per_axis, num_states_per_axis)],
                10,
            )
            spec = Specification(prop, Pessimistic, Maximize)
            prob = ControlSynthesisProblem(mdp, spec)

            policy, V, it, res = solve(prob)
            @test it == 10
            @test all(V .≥ 0.0)

            # Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
            prob = VerificationProblem(mdp, spec, policy)
            V_mc, k, res = solve(prob)
            @test V ≈ V_mc
        end
    end
end

# @testset "Orthogonal abstraction" begin
#     using LazySets
#     using SpecialFunctions: erf
#     using LinearAlgebra: I

#     function IMDP_orthogonal_abstraction()
#         A = 0.9 * I
#         B = 0.7 * I
#         sigma = 2.0

#         X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
#         X1 = Interval(-10.0, 10.0)
#         X2 = Interval(-10.0, 10.0)
#         U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])

#         reach_region = Hyperrectangle(; low = [4.0, -6.0], high = [10.0, -2.0])

#         l = [5, 5]
#         X1_split = split(X1, l[1])
#         X2_split = split(X2, l[2])

#         X_split = Matrix{LazySet}(undef, l[1], l[2])
#         for j in 1:l[2]
#             for i in 1:l[1]
#                 x1 = X1_split[i]
#                 x2 = X2_split[j]
#                 X_split[i, j] = Hyperrectangle(
#                     [center(x1)[1], center(x2)[1]],
#                     [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]],
#                 )
#             end
#         end

#         U_split = split(U, [3, 3])

#         transition_prob(x, v_lower, v_upper) =
#             0.5 *
#             erf((x - v_upper) / (sigma * sqrt(2.0)), (x - v_lower) / (sigma * sqrt(2.0)))

#         probs1 = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
#         probs2 = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
#         stateptr = Int32[1]

#         for source2 in 1:(l[2] + 1)
#             for source1 in 1:(l[1] + 1)
#                 if source1 == 1 || source2 == 1
#                     probs1_lower = zeros(l[1] + 1, 1)
#                     probs1_upper = zeros(l[1] + 1, 1)

#                     probs1_upper[source1, 1] = 1
#                     probs1_lower[source1, 1] = 1

#                     probs2_lower = zeros(l[2] + 1, 1)
#                     probs2_upper = zeros(l[2] + 1, 1)

#                     probs2_upper[source2, 1] = 1
#                     probs2_lower[source2, 1] = 1

#                     push!(
#                         probs1,
#                         IntervalProbabilities(; lower = probs1_lower, upper = probs1_upper),
#                     )
#                     push!(
#                         probs2,
#                         IntervalProbabilities(; lower = probs2_lower, upper = probs2_upper),
#                     )
#                 else
#                     Xij = X_split[source1 - 1, source2 - 1]

#                     for u in U_split
#                         Xij_u = A * Xij + B * u
#                         Xij_u = box_approximation(Xij_u)

#                         probs1_lower = zeros(l[1] + 1, 1)
#                         probs1_upper = zeros(l[1] + 1, 1)

#                         for target1 in 1:(l[1] + 1)
#                             if target1 == 1
#                                 probs1_upper[target1, 1] =
#                                     max(
#                                         1 - transition_prob(
#                                             low(Xij_u)[1],
#                                             low(X)[1],
#                                             high(X)[1],
#                                         ),
#                                         1 - transition_prob(
#                                             high(Xij_u)[1],
#                                             low(X)[1],
#                                             high(X)[1],
#                                         ),
#                                     ) + eps(Float64)
#                                 probs1_lower[target1, 1] = min(
#                                     1 - transition_prob(
#                                         center(Xij_u)[1],
#                                         low(X)[1],
#                                         high(X)[1],
#                                     ),
#                                     1 - transition_prob(
#                                         low(Xij_u)[1],
#                                         low(X)[1],
#                                         high(X)[1],
#                                     ),
#                                     1 - transition_prob(
#                                         high(Xij_u)[1],
#                                         low(X)[1],
#                                         high(X)[1],
#                                     ),
#                                 )
#                             else
#                                 probs1_upper[target1, 1] = max(
#                                     transition_prob(
#                                         center(Xij_u)[1],
#                                         low(X1_split[target1 - 1])[1],
#                                         high(X1_split[target1 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         low(Xij_u)[1],
#                                         low(X1_split[target1 - 1])[1],
#                                         high(X1_split[target1 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         high(Xij_u)[1],
#                                         low(X1_split[target1 - 1])[1],
#                                         high(X1_split[target1 - 1])[1],
#                                     ),
#                                 )
#                                 probs1_lower[target1, 1] = min(
#                                     transition_prob(
#                                         low(Xij_u)[1],
#                                         low(X1_split[target1 - 1])[1],
#                                         high(X1_split[target1 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         high(Xij_u)[1],
#                                         low(X1_split[target1 - 1])[1],
#                                         high(X1_split[target1 - 1])[1],
#                                     ),
#                                 )
#                             end
#                         end

#                         probs2_lower = zeros(l[2] + 1, 1)
#                         probs2_upper = zeros(l[2] + 1, 1)

#                         for target2 in 1:(l[2] + 1)
#                             if target2 == 1
#                                 probs2_upper[target2, 1] =
#                                     max(
#                                         1 - transition_prob(
#                                             low(Xij_u)[2],
#                                             low(X)[2],
#                                             high(X)[2],
#                                         ),
#                                         1 - transition_prob(
#                                             high(Xij_u)[2],
#                                             low(X)[2],
#                                             high(X)[2],
#                                         ),
#                                     ) + eps(Float64)
#                                 probs2_lower[target2, 1] = min(
#                                     1 - transition_prob(
#                                         center(Xij_u)[2],
#                                         low(X)[2],
#                                         high(X)[2],
#                                     ),
#                                     1 - transition_prob(
#                                         low(Xij_u)[2],
#                                         low(X)[2],
#                                         high(X)[2],
#                                     ),
#                                     1 - transition_prob(
#                                         high(Xij_u)[2],
#                                         low(X)[2],
#                                         high(X)[2],
#                                     ),
#                                 )
#                             else
#                                 probs2_upper[target2, 1] = max(
#                                     transition_prob(
#                                         center(Xij_u)[2],
#                                         low(X2_split[target2 - 1])[1],
#                                         high(X2_split[target2 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         low(Xij_u)[2],
#                                         low(X2_split[target2 - 1])[1],
#                                         high(X2_split[target2 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         high(Xij_u)[2],
#                                         low(X2_split[target2 - 1])[1],
#                                         high(X2_split[target2 - 1])[1],
#                                     ),
#                                 )
#                                 probs2_lower[target2, 1] = min(
#                                     transition_prob(
#                                         low(Xij_u)[2],
#                                         low(X2_split[target2 - 1])[1],
#                                         high(X2_split[target2 - 1])[1],
#                                     ),
#                                     transition_prob(
#                                         high(Xij_u)[2],
#                                         low(X2_split[target2 - 1])[1],
#                                         high(X2_split[target2 - 1])[1],
#                                     ),
#                                 )
#                             end
#                         end

#                         push!(
#                             probs1,
#                             IntervalProbabilities(;
#                                 lower = probs1_lower,
#                                 upper = probs1_upper,
#                             ),
#                         )
#                         push!(
#                             probs2,
#                             IntervalProbabilities(;
#                                 lower = probs2_lower,
#                                 upper = probs2_upper,
#                             ),
#                         )
#                     end
#                 end

#                 push!(stateptr, length(probs1) + 1)
#             end
#         end

#         probs1, _ = IntervalMDP.interval_prob_hcat(probs1)
#         probs2, _ = IntervalMDP.interval_prob_hcat(probs2)
#         probs = OrthogonalIntervalProbabilities(
#             (probs1, probs2),
#             (Int32(l[1] + 1), Int32(l[2] + 1)),
#         )
#         pmdp = OrthogonalIntervalMarkovDecisionProcess(probs, stateptr)

#         reach = Tuple{Int32, Int32}[]
#         avoid = Tuple{Int32, Int32}[]

#         for j in 1:(l[2] + 1)
#             for i in 1:(l[1] + 1)
#                 if j == 1 || i == 1
#                     push!(avoid, (i, j))
#                 elseif X_split[i - 1, j - 1] ⊆ reach_region
#                     push!(reach, (i, j))
#                 end
#             end
#         end

#         return pmdp, reach, avoid
#     end

#     function IMDP_direct_abstraction()
#         A = 0.9I(2)
#         B = 0.7I(2)
#         sigma = 2.0

#         X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
#         X1 = Interval(-10.0, 10.0)
#         X2 = Interval(-10.0, 10.0)
#         U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])

#         reach_region = Hyperrectangle(; low = [4.0, -6.0], high = [10.0, -2.0])

#         l = [5, 5]
#         X1_split = split(X1, l[1])
#         X2_split = split(X2, l[2])

#         X_split = Matrix{LazySet}(undef, l[1] + 1, l[2] + 1)
#         for j in 1:(l[2] + 1)
#             for i in 1:(l[1] + 1)
#                 if i == 1 && j == 1
#                     X_split[i, j] = CartesianProduct(
#                         Complement(Interval(low(X, 1), high(X, 1))),
#                         Complement(Interval(low(X, 2), high(X, 2))),
#                     )
#                 elseif i == 1
#                     x2 = X2_split[j - 1]
#                     X_split[i, j] = CartesianProduct(
#                         Complement(Interval(low(X, 1), high(X, 1))),
#                         Interval(low(x2, 1), high(x2, 1)),
#                     )
#                 elseif j == 1
#                     x1 = X1_split[i - 1]
#                     X_split[i, j] = CartesianProduct(
#                         Interval(low(x1, 1), high(x1, 1)),
#                         Complement(Interval(low(X, 2), high(X, 2))),
#                     )
#                 else
#                     x1 = X1_split[i - 1]
#                     x2 = X2_split[j - 1]
#                     X_split[i, j] = Hyperrectangle(
#                         [center(x1)[1], center(x2)[1]],
#                         [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]],
#                     )
#                 end
#             end
#         end

#         U_split = split(U, [3, 3])

#         transition_prob(x, v_lower, v_upper) =
#             0.5 *
#             erf((x - v_upper) / (sigma * sqrt(2.0)), (x - v_lower) / (sigma * sqrt(2.0)))

#         probs = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
#         for source2 in 1:(l[2] + 1)
#             for source1 in 1:(l[1] + 1)
#                 source = (source2 - 1) * (l[1] + 1) + source1

#                 probs_lower = Vector{Float64}[]
#                 probs_upper = Vector{Float64}[]

#                 if source1 == 1 || source2 == 1
#                     prob_upper = zeros(prod(l .+ 1))
#                     prob_lower = zeros(prod(l .+ 1))

#                     prob_upper[source] = 1
#                     prob_lower[source] = 1

#                     push!(probs_lower, prob_lower)
#                     push!(probs_upper, prob_upper)
#                 else
#                     Xij = X_split[source1, source2]

#                     for u in U_split
#                         Xij_u = A * Xij + B * u
#                         box_Xij_u = box_approximation(Xij_u)

#                         prob_upper = zeros(prod(l .+ 1))
#                         prob_lower = zeros(prod(l .+ 1))

#                         for target2 in 1:(l[2] + 1)
#                             for target1 in 1:(l[1] + 1)
#                                 Xij_target = X_split[target1, target2]
#                                 target = (target2 - 1) * (l[1] + 1) + target1

#                                 if target1 == 1 && target2 == 1
#                                     prob_upper[target] =
#                                         max(
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                         ) * max(
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                         )
#                                     prob_lower[target] =
#                                         min(
#                                             1 - transition_prob(
#                                                 center(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                         ) * min(
#                                             1 - transition_prob(
#                                                 center(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                         )
#                                 elseif target1 == 1
#                                     prob_upper[target] =
#                                         max(
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                         ) * max(
#                                             transition_prob(
#                                                 center(box_Xij_u)[2],
#                                                 low(Xij_target.Y)[1],
#                                                 high(Xij_target.Y)[1],
#                                             ),
#                                             transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(Xij_target.Y)[1],
#                                                 high(Xij_target.Y)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(Xij_target.Y)[1],
#                                                 high(Xij_target.Y)[1],
#                                             ),
#                                         )
#                                     prob_lower[target] =
#                                         min(
#                                             1 - transition_prob(
#                                                 center(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                         ) * min(
#                                             transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(Xij_target.Y)[1],
#                                                 high(Xij_target.Y)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(Xij_target.Y)[1],
#                                                 high(Xij_target.Y)[1],
#                                             ),
#                                         )
#                                 elseif target2 == 1
#                                     prob_upper[target] =
#                                         max(
#                                             transition_prob(
#                                                 center(box_Xij_u)[1],
#                                                 low(Xij_target.X)[1],
#                                                 high(Xij_target.X)[1],
#                                             ),
#                                             transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(Xij_target.X)[1],
#                                                 high(Xij_target.X)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(Xij_target.X)[1],
#                                                 high(Xij_target.X)[1],
#                                             ),
#                                         ) * max(
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(X)[2],
#                                                 high(X)[2],
#                                             ),
#                                         )
#                                     prob_lower[target] =
#                                         min(
#                                             transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(Xij_target.X)[1],
#                                                 high(Xij_target.X)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(Xij_target.X)[1],
#                                                 high(Xij_target.X)[1],
#                                             ),
#                                         ) * min(
#                                             1 - transition_prob(
#                                                 center(box_Xij_u)[2],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                             1 - transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(X)[1],
#                                                 high(X)[1],
#                                             ),
#                                         )
#                                 else
#                                     prob_upper[target] =
#                                         max(
#                                             transition_prob(
#                                                 center(box_Xij_u)[1],
#                                                 low(Xij_target)[1],
#                                                 high(Xij_target)[1],
#                                             ),
#                                             transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(Xij_target)[1],
#                                                 high(Xij_target)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(Xij_target)[1],
#                                                 high(Xij_target)[1],
#                                             ),
#                                         ) * max(
#                                             transition_prob(
#                                                 center(box_Xij_u)[2],
#                                                 low(Xij_target)[2],
#                                                 high(Xij_target)[2],
#                                             ),
#                                             transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(Xij_target)[2],
#                                                 high(Xij_target)[2],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(Xij_target)[2],
#                                                 high(Xij_target)[2],
#                                             ),
#                                         )
#                                     prob_lower[target] =
#                                         min(
#                                             transition_prob(
#                                                 low(box_Xij_u)[1],
#                                                 low(Xij_target)[1],
#                                                 high(Xij_target)[1],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[1],
#                                                 low(Xij_target)[1],
#                                                 high(Xij_target)[1],
#                                             ),
#                                         ) * min(
#                                             transition_prob(
#                                                 low(box_Xij_u)[2],
#                                                 low(Xij_target)[2],
#                                                 high(Xij_target)[2],
#                                             ),
#                                             transition_prob(
#                                                 high(box_Xij_u)[2],
#                                                 low(Xij_target)[2],
#                                                 high(Xij_target)[2],
#                                             ),
#                                         )
#                                 end
#                             end
#                         end

#                         push!(probs_lower, prob_lower)
#                         push!(probs_upper, prob_upper)
#                     end
#                 end

#                 prob = IntervalProbabilities(;
#                     lower = reduce(hcat, probs_lower),
#                     upper = reduce(hcat, probs_upper),
#                 )
#                 push!(probs, prob)
#             end
#         end
#         mdp = IntervalMarkovDecisionProcess(probs)

#         reach = Int32[]
#         avoid = Int32[]

#         for source2 in 1:(l[2] + 1)
#             for source1 in 1:(l[1] + 1)
#                 Xij = X_split[source1, source2]
#                 source = (source2 - 1) * (l[1] + 1) + source1

#                 if source1 == 1 || source2 == 1
#                     push!(avoid, source)
#                 elseif Xij ⊆ reach_region
#                     push!(reach, source)
#                 end
#             end
#         end

#         return mdp, reach, avoid
#     end

#     # Orthogonal abstraction
#     pmdp, reach_set, avoid_set = IMDP_orthogonal_abstraction()

#     prop = FiniteTimeReachAvoid(reach_set, avoid_set, 10)
#     spec = Specification(prop, Pessimistic, Maximize)
#     prob_ortho = VerificationProblem(pmdp, spec)

#     V_ortho, it_ortho, res_ortho = solve(prob_ortho)
#     @test all(V_ortho .≥ 0.0)

#     # Direct abstraction
#     mdp, reach_set, avoid_set = IMDP_direct_abstraction()

#     prop = FiniteTimeReachAvoid(reach_set, avoid_set, 10)
#     spec = Specification(prop, Pessimistic, Maximize)
#     prob_direct = VerificationProblem(mdp, spec)

#     V_direct, it_direct, res_direct = solve(prob_direct)

#     @test it_ortho == it_direct
#     @test all(V_ortho .≥ reshape(V_direct, 6, 6))
# end
