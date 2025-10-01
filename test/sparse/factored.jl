using Revise, Test
using IntervalMDP, SparseArrays
using Random: MersenneTwister

@testset for N in [Float32, Float64]
    @testset "bellman 1d" begin
        ambiguity_sets = IntervalAmbiguitySets(;
            lower = sparse(N[
                0 5//10 2//10
                1//10 0 3//10
                2//10 1//10 5//10
            ]),
            upper = sparse(N[
                0 7//10 3//10
                6//10 5//10 4//10
                7//10 3//10 5//10
            ]),
        )
        imc = IntervalMarkovChain(ambiguity_sets)

        V = N[1, 2, 3]

        @testset "vertices" begin
            verts = IntervalMDP.vertices(ambiguity_sets[1])
            @test length(verts) <= 2  # = number of permutations of 2 elements

            expected_verts = N[
                0 6//10 4//10
                0 3//10 7//10
            ]
            @test length(verts) ≥ size(expected_verts, 1)  # at least the unique ones
            @test all(any(v2 -> v1 ≈ v2, verts) for v1 in eachrow(expected_verts))

            verts = IntervalMDP.vertices(ambiguity_sets[2])
            @test length(verts) <= 6  # = number of permutations of 3 elements 

            expected_verts = N[  # duplicates due to budget < gap for all elements
                7 // 10 2//10 1//10
                7//10 0 3//10
                5//10 4//10 1//10
                7//10 0//10 3//10
                5//10 2//10 3//10
            ]
            @test length(verts) ≥ size(expected_verts, 1)  # at least the unique ones
            @test all(any(v2 -> v1 ≈ v2, verts) for v1 in eachrow(expected_verts))

            verts = IntervalMDP.vertices(ambiguity_sets[3])
            @test length(verts) <= 6  # = number of permutations of 3 elements

            expected_verts = N[2 // 10 3//10 5//10]
            @test length(verts) ≥ size(expected_verts, 1)  # at least the unique ones
            @test all(any(v2 -> v1 ≈ v2, verts) for v1 in eachrow(expected_verts))
        end

        @testset "maximization" begin
            Vexpected = IntervalMDP.bellman(V, imc; upper_bound = true) # Using O-maximization, should be equivalent

            ws = IntervalMDP.construct_workspace(imc, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = zeros(N, 3)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = true)
            @test Vres ≈ Vexpected

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(imc, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = similar(Vres)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = true)
            @test Vres ≈ Vexpected

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                imc,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = similar(Vres)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = true)
            @test Vres ≈ Vexpected

            ws = IntervalMDP.construct_workspace(imc, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = zeros(N, 3)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = true)
            @test Vres ≈ Vexpected

            ws = IntervalMDP.FactoredVertexIteratorWorkspace(imc)
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = similar(Vres)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = true)
            @test Vres ≈ Vexpected

            ws = IntervalMDP.ThreadedFactoredVertexIteratorWorkspace(imc)
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = similar(Vres)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = true)
            @test Vres ≈ Vexpected
        end

        @testset "minimization" begin
            Vexpected = IntervalMDP.bellman(V, imc; upper_bound = false) # Using O-maximization, should be equivalent

            ws = IntervalMDP.construct_workspace(imc, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = zeros(N, 3)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = false)
            @test Vres ≈ Vexpected

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(imc, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = similar(Vres)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = false)
            @test Vres ≈ Vexpected

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                imc,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = similar(Vres)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = false)
            @test Vres ≈ Vexpected

            ws = IntervalMDP.construct_workspace(imc, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = zeros(N, 3)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = false)
            @test Vres ≈ Vexpected

            ws = IntervalMDP.FactoredVertexIteratorWorkspace(imc)
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = similar(Vres)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = false)
            @test Vres ≈ Vexpected

            ws = IntervalMDP.ThreadedFactoredVertexIteratorWorkspace(imc)
            strategy_cache = IntervalMDP.construct_strategy_cache(imc)
            Vres = similar(Vres)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, imc; upper_bound = false)
            @test Vres ≈ Vexpected
        end
    end

    @testset "bellman 2d" begin
        state_indices = (1, 2)
        action_indices = (1,)
        state_vars = (2, 3)
        action_vars = (1,)

        marginal1 = Marginal(
            IntervalAmbiguitySets(;
                lower = sparse(
                    N[
                        1//15 0 1//15 13//30 4//15 0
                        2//5 7//30 0 11//30 2//15 1//10
                    ],
                ),
                upper = sparse(
                    N[
                        17//30 7//10 2//3 4//5 7//10 2//3
                        9//10 13//15 9//10 5//6 4//5 14//15
                    ],
                ),
            ),
            state_indices,
            action_indices,
            state_vars,
            action_vars,
        )

        marginal2 = Marginal(
            IntervalAmbiguitySets(;
                lower = sparse(
                    N[
                        1//30 1//3 1//6 1//15 0 2//15
                        0 1//4 1//6 1//30 2//15 1//30
                        2//15 7//30 1//10 7//30 7//15 1//5
                    ],
                ),
                upper = sparse(
                    N[
                        2//3 7//15 4//5 11//30 19//30 1//2
                        23//30 4//5 23//30 3//5 7//10 8//15
                        7//15 4//5 23//30 7//10 7//15 23//30
                    ],
                ),
            ),
            state_indices,
            action_indices,
            state_vars,
            action_vars,
        )

        mdp = FactoredRobustMarkovDecisionProcess(
            state_vars,
            action_vars,
            (marginal1, marginal2),
        )

        V = N[
            3 13 18
            12 16 8
        ]

        #### Maximization
        @testset "maximization" begin
            ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            V_vertex = zeros(N, 2, 3)
            IntervalMDP.bellman!(ws, strategy_cache, V_vertex, V, mdp; upper_bound = true)

            @test V_vertex ≈ N[
                1076//75 4279//300 1081//75
                2821//225 4123//300 121//9
            ]

            ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_McCormick = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_McCormick,
                V,
                mdp;
                upper_bound = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_McCormick .>= 0.0)
            @test all(Vres_first_McCormick .<= maximum(V))
            @test all(Vres_first_McCormick .+ epsilon .>= V_vertex)

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                mdp,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_OMax = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_OMax,
                V,
                mdp;
                upper_bound = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_OMax .>= 0.0)
            @test all(Vres_first_OMax .<= maximum(V))
            @test all(Vres_first_OMax .+ epsilon .>= V_vertex)

            ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_OMax

            ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_OMax
        end

        #### Minimization
        @testset "minimization" begin
            ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            V_vertex = zeros(N, 2, 3)
            IntervalMDP.bellman!(ws, strategy_cache, V_vertex, V, mdp; upper_bound = false)

            @test V_vertex ≈ N[
                412//45 41//5 488//45
                1033//100 543//50 4253//450
            ]

            ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_McCormick = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_McCormick,
                V,
                mdp;
                upper_bound = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_McCormick .>= 0.0)
            @test all(Vres_first_McCormick .<= maximum(V))
            @test all(Vres_first_McCormick .- epsilon .<= V_vertex)

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                mdp,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_OMax = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_OMax,
                V,
                mdp;
                upper_bound = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_OMax .>= 0.0)
            @test all(Vres_first_OMax .<= maximum(V))
            @test all(Vres_first_OMax .- epsilon .<= V_vertex)

            ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_OMax

            ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_OMax
        end
    end

    @testset "bellman 2d partial dependence" begin
        state_vars = (2, 3)
        action_vars = (1, 2)

        marginal1 = Marginal(
            IntervalAmbiguitySets(;
                lower = N[
                    0 7//30 0 13//30 4//15 1//6
                    2//5 7//30 0 11//30 2//15 0
                ],
                upper = N[
                    17//30 7//10 2//3 4//5 7//10 2//3
                    9//10 13//15 9//10 5//6 4//5 14//15
                ],
            ),
            (1, 2),
            (1,),
            (2, 3),
            (1,),
        )

        marginal2 = Marginal(
            IntervalAmbiguitySets(;
                lower = N[
                    0 1//3 1//6 1//15 2//5 2//15
                    4//15 1//4 1//6 0 2//15 0
                    2//15 7//30 0 7//30 7//15 1//5
                ],
                upper = N[
                    2//3 7//15 4//5 11//30 19//30 1//2
                    23//30 4//5 23//30 3//5 7//10 8//15
                    7//15 4//5 23//30 7//10 7//15 23//30
                ],
            ),
            (2,),
            (2,),
            (3,),
            (2,),
        )

        mdp = FactoredRobustMarkovDecisionProcess(
            state_vars,
            action_vars,
            (marginal1, marginal2),
        )

        V = N[
            3 13 18
            12 16 8
        ]

        #### Maximization
        @testset "max/max" begin
            ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            V_vertex = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                V_vertex,
                V,
                mdp;
                upper_bound = true,
                maximize = true,
            )

            ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_McCormick = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_McCormick,
                V,
                mdp;
                upper_bound = true,
                maximize = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_McCormick .>= 0.0)
            @test all(Vres_first_McCormick .<= maximum(V))
            @test all(Vres_first_McCormick .+ epsilon .>= V_vertex)

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = true,
                maximize = true,
            )
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                mdp,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = true,
                maximize = true,
            )
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_OMax = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_OMax,
                V,
                mdp;
                upper_bound = true,
                maximize = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_OMax .>= 0.0)
            @test all(Vres_first_OMax .<= maximum(V))
            @test all(Vres_first_OMax .+ epsilon .>= V_vertex)

            ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = true,
                maximize = true,
            )
            @test Vres ≈ Vres_first_OMax

            ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = true,
                maximize = true,
            )
            @test Vres ≈ Vres_first_OMax
        end

        @testset "min/max" begin
            ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            V_vertex = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                V_vertex,
                V,
                mdp;
                upper_bound = true,
                maximize = false,
            )

            ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_McCormick = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_McCormick,
                V,
                mdp;
                upper_bound = true,
                maximize = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_McCormick .>= 0.0)
            @test all(Vres_first_McCormick .<= maximum(V))
            @test all(Vres_first_McCormick .+ epsilon .>= V_vertex)

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = true,
                maximize = false,
            )
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                mdp,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = true,
                maximize = false,
            )
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_OMax = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_OMax,
                V,
                mdp;
                upper_bound = true,
                maximize = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_OMax .>= 0.0)
            @test all(Vres_first_OMax .<= maximum(V))
            @test all(Vres_first_OMax .+ epsilon .>= V_vertex)

            ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = true,
                maximize = false,
            )
            @test Vres ≈ Vres_first_OMax

            ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = true,
                maximize = false,
            )
            @test Vres ≈ Vres_first_OMax
        end

        #### Minimization
        @testset "min/min" begin
            ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            V_vertex = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                V_vertex,
                V,
                mdp;
                upper_bound = false,
                maximize = false,
            )

            ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_McCormick = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_McCormick,
                V,
                mdp;
                upper_bound = false,
                maximize = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_McCormick .>= 0.0)
            @test all(Vres_first_McCormick .<= maximum(V))
            @test all(Vres_first_McCormick .- epsilon .<= V_vertex)

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = false,
                maximize = false,
            )
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                mdp,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = false,
                maximize = false,
            )
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_OMax = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_OMax,
                V,
                mdp;
                upper_bound = false,
                maximize = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_OMax .>= 0.0)
            @test all(Vres_first_OMax .<= maximum(V))
            @test all(Vres_first_OMax .- epsilon .<= V_vertex)

            ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = false,
                maximize = false,
            )
            @test Vres ≈ Vres_first_OMax

            ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = false,
                maximize = false,
            )
            @test Vres ≈ Vres_first_OMax
        end

        @testset "max/min" begin
            ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            V_vertex = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                V_vertex,
                V,
                mdp;
                upper_bound = false,
                maximize = true,
            )

            ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_McCormick = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_McCormick,
                V,
                mdp;
                upper_bound = false,
                maximize = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_McCormick .>= 0.0)
            @test all(Vres_first_McCormick .<= maximum(V))
            @test all(Vres_first_McCormick .- epsilon .<= V_vertex)

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = false,
                maximize = true,
            )
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                mdp,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = false,
                maximize = true,
            )
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_OMax = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_OMax,
                V,
                mdp;
                upper_bound = false,
                maximize = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_OMax .>= 0.0)
            @test all(Vres_first_OMax .<= maximum(V))
            @test all(Vres_first_OMax .- epsilon .<= V_vertex)

            ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = false,
                maximize = true,
            )
            @test Vres ≈ Vres_first_OMax

            ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                V,
                mdp;
                upper_bound = false,
                maximize = true,
            )
            @test Vres ≈ Vres_first_OMax
        end
    end

    @testset "bellman 3d" begin
        state_indices = (1, 2, 3)
        action_indices = (1,)
        state_vars = (3, 3, 3)
        action_vars = (1,)

        marginal1 = Marginal(
            IntervalAmbiguitySets(;
                lower = sparse(
                    N[
                        1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6 1//10 1//30 1//10 1//15 1//10 1//15 4//15 4//15 1//3
                        1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10 1//5 1//5 1//10 1//30 4//15 1//10 1//5 1//6 7//30
                        4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10 3//10 1//10 1//15 1//30 2//15 1//6 1//5 1//10 4//15
                    ],
                ),
                upper = sparse(
                    N[
                        7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30 17//30 13//30 2//5 2//5 2//3 2//5 17//30 2//5 19//30
                        8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30 8//15 7//15 7//15 13//30 8//15 2//5 8//15 17//30 3//5
                        11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15 2//5 2//5 11//30 17//30 17//30 1//2 2//5 19//30 13//30
                    ],
                ),
            ),
            state_indices,
            action_indices,
            state_vars,
            action_vars,
        )

        marginal2 = Marginal(
            IntervalAmbiguitySets(;
                lower = sparse(
                    N[
                        1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15 1//6 7//30 1//15 2//15 1//10 1//3 7//30 1//30 7//30
                        3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15 3//10 0 3//10 1//6 3//10 1//5 0 7//30 2//15
                        3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15 1//15 1//3 2//15 1//15 1//5 1//5 1//15 7//30 1//15
                    ],
                ),
                upper = sparse(
                    N[
                        2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30 13//30 17//30 17//30 13//30 11//30 19//30 8//15 2//5 8//15
                        1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3 1//2 8//15 8//15 8//15 8//15 2//5 3//5 2//3 13//30
                        17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3 1//3 2//3 17//30 8//15 17//30 3//5 2//5 19//30 11//30
                    ],
                ),
            ),
            state_indices,
            action_indices,
            state_vars,
            action_vars,
        )

        marginal3 = Marginal(
            IntervalAmbiguitySets(;
                lower = sparse(
                    N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15 7//30 4//15 1//10 1//3 1//5 7//30 1//30 1//5 7//30
                        2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15 1//5 4//15 1//10 1//10 1//3 7//30 3//10 1//3 3//10
                        1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30 0 1//30 1//15 2//15 1//6 7//30 4//15 4//15 7//30
                    ],
                ),
                upper = sparse(
                    N[
                        3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30 2//3 17//30 3//5 7//15 19//30 1//2 3//5 1//3 19//30
                        3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30 2//3 17//30 8//15 13//30 13//30 3//5 1//2 8//15 8//15
                        3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30 0 2//5 11//30 19//30 19//30 2//5 1//2 7//15 7//15
                    ],
                ),
            ),
            state_indices,
            action_indices,
            state_vars,
            action_vars,
        )

        mdp = FactoredRobustMarkovDecisionProcess(
            state_vars,
            action_vars,
            (marginal1, marginal2, marginal3),
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
            ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            V_vertex = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(ws, strategy_cache, V_vertex, V, mdp; upper_bound = true)

            ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_McCormick = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_McCormick,
                V,
                mdp;
                upper_bound = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_McCormick .>= 0.0)
            @test all(Vres_first_McCormick .<= maximum(V))
            @test all(Vres_first_McCormick .+ epsilon .>= V_vertex)

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                mdp,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_OMax = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_OMax,
                V,
                mdp;
                upper_bound = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_OMax .>= 0.0)
            @test all(Vres_first_OMax .<= maximum(V))
            @test all(Vres_first_OMax .+ epsilon .>= V_vertex)

            ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_OMax

            ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_OMax
        end

        #### Minimization
        @testset "minimization" begin
            ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            V_vertex = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(ws, strategy_cache, V_vertex, V, mdp; upper_bound = false)

            ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_McCormick = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_McCormick,
                V,
                mdp;
                upper_bound = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_McCormick .>= 0.0)
            @test all(Vres_first_McCormick .<= maximum(V))
            @test all(Vres_first_McCormick .- epsilon .<= V_vertex)

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                mdp,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_OMax = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_OMax,
                V,
                mdp;
                upper_bound = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_OMax .>= 0.0)
            @test all(Vres_first_OMax .<= maximum(V))
            @test all(Vres_first_OMax .- epsilon .<= V_vertex)

            ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_OMax

            ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_OMax
        end
    end

    @testset "bellman 3d mixed sparse/dense" begin
        state_indices = (1, 2, 3)
        action_indices = (1,)
        state_vars = (3, 3, 3)
        action_vars = (1,)

        marginal1 = Marginal(
            IntervalAmbiguitySets(;
                lower = sparse(
                    N[
                        1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6 1//10 1//30 1//10 1//15 1//10 1//15 4//15 4//15 1//3
                        1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10 1//5 1//5 1//10 1//30 4//15 1//10 1//5 1//6 7//30
                        4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10 3//10 1//10 1//15 1//30 2//15 1//6 1//5 1//10 4//15
                    ],
                ),
                upper = sparse(
                    N[
                        7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30 17//30 13//30 2//5 2//5 2//3 2//5 17//30 2//5 19//30
                        8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30 8//15 7//15 7//15 13//30 8//15 2//5 8//15 17//30 3//5
                        11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15 2//5 2//5 11//30 17//30 17//30 1//2 2//5 19//30 13//30
                    ],
                ),
            ),
            state_indices,
            action_indices,
            state_vars,
            action_vars,
        )

        marginal2 = Marginal(
            IntervalAmbiguitySets(;
                lower = N[
                    1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15 1//6 7//30 1//15 2//15 1//10 1//3 7//30 1//30 7//30
                    3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15 3//10 0 3//10 1//6 3//10 1//5 0 7//30 2//15
                    3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15 1//15 1//3 2//15 1//15 1//5 1//5 1//15 7//30 1//15
                ],
                upper = N[
                    2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30 13//30 17//30 17//30 13//30 11//30 19//30 8//15 2//5 8//15
                    1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3 1//2 8//15 8//15 8//15 8//15 2//5 3//5 2//3 13//30
                    17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3 1//3 2//3 17//30 8//15 17//30 3//5 2//5 19//30 11//30
                ],
            ),
            state_indices,
            action_indices,
            state_vars,
            action_vars,
        )

        marginal3 = Marginal(
            IntervalAmbiguitySets(;
                lower = sparse(
                    N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15 7//30 4//15 1//10 1//3 1//5 7//30 1//30 1//5 7//30
                        2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15 1//5 4//15 1//10 1//10 1//3 7//30 3//10 1//3 3//10
                        1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30 0 1//30 1//15 2//15 1//6 7//30 4//15 4//15 7//30
                    ],
                ),
                upper = sparse(
                    N[
                        3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30 2//3 17//30 3//5 7//15 19//30 1//2 3//5 1//3 19//30
                        3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30 2//3 17//30 8//15 13//30 13//30 3//5 1//2 8//15 8//15
                        3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30 0 2//5 11//30 19//30 19//30 2//5 1//2 7//15 7//15
                    ],
                ),
            ),
            state_indices,
            action_indices,
            state_vars,
            action_vars,
        )

        mdp = FactoredRobustMarkovDecisionProcess(
            state_vars,
            action_vars,
            (marginal1, marginal2, marginal3),
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
            ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            V_vertex = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(ws, strategy_cache, V_vertex, V, mdp; upper_bound = true)

            ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_McCormick = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_McCormick,
                V,
                mdp;
                upper_bound = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_McCormick .>= 0.0)
            @test all(Vres_first_McCormick .<= maximum(V))
            @test all(Vres_first_McCormick .+ epsilon .>= V_vertex)

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                mdp,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_OMax = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_OMax,
                V,
                mdp;
                upper_bound = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_OMax .>= 0.0)
            @test all(Vres_first_OMax .<= maximum(V))
            @test all(Vres_first_OMax .+ epsilon .>= V_vertex)

            ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_OMax

            ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = true)
            @test Vres ≈ Vres_first_OMax
        end

        #### Minimization
        @testset "minimization" begin
            ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            V_vertex = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(ws, strategy_cache, V_vertex, V, mdp; upper_bound = false)

            ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_McCormick = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_McCormick,
                V,
                mdp;
                upper_bound = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_McCormick .>= 0.0)
            @test all(Vres_first_McCormick .<= maximum(V))
            @test all(Vres_first_McCormick .- epsilon .<= V_vertex)

            ws =
                IntervalMDP.FactoredIntervalMcCormickWorkspace(mdp, LPMcCormickRelaxation())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                mdp,
                LPMcCormickRelaxation(),
            )
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_McCormick)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_McCormick

            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres_first_OMax = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres_first_OMax,
                V,
                mdp;
                upper_bound = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test all(Vres_first_OMax .>= 0.0)
            @test all(Vres_first_OMax .<= maximum(V))
            @test all(Vres_first_OMax .- epsilon .<= V_vertex)

            ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_OMax

            ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vres = similar(Vres_first_OMax)
            IntervalMDP.bellman!(ws, strategy_cache, Vres, V, mdp; upper_bound = false)
            @test Vres ≈ Vres_first_OMax
        end
    end

    @testset for alg in [
        RobustValueIteration(LPMcCormickRelaxation()),
        RobustValueIteration(OMaximization()),
        RobustValueIteration(VertexEnumeration()),
    ]
        @testset "implicit sink state" begin
            @testset "first dimension" begin
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = (3, 3, 3)
                source_dims = (2, 3, 3)
                action_vars = (1,)

                # Explicit
                marginal1 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//15 3//10 0 1//15 3//10 0 1//30 1//3 0 7//30 4//15 0 1//6 1//5 0 1//10 1//5 0 0 7//30 0 7//30 1//5 0 2//15 1//6 0
                                1//5 4//15 0 1//10 1//5 0 3//10 3//10 0 1//10 1//15 0 3//10 3//10 0 7//30 1//5 0 1//10 1//5 0 1//5 1//30 0 1//5 3//10 0
                                4//15 1//30 1 1//5 1//5 1 7//30 4//15 1 2//15 7//30 1 1//5 1//3 1 2//15 1//6 1 1//6 1//3 1 4//15 3//10 1 1//30 3//10 1
                            ],
                        ),
                        upper = sparse(
                            N[
                                7//15 17//30 0 13//30 3//5 0 17//30 17//30 0 17//30 13//30 0 3//5 2//3 0 11//30 7//15 0 0 1//2 0 17//30 13//30 0 7//15 13//30 0
                                8//15 1//2 0 3//5 7//15 0 8//15 17//30 0 2//3 17//30 0 11//30 7//15 0 19//30 19//30 0 13//15 1//2 0 17//30 13//30 0 3//5 11//30 0
                                11//30 1//3 1 2//5 8//15 1 7//15 3//5 1 2//3 17//30 1 2//3 8//15 1 2//15 3//5 1 2//3 3//5 1 17//30 2//3 1 7//15 8//15 1
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    state_vars,
                    action_vars,
                )

                marginal2 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//10 1//15 1 3//10 0 0 1//6 1//15 0 1//15 1//6 1 1//6 1//30 0 1//10 1//10 0 1//3 2//15 1 3//10 4//15 0 2//15 2//15 0
                                3//10 1//5 0 3//10 2//15 1 0 1//30 0 0 1//15 0 1//30 7//30 1 1//30 1//15 0 7//30 1//15 0 1//6 1//30 1 1//10 1//15 0
                                3//10 4//15 0 1//10 3//10 0 2//15 1//3 1 3//10 1//10 0 1//6 3//10 0 7//30 1//6 1 1//15 1//15 0 1//10 1//5 0 1//5 4//15 1
                            ],
                        ),
                        upper = sparse(
                            N[
                                2//5 17//30 1 3//5 11//30 0 3//5 7//15 0 19//30 2//5 1 3//5 2//3 0 2//3 8//15 0 8//15 19//30 1 8//15 8//15 0 13//30 13//30 0
                                1//3 13//30 0 11//30 2//5 1 2//3 2//3 0 0 13//30 0 1//2 17//30 1 17//30 1//3 0 2//5 1//3 0 13//30 11//30 1 8//15 1//3 0
                                17//30 3//5 0 8//15 1//2 0 7//15 1//2 1 2//3 17//30 0 11//30 2//5 0 1//2 7//15 1 2//5 17//30 0 11//30 2//5 0 11//30 2//3 1
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    state_vars,
                    action_vars,
                )

                marginal3 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                4//15 1//5 1 3//10 3//10 1 4//15 7//30 1 1//5 4//15 0 7//30 1//6 0 1//5 0 0 1//15 1//30 0 3//10 1//3 0 2//15 1//15 0
                                2//15 4//15 0 1//10 1//30 0 7//30 2//15 0 1//15 1//30 1 3//10 1//3 1 1//5 1//10 1 2//15 1//30 0 2//15 4//15 0 0 4//15 0
                                1//5 1//3 0 3//10 1//10 0 1//15 1//10 0 1//30 1//5 0 2//15 7//30 0 1//3 2//15 0 1//10 1//6 1 3//10 1//5 1 7//30 1//30 1
                            ],
                        ),
                        upper = sparse(
                            N[
                                3//5 17//30 1 1//2 3//5 1 19//30 2//5 1 8//15 1//3 0 11//30 2//5 0 17//30 13//30 0 2//5 3//5 0 3//5 11//30 0 1//2 11//30 0
                                3//5 2//3 0 13//30 19//30 0 1//3 2//5 0 17//30 7//15 1 11//30 3//5 1 19//30 7//15 1 2//5 8//15 0 17//30 11//30 0 19//30 13//30 0
                                3//5 2//3 0 1//2 1//2 0 2//3 7//15 0 3//5 3//5 0 1//2 1//3 0 2//5 8//15 0 2//5 11//30 1 1//3 8//15 1 7//15 13//30 1
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    state_vars,
                    action_vars,
                )

                mdp = FactoredRobustMarkovDecisionProcess(
                    state_vars,
                    action_vars,
                    (marginal1, marginal2, marginal3),
                )

                # Implicit
                marginal1 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6
                                1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10
                                4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10
                            ],
                        ),
                        upper = sparse(
                            N[
                                7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30
                                8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30
                                11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    source_dims,
                    action_vars,
                )

                marginal2 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15
                                3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15
                                3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15
                            ],
                        ),
                        upper = sparse(
                            N[
                                2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30
                                1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3
                                17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    source_dims,
                    action_vars,
                )

                marginal3 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15
                                2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15
                                1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30
                            ],
                        ),
                        upper = sparse(
                            N[
                                3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30
                                3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30
                                3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    source_dims,
                    action_vars,
                )

                implicit_mdp = FactoredRobustMarkovDecisionProcess(
                    state_vars,
                    action_vars,
                    source_dims,
                    (marginal1, marginal2, marginal3),
                )

                prop = FiniteTimeSafety([(3, i, j) for i in 1:3 for j in 1:3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                implicit_prob = VerificationProblem(implicit_mdp, spec)

                V, k, res = solve(prob, alg)
                V_implicit, k_implicit, res_implicit = solve(implicit_prob, alg)

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
                marginal1 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//15 3//10 1//15 3//10 1//30 1//3 1 0 0 7//30 4//15 1//6 1//5 1//10 1//5 1 0 0 0 7//30 7//30 1//5 2//15 1//6 1 0 0
                                1//5 4//15 1//10 1//5 3//10 3//10 0 1 0 1//10 1//15 3//10 3//10 7//30 1//5 0 1 0 1//10 1//5 1//5 1//30 1//5 3//10 0 1 0
                                4//15 1//30 1//5 1//5 7//30 4//15 0 0 1 2//15 7//30 1//5 1//3 2//15 1//6 0 0 1 1//6 1//3 4//15 3//10 1//30 3//10 0 0 1
                            ],
                        ),
                        upper = sparse(
                            N[
                                7//15 17//30 13//30 3//5 17//30 17//30 1 0 0 17//30 13//30 3//5 2//3 11//30 7//15 1 0 0 0 1//2 17//30 13//30 7//15 13//30 1 0 0
                                8//15 1//2 3//5 7//15 8//15 17//30 0 1 0 2//3 17//30 11//30 7//15 19//30 19//30 0 1 0 13//15 1//2 17//30 13//30 3//5 11//30 0 1 0
                                11//30 1//3 2//5 8//15 7//15 3//5 0 0 1 2//3 17//30 2//3 8//15 2//15 3//5 0 0 1 2//3 3//5 17//30 2//3 7//15 8//15 0 0 1
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    state_vars,
                    action_vars,
                )

                marginal2 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//10 1//15 3//10 0 1//6 1//15 0 0 0 1//15 1//6 1//6 1//30 1//10 1//10 0 0 0 1//3 2//15 3//10 4//15 2//15 2//15 0 0 0
                                3//10 1//5 3//10 2//15 0 1//30 0 0 0 0 1//15 1//30 7//30 1//30 1//15 0 0 0 7//30 1//15 1//6 1//30 1//10 1//15 0 0 0
                                3//10 4//15 1//10 3//10 2//15 1//3 1 1 1 3//10 1//10 1//6 3//10 7//30 1//6 1 1 1 1//15 1//15 1//10 1//5 1//5 4//15 1 1 1
                            ],
                        ),
                        upper = sparse(
                            N[
                                2//5 17//30 3//5 11//30 3//5 7//15 0 0 0 19//30 2//5 3//5 2//3 2//3 8//15 0 0 0 8//15 19//30 8//15 8//15 13//30 13//30 0 0 0
                                1//3 13//30 11//30 2//5 2//3 2//3 0 0 0 0 13//30 1//2 17//30 17//30 1//3 0 0 0 2//5 1//3 13//30 11//30 8//15 1//3 0 0 0
                                17//30 3//5 8//15 1//2 7//15 1//2 1 1 1 2//3 17//30 11//30 2//5 1//2 7//15 1 1 1 2//5 17//30 11//30 2//5 11//30 2//3 1 1 1
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    state_vars,
                    action_vars,
                )

                marginal3 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                4//15 1//5 3//10 3//10 4//15 7//30 1 1 1 1//5 4//15 7//30 1//6 1//5 0 0 0 0 1//15 1//30 3//10 1//3 2//15 1//15 0 0 0
                                2//15 4//15 1//10 1//30 7//30 2//15 0 0 0 1//15 1//30 3//10 1//3 1//5 1//10 1 1 1 2//15 1//30 2//15 4//15 0 4//15 0 0 0
                                1//5 1//3 3//10 1//10 1//15 1//10 0 0 0 1//30 1//5 2//15 7//30 1//3 2//15 0 0 0 1//10 1//6 3//10 1//5 7//30 1//30 1 1 1
                            ],
                        ),
                        upper = sparse(
                            N[
                                3//5 17//30 1//2 3//5 19//30 2//5 1 1 1 8//15 1//3 11//30 2//5 17//30 13//30 0 0 0 2//5 3//5 3//5 11//30 1//2 11//30 0 0 0
                                3//5 2//3 13//30 19//30 1//3 2//5 0 0 0 17//30 7//15 11//30 3//5 19//30 7//15 1 1 1 2//5 8//15 17//30 11//30 19//30 13//30 0 0 0
                                3//5 2//3 1//2 1//2 2//3 7//15 0 0 0 3//5 3//5 1//2 1//3 2//5 8//15 0 0 0 2//5 11//30 1//3 8//15 7//15 13//30 1 1 1
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    state_vars,
                    action_vars,
                )

                mdp = FactoredRobustMarkovDecisionProcess(
                    state_vars,
                    action_vars,
                    (marginal1, marginal2, marginal3),
                )

                # Implicit
                marginal1 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6
                                1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10
                                4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10
                            ],
                        ),
                        upper = sparse(
                            N[
                                7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30
                                8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30
                                11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    source_dims,
                    action_vars,
                )

                marginal2 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15
                                3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15
                                3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15
                            ],
                        ),
                        upper = sparse(
                            N[
                                2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30
                                1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3
                                17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    source_dims,
                    action_vars,
                )

                marginal3 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15
                                2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15
                                1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30
                            ],
                        ),
                        upper = sparse(
                            N[
                                3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30
                                3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30
                                3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    source_dims,
                    action_vars,
                )

                implicit_mdp = FactoredRobustMarkovDecisionProcess(
                    state_vars,
                    action_vars,
                    source_dims,
                    (marginal1, marginal2, marginal3),
                )

                prop = FiniteTimeSafety([(i, 3, j) for i in 1:3 for j in 1:3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                implicit_prob = VerificationProblem(implicit_mdp, spec)

                V, k, res = solve(prob, alg)
                V_implicit, k_implicit, res_implicit = solve(implicit_prob, alg)

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
                marginal1 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6 1 0 0 1 0 0 1 0 0
                                1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10 0 1 0 0 1 0 0 1 0
                                4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10 0 0 1 0 0 1 0 0 1
                            ],
                        ),
                        upper = sparse(
                            N[
                                7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30 1 0 0 1 0 0 1 0 0
                                8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30 0 1 0 0 1 0 0 1 0
                                11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15 0 0 1 0 0 1 0 0 1
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    state_vars,
                    action_vars,
                )

                marginal2 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15 1 1 1 0 0 0 0 0 0
                                3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15 0 0 0 1 1 1 0 0 0
                                3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15 0 0 0 0 0 0 1 1 1
                            ],
                        ),
                        upper = sparse(
                            N[
                                2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30 1 1 1 0 0 0 0 0 0
                                1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3 0 0 0 1 1 1 0 0 0
                                17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3 0 0 0 0 0 0 1 1 1
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    state_vars,
                    action_vars,
                )

                marginal3 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15 0 0 0 0 0 0 0 0 0
                                2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15 0 0 0 0 0 0 0 0 0
                                1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30 1 1 1 1 1 1 1 1 1
                            ],
                        ),
                        upper = sparse(
                            N[
                                3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30 0 0 0 0 0 0 0 0 0
                                3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30 0 0 0 0 0 0 0 0 0
                                3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30 1 1 1 1 1 1 1 1 1
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    state_vars,
                    action_vars,
                )

                mdp = FactoredRobustMarkovDecisionProcess(
                    state_vars,
                    action_vars,
                    (marginal1, marginal2, marginal3),
                )

                # Implicit
                marginal1 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6
                                1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10
                                4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10
                            ],
                        ),
                        upper = sparse(
                            N[
                                7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30
                                8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30
                                11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    source_dims,
                    action_vars,
                )

                marginal2 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15
                                3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15
                                3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15
                            ],
                        ),
                        upper = sparse(
                            N[
                                2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30
                                1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3
                                17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    source_dims,
                    action_vars,
                )

                marginal3 = Marginal(
                    IntervalAmbiguitySets(;
                        lower = sparse(
                            N[
                                4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15
                                2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15
                                1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30
                            ],
                        ),
                        upper = sparse(
                            N[
                                3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30
                                3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30
                                3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30
                            ],
                        ),
                    ),
                    state_indices,
                    action_indices,
                    source_dims,
                    action_vars,
                )

                implicit_mdp = FactoredRobustMarkovDecisionProcess(
                    state_vars,
                    action_vars,
                    source_dims,
                    (marginal1, marginal2, marginal3),
                )

                prop = FiniteTimeSafety([(i, j, 3) for i in 1:3 for j in 1:3], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                implicit_prob = VerificationProblem(implicit_mdp, spec)

                V, k, res = solve(prob, alg)
                V_implicit, k_implicit, res_implicit = solve(implicit_prob, alg)

                @test V ≈ V_implicit
                @test k == k_implicit
                @test res ≈ res_implicit
            end
        end
    end
end
