using Revise, Test
using IntervalMDP, CUDA
using Random: MersenneTwister

@testset "show 1d" begin
    N = Float64
    ambiguity_sets = IntervalAmbiguitySets(;
        lower = N[
            0 5//10 2//10
            1//10 3//10 3//10
            2//10 1//10 5//10
        ],
        upper = N[
            5//10 7//10 3//10
            6//10 5//10 4//10
            7//10 3//10 5//10
        ],
    )
    imc = IntervalMarkovChain(ambiguity_sets, [CartesianIndex(2)])
    imc = IntervalMDP.cu(imc)

    io = IOBuffer()
    show(io, MIME("text/plain"), imc)
    str = String(take!(io))
    @test occursin("FactoredRobustMarkovDecisionProcess", str)
    @test occursin("1 state variables with cardinality: (3,)", str)
    @test occursin("1 action variables with cardinality: (1,)", str)
    @test occursin("Initial states: CartesianIndex{1}[$(CartesianIndex(2))]", str)
    @test occursin("Marginal 1:", str)
    @test occursin(
        "Ambiguity set type: Interval (dense, CuArray{Float64, 2, CUDA.DeviceMemory})",
        str,
    )
    @test !occursin("Marginal 2:", str)
    @test occursin("Inferred properties", str)
    @test occursin("Model type: Interval MDP", str)
    @test occursin("Number of states: 3", str)
    @test occursin("Number of actions: 1", str)
    @test occursin("Default model checking algorithm: Robust Value Iteration", str)
    @test occursin("Default Bellman operator algorithm: O-Maximization", str)
end

@testset "show 3d" begin
    N = Float64
    state_indices = (1, 2, 3)
    action_indices = (1,)
    state_vars = (3, 3, 3)
    source_dims = (2, 3, 3)
    action_vars = (1,)

    marginal1 = Marginal(
        IntervalAmbiguitySets(;
            lower = N[
                1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6
                1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10
                4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10
            ],
            upper = N[
                7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30
                8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30
                11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15
            ],
        ),
        state_indices,
        action_indices,
        source_dims,
        action_vars,
    )

    marginal2 = Marginal(
        IntervalAmbiguitySets(;
            lower = N[
                1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15
                3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15
                3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15
            ],
            upper = N[
                2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30
                1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3
                17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3
            ],
        ),
        state_indices,
        action_indices,
        source_dims,
        action_vars,
    )

    marginal3 = Marginal(
        IntervalAmbiguitySets(;
            lower = N[
                4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15
                2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15
                1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30
            ],
            upper = N[
                3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30
                3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30
                3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30
            ],
        ),
        state_indices,
        action_indices,
        source_dims,
        action_vars,
    )

    mdp = FactoredRobustMarkovDecisionProcess(
        state_vars,
        action_vars,
        source_dims,
        (marginal1, marginal2, marginal3),
    )
    mdp = IntervalMDP.cu(mdp)

    io = IOBuffer()
    show(io, MIME("text/plain"), mdp)
    str = String(take!(io))
    @test occursin("FactoredRobustMarkovDecisionProcess", str)
    @test occursin("3 state variables with cardinality: (3, 3, 3)", str)
    @test occursin("1 action variables with cardinality: (1,)", str)
    @test occursin("Initial states: All states", str)
    @test occursin("Marginal 1:", str)
    @test occursin(
        "Ambiguity set type: Interval (dense, CuArray{Float64, 2, CUDA.DeviceMemory})",
        str,
    )
    @test occursin("Marginal 2:", str)
    @test occursin("Marginal 3:", str)
    @test occursin("Inferred properties", str)
    @test occursin("Model type: Factored Interval MDP", str)
    @test occursin("Number of states: 27", str)
    @test occursin("Number of actions: 1", str)
    @test occursin("Default model checking algorithm: Robust Value Iteration", str)
    @test occursin("Default Bellman operator algorithm: Recursive O-Maximization", str)
end

@testset for N in [Float32, Float64]
    @testset "bellman 2d" begin
        state_indices = (1, 2)
        action_indices = (1,)
        state_vars = (2, 3)
        action_vars = (1,)

        marginal1 = Marginal(
            IntervalAmbiguitySets(;
                lower = N[
                    1//15 7//30 1//15 13//30 4//15 1//6
                    2//5 7//30 1//30 11//30 2//15 1//10
                ],
                upper = N[
                    17//30 7//10 2//3 4//5 7//10 2//3
                    9//10 13//15 9//10 5//6 4//5 14//15
                ],
            ),
            state_indices,
            action_indices,
            state_vars,
            action_vars,
        )

        marginal2 = Marginal(
            IntervalAmbiguitySets(;
                lower = N[
                    1//30 1//3 1//6 1//15 2//5 2//15
                    4//15 1//4 1//6 1//30 2//15 1//30
                    2//15 7//30 1//10 7//30 7//15 1//5
                ],
                upper = N[
                    2//3 7//15 4//5 11//30 19//30 1//2
                    23//30 4//5 23//30 3//5 7//10 8//15
                    7//15 4//5 23//30 7//10 7//15 23//30
                ],
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
        cuda_mdp = IntervalMDP.cu(mdp)

        V = N[
            3 13 18
            12 16 8
        ]
        cuda_V = IntervalMDP.cu(V)

        #### Maximization
        @testset "maximization" begin
            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vexpected = similar(V)
            IntervalMDP.bellman!(ws, strategy_cache, Vexpected, V, mdp; upper_bound = true)

            ws = IntervalMDP.construct_workspace(cuda_mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(cuda_mdp)
            Vres = similar(cuda_V)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                cuda_V,
                cuda_mdp;
                upper_bound = true,
            )
            @test IntervalMDP.cpu(Vres) ≈ Vexpected
        end

        #### Minimization
        @testset "minimization" begin
            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vexpected = similar(V)
            IntervalMDP.bellman!(ws, strategy_cache, Vexpected, V, mdp; upper_bound = false)

            ws = IntervalMDP.construct_workspace(cuda_mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(cuda_mdp)
            Vres = similar(cuda_V)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                cuda_V,
                cuda_mdp;
                upper_bound = false,
            )
            @test IntervalMDP.cpu(Vres) ≈ Vexpected
        end
    end

    @testset "bellman 2d partial dependence" begin
        state_vars = (2, 3)
        action_vars = (1, 2)

        marginal1 = Marginal(
            IntervalAmbiguitySets(;
                lower = N[
                    1//15 7//30 1//15 13//30 4//15 1//6
                    2//5 7//30 1//30 11//30 2//15 1//10
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
                    1//30 1//3 1//6 1//15 2//5 2//15
                    4//15 1//4 1//6 1//30 2//15 1//30
                    2//15 7//30 1//10 7//30 7//15 1//5
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
        cuda_mdp = IntervalMDP.cu(mdp)

        V = N[
            3 13 18
            12 16 8
        ]
        cuda_V = IntervalMDP.cu(V)

        #### Maximization
        @testset "max/max" begin
            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vexpected = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vexpected,
                V,
                mdp;
                upper_bound = true,
                maximize = true,
            )

            ws = IntervalMDP.construct_workspace(cuda_mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(cuda_mdp)
            Vres = CUDA.zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                cuda_V,
                cuda_mdp;
                upper_bound = true,
                maximize = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test IntervalMDP.cpu(Vres) ≈ Vexpected atol=epsilon
        end

        @testset "min/max" begin
            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vexpected = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vexpected,
                V,
                mdp;
                upper_bound = true,
                maximize = false,
            )

            ws = IntervalMDP.construct_workspace(cuda_mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(cuda_mdp)
            Vres = CUDA.zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                cuda_V,
                cuda_mdp;
                upper_bound = true,
                maximize = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test IntervalMDP.cpu(Vres) ≈ Vexpected atol=epsilon
        end

        #### Minimization
        @testset "min/min" begin
            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vexpected = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vexpected,
                V,
                mdp;
                upper_bound = false,
                maximize = false,
            )

            ws = IntervalMDP.construct_workspace(cuda_mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(cuda_mdp)
            Vres = CUDA.zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                cuda_V,
                cuda_mdp;
                upper_bound = false,
                maximize = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test IntervalMDP.cpu(Vres) ≈ Vexpected atol=epsilon
        end

        @testset "max/min" begin
            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vexpected = zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vexpected,
                V,
                mdp;
                upper_bound = false,
                maximize = true,
            )

            ws = IntervalMDP.construct_workspace(cuda_mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(cuda_mdp)
            Vres = CUDA.zeros(N, 2, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                cuda_V,
                cuda_mdp;
                upper_bound = false,
                maximize = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test IntervalMDP.cpu(Vres) ≈ Vexpected atol=epsilon
        end
    end

    @testset "bellman 3d" begin
        state_indices = (1, 2, 3)
        action_indices = (1,)
        state_vars = (3, 3, 3)
        action_vars = (1,)

        marginal1 = Marginal(
            IntervalAmbiguitySets(;
                lower = N[
                    1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6 1//10 1//30 1//10 1//15 1//10 1//15 4//15 4//15 1//3
                    1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10 1//5 1//5 1//10 1//30 4//15 1//10 1//5 1//6 7//30
                    4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10 3//10 1//10 1//15 1//30 2//15 1//6 1//5 1//10 4//15
                ],
                upper = N[
                    7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30 17//30 13//30 2//5 2//5 2//3 2//5 17//30 2//5 19//30
                    8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30 8//15 7//15 7//15 13//30 8//15 2//5 8//15 17//30 3//5
                    11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15 2//5 2//5 11//30 17//30 17//30 1//2 2//5 19//30 13//30
                ],
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
                lower = N[
                    4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15 7//30 4//15 1//10 1//3 1//5 7//30 1//30 1//5 7//30
                    2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15 1//5 4//15 1//10 1//10 1//3 7//30 3//10 1//3 3//10
                    1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30 0 1//30 1//15 2//15 1//6 7//30 4//15 4//15 7//30
                ],
                upper = N[
                    3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30 2//3 17//30 3//5 7//15 19//30 1//2 3//5 1//3 19//30
                    3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30 2//3 17//30 8//15 13//30 13//30 3//5 1//2 8//15 8//15
                    3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30 0 2//5 11//30 19//30 19//30 2//5 1//2 7//15 7//15
                ],
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
        cuda_mdp = IntervalMDP.cu(mdp)

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
        cuda_V = IntervalMDP.cu(V)

        #### Maximization
        @testset "maximization" begin
            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vexpected = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(ws, strategy_cache, Vexpected, V, mdp; upper_bound = true)

            ws = IntervalMDP.construct_workspace(cuda_mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(cuda_mdp)
            Vres = CUDA.zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                cuda_V,
                cuda_mdp;
                upper_bound = true,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test IntervalMDP.cpu(Vres) ≈ Vexpected atol=epsilon
        end

        #### Minimization
        @testset "minimization" begin
            ws = IntervalMDP.construct_workspace(mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
            Vexpected = zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(ws, strategy_cache, Vexpected, V, mdp; upper_bound = false)

            ws = IntervalMDP.construct_workspace(cuda_mdp, OMaximization())
            strategy_cache = IntervalMDP.construct_strategy_cache(cuda_mdp)
            Vres = CUDA.zeros(N, 3, 3, 3)
            IntervalMDP.bellman!(
                ws,
                strategy_cache,
                Vres,
                cuda_V,
                cuda_mdp;
                upper_bound = false,
            )

            epsilon = N == Float32 ? 1e-5 : 1e-8
            @test IntervalMDP.cpu(Vres) ≈ Vexpected atol=epsilon
        end
    end

    alg = RobustValueIteration(OMaximization())
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
                    lower = N[
                        1//15 3//10 0 1//15 3//10 0 1//30 1//3 0 7//30 4//15 0 1//6 1//5 0 1//10 1//5 0 0 7//30 0 7//30 1//5 0 2//15 1//6 0
                        1//5 4//15 0 1//10 1//5 0 3//10 3//10 0 1//10 1//15 0 3//10 3//10 0 7//30 1//5 0 1//10 1//5 0 1//5 1//30 0 1//5 3//10 0
                        4//15 1//30 1 1//5 1//5 1 7//30 4//15 1 2//15 7//30 1 1//5 1//3 1 2//15 1//6 1 1//6 1//3 1 4//15 3//10 1 1//30 3//10 1
                    ],
                    upper = N[
                        7//15 17//30 0 13//30 3//5 0 17//30 17//30 0 17//30 13//30 0 3//5 2//3 0 11//30 7//15 0 0 1//2 0 17//30 13//30 0 7//15 13//30 0
                        8//15 1//2 0 3//5 7//15 0 8//15 17//30 0 2//3 17//30 0 11//30 7//15 0 19//30 19//30 0 13//15 1//2 0 17//30 13//30 0 3//5 11//30 0
                        11//30 1//3 1 2//5 8//15 1 7//15 3//5 1 2//3 17//30 1 2//3 8//15 1 2//15 3//5 1 2//3 3//5 1 17//30 2//3 1 7//15 8//15 1
                    ],
                ),
                state_indices,
                action_indices,
                state_vars,
                action_vars,
            )

            marginal2 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 1 3//10 0 0 1//6 1//15 0 1//15 1//6 1 1//6 1//30 0 1//10 1//10 0 1//3 2//15 1 3//10 4//15 0 2//15 2//15 0
                        3//10 1//5 0 3//10 2//15 1 0 1//30 0 0 1//15 0 1//30 7//30 1 1//30 1//15 0 7//30 1//15 0 1//6 1//30 1 1//10 1//15 0
                        3//10 4//15 0 1//10 3//10 0 2//15 1//3 1 3//10 1//10 0 1//6 3//10 0 7//30 1//6 1 1//15 1//15 0 1//10 1//5 0 1//5 4//15 1
                    ],
                    upper = N[
                        2//5 17//30 1 3//5 11//30 0 3//5 7//15 0 19//30 2//5 1 3//5 2//3 0 2//3 8//15 0 8//15 19//30 1 8//15 8//15 0 13//30 13//30 0
                        1//3 13//30 0 11//30 2//5 1 2//3 2//3 0 0 13//30 0 1//2 17//30 1 17//30 1//3 0 2//5 1//3 0 13//30 11//30 1 8//15 1//3 0
                        17//30 3//5 0 8//15 1//2 0 7//15 1//2 1 2//3 17//30 0 11//30 2//5 0 1//2 7//15 1 2//5 17//30 0 11//30 2//5 0 11//30 2//3 1
                    ],
                ),
                state_indices,
                action_indices,
                state_vars,
                action_vars,
            )

            marginal3 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 1 3//10 3//10 1 4//15 7//30 1 1//5 4//15 0 7//30 1//6 0 1//5 0 0 1//15 1//30 0 3//10 1//3 0 2//15 1//15 0
                        2//15 4//15 0 1//10 1//30 0 7//30 2//15 0 1//15 1//30 1 3//10 1//3 1 1//5 1//10 1 2//15 1//30 0 2//15 4//15 0 0 4//15 0
                        1//5 1//3 0 3//10 1//10 0 1//15 1//10 0 1//30 1//5 0 2//15 7//30 0 1//3 2//15 0 1//10 1//6 1 3//10 1//5 1 7//30 1//30 1
                    ],
                    upper = N[
                        3//5 17//30 1 1//2 3//5 1 19//30 2//5 1 8//15 1//3 0 11//30 2//5 0 17//30 13//30 0 2//5 3//5 0 3//5 11//30 0 1//2 11//30 0
                        3//5 2//3 0 13//30 19//30 0 1//3 2//5 0 17//30 7//15 1 11//30 3//5 1 19//30 7//15 1 2//5 8//15 0 17//30 11//30 0 19//30 13//30 0
                        3//5 2//3 0 1//2 1//2 0 2//3 7//15 0 3//5 3//5 0 1//2 1//3 0 2//5 8//15 0 2//5 11//30 1 1//3 8//15 1 7//15 13//30 1
                    ],
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
                    lower = N[
                        1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6
                        1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10
                        4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10
                    ],
                    upper = N[
                        7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30
                        8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30
                        11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15
                    ],
                ),
                state_indices,
                action_indices,
                source_dims,
                action_vars,
            )

            marginal2 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15
                        3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15
                        3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15
                    ],
                    upper = N[
                        2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30
                        1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3
                        17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3
                    ],
                ),
                state_indices,
                action_indices,
                source_dims,
                action_vars,
            )

            marginal3 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15
                        2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15
                        1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30
                    ],
                    upper = N[
                        3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30
                        3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30
                        3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30
                    ],
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
            implicit_mdp = IntervalMDP.cu(implicit_mdp)

            prop = FiniteTimeSafety([(3, i, j) for i in 1:3 for j in 1:3], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            prob = VerificationProblem(mdp, spec)
            implicit_prob = VerificationProblem(implicit_mdp, spec)

            V, k, res = solve(prob, alg)
            V_implicit, k_implicit, res_implicit = solve(implicit_prob, alg)

            @test V ≈ IntervalMDP.cpu(V_implicit)
            @test k == k_implicit
            @test res ≈ IntervalMDP.cpu(res_implicit)
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
                    lower = N[
                        1//15 3//10 1//15 3//10 1//30 1//3 1 0 0 7//30 4//15 1//6 1//5 1//10 1//5 1 0 0 0 7//30 7//30 1//5 2//15 1//6 1 0 0
                        1//5 4//15 1//10 1//5 3//10 3//10 0 1 0 1//10 1//15 3//10 3//10 7//30 1//5 0 1 0 1//10 1//5 1//5 1//30 1//5 3//10 0 1 0
                        4//15 1//30 1//5 1//5 7//30 4//15 0 0 1 2//15 7//30 1//5 1//3 2//15 1//6 0 0 1 1//6 1//3 4//15 3//10 1//30 3//10 0 0 1
                    ],
                    upper = N[
                        7//15 17//30 13//30 3//5 17//30 17//30 1 0 0 17//30 13//30 3//5 2//3 11//30 7//15 1 0 0 0 1//2 17//30 13//30 7//15 13//30 1 0 0
                        8//15 1//2 3//5 7//15 8//15 17//30 0 1 0 2//3 17//30 11//30 7//15 19//30 19//30 0 1 0 13//15 1//2 17//30 13//30 3//5 11//30 0 1 0
                        11//30 1//3 2//5 8//15 7//15 3//5 0 0 1 2//3 17//30 2//3 8//15 2//15 3//5 0 0 1 2//3 3//5 17//30 2//3 7//15 8//15 0 0 1
                    ],
                ),
                state_indices,
                action_indices,
                state_vars,
                action_vars,
            )

            marginal2 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 3//10 0 1//6 1//15 0 0 0 1//15 1//6 1//6 1//30 1//10 1//10 0 0 0 1//3 2//15 3//10 4//15 2//15 2//15 0 0 0
                        3//10 1//5 3//10 2//15 0 1//30 0 0 0 0 1//15 1//30 7//30 1//30 1//15 0 0 0 7//30 1//15 1//6 1//30 1//10 1//15 0 0 0
                        3//10 4//15 1//10 3//10 2//15 1//3 1 1 1 3//10 1//10 1//6 3//10 7//30 1//6 1 1 1 1//15 1//15 1//10 1//5 1//5 4//15 1 1 1
                    ],
                    upper = N[
                        2//5 17//30 3//5 11//30 3//5 7//15 0 0 0 19//30 2//5 3//5 2//3 2//3 8//15 0 0 0 8//15 19//30 8//15 8//15 13//30 13//30 0 0 0
                        1//3 13//30 11//30 2//5 2//3 2//3 0 0 0 0 13//30 1//2 17//30 17//30 1//3 0 0 0 2//5 1//3 13//30 11//30 8//15 1//3 0 0 0
                        17//30 3//5 8//15 1//2 7//15 1//2 1 1 1 2//3 17//30 11//30 2//5 1//2 7//15 1 1 1 2//5 17//30 11//30 2//5 11//30 2//3 1 1 1
                    ],
                ),
                state_indices,
                action_indices,
                state_vars,
                action_vars,
            )

            marginal3 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1 1 1 1//5 4//15 7//30 1//6 1//5 0 0 0 0 1//15 1//30 3//10 1//3 2//15 1//15 0 0 0
                        2//15 4//15 1//10 1//30 7//30 2//15 0 0 0 1//15 1//30 3//10 1//3 1//5 1//10 1 1 1 2//15 1//30 2//15 4//15 0 4//15 0 0 0
                        1//5 1//3 3//10 1//10 1//15 1//10 0 0 0 1//30 1//5 2//15 7//30 1//3 2//15 0 0 0 1//10 1//6 3//10 1//5 7//30 1//30 1 1 1
                    ],
                    upper = N[
                        3//5 17//30 1//2 3//5 19//30 2//5 1 1 1 8//15 1//3 11//30 2//5 17//30 13//30 0 0 0 2//5 3//5 3//5 11//30 1//2 11//30 0 0 0
                        3//5 2//3 13//30 19//30 1//3 2//5 0 0 0 17//30 7//15 11//30 3//5 19//30 7//15 1 1 1 2//5 8//15 17//30 11//30 19//30 13//30 0 0 0
                        3//5 2//3 1//2 1//2 2//3 7//15 0 0 0 3//5 3//5 1//2 1//3 2//5 8//15 0 0 0 2//5 11//30 1//3 8//15 7//15 13//30 1 1 1
                    ],
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
                    lower = N[
                        1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6
                        1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10
                        4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10
                    ],
                    upper = N[
                        7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30
                        8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30
                        11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15
                    ],
                ),
                state_indices,
                action_indices,
                source_dims,
                action_vars,
            )

            marginal2 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15
                        3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15
                        3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15
                    ],
                    upper = N[
                        2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30
                        1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3
                        17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3
                    ],
                ),
                state_indices,
                action_indices,
                source_dims,
                action_vars,
            )

            marginal3 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15
                        2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15
                        1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30
                    ],
                    upper = N[
                        3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30
                        3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30
                        3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30
                    ],
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
            implicit_mdp = IntervalMDP.cu(implicit_mdp)

            prop = FiniteTimeSafety([(i, 3, j) for i in 1:3 for j in 1:3], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            prob = VerificationProblem(mdp, spec)
            implicit_prob = VerificationProblem(implicit_mdp, spec)

            V, k, res = solve(prob, alg)
            V_implicit, k_implicit, res_implicit = solve(implicit_prob, alg)

            @test V ≈ IntervalMDP.cpu(V_implicit)
            @test k == k_implicit
            @test res ≈ IntervalMDP.cpu(res_implicit)
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
                    lower = N[
                        1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6 1 0 0 1 0 0 1 0 0
                        1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10 0 1 0 0 1 0 0 1 0
                        4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10 0 0 1 0 0 1 0 0 1
                    ],
                    upper = N[
                        7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30 1 0 0 1 0 0 1 0 0
                        8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30 0 1 0 0 1 0 0 1 0
                        11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15 0 0 1 0 0 1 0 0 1
                    ],
                ),
                state_indices,
                action_indices,
                state_vars,
                action_vars,
            )

            marginal2 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15 1 1 1 0 0 0 0 0 0
                        3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15 0 0 0 1 1 1 0 0 0
                        3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15 0 0 0 0 0 0 1 1 1
                    ],
                    upper = N[
                        2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30 1 1 1 0 0 0 0 0 0
                        1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3 0 0 0 1 1 1 0 0 0
                        17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3 0 0 0 0 0 0 1 1 1
                    ],
                ),
                state_indices,
                action_indices,
                state_vars,
                action_vars,
            )

            marginal3 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15 0 0 0 0 0 0 0 0 0
                        2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15 0 0 0 0 0 0 0 0 0
                        1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30 1 1 1 1 1 1 1 1 1
                    ],
                    upper = N[
                        3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30 0 0 0 0 0 0 0 0 0
                        3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30 0 0 0 0 0 0 0 0 0
                        3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30 1 1 1 1 1 1 1 1 1
                    ],
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
                    lower = N[
                        1//15 3//10 1//15 3//10 1//30 1//3 7//30 4//15 1//6 1//5 1//10 1//5 0 7//30 7//30 1//5 2//15 1//6
                        1//5 4//15 1//10 1//5 3//10 3//10 1//10 1//15 3//10 3//10 7//30 1//5 1//10 1//5 1//5 1//30 1//5 3//10
                        4//15 1//30 1//5 1//5 7//30 4//15 2//15 7//30 1//5 1//3 2//15 1//6 1//6 1//3 4//15 3//10 1//30 3//10
                    ],
                    upper = N[
                        7//15 17//30 13//30 3//5 17//30 17//30 17//30 13//30 3//5 2//3 11//30 7//15 0 1//2 17//30 13//30 7//15 13//30
                        8//15 1//2 3//5 7//15 8//15 17//30 2//3 17//30 11//30 7//15 19//30 19//30 13//15 1//2 17//30 13//30 3//5 11//30
                        11//30 1//3 2//5 8//15 7//15 3//5 2//3 17//30 2//3 8//15 2//15 3//5 2//3 3//5 17//30 2//3 7//15 8//15
                    ],
                ),
                state_indices,
                action_indices,
                source_dims,
                action_vars,
            )

            marginal2 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        1//10 1//15 3//10 0 1//6 1//15 1//15 1//6 1//6 1//30 1//10 1//10 1//3 2//15 3//10 4//15 2//15 2//15
                        3//10 1//5 3//10 2//15 0 1//30 0 1//15 1//30 7//30 1//30 1//15 7//30 1//15 1//6 1//30 1//10 1//15
                        3//10 4//15 1//10 3//10 2//15 1//3 3//10 1//10 1//6 3//10 7//30 1//6 1//15 1//15 1//10 1//5 1//5 4//15
                    ],
                    upper = N[
                        2//5 17//30 3//5 11//30 3//5 7//15 19//30 2//5 3//5 2//3 2//3 8//15 8//15 19//30 8//15 8//15 13//30 13//30
                        1//3 13//30 11//30 2//5 2//3 2//3 0 13//30 1//2 17//30 17//30 1//3 2//5 1//3 13//30 11//30 8//15 1//3
                        17//30 3//5 8//15 1//2 7//15 1//2 2//3 17//30 11//30 2//5 1//2 7//15 2//5 17//30 11//30 2//5 11//30 2//3
                    ],
                ),
                state_indices,
                action_indices,
                source_dims,
                action_vars,
            )

            marginal3 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        4//15 1//5 3//10 3//10 4//15 7//30 1//5 4//15 7//30 1//6 1//5 0 1//15 1//30 3//10 1//3 2//15 1//15
                        2//15 4//15 1//10 1//30 7//30 2//15 1//15 1//30 3//10 1//3 1//5 1//10 2//15 1//30 2//15 4//15 0 4//15
                        1//5 1//3 3//10 1//10 1//15 1//10 1//30 1//5 2//15 7//30 1//3 2//15 1//10 1//6 3//10 1//5 7//30 1//30
                    ],
                    upper = N[
                        3//5 17//30 1//2 3//5 19//30 2//5 8//15 1//3 11//30 2//5 17//30 13//30 2//5 3//5 3//5 11//30 1//2 11//30
                        3//5 2//3 13//30 19//30 1//3 2//5 17//30 7//15 11//30 3//5 19//30 7//15 2//5 8//15 17//30 11//30 19//30 13//30
                        3//5 2//3 1//2 1//2 2//3 7//15 3//5 3//5 1//2 1//3 2//5 8//15 2//5 11//30 1//3 8//15 7//15 13//30
                    ],
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
            implicit_mdp = IntervalMDP.cu(implicit_mdp)

            prop = FiniteTimeSafety([(i, j, 3) for i in 1:3 for j in 1:3], 10)
            spec = Specification(prop, Pessimistic, Maximize)
            prob = VerificationProblem(mdp, spec)
            implicit_prob = VerificationProblem(implicit_mdp, spec)

            V, k, res = solve(prob, alg)
            V_implicit, k_implicit, res_implicit = solve(implicit_prob, alg)

            @test V ≈ IntervalMDP.cpu(V_implicit)
            @test k == k_implicit
            @test res ≈ IntervalMDP.cpu(res_implicit)
        end
    end

    # 4-D rand
    @testset "4D rand" begin
        rng = MersenneTwister(995)

        prob_lower = [rand(rng, N, 3, 81) ./ N(3) for _ in 1:4]
        prob_upper = [(rand(rng, N, 3, 81) .+ N(1)) ./ N(3) for _ in 1:4]

        ambs = ntuple(
            i -> IntervalAmbiguitySets(; lower = prob_lower[i], upper = prob_upper[i]),
            4,
        )

        margs = ntuple(i -> Marginal(ambs[i], (1, 2, 3, 4), (1,), (3, 3, 3, 3), (1,)), 4)

        mdp = FactoredRobustMarkovDecisionProcess((3, 3, 3, 3), (1,), margs)
        cuda_mdp = IntervalMDP.cu(mdp)

        prop = FiniteTimeReachability([(3, 3, 3, 3)], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        prob = VerificationProblem(mdp, spec)
        cuda_prob = VerificationProblem(cuda_mdp, spec)

        V, it, res = solve(prob, alg)
        V_cuda, it_cuda, res_cuda = solve(cuda_prob, alg)

        @test V ≈ IntervalMDP.cpu(V_cuda)
        @test it == it_cuda
        @test res ≈ IntervalMDP.cpu(res_cuda)
    end

    # 5-D rand
    @testset "5D rand" begin
        rng = MersenneTwister(995)

        prob_lower = [rand(rng, N, 5, 3125) ./ N(5) for _ in 1:5]
        prob_upper = [(rand(rng, N, 5, 3125) .+ N(1)) ./ N(5) for _ in 1:5]

        ambs = ntuple(
            i -> IntervalAmbiguitySets(; lower = prob_lower[i], upper = prob_upper[i]),
            5,
        )

        margs =
            ntuple(i -> Marginal(ambs[i], (1, 2, 3, 4, 5), (1,), (5, 5, 5, 5, 5), (1,)), 5)

        mdp = FactoredRobustMarkovDecisionProcess((5, 5, 5, 5, 5), (1,), margs)
        cuda_mdp = IntervalMDP.cu(mdp)

        prop = FiniteTimeReachability([(5, 5, 5, 5, 5)], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        prob = VerificationProblem(mdp, spec)
        cuda_prob = VerificationProblem(cuda_mdp, spec)

        V, it, res = solve(prob, alg)
        V_cuda, it_cuda, res_cuda = solve(cuda_prob, alg)

        @test V ≈ IntervalMDP.cpu(V_cuda)
        @test it == it_cuda
        @test res ≈ IntervalMDP.cpu(res_cuda)
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
            rand(rng, N, num_states_per_axis, num_choices) ./ num_states_per_axis for
            _ in 1:num_axis
        ]
        prob_upper = [
            (rand(rng, N, num_states_per_axis, num_choices) .+ N(1)) ./ num_states_per_axis for _ in 1:num_axis
        ]

        ambiguity_sets = ntuple(
            i -> IntervalAmbiguitySets(; lower = prob_lower[i], upper = prob_upper[i]),
            num_axis,
        )

        marginals = ntuple(
            i -> Marginal(
                ambiguity_sets[i],
                state_indices,
                action_indices,
                state_vars,
                action_vars,
            ),
            num_axis,
        )

        mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)
        cuda_mdp = IntervalMDP.cu(mdp)

        prop = FiniteTimeReachability(
            [(num_states_per_axis, num_states_per_axis, num_states_per_axis)],
            10,
        )
        spec = Specification(prop, Pessimistic, Maximize)
        prob = ControlSynthesisProblem(mdp, spec)
        cuda_prob = ControlSynthesisProblem(cuda_mdp, spec)

        policy, V, it, res = solve(prob, alg)
        cuda_policy, V_cuda, it_cuda, res_cuda = solve(cuda_prob, alg)
        @test all(
            p -> all(splat(isequal), zip(p...)),
            zip(policy.strategy, IntervalMDP.cpu(cuda_policy).strategy),
        )
        @test V ≈ IntervalMDP.cpu(V_cuda)

        # Check if the value iteration for the IMDP with the policy applied is the same as the value iteration for the original IMDP
        cuda_prob = VerificationProblem(cuda_mdp, spec, cuda_policy)
        V_mc, k, res = solve(cuda_prob, alg)
        @test V_cuda ≈ V_mc
    end
end
