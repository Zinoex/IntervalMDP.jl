@testitem "invalid factored MDPs" tags = [:base, :invalid_factored_mdps] begin
    using IntervalMDP
    using Random: MersenneTwister
    @testset for N in [Float32, Float64]
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 0 1 // 15 3 // 10 0 1 // 30 1 // 3 0 7 // 30 4 // 15 0 1 // 6 1 // 5 0 1 // 10 1 // 5 0 0 7 // 30 0 7 // 30 1 // 5 0 2 // 15 1 // 6 0;
                                1 // 5 4 // 15 0 1 // 10 1 // 5 0 3 // 10 3 // 10 0 1 // 10 1 // 15 0 3 // 10 3 // 10 0 7 // 30 1 // 5 0 1 // 10 1 // 5 0 1 // 5 1 // 30 0 1 // 5 3 // 10 0;
                                4 // 15 1 // 30 1 1 // 5 1 // 5 1 7 // 30 4 // 15 1 2 // 15 7 // 30 1 1 // 5 1 // 3 1 2 // 15 1 // 6 1 1 // 6 1 // 3 1 4 // 15 3 // 10 1 1 // 30 3 // 10 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 0 13 // 30 3 // 5 0 17 // 30 17 // 30 0 17 // 30 13 // 30 0 3 // 5 2 // 3 0 11 // 30 7 // 15 0 0 1 // 2 0 17 // 30 13 // 30 0 7 // 15 13 // 30 0;
                                8 // 15 1 // 2 0 3 // 5 7 // 15 0 8 // 15 17 // 30 0 2 // 3 17 // 30 0 11 // 30 7 // 15 0 19 // 30 19 // 30 0 13 // 15 1 // 2 0 17 // 30 13 // 30 0 3 // 5 11 // 30 0;
                                11 // 30 1 // 3 1 2 // 5 8 // 15 1 7 // 15 3 // 5 1 2 // 3 17 // 30 1 2 // 3 8 // 15 1 2 // 15 3 // 5 1 2 // 3 3 // 5 1 17 // 30 2 // 3 1 7 // 15 8 // 15 1
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
                                1 // 10 1 // 15 1 3 // 10 0 0 1 // 6 1 // 15 0 1 // 15 1 // 6 1 1 // 6 1 // 30 0 1 // 10 1 // 10 0 1 // 3 2 // 15 1 3 // 10 4 // 15 0 2 // 15 2 // 15 0;
                                3 // 10 1 // 5 0 3 // 10 2 // 15 1 0 1 // 30 0 0 1 // 15 0 1 // 30 7 // 30 1 1 // 30 1 // 15 0 7 // 30 1 // 15 0 1 // 6 1 // 30 1 1 // 10 1 // 15 0;
                                3 // 10 4 // 15 0 1 // 10 3 // 10 0 2 // 15 1 // 3 1 3 // 10 1 // 10 0 1 // 6 3 // 10 0 7 // 30 1 // 6 1 1 // 15 1 // 15 0 1 // 10 1 // 5 0 1 // 5 4 // 15 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 1 3 // 5 11 // 30 0 3 // 5 7 // 15 0 19 // 30 2 // 5 1 3 // 5 2 // 3 0 2 // 3 8 // 15 0 8 // 15 19 // 30 1 8 // 15 8 // 15 0 13 // 30 13 // 30 0;
                                1 // 3 13 // 30 0 11 // 30 2 // 5 1 2 // 3 2 // 3 0 0 13 // 30 0 1 // 2 17 // 30 1 17 // 30 1 // 3 0 2 // 5 1 // 3 0 13 // 30 11 // 30 1 8 // 15 1 // 3 0;
                                17 // 30 3 // 5 0 8 // 15 1 // 2 0 7 // 15 1 // 2 1 2 // 3 17 // 30 0 11 // 30 2 // 5 0 1 // 2 7 // 15 1 2 // 5 17 // 30 0 11 // 30 2 // 5 0 11 // 30 2 // 3 1
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
                                4 // 15 1 // 5 1 3 // 10 3 // 10 1 4 // 15 7 // 30 1 1 // 5 4 // 15 0 7 // 30 1 // 6 0 1 // 5 0 0 1 // 15 1 // 30 0 3 // 10 1 // 3 0 2 // 15 1 // 15 0;
                                2 // 15 4 // 15 0 1 // 10 1 // 30 0 7 // 30 2 // 15 0 1 // 15 1 // 30 1 3 // 10 1 // 3 1 1 // 5 1 // 10 1 2 // 15 1 // 30 0 2 // 15 4 // 15 0 0 4 // 15 0;
                                1 // 5 1 // 3 0 3 // 10 1 // 10 0 1 // 15 1 // 10 0 1 // 30 1 // 5 0 2 // 15 7 // 30 0 1 // 3 2 // 15 0 1 // 10 1 // 6 1 3 // 10 1 // 5 1 7 // 30 1 // 30 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 1 // 2 3 // 5 1 19 // 30 2 // 5 1 8 // 15 1 // 3 0 11 // 30 2 // 5 0 17 // 30 13 // 30 0 2 // 5 3 // 5 0 3 // 5 11 // 30 0 1 // 2 11 // 30 0;
                                3 // 5 2 // 3 0 13 // 30 19 // 30 0 1 // 3 2 // 5 0 17 // 30 7 // 15 1 11 // 30 3 // 5 1 19 // 30 7 // 15 1 2 // 5 8 // 15 0 17 // 30 11 // 30 0 19 // 30 13 // 30 0;
                                3 // 5 2 // 3 0 1 // 2 1 // 2 0 2 // 3 7 // 15 0 3 // 5 3 // 5 0 1 // 2 1 // 3 0 2 // 5 8 // 15 0 2 // 5 11 // 30 1 1 // 3 8 // 15 1 7 // 15 13 // 30 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(3, i, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 1 0 0 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 1 0 0 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 0 1 0 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 0 1 0 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 0 0 1 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 0 0 1 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 1 0 0 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 1 0 0 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 0 1 0 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 0 1 0 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 0 0 1 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 0 0 1 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 0 0 0 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 0 0 0 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 0 0 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 0 0 0 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 1 1 1 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 1 1 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 0 0 0 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 0 0 0 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 0 0 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 0 0 0 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 1 1 1 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 1 1 1 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 1 1 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 1 1 1 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 0 0 0 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 0 0 0 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 1 1 1 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 0 0 0 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 0 0 0 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 1 1 1 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 0 0 0 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 0 0 0 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, 3, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0 1 0 0 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0 0 1 0 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1 0 0 1 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0 1 0 0 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0 0 1 0 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1 0 0 1 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 1 1 1 0 0 0 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0 1 1 1 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 0 0 0 0 0 0 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 1 1 1 0 0 0 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0 1 1 1 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 0 0 0 0 0 0 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0 0 0 0 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0 0 0 0 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1 1 1 1 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1 1 1 1 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, j, 3) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
                    @test V ≈ V_implicit
                    @test k == k_implicit
                    @test res ≈ res_implicit
                end
            end
            @testset "4D abstraction" begin
                rng = MersenneTwister(995)
                prob_lower = [rand(rng, N, 3, 81) ./ N(3) for _ in 1:4]
                prob_upper = [(rand(rng, N, 3, 81) .+ N(1)) ./ N(3) for _ in 1:4]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    4,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(ambiguity_sets[i], (1, 2, 3, 4), (1,), (3, 3, 3, 3), (1,))
                        end
                    ),
                    4,
                )
                mdp = FactoredRobustMarkovDecisionProcess((3, 3, 3, 3), (1,), marginals)
                prop = FiniteTimeReachability([(3, 3, 3, 3)], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                (V_ortho, it_ortho, res_ortho) = solve(prob, alg)
                @test V_ortho[3, 3, 3, 3] ≈ one(N)
                @test all(V_ortho .>= zero(N))
                @test all(V_ortho .<= one(N))
                if !(bellman_algorithm(alg) isa VertexEnumeration)
                    prob_lower_simple = zeros(N, 81, 81)
                    prob_upper_simple = zeros(N, 81, 81)
                    lin = LinearIndices((3, 3, 3, 3))
                    act_idx = CartesianIndex(1)
                    for I in CartesianIndices((3, 3, 3, 3))
                        for J in CartesianIndices((3, 3, 3, 3))
                            marginal_ambiguity_sets = map((marginal->begin
                                marginal[act_idx, I]
                            end), marginals)
                            prob_lower_simple[lin[J], lin[I]] =
                                prod((lower(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
                            prob_upper_simple[lin[J], lin[I]] =
                                prod((upper(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
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
                    (V_direct, it_direct, res_direct) = solve(prob, alg)
                    @test V_direct[81] ≈ one(N)
                    @test all(V_ortho .≥ reshape(V_direct, 3, 3, 3, 3))
                end
            end
            @testset "synthesis" begin
                rng = MersenneTwister(3286)
                num_states_per_axis = 3
                num_axis = 3
                num_states = num_states_per_axis ^ num_axis
                num_actions = 2
                num_choices = num_states * num_actions
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = ntuple((_->begin
                    num_states_per_axis
                end), num_axis)
                action_vars = (num_actions,)
                prob_lower = [
                    rand(rng, N, num_states_per_axis, num_choices) ./ num_states_per_axis for _ in 1:num_axis
                ]
                prob_upper = [
                    (rand(rng, N, num_states_per_axis, num_choices) .+ N(1)) ./
                    num_states_per_axis for _ in 1:num_axis
                ]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    num_axis,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(
                                ambiguity_sets[i],
                                state_indices,
                                action_indices,
                                state_vars,
                                action_vars,
                            )
                        end
                    ),
                    num_axis,
                )
                mdp =
                    FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)
                prop = FiniteTimeReachability(
                    [(num_states_per_axis, num_states_per_axis, num_states_per_axis)],
                    10,
                )
                spec = Specification(prop, Pessimistic, Maximize)
                prob = ControlSynthesisProblem(mdp, spec)
                (policy, V, it, res) = solve(prob, alg)
                @test it == 10
                @test all(V .≥ 0.0)
                prob = VerificationProblem(mdp, spec, policy)
                (V_mc, k, res) = solve(prob, alg)
                @test V ≈ V_mc
            end
        end
        @testset "invalid factored MDPs" begin
            N = Float64
            state_indices = (1, 2, 3)
            action_indices = (1,)
            state_vars = (3, 3, 3)
            source_dims = (2, 3, 3)
            action_vars = (1,)
            marginal1 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                        1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                        4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                    ],
                    upper = N[
                        7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                        8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                        11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                        1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                        3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                        3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                    ],
                    upper = N[
                        2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                        1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                        17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                        4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                        2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                        1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                    ],
                    upper = N[
                        3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                        3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                        3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
                    ],
                ),
                state_indices,
                action_indices,
                source_dims,
                action_vars,
            )
            @test_throws ArgumentError FactoredRobustMarkovDecisionProcess(
                (-3, 3, 3),
                action_vars,
                source_dims,
                (marginal1, marginal2, marginal3),
            )
            @test_throws ArgumentError FactoredRobustMarkovDecisionProcess(
                state_vars,
                action_vars,
                (-2, 3, 3),
                (marginal1, marginal2, marginal3),
            )
            @test_throws ArgumentError FactoredRobustMarkovDecisionProcess(
                state_vars,
                action_vars,
                (2, 4, 3),
                (marginal1, marginal2, marginal3),
            )
            @test_throws ArgumentError FactoredRobustMarkovDecisionProcess(
                state_vars,
                (-1,),
                source_dims,
                (marginal1, marginal2, marginal3),
            )
            @test_throws DimensionMismatch FactoredRobustMarkovDecisionProcess(
                (2, 3, 3),
                action_vars,
                source_dims,
                (marginal1, marginal2, marginal3),
            )
            malformed_marginal3 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0;
                        2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10;
                        1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15
                    ],
                    upper = N[
                        3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30;
                        3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15;
                        3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15
                    ],
                ),
                state_indices,
                action_indices,
                (2, 2, 3),
                action_vars,
            )
            @test_throws DimensionMismatch FactoredRobustMarkovDecisionProcess(
                state_vars,
                action_vars,
                source_dims,
                (marginal1, marginal2, malformed_marginal3),
            )
            malformed_marginal3 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                        2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                        1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                    ],
                    upper = N[
                        3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                        3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                        3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
                    ],
                ),
                state_indices,
                action_indices,
                source_dims,
                (2,),
            )
            @test_throws DimensionMismatch FactoredRobustMarkovDecisionProcess(
                state_vars,
                action_vars,
                source_dims,
                (marginal1, marginal2, malformed_marginal3),
            )
            @test_throws DimensionMismatch FactoredRobustMarkovDecisionProcess(
                state_vars,
                action_vars,
                source_dims,
                (marginal1, marginal2, marginal3),
                [CartesianIndex(1, 2)],
            )
            @test_throws DimensionMismatch FactoredRobustMarkovDecisionProcess(
                state_vars,
                action_vars,
                source_dims,
                (marginal1, marginal2, marginal3),
                [CartesianIndex(1, 2, 3, 4)],
            )
            @test_throws DimensionMismatch FactoredRobustMarkovDecisionProcess(
                state_vars,
                action_vars,
                source_dims,
                (marginal1, marginal2, marginal3),
                [CartesianIndex(1, -2, 3)],
            )
            @test_throws DimensionMismatch FactoredRobustMarkovDecisionProcess(
                state_vars,
                action_vars,
                source_dims,
                (marginal1, marginal2, marginal3),
                [CartesianIndex(1, 4, 3)],
            )
        end
    end
end

@testitem "non-interval" tags = [:base, :non_interval] begin
    using IntervalMDP
    using Random: MersenneTwister
    @testset for N in [Float32, Float64]
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 0 1 // 15 3 // 10 0 1 // 30 1 // 3 0 7 // 30 4 // 15 0 1 // 6 1 // 5 0 1 // 10 1 // 5 0 0 7 // 30 0 7 // 30 1 // 5 0 2 // 15 1 // 6 0;
                                1 // 5 4 // 15 0 1 // 10 1 // 5 0 3 // 10 3 // 10 0 1 // 10 1 // 15 0 3 // 10 3 // 10 0 7 // 30 1 // 5 0 1 // 10 1 // 5 0 1 // 5 1 // 30 0 1 // 5 3 // 10 0;
                                4 // 15 1 // 30 1 1 // 5 1 // 5 1 7 // 30 4 // 15 1 2 // 15 7 // 30 1 1 // 5 1 // 3 1 2 // 15 1 // 6 1 1 // 6 1 // 3 1 4 // 15 3 // 10 1 1 // 30 3 // 10 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 0 13 // 30 3 // 5 0 17 // 30 17 // 30 0 17 // 30 13 // 30 0 3 // 5 2 // 3 0 11 // 30 7 // 15 0 0 1 // 2 0 17 // 30 13 // 30 0 7 // 15 13 // 30 0;
                                8 // 15 1 // 2 0 3 // 5 7 // 15 0 8 // 15 17 // 30 0 2 // 3 17 // 30 0 11 // 30 7 // 15 0 19 // 30 19 // 30 0 13 // 15 1 // 2 0 17 // 30 13 // 30 0 3 // 5 11 // 30 0;
                                11 // 30 1 // 3 1 2 // 5 8 // 15 1 7 // 15 3 // 5 1 2 // 3 17 // 30 1 2 // 3 8 // 15 1 2 // 15 3 // 5 1 2 // 3 3 // 5 1 17 // 30 2 // 3 1 7 // 15 8 // 15 1
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
                                1 // 10 1 // 15 1 3 // 10 0 0 1 // 6 1 // 15 0 1 // 15 1 // 6 1 1 // 6 1 // 30 0 1 // 10 1 // 10 0 1 // 3 2 // 15 1 3 // 10 4 // 15 0 2 // 15 2 // 15 0;
                                3 // 10 1 // 5 0 3 // 10 2 // 15 1 0 1 // 30 0 0 1 // 15 0 1 // 30 7 // 30 1 1 // 30 1 // 15 0 7 // 30 1 // 15 0 1 // 6 1 // 30 1 1 // 10 1 // 15 0;
                                3 // 10 4 // 15 0 1 // 10 3 // 10 0 2 // 15 1 // 3 1 3 // 10 1 // 10 0 1 // 6 3 // 10 0 7 // 30 1 // 6 1 1 // 15 1 // 15 0 1 // 10 1 // 5 0 1 // 5 4 // 15 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 1 3 // 5 11 // 30 0 3 // 5 7 // 15 0 19 // 30 2 // 5 1 3 // 5 2 // 3 0 2 // 3 8 // 15 0 8 // 15 19 // 30 1 8 // 15 8 // 15 0 13 // 30 13 // 30 0;
                                1 // 3 13 // 30 0 11 // 30 2 // 5 1 2 // 3 2 // 3 0 0 13 // 30 0 1 // 2 17 // 30 1 17 // 30 1 // 3 0 2 // 5 1 // 3 0 13 // 30 11 // 30 1 8 // 15 1 // 3 0;
                                17 // 30 3 // 5 0 8 // 15 1 // 2 0 7 // 15 1 // 2 1 2 // 3 17 // 30 0 11 // 30 2 // 5 0 1 // 2 7 // 15 1 2 // 5 17 // 30 0 11 // 30 2 // 5 0 11 // 30 2 // 3 1
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
                                4 // 15 1 // 5 1 3 // 10 3 // 10 1 4 // 15 7 // 30 1 1 // 5 4 // 15 0 7 // 30 1 // 6 0 1 // 5 0 0 1 // 15 1 // 30 0 3 // 10 1 // 3 0 2 // 15 1 // 15 0;
                                2 // 15 4 // 15 0 1 // 10 1 // 30 0 7 // 30 2 // 15 0 1 // 15 1 // 30 1 3 // 10 1 // 3 1 1 // 5 1 // 10 1 2 // 15 1 // 30 0 2 // 15 4 // 15 0 0 4 // 15 0;
                                1 // 5 1 // 3 0 3 // 10 1 // 10 0 1 // 15 1 // 10 0 1 // 30 1 // 5 0 2 // 15 7 // 30 0 1 // 3 2 // 15 0 1 // 10 1 // 6 1 3 // 10 1 // 5 1 7 // 30 1 // 30 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 1 // 2 3 // 5 1 19 // 30 2 // 5 1 8 // 15 1 // 3 0 11 // 30 2 // 5 0 17 // 30 13 // 30 0 2 // 5 3 // 5 0 3 // 5 11 // 30 0 1 // 2 11 // 30 0;
                                3 // 5 2 // 3 0 13 // 30 19 // 30 0 1 // 3 2 // 5 0 17 // 30 7 // 15 1 11 // 30 3 // 5 1 19 // 30 7 // 15 1 2 // 5 8 // 15 0 17 // 30 11 // 30 0 19 // 30 13 // 30 0;
                                3 // 5 2 // 3 0 1 // 2 1 // 2 0 2 // 3 7 // 15 0 3 // 5 3 // 5 0 1 // 2 1 // 3 0 2 // 5 8 // 15 0 2 // 5 11 // 30 1 1 // 3 8 // 15 1 7 // 15 13 // 30 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(3, i, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 1 0 0 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 1 0 0 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 0 1 0 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 0 1 0 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 0 0 1 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 0 0 1 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 1 0 0 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 1 0 0 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 0 1 0 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 0 1 0 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 0 0 1 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 0 0 1 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 0 0 0 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 0 0 0 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 0 0 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 0 0 0 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 1 1 1 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 1 1 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 0 0 0 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 0 0 0 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 0 0 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 0 0 0 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 1 1 1 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 1 1 1 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 1 1 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 1 1 1 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 0 0 0 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 0 0 0 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 1 1 1 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 0 0 0 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 0 0 0 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 1 1 1 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 0 0 0 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 0 0 0 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, 3, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0 1 0 0 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0 0 1 0 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1 0 0 1 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0 1 0 0 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0 0 1 0 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1 0 0 1 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 1 1 1 0 0 0 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0 1 1 1 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 0 0 0 0 0 0 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 1 1 1 0 0 0 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0 1 1 1 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 0 0 0 0 0 0 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0 0 0 0 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0 0 0 0 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1 1 1 1 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1 1 1 1 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, j, 3) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
                    @test V ≈ V_implicit
                    @test k == k_implicit
                    @test res ≈ res_implicit
                end
            end
            @testset "4D abstraction" begin
                rng = MersenneTwister(995)
                prob_lower = [rand(rng, N, 3, 81) ./ N(3) for _ in 1:4]
                prob_upper = [(rand(rng, N, 3, 81) .+ N(1)) ./ N(3) for _ in 1:4]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    4,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(ambiguity_sets[i], (1, 2, 3, 4), (1,), (3, 3, 3, 3), (1,))
                        end
                    ),
                    4,
                )
                mdp = FactoredRobustMarkovDecisionProcess((3, 3, 3, 3), (1,), marginals)
                prop = FiniteTimeReachability([(3, 3, 3, 3)], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                (V_ortho, it_ortho, res_ortho) = solve(prob, alg)
                @test V_ortho[3, 3, 3, 3] ≈ one(N)
                @test all(V_ortho .>= zero(N))
                @test all(V_ortho .<= one(N))
                if !(bellman_algorithm(alg) isa VertexEnumeration)
                    prob_lower_simple = zeros(N, 81, 81)
                    prob_upper_simple = zeros(N, 81, 81)
                    lin = LinearIndices((3, 3, 3, 3))
                    act_idx = CartesianIndex(1)
                    for I in CartesianIndices((3, 3, 3, 3))
                        for J in CartesianIndices((3, 3, 3, 3))
                            marginal_ambiguity_sets = map((marginal->begin
                                marginal[act_idx, I]
                            end), marginals)
                            prob_lower_simple[lin[J], lin[I]] =
                                prod((lower(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
                            prob_upper_simple[lin[J], lin[I]] =
                                prod((upper(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
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
                    (V_direct, it_direct, res_direct) = solve(prob, alg)
                    @test V_direct[81] ≈ one(N)
                    @test all(V_ortho .≥ reshape(V_direct, 3, 3, 3, 3))
                end
            end
            @testset "synthesis" begin
                rng = MersenneTwister(3286)
                num_states_per_axis = 3
                num_axis = 3
                num_states = num_states_per_axis ^ num_axis
                num_actions = 2
                num_choices = num_states * num_actions
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = ntuple((_->begin
                    num_states_per_axis
                end), num_axis)
                action_vars = (num_actions,)
                prob_lower = [
                    rand(rng, N, num_states_per_axis, num_choices) ./ num_states_per_axis for _ in 1:num_axis
                ]
                prob_upper = [
                    (rand(rng, N, num_states_per_axis, num_choices) .+ N(1)) ./
                    num_states_per_axis for _ in 1:num_axis
                ]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    num_axis,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(
                                ambiguity_sets[i],
                                state_indices,
                                action_indices,
                                state_vars,
                                action_vars,
                            )
                        end
                    ),
                    num_axis,
                )
                mdp =
                    FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)
                prop = FiniteTimeReachability(
                    [(num_states_per_axis, num_states_per_axis, num_states_per_axis)],
                    10,
                )
                spec = Specification(prop, Pessimistic, Maximize)
                prob = ControlSynthesisProblem(mdp, spec)
                (policy, V, it, res) = solve(prob, alg)
                @test it == 10
                @test all(V .≥ 0.0)
                prob = VerificationProblem(mdp, spec, policy)
                (V_mc, k, res) = solve(prob, alg)
                @test V ≈ V_mc
            end
        end
        @testset "non-interval" begin
            N = Float64
            struct SingletonAmbiguitySets <: IntervalMDP.AbstractAmbiguitySets
                transition_matrix::Matrix{N}
            end
            IntervalMDP.num_target(p::SingletonAmbiguitySets) = begin
                size(p.transition_matrix, 1)
            end
            IntervalMDP.num_sets(p::SingletonAmbiguitySets) = begin
                size(p.transition_matrix, 2)
            end
            IntervalMDP.showambiguitysets(
                io::IO,
                prefix::AbstractString,
                p::SingletonAmbiguitySets,
            ) = begin
                println(
                    io,
                    prefix,
                    "SingletonAmbiguitySets with Storage type: Matrix{$(eltype(p.transition_matrix))}, Number of target states: $(size(p.transition_matrix, 1)), Number of ambiguity sets: $(size(p.transition_matrix, 2))",
                )
            end
            @testset "1d" begin
                marginal = Marginal(
                    SingletonAmbiguitySets(N[0.2 0.8; 0.5 0.5]),
                    (1,),
                    (1,),
                    (2,),
                    (1,),
                )
                mc = FactoredRobustMarkovDecisionProcess((2,), (1,), (marginal,))
                io = IOBuffer()
                show(io, MIME("text/plain"), mc)
                str = String(take!(io))
                @test occursin("Model type: Robust MDP", str)
                @test occursin(
                    "Default model checking algorithm: Robust Value Iteration",
                    str,
                )
                @test occursin("Default Bellman operator algorithm: None", str)
            end
            @testset "2d" begin
                marginal1 = Marginal(
                    SingletonAmbiguitySets(N[0.2 0.8; 0.5 0.5]),
                    (1,),
                    (1,),
                    (2,),
                    (1,),
                )
                marginal2 = Marginal(
                    SingletonAmbiguitySets(N[0.2 0.8; 0.5 0.5]),
                    (2,),
                    (1,),
                    (2,),
                    (1,),
                )
                mc = FactoredRobustMarkovDecisionProcess(
                    (2, 2),
                    (1,),
                    (marginal1, marginal2),
                )
                io = IOBuffer()
                show(io, MIME("text/plain"), mc)
                str = String(take!(io))
                @test occursin("Model type: Factored Robust MDP", str)
                @test occursin(
                    "Default model checking algorithm: Robust Value Iteration",
                    str,
                )
                @test occursin("Default Bellman operator algorithm: None", str)
            end
        end
    end
end

@testitem "show 1d" tags = [:base, :show_1d] begin
    using IntervalMDP
    using Random: MersenneTwister
    @testset for N in [Float32, Float64]
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 0 1 // 15 3 // 10 0 1 // 30 1 // 3 0 7 // 30 4 // 15 0 1 // 6 1 // 5 0 1 // 10 1 // 5 0 0 7 // 30 0 7 // 30 1 // 5 0 2 // 15 1 // 6 0;
                                1 // 5 4 // 15 0 1 // 10 1 // 5 0 3 // 10 3 // 10 0 1 // 10 1 // 15 0 3 // 10 3 // 10 0 7 // 30 1 // 5 0 1 // 10 1 // 5 0 1 // 5 1 // 30 0 1 // 5 3 // 10 0;
                                4 // 15 1 // 30 1 1 // 5 1 // 5 1 7 // 30 4 // 15 1 2 // 15 7 // 30 1 1 // 5 1 // 3 1 2 // 15 1 // 6 1 1 // 6 1 // 3 1 4 // 15 3 // 10 1 1 // 30 3 // 10 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 0 13 // 30 3 // 5 0 17 // 30 17 // 30 0 17 // 30 13 // 30 0 3 // 5 2 // 3 0 11 // 30 7 // 15 0 0 1 // 2 0 17 // 30 13 // 30 0 7 // 15 13 // 30 0;
                                8 // 15 1 // 2 0 3 // 5 7 // 15 0 8 // 15 17 // 30 0 2 // 3 17 // 30 0 11 // 30 7 // 15 0 19 // 30 19 // 30 0 13 // 15 1 // 2 0 17 // 30 13 // 30 0 3 // 5 11 // 30 0;
                                11 // 30 1 // 3 1 2 // 5 8 // 15 1 7 // 15 3 // 5 1 2 // 3 17 // 30 1 2 // 3 8 // 15 1 2 // 15 3 // 5 1 2 // 3 3 // 5 1 17 // 30 2 // 3 1 7 // 15 8 // 15 1
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
                                1 // 10 1 // 15 1 3 // 10 0 0 1 // 6 1 // 15 0 1 // 15 1 // 6 1 1 // 6 1 // 30 0 1 // 10 1 // 10 0 1 // 3 2 // 15 1 3 // 10 4 // 15 0 2 // 15 2 // 15 0;
                                3 // 10 1 // 5 0 3 // 10 2 // 15 1 0 1 // 30 0 0 1 // 15 0 1 // 30 7 // 30 1 1 // 30 1 // 15 0 7 // 30 1 // 15 0 1 // 6 1 // 30 1 1 // 10 1 // 15 0;
                                3 // 10 4 // 15 0 1 // 10 3 // 10 0 2 // 15 1 // 3 1 3 // 10 1 // 10 0 1 // 6 3 // 10 0 7 // 30 1 // 6 1 1 // 15 1 // 15 0 1 // 10 1 // 5 0 1 // 5 4 // 15 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 1 3 // 5 11 // 30 0 3 // 5 7 // 15 0 19 // 30 2 // 5 1 3 // 5 2 // 3 0 2 // 3 8 // 15 0 8 // 15 19 // 30 1 8 // 15 8 // 15 0 13 // 30 13 // 30 0;
                                1 // 3 13 // 30 0 11 // 30 2 // 5 1 2 // 3 2 // 3 0 0 13 // 30 0 1 // 2 17 // 30 1 17 // 30 1 // 3 0 2 // 5 1 // 3 0 13 // 30 11 // 30 1 8 // 15 1 // 3 0;
                                17 // 30 3 // 5 0 8 // 15 1 // 2 0 7 // 15 1 // 2 1 2 // 3 17 // 30 0 11 // 30 2 // 5 0 1 // 2 7 // 15 1 2 // 5 17 // 30 0 11 // 30 2 // 5 0 11 // 30 2 // 3 1
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
                                4 // 15 1 // 5 1 3 // 10 3 // 10 1 4 // 15 7 // 30 1 1 // 5 4 // 15 0 7 // 30 1 // 6 0 1 // 5 0 0 1 // 15 1 // 30 0 3 // 10 1 // 3 0 2 // 15 1 // 15 0;
                                2 // 15 4 // 15 0 1 // 10 1 // 30 0 7 // 30 2 // 15 0 1 // 15 1 // 30 1 3 // 10 1 // 3 1 1 // 5 1 // 10 1 2 // 15 1 // 30 0 2 // 15 4 // 15 0 0 4 // 15 0;
                                1 // 5 1 // 3 0 3 // 10 1 // 10 0 1 // 15 1 // 10 0 1 // 30 1 // 5 0 2 // 15 7 // 30 0 1 // 3 2 // 15 0 1 // 10 1 // 6 1 3 // 10 1 // 5 1 7 // 30 1 // 30 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 1 // 2 3 // 5 1 19 // 30 2 // 5 1 8 // 15 1 // 3 0 11 // 30 2 // 5 0 17 // 30 13 // 30 0 2 // 5 3 // 5 0 3 // 5 11 // 30 0 1 // 2 11 // 30 0;
                                3 // 5 2 // 3 0 13 // 30 19 // 30 0 1 // 3 2 // 5 0 17 // 30 7 // 15 1 11 // 30 3 // 5 1 19 // 30 7 // 15 1 2 // 5 8 // 15 0 17 // 30 11 // 30 0 19 // 30 13 // 30 0;
                                3 // 5 2 // 3 0 1 // 2 1 // 2 0 2 // 3 7 // 15 0 3 // 5 3 // 5 0 1 // 2 1 // 3 0 2 // 5 8 // 15 0 2 // 5 11 // 30 1 1 // 3 8 // 15 1 7 // 15 13 // 30 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(3, i, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 1 0 0 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 1 0 0 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 0 1 0 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 0 1 0 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 0 0 1 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 0 0 1 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 1 0 0 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 1 0 0 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 0 1 0 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 0 1 0 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 0 0 1 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 0 0 1 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 0 0 0 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 0 0 0 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 0 0 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 0 0 0 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 1 1 1 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 1 1 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 0 0 0 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 0 0 0 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 0 0 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 0 0 0 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 1 1 1 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 1 1 1 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 1 1 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 1 1 1 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 0 0 0 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 0 0 0 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 1 1 1 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 0 0 0 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 0 0 0 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 1 1 1 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 0 0 0 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 0 0 0 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, 3, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0 1 0 0 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0 0 1 0 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1 0 0 1 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0 1 0 0 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0 0 1 0 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1 0 0 1 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 1 1 1 0 0 0 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0 1 1 1 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 0 0 0 0 0 0 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 1 1 1 0 0 0 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0 1 1 1 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 0 0 0 0 0 0 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0 0 0 0 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0 0 0 0 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1 1 1 1 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1 1 1 1 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, j, 3) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
                    @test V ≈ V_implicit
                    @test k == k_implicit
                    @test res ≈ res_implicit
                end
            end
            @testset "4D abstraction" begin
                rng = MersenneTwister(995)
                prob_lower = [rand(rng, N, 3, 81) ./ N(3) for _ in 1:4]
                prob_upper = [(rand(rng, N, 3, 81) .+ N(1)) ./ N(3) for _ in 1:4]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    4,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(ambiguity_sets[i], (1, 2, 3, 4), (1,), (3, 3, 3, 3), (1,))
                        end
                    ),
                    4,
                )
                mdp = FactoredRobustMarkovDecisionProcess((3, 3, 3, 3), (1,), marginals)
                prop = FiniteTimeReachability([(3, 3, 3, 3)], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                (V_ortho, it_ortho, res_ortho) = solve(prob, alg)
                @test V_ortho[3, 3, 3, 3] ≈ one(N)
                @test all(V_ortho .>= zero(N))
                @test all(V_ortho .<= one(N))
                if !(bellman_algorithm(alg) isa VertexEnumeration)
                    prob_lower_simple = zeros(N, 81, 81)
                    prob_upper_simple = zeros(N, 81, 81)
                    lin = LinearIndices((3, 3, 3, 3))
                    act_idx = CartesianIndex(1)
                    for I in CartesianIndices((3, 3, 3, 3))
                        for J in CartesianIndices((3, 3, 3, 3))
                            marginal_ambiguity_sets = map((marginal->begin
                                marginal[act_idx, I]
                            end), marginals)
                            prob_lower_simple[lin[J], lin[I]] =
                                prod((lower(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
                            prob_upper_simple[lin[J], lin[I]] =
                                prod((upper(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
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
                    (V_direct, it_direct, res_direct) = solve(prob, alg)
                    @test V_direct[81] ≈ one(N)
                    @test all(V_ortho .≥ reshape(V_direct, 3, 3, 3, 3))
                end
            end
            @testset "synthesis" begin
                rng = MersenneTwister(3286)
                num_states_per_axis = 3
                num_axis = 3
                num_states = num_states_per_axis ^ num_axis
                num_actions = 2
                num_choices = num_states * num_actions
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = ntuple((_->begin
                    num_states_per_axis
                end), num_axis)
                action_vars = (num_actions,)
                prob_lower = [
                    rand(rng, N, num_states_per_axis, num_choices) ./ num_states_per_axis for _ in 1:num_axis
                ]
                prob_upper = [
                    (rand(rng, N, num_states_per_axis, num_choices) .+ N(1)) ./
                    num_states_per_axis for _ in 1:num_axis
                ]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    num_axis,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(
                                ambiguity_sets[i],
                                state_indices,
                                action_indices,
                                state_vars,
                                action_vars,
                            )
                        end
                    ),
                    num_axis,
                )
                mdp =
                    FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)
                prop = FiniteTimeReachability(
                    [(num_states_per_axis, num_states_per_axis, num_states_per_axis)],
                    10,
                )
                spec = Specification(prop, Pessimistic, Maximize)
                prob = ControlSynthesisProblem(mdp, spec)
                (policy, V, it, res) = solve(prob, alg)
                @test it == 10
                @test all(V .≥ 0.0)
                prob = VerificationProblem(mdp, spec, policy)
                (V_mc, k, res) = solve(prob, alg)
                @test V ≈ V_mc
            end
        end
        @testset "show 1d" begin
            N = Float64
            ambiguity_sets = IntervalAmbiguitySets(;
                lower = N[
                    0 5 // 10 2 // 10;
                    1 // 10 3 // 10 3 // 10;
                    2 // 10 1 // 10 5 // 10
                ],
                upper = N[
                    5 // 10 7 // 10 3 // 10;
                    6 // 10 5 // 10 4 // 10;
                    7 // 10 3 // 10 5 // 10
                ],
            )
            imc = IntervalMarkovChain(ambiguity_sets, [CartesianIndex(2)])
            io = IOBuffer()
            show(io, MIME("text/plain"), imc)
            str = String(take!(io))
            @test occursin("FactoredRobustMarkovDecisionProcess", str)
            @test occursin("1 state variables with cardinality: (3,)", str)
            @test occursin("1 action variables with cardinality: (1,)", str)
            @test occursin("Initial states: CartesianIndex{1}[$(CartesianIndex(2))]", str)
            @test occursin("Marginal 1:", str)
            @test occursin("Ambiguity set type: Interval (dense, Matrix{Float64})", str)
            @test !(occursin("Marginal 2:", str))
            @test occursin("Inferred properties", str)
            @test occursin("Model type: Interval MDP", str)
            @test occursin("Number of states: 3", str)
            @test occursin("Number of actions: 1", str)
            @test occursin("Default model checking algorithm: Robust Value Iteration", str)
            @test occursin("Default Bellman operator algorithm: O-Maximization", str)
        end
    end
end

@testitem "show 3d" tags = [:base, :show_3d] begin
    using IntervalMDP
    using Random: MersenneTwister
    @testset for N in [Float32, Float64]
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 0 1 // 15 3 // 10 0 1 // 30 1 // 3 0 7 // 30 4 // 15 0 1 // 6 1 // 5 0 1 // 10 1 // 5 0 0 7 // 30 0 7 // 30 1 // 5 0 2 // 15 1 // 6 0;
                                1 // 5 4 // 15 0 1 // 10 1 // 5 0 3 // 10 3 // 10 0 1 // 10 1 // 15 0 3 // 10 3 // 10 0 7 // 30 1 // 5 0 1 // 10 1 // 5 0 1 // 5 1 // 30 0 1 // 5 3 // 10 0;
                                4 // 15 1 // 30 1 1 // 5 1 // 5 1 7 // 30 4 // 15 1 2 // 15 7 // 30 1 1 // 5 1 // 3 1 2 // 15 1 // 6 1 1 // 6 1 // 3 1 4 // 15 3 // 10 1 1 // 30 3 // 10 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 0 13 // 30 3 // 5 0 17 // 30 17 // 30 0 17 // 30 13 // 30 0 3 // 5 2 // 3 0 11 // 30 7 // 15 0 0 1 // 2 0 17 // 30 13 // 30 0 7 // 15 13 // 30 0;
                                8 // 15 1 // 2 0 3 // 5 7 // 15 0 8 // 15 17 // 30 0 2 // 3 17 // 30 0 11 // 30 7 // 15 0 19 // 30 19 // 30 0 13 // 15 1 // 2 0 17 // 30 13 // 30 0 3 // 5 11 // 30 0;
                                11 // 30 1 // 3 1 2 // 5 8 // 15 1 7 // 15 3 // 5 1 2 // 3 17 // 30 1 2 // 3 8 // 15 1 2 // 15 3 // 5 1 2 // 3 3 // 5 1 17 // 30 2 // 3 1 7 // 15 8 // 15 1
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
                                1 // 10 1 // 15 1 3 // 10 0 0 1 // 6 1 // 15 0 1 // 15 1 // 6 1 1 // 6 1 // 30 0 1 // 10 1 // 10 0 1 // 3 2 // 15 1 3 // 10 4 // 15 0 2 // 15 2 // 15 0;
                                3 // 10 1 // 5 0 3 // 10 2 // 15 1 0 1 // 30 0 0 1 // 15 0 1 // 30 7 // 30 1 1 // 30 1 // 15 0 7 // 30 1 // 15 0 1 // 6 1 // 30 1 1 // 10 1 // 15 0;
                                3 // 10 4 // 15 0 1 // 10 3 // 10 0 2 // 15 1 // 3 1 3 // 10 1 // 10 0 1 // 6 3 // 10 0 7 // 30 1 // 6 1 1 // 15 1 // 15 0 1 // 10 1 // 5 0 1 // 5 4 // 15 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 1 3 // 5 11 // 30 0 3 // 5 7 // 15 0 19 // 30 2 // 5 1 3 // 5 2 // 3 0 2 // 3 8 // 15 0 8 // 15 19 // 30 1 8 // 15 8 // 15 0 13 // 30 13 // 30 0;
                                1 // 3 13 // 30 0 11 // 30 2 // 5 1 2 // 3 2 // 3 0 0 13 // 30 0 1 // 2 17 // 30 1 17 // 30 1 // 3 0 2 // 5 1 // 3 0 13 // 30 11 // 30 1 8 // 15 1 // 3 0;
                                17 // 30 3 // 5 0 8 // 15 1 // 2 0 7 // 15 1 // 2 1 2 // 3 17 // 30 0 11 // 30 2 // 5 0 1 // 2 7 // 15 1 2 // 5 17 // 30 0 11 // 30 2 // 5 0 11 // 30 2 // 3 1
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
                                4 // 15 1 // 5 1 3 // 10 3 // 10 1 4 // 15 7 // 30 1 1 // 5 4 // 15 0 7 // 30 1 // 6 0 1 // 5 0 0 1 // 15 1 // 30 0 3 // 10 1 // 3 0 2 // 15 1 // 15 0;
                                2 // 15 4 // 15 0 1 // 10 1 // 30 0 7 // 30 2 // 15 0 1 // 15 1 // 30 1 3 // 10 1 // 3 1 1 // 5 1 // 10 1 2 // 15 1 // 30 0 2 // 15 4 // 15 0 0 4 // 15 0;
                                1 // 5 1 // 3 0 3 // 10 1 // 10 0 1 // 15 1 // 10 0 1 // 30 1 // 5 0 2 // 15 7 // 30 0 1 // 3 2 // 15 0 1 // 10 1 // 6 1 3 // 10 1 // 5 1 7 // 30 1 // 30 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 1 // 2 3 // 5 1 19 // 30 2 // 5 1 8 // 15 1 // 3 0 11 // 30 2 // 5 0 17 // 30 13 // 30 0 2 // 5 3 // 5 0 3 // 5 11 // 30 0 1 // 2 11 // 30 0;
                                3 // 5 2 // 3 0 13 // 30 19 // 30 0 1 // 3 2 // 5 0 17 // 30 7 // 15 1 11 // 30 3 // 5 1 19 // 30 7 // 15 1 2 // 5 8 // 15 0 17 // 30 11 // 30 0 19 // 30 13 // 30 0;
                                3 // 5 2 // 3 0 1 // 2 1 // 2 0 2 // 3 7 // 15 0 3 // 5 3 // 5 0 1 // 2 1 // 3 0 2 // 5 8 // 15 0 2 // 5 11 // 30 1 1 // 3 8 // 15 1 7 // 15 13 // 30 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(3, i, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 1 0 0 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 1 0 0 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 0 1 0 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 0 1 0 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 0 0 1 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 0 0 1 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 1 0 0 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 1 0 0 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 0 1 0 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 0 1 0 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 0 0 1 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 0 0 1 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 0 0 0 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 0 0 0 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 0 0 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 0 0 0 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 1 1 1 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 1 1 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 0 0 0 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 0 0 0 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 0 0 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 0 0 0 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 1 1 1 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 1 1 1 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 1 1 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 1 1 1 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 0 0 0 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 0 0 0 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 1 1 1 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 0 0 0 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 0 0 0 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 1 1 1 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 0 0 0 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 0 0 0 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, 3, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0 1 0 0 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0 0 1 0 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1 0 0 1 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0 1 0 0 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0 0 1 0 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1 0 0 1 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 1 1 1 0 0 0 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0 1 1 1 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 0 0 0 0 0 0 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 1 1 1 0 0 0 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0 1 1 1 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 0 0 0 0 0 0 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0 0 0 0 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0 0 0 0 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1 1 1 1 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1 1 1 1 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, j, 3) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
                    @test V ≈ V_implicit
                    @test k == k_implicit
                    @test res ≈ res_implicit
                end
            end
            @testset "4D abstraction" begin
                rng = MersenneTwister(995)
                prob_lower = [rand(rng, N, 3, 81) ./ N(3) for _ in 1:4]
                prob_upper = [(rand(rng, N, 3, 81) .+ N(1)) ./ N(3) for _ in 1:4]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    4,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(ambiguity_sets[i], (1, 2, 3, 4), (1,), (3, 3, 3, 3), (1,))
                        end
                    ),
                    4,
                )
                mdp = FactoredRobustMarkovDecisionProcess((3, 3, 3, 3), (1,), marginals)
                prop = FiniteTimeReachability([(3, 3, 3, 3)], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                (V_ortho, it_ortho, res_ortho) = solve(prob, alg)
                @test V_ortho[3, 3, 3, 3] ≈ one(N)
                @test all(V_ortho .>= zero(N))
                @test all(V_ortho .<= one(N))
                if !(bellman_algorithm(alg) isa VertexEnumeration)
                    prob_lower_simple = zeros(N, 81, 81)
                    prob_upper_simple = zeros(N, 81, 81)
                    lin = LinearIndices((3, 3, 3, 3))
                    act_idx = CartesianIndex(1)
                    for I in CartesianIndices((3, 3, 3, 3))
                        for J in CartesianIndices((3, 3, 3, 3))
                            marginal_ambiguity_sets = map((marginal->begin
                                marginal[act_idx, I]
                            end), marginals)
                            prob_lower_simple[lin[J], lin[I]] =
                                prod((lower(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
                            prob_upper_simple[lin[J], lin[I]] =
                                prod((upper(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
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
                    (V_direct, it_direct, res_direct) = solve(prob, alg)
                    @test V_direct[81] ≈ one(N)
                    @test all(V_ortho .≥ reshape(V_direct, 3, 3, 3, 3))
                end
            end
            @testset "synthesis" begin
                rng = MersenneTwister(3286)
                num_states_per_axis = 3
                num_axis = 3
                num_states = num_states_per_axis ^ num_axis
                num_actions = 2
                num_choices = num_states * num_actions
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = ntuple((_->begin
                    num_states_per_axis
                end), num_axis)
                action_vars = (num_actions,)
                prob_lower = [
                    rand(rng, N, num_states_per_axis, num_choices) ./ num_states_per_axis for _ in 1:num_axis
                ]
                prob_upper = [
                    (rand(rng, N, num_states_per_axis, num_choices) .+ N(1)) ./
                    num_states_per_axis for _ in 1:num_axis
                ]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    num_axis,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(
                                ambiguity_sets[i],
                                state_indices,
                                action_indices,
                                state_vars,
                                action_vars,
                            )
                        end
                    ),
                    num_axis,
                )
                mdp =
                    FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)
                prop = FiniteTimeReachability(
                    [(num_states_per_axis, num_states_per_axis, num_states_per_axis)],
                    10,
                )
                spec = Specification(prop, Pessimistic, Maximize)
                prob = ControlSynthesisProblem(mdp, spec)
                (policy, V, it, res) = solve(prob, alg)
                @test it == 10
                @test all(V .≥ 0.0)
                prob = VerificationProblem(mdp, spec, policy)
                (V_mc, k, res) = solve(prob, alg)
                @test V ≈ V_mc
            end
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
                        1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                        1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                        4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                    ],
                    upper = N[
                        7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                        8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                        11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                        1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                        3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                        3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                    ],
                    upper = N[
                        2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                        1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                        17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                        4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                        2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                        1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                    ],
                    upper = N[
                        3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                        3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                        3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
            io = IOBuffer()
            show(io, MIME("text/plain"), mdp)
            str = String(take!(io))
            @test occursin("FactoredRobustMarkovDecisionProcess", str)
            @test occursin("3 state variables with cardinality: (3, 3, 3)", str)
            @test occursin("1 action variables with cardinality: (1,)", str)
            @test occursin("Initial states: All states", str)
            @test occursin("Marginal 1:", str)
            @test occursin("Marginal 2:", str)
            @test occursin("Marginal 3:", str)
            @test occursin("Inferred properties", str)
            @test occursin("Model type: Factored Interval MDP", str)
            @test occursin("Number of states: 27", str)
            @test occursin("Number of actions: 1", str)
            @test occursin("Default model checking algorithm: Robust Value Iteration", str)
            @test occursin(
                "Default Bellman operator algorithm: Recursive O-Maximization",
                str,
            )
        end
    end
end

@testitem "bellman 1d" tags = [:base, :bellman_1d] begin
    using IntervalMDP
    using Random: MersenneTwister
    @testset for N in [Float32, Float64]
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 0 1 // 15 3 // 10 0 1 // 30 1 // 3 0 7 // 30 4 // 15 0 1 // 6 1 // 5 0 1 // 10 1 // 5 0 0 7 // 30 0 7 // 30 1 // 5 0 2 // 15 1 // 6 0;
                                1 // 5 4 // 15 0 1 // 10 1 // 5 0 3 // 10 3 // 10 0 1 // 10 1 // 15 0 3 // 10 3 // 10 0 7 // 30 1 // 5 0 1 // 10 1 // 5 0 1 // 5 1 // 30 0 1 // 5 3 // 10 0;
                                4 // 15 1 // 30 1 1 // 5 1 // 5 1 7 // 30 4 // 15 1 2 // 15 7 // 30 1 1 // 5 1 // 3 1 2 // 15 1 // 6 1 1 // 6 1 // 3 1 4 // 15 3 // 10 1 1 // 30 3 // 10 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 0 13 // 30 3 // 5 0 17 // 30 17 // 30 0 17 // 30 13 // 30 0 3 // 5 2 // 3 0 11 // 30 7 // 15 0 0 1 // 2 0 17 // 30 13 // 30 0 7 // 15 13 // 30 0;
                                8 // 15 1 // 2 0 3 // 5 7 // 15 0 8 // 15 17 // 30 0 2 // 3 17 // 30 0 11 // 30 7 // 15 0 19 // 30 19 // 30 0 13 // 15 1 // 2 0 17 // 30 13 // 30 0 3 // 5 11 // 30 0;
                                11 // 30 1 // 3 1 2 // 5 8 // 15 1 7 // 15 3 // 5 1 2 // 3 17 // 30 1 2 // 3 8 // 15 1 2 // 15 3 // 5 1 2 // 3 3 // 5 1 17 // 30 2 // 3 1 7 // 15 8 // 15 1
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
                                1 // 10 1 // 15 1 3 // 10 0 0 1 // 6 1 // 15 0 1 // 15 1 // 6 1 1 // 6 1 // 30 0 1 // 10 1 // 10 0 1 // 3 2 // 15 1 3 // 10 4 // 15 0 2 // 15 2 // 15 0;
                                3 // 10 1 // 5 0 3 // 10 2 // 15 1 0 1 // 30 0 0 1 // 15 0 1 // 30 7 // 30 1 1 // 30 1 // 15 0 7 // 30 1 // 15 0 1 // 6 1 // 30 1 1 // 10 1 // 15 0;
                                3 // 10 4 // 15 0 1 // 10 3 // 10 0 2 // 15 1 // 3 1 3 // 10 1 // 10 0 1 // 6 3 // 10 0 7 // 30 1 // 6 1 1 // 15 1 // 15 0 1 // 10 1 // 5 0 1 // 5 4 // 15 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 1 3 // 5 11 // 30 0 3 // 5 7 // 15 0 19 // 30 2 // 5 1 3 // 5 2 // 3 0 2 // 3 8 // 15 0 8 // 15 19 // 30 1 8 // 15 8 // 15 0 13 // 30 13 // 30 0;
                                1 // 3 13 // 30 0 11 // 30 2 // 5 1 2 // 3 2 // 3 0 0 13 // 30 0 1 // 2 17 // 30 1 17 // 30 1 // 3 0 2 // 5 1 // 3 0 13 // 30 11 // 30 1 8 // 15 1 // 3 0;
                                17 // 30 3 // 5 0 8 // 15 1 // 2 0 7 // 15 1 // 2 1 2 // 3 17 // 30 0 11 // 30 2 // 5 0 1 // 2 7 // 15 1 2 // 5 17 // 30 0 11 // 30 2 // 5 0 11 // 30 2 // 3 1
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
                                4 // 15 1 // 5 1 3 // 10 3 // 10 1 4 // 15 7 // 30 1 1 // 5 4 // 15 0 7 // 30 1 // 6 0 1 // 5 0 0 1 // 15 1 // 30 0 3 // 10 1 // 3 0 2 // 15 1 // 15 0;
                                2 // 15 4 // 15 0 1 // 10 1 // 30 0 7 // 30 2 // 15 0 1 // 15 1 // 30 1 3 // 10 1 // 3 1 1 // 5 1 // 10 1 2 // 15 1 // 30 0 2 // 15 4 // 15 0 0 4 // 15 0;
                                1 // 5 1 // 3 0 3 // 10 1 // 10 0 1 // 15 1 // 10 0 1 // 30 1 // 5 0 2 // 15 7 // 30 0 1 // 3 2 // 15 0 1 // 10 1 // 6 1 3 // 10 1 // 5 1 7 // 30 1 // 30 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 1 // 2 3 // 5 1 19 // 30 2 // 5 1 8 // 15 1 // 3 0 11 // 30 2 // 5 0 17 // 30 13 // 30 0 2 // 5 3 // 5 0 3 // 5 11 // 30 0 1 // 2 11 // 30 0;
                                3 // 5 2 // 3 0 13 // 30 19 // 30 0 1 // 3 2 // 5 0 17 // 30 7 // 15 1 11 // 30 3 // 5 1 19 // 30 7 // 15 1 2 // 5 8 // 15 0 17 // 30 11 // 30 0 19 // 30 13 // 30 0;
                                3 // 5 2 // 3 0 1 // 2 1 // 2 0 2 // 3 7 // 15 0 3 // 5 3 // 5 0 1 // 2 1 // 3 0 2 // 5 8 // 15 0 2 // 5 11 // 30 1 1 // 3 8 // 15 1 7 // 15 13 // 30 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(3, i, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 1 0 0 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 1 0 0 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 0 1 0 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 0 1 0 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 0 0 1 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 0 0 1 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 1 0 0 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 1 0 0 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 0 1 0 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 0 1 0 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 0 0 1 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 0 0 1 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 0 0 0 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 0 0 0 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 0 0 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 0 0 0 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 1 1 1 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 1 1 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 0 0 0 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 0 0 0 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 0 0 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 0 0 0 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 1 1 1 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 1 1 1 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 1 1 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 1 1 1 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 0 0 0 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 0 0 0 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 1 1 1 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 0 0 0 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 0 0 0 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 1 1 1 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 0 0 0 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 0 0 0 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, 3, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0 1 0 0 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0 0 1 0 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1 0 0 1 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0 1 0 0 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0 0 1 0 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1 0 0 1 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 1 1 1 0 0 0 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0 1 1 1 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 0 0 0 0 0 0 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 1 1 1 0 0 0 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0 1 1 1 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 0 0 0 0 0 0 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0 0 0 0 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0 0 0 0 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1 1 1 1 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1 1 1 1 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, j, 3) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
                    @test V ≈ V_implicit
                    @test k == k_implicit
                    @test res ≈ res_implicit
                end
            end
            @testset "4D abstraction" begin
                rng = MersenneTwister(995)
                prob_lower = [rand(rng, N, 3, 81) ./ N(3) for _ in 1:4]
                prob_upper = [(rand(rng, N, 3, 81) .+ N(1)) ./ N(3) for _ in 1:4]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    4,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(ambiguity_sets[i], (1, 2, 3, 4), (1,), (3, 3, 3, 3), (1,))
                        end
                    ),
                    4,
                )
                mdp = FactoredRobustMarkovDecisionProcess((3, 3, 3, 3), (1,), marginals)
                prop = FiniteTimeReachability([(3, 3, 3, 3)], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                (V_ortho, it_ortho, res_ortho) = solve(prob, alg)
                @test V_ortho[3, 3, 3, 3] ≈ one(N)
                @test all(V_ortho .>= zero(N))
                @test all(V_ortho .<= one(N))
                if !(bellman_algorithm(alg) isa VertexEnumeration)
                    prob_lower_simple = zeros(N, 81, 81)
                    prob_upper_simple = zeros(N, 81, 81)
                    lin = LinearIndices((3, 3, 3, 3))
                    act_idx = CartesianIndex(1)
                    for I in CartesianIndices((3, 3, 3, 3))
                        for J in CartesianIndices((3, 3, 3, 3))
                            marginal_ambiguity_sets = map((marginal->begin
                                marginal[act_idx, I]
                            end), marginals)
                            prob_lower_simple[lin[J], lin[I]] =
                                prod((lower(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
                            prob_upper_simple[lin[J], lin[I]] =
                                prod((upper(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
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
                    (V_direct, it_direct, res_direct) = solve(prob, alg)
                    @test V_direct[81] ≈ one(N)
                    @test all(V_ortho .≥ reshape(V_direct, 3, 3, 3, 3))
                end
            end
            @testset "synthesis" begin
                rng = MersenneTwister(3286)
                num_states_per_axis = 3
                num_axis = 3
                num_states = num_states_per_axis ^ num_axis
                num_actions = 2
                num_choices = num_states * num_actions
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = ntuple((_->begin
                    num_states_per_axis
                end), num_axis)
                action_vars = (num_actions,)
                prob_lower = [
                    rand(rng, N, num_states_per_axis, num_choices) ./ num_states_per_axis for _ in 1:num_axis
                ]
                prob_upper = [
                    (rand(rng, N, num_states_per_axis, num_choices) .+ N(1)) ./
                    num_states_per_axis for _ in 1:num_axis
                ]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    num_axis,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(
                                ambiguity_sets[i],
                                state_indices,
                                action_indices,
                                state_vars,
                                action_vars,
                            )
                        end
                    ),
                    num_axis,
                )
                mdp =
                    FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)
                prop = FiniteTimeReachability(
                    [(num_states_per_axis, num_states_per_axis, num_states_per_axis)],
                    10,
                )
                spec = Specification(prop, Pessimistic, Maximize)
                prob = ControlSynthesisProblem(mdp, spec)
                (policy, V, it, res) = solve(prob, alg)
                @test it == 10
                @test all(V .≥ 0.0)
                prob = VerificationProblem(mdp, spec, policy)
                (V_mc, k, res) = solve(prob, alg)
                @test V ≈ V_mc
            end
        end
        @testset "bellman 1d" begin
            ambiguity_sets = IntervalAmbiguitySets(;
                lower = N[
                    0 5 // 10 2 // 10;
                    1 // 10 3 // 10 3 // 10;
                    2 // 10 1 // 10 5 // 10
                ],
                upper = N[
                    5 // 10 7 // 10 3 // 10;
                    6 // 10 5 // 10 4 // 10;
                    7 // 10 3 // 10 5 // 10
                ],
            )
            imc = IntervalMarkovChain(ambiguity_sets)
            V = N[1, 2, 3]
            @testset "vertices" begin
                verts = IntervalMDP.vertices(ambiguity_sets[1])
                @test length(verts) <= 6
                expected_verts = N[
                    5 // 10 3 // 10 2 // 10;
                    5 // 10 1 // 10 4 // 10;
                    2 // 10 6 // 10 2 // 10;
                    0 6 // 10 4 // 10;
                    2 // 10 1 // 10 7 // 10;
                    0 3 // 10 7 // 10
                ]
                @test length(verts) ≥ size(expected_verts, 1)
                @test all((any((v2->begin
                    v1 ≈ v2
                end), verts) for v1 in eachrow(expected_verts)))
                verts = IntervalMDP.vertices(ambiguity_sets[2])
                @test length(verts) <= 6
                expected_verts = N[
                    6 // 10 3 // 10 1 // 10;
                    5 // 10 4 // 10 1 // 10;
                    5 // 10 3 // 10 2 // 10
                ]
                @test length(verts) ≥ size(expected_verts, 1)
                @test all((any((v2->begin
                    v1 ≈ v2
                end), verts) for v1 in eachrow(expected_verts)))
                verts = IntervalMDP.vertices(ambiguity_sets[3])
                @test length(verts) <= 6
                expected_verts = N[2 // 10 3 // 10 5 // 10]
                @test length(verts) ≥ size(expected_verts, 1)
                @test all((any((v2->begin
                    v1 ≈ v2
                end), verts) for v1 in eachrow(expected_verts)))
            end
            @testset "maximization" begin
                Vexpected = let
    _Qres = Array{eltype(V)}(undef, (IntervalMDP.action_values(imc)..., size(V)...))
    _ws = IntervalMDP.construct_workspace(imc)
    _sc = IntervalMDP.construct_strategy_cache(imc)
    IntervalMDP.bellman_q!(_ws, _sc, IntervalMDP.StateActionValueArray(_Qres), IntervalMDP.StateValueArray(V), imc; upper_bound = true)
    _Qres
end
                ws = IntervalMDP.construct_workspace(imc, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = zeros(N, 1, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = true,)
                @test Vres ≈ Vexpected
                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(
                    imc,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = true,)
                @test Vres ≈ Vexpected
                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                    imc,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = true,)
                @test Vres ≈ Vexpected
                ws = IntervalMDP.construct_workspace(imc, VertexEnumeration())
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = zeros(N, 1, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = true,)
                @test Vres ≈ Vexpected
                ws = IntervalMDP.FactoredVertexIteratorWorkspace(imc)
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = true,)
                @test Vres ≈ Vexpected
                ws = IntervalMDP.ThreadedFactoredVertexIteratorWorkspace(imc)
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = true,)
                @test Vres ≈ Vexpected
            end
            @testset "minimization" begin
                Vexpected = let
    _Qres = Array{eltype(V)}(undef, (IntervalMDP.action_values(imc)..., size(V)...))
    _ws = IntervalMDP.construct_workspace(imc)
    _sc = IntervalMDP.construct_strategy_cache(imc)
    IntervalMDP.bellman_q!(_ws, _sc, IntervalMDP.StateActionValueArray(_Qres), IntervalMDP.StateValueArray(V), imc; upper_bound = false)
    _Qres
end
                ws = IntervalMDP.construct_workspace(imc, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = zeros(N, 1, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = false,)
                @test Vres ≈ Vexpected
                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(
                    imc,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = false,)
                @test Vres ≈ Vexpected
                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                    imc,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = false,)
                @test Vres ≈ Vexpected
                ws = IntervalMDP.construct_workspace(imc, VertexEnumeration())
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = zeros(N, 1, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = false,)
                @test Vres ≈ Vexpected
                ws = IntervalMDP.FactoredVertexIteratorWorkspace(imc)
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = false,)
                @test Vres ≈ Vexpected
                ws = IntervalMDP.ThreadedFactoredVertexIteratorWorkspace(imc)
                strategy_cache = IntervalMDP.construct_strategy_cache(imc)
                Vres = similar(Vres)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), imc; upper_bound = false,)
                @test Vres ≈ Vexpected
            end
        end
    end
end

@testitem "bellman 2d" tags = [:base, :bellman_2d] begin
    using IntervalMDP
    using Random: MersenneTwister
    @testset for N in [Float32, Float64]
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 0 1 // 15 3 // 10 0 1 // 30 1 // 3 0 7 // 30 4 // 15 0 1 // 6 1 // 5 0 1 // 10 1 // 5 0 0 7 // 30 0 7 // 30 1 // 5 0 2 // 15 1 // 6 0;
                                1 // 5 4 // 15 0 1 // 10 1 // 5 0 3 // 10 3 // 10 0 1 // 10 1 // 15 0 3 // 10 3 // 10 0 7 // 30 1 // 5 0 1 // 10 1 // 5 0 1 // 5 1 // 30 0 1 // 5 3 // 10 0;
                                4 // 15 1 // 30 1 1 // 5 1 // 5 1 7 // 30 4 // 15 1 2 // 15 7 // 30 1 1 // 5 1 // 3 1 2 // 15 1 // 6 1 1 // 6 1 // 3 1 4 // 15 3 // 10 1 1 // 30 3 // 10 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 0 13 // 30 3 // 5 0 17 // 30 17 // 30 0 17 // 30 13 // 30 0 3 // 5 2 // 3 0 11 // 30 7 // 15 0 0 1 // 2 0 17 // 30 13 // 30 0 7 // 15 13 // 30 0;
                                8 // 15 1 // 2 0 3 // 5 7 // 15 0 8 // 15 17 // 30 0 2 // 3 17 // 30 0 11 // 30 7 // 15 0 19 // 30 19 // 30 0 13 // 15 1 // 2 0 17 // 30 13 // 30 0 3 // 5 11 // 30 0;
                                11 // 30 1 // 3 1 2 // 5 8 // 15 1 7 // 15 3 // 5 1 2 // 3 17 // 30 1 2 // 3 8 // 15 1 2 // 15 3 // 5 1 2 // 3 3 // 5 1 17 // 30 2 // 3 1 7 // 15 8 // 15 1
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
                                1 // 10 1 // 15 1 3 // 10 0 0 1 // 6 1 // 15 0 1 // 15 1 // 6 1 1 // 6 1 // 30 0 1 // 10 1 // 10 0 1 // 3 2 // 15 1 3 // 10 4 // 15 0 2 // 15 2 // 15 0;
                                3 // 10 1 // 5 0 3 // 10 2 // 15 1 0 1 // 30 0 0 1 // 15 0 1 // 30 7 // 30 1 1 // 30 1 // 15 0 7 // 30 1 // 15 0 1 // 6 1 // 30 1 1 // 10 1 // 15 0;
                                3 // 10 4 // 15 0 1 // 10 3 // 10 0 2 // 15 1 // 3 1 3 // 10 1 // 10 0 1 // 6 3 // 10 0 7 // 30 1 // 6 1 1 // 15 1 // 15 0 1 // 10 1 // 5 0 1 // 5 4 // 15 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 1 3 // 5 11 // 30 0 3 // 5 7 // 15 0 19 // 30 2 // 5 1 3 // 5 2 // 3 0 2 // 3 8 // 15 0 8 // 15 19 // 30 1 8 // 15 8 // 15 0 13 // 30 13 // 30 0;
                                1 // 3 13 // 30 0 11 // 30 2 // 5 1 2 // 3 2 // 3 0 0 13 // 30 0 1 // 2 17 // 30 1 17 // 30 1 // 3 0 2 // 5 1 // 3 0 13 // 30 11 // 30 1 8 // 15 1 // 3 0;
                                17 // 30 3 // 5 0 8 // 15 1 // 2 0 7 // 15 1 // 2 1 2 // 3 17 // 30 0 11 // 30 2 // 5 0 1 // 2 7 // 15 1 2 // 5 17 // 30 0 11 // 30 2 // 5 0 11 // 30 2 // 3 1
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
                                4 // 15 1 // 5 1 3 // 10 3 // 10 1 4 // 15 7 // 30 1 1 // 5 4 // 15 0 7 // 30 1 // 6 0 1 // 5 0 0 1 // 15 1 // 30 0 3 // 10 1 // 3 0 2 // 15 1 // 15 0;
                                2 // 15 4 // 15 0 1 // 10 1 // 30 0 7 // 30 2 // 15 0 1 // 15 1 // 30 1 3 // 10 1 // 3 1 1 // 5 1 // 10 1 2 // 15 1 // 30 0 2 // 15 4 // 15 0 0 4 // 15 0;
                                1 // 5 1 // 3 0 3 // 10 1 // 10 0 1 // 15 1 // 10 0 1 // 30 1 // 5 0 2 // 15 7 // 30 0 1 // 3 2 // 15 0 1 // 10 1 // 6 1 3 // 10 1 // 5 1 7 // 30 1 // 30 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 1 // 2 3 // 5 1 19 // 30 2 // 5 1 8 // 15 1 // 3 0 11 // 30 2 // 5 0 17 // 30 13 // 30 0 2 // 5 3 // 5 0 3 // 5 11 // 30 0 1 // 2 11 // 30 0;
                                3 // 5 2 // 3 0 13 // 30 19 // 30 0 1 // 3 2 // 5 0 17 // 30 7 // 15 1 11 // 30 3 // 5 1 19 // 30 7 // 15 1 2 // 5 8 // 15 0 17 // 30 11 // 30 0 19 // 30 13 // 30 0;
                                3 // 5 2 // 3 0 1 // 2 1 // 2 0 2 // 3 7 // 15 0 3 // 5 3 // 5 0 1 // 2 1 // 3 0 2 // 5 8 // 15 0 2 // 5 11 // 30 1 1 // 3 8 // 15 1 7 // 15 13 // 30 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(3, i, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 1 0 0 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 1 0 0 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 0 1 0 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 0 1 0 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 0 0 1 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 0 0 1 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 1 0 0 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 1 0 0 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 0 1 0 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 0 1 0 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 0 0 1 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 0 0 1 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 0 0 0 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 0 0 0 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 0 0 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 0 0 0 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 1 1 1 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 1 1 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 0 0 0 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 0 0 0 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 0 0 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 0 0 0 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 1 1 1 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 1 1 1 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 1 1 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 1 1 1 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 0 0 0 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 0 0 0 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 1 1 1 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 0 0 0 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 0 0 0 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 1 1 1 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 0 0 0 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 0 0 0 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, 3, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0 1 0 0 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0 0 1 0 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1 0 0 1 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0 1 0 0 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0 0 1 0 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1 0 0 1 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 1 1 1 0 0 0 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0 1 1 1 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 0 0 0 0 0 0 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 1 1 1 0 0 0 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0 1 1 1 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 0 0 0 0 0 0 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0 0 0 0 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0 0 0 0 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1 1 1 1 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1 1 1 1 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, j, 3) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
                    @test V ≈ V_implicit
                    @test k == k_implicit
                    @test res ≈ res_implicit
                end
            end
            @testset "4D abstraction" begin
                rng = MersenneTwister(995)
                prob_lower = [rand(rng, N, 3, 81) ./ N(3) for _ in 1:4]
                prob_upper = [(rand(rng, N, 3, 81) .+ N(1)) ./ N(3) for _ in 1:4]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    4,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(ambiguity_sets[i], (1, 2, 3, 4), (1,), (3, 3, 3, 3), (1,))
                        end
                    ),
                    4,
                )
                mdp = FactoredRobustMarkovDecisionProcess((3, 3, 3, 3), (1,), marginals)
                prop = FiniteTimeReachability([(3, 3, 3, 3)], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                (V_ortho, it_ortho, res_ortho) = solve(prob, alg)
                @test V_ortho[3, 3, 3, 3] ≈ one(N)
                @test all(V_ortho .>= zero(N))
                @test all(V_ortho .<= one(N))
                if !(bellman_algorithm(alg) isa VertexEnumeration)
                    prob_lower_simple = zeros(N, 81, 81)
                    prob_upper_simple = zeros(N, 81, 81)
                    lin = LinearIndices((3, 3, 3, 3))
                    act_idx = CartesianIndex(1)
                    for I in CartesianIndices((3, 3, 3, 3))
                        for J in CartesianIndices((3, 3, 3, 3))
                            marginal_ambiguity_sets = map((marginal->begin
                                marginal[act_idx, I]
                            end), marginals)
                            prob_lower_simple[lin[J], lin[I]] =
                                prod((lower(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
                            prob_upper_simple[lin[J], lin[I]] =
                                prod((upper(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
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
                    (V_direct, it_direct, res_direct) = solve(prob, alg)
                    @test V_direct[81] ≈ one(N)
                    @test all(V_ortho .≥ reshape(V_direct, 3, 3, 3, 3))
                end
            end
            @testset "synthesis" begin
                rng = MersenneTwister(3286)
                num_states_per_axis = 3
                num_axis = 3
                num_states = num_states_per_axis ^ num_axis
                num_actions = 2
                num_choices = num_states * num_actions
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = ntuple((_->begin
                    num_states_per_axis
                end), num_axis)
                action_vars = (num_actions,)
                prob_lower = [
                    rand(rng, N, num_states_per_axis, num_choices) ./ num_states_per_axis for _ in 1:num_axis
                ]
                prob_upper = [
                    (rand(rng, N, num_states_per_axis, num_choices) .+ N(1)) ./
                    num_states_per_axis for _ in 1:num_axis
                ]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    num_axis,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(
                                ambiguity_sets[i],
                                state_indices,
                                action_indices,
                                state_vars,
                                action_vars,
                            )
                        end
                    ),
                    num_axis,
                )
                mdp =
                    FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)
                prop = FiniteTimeReachability(
                    [(num_states_per_axis, num_states_per_axis, num_states_per_axis)],
                    10,
                )
                spec = Specification(prop, Pessimistic, Maximize)
                prob = ControlSynthesisProblem(mdp, spec)
                (policy, V, it, res) = solve(prob, alg)
                @test it == 10
                @test all(V .≥ 0.0)
                prob = VerificationProblem(mdp, spec, policy)
                (V_mc, k, res) = solve(prob, alg)
                @test V ≈ V_mc
            end
        end
        @testset "bellman 2d" begin
            state_indices = (1, 2)
            action_indices = (1,)
            state_vars = (2, 3)
            action_vars = (1,)
            marginal1 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        1 // 15 7 // 30 1 // 15 13 // 30 4 // 15 1 // 6;
                        2 // 5 7 // 30 1 // 30 11 // 30 2 // 15 1 // 10
                    ],
                    upper = N[
                        17 // 30 7 // 10 2 // 3 4 // 5 7 // 10 2 // 3;
                        9 // 10 13 // 15 9 // 10 5 // 6 4 // 5 14 // 15
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
                        1 // 30 1 // 3 1 // 6 1 // 15 2 // 5 2 // 15;
                        4 // 15 1 // 4 1 // 6 1 // 30 2 // 15 1 // 30;
                        2 // 15 7 // 30 1 // 10 7 // 30 7 // 15 1 // 5
                    ],
                    upper = N[
                        2 // 3 7 // 15 4 // 5 11 // 30 19 // 30 1 // 2;
                        23 // 30 4 // 5 23 // 30 3 // 5 7 // 10 8 // 15;
                        7 // 15 4 // 5 23 // 30 7 // 10 7 // 15 23 // 30
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
            mdp_nonchecked = FactoredRobustMarkovDecisionProcess(
                Int32.(state_vars),
                Int32.(action_vars),
                Int32.(state_vars),
                (marginal1, marginal2),
                AllAvailableActions(Int32.(action_vars)),
                AllStates(),
                Val(false),
            )
            @test state_values(mdp) == state_values(mdp_nonchecked)
            @test action_values(mdp) == action_values(mdp_nonchecked)
            @test marginals(mdp) == marginals(mdp_nonchecked)
            V = N[3 13 18; 12 16 8]
            @testset "maximization" begin
                ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                V_vertex = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(V_vertex), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                @test V_vertex ≈
                      N[1076 // 75 4279 // 300 167 // 15; 11107 // 900 4123 // 300 121 // 9]
                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_McCormick = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_McCormick), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_McCormick .>= 0.0)
                @test all(Vres_first_McCormick .<= maximum(V))
                @test all(Vres_first_McCormick .+ epsilon .>= V_vertex)
                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.construct_workspace(mdp, OMaximization())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_OMax = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_OMax), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_OMax .>= 0.0)
                @test all(Vres_first_OMax .<= maximum(V))
                @test all(Vres_first_OMax .+ epsilon .>= V_vertex)
                ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                @test Vres ≈ Vres_first_OMax
                ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                @test Vres ≈ Vres_first_OMax
            end
            @testset "minimization" begin
                ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                V_vertex = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(V_vertex), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                @test V_vertex ≈
                      N[4399 // 450 41 // 5 488 // 45; 1033 // 100 543 // 50 361 // 36]
                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_McCormick = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_McCormick), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_McCormick .>= 0.0)
                @test all(Vres_first_McCormick .<= maximum(V))
                @test all(Vres_first_McCormick .- epsilon .<= V_vertex)
                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.construct_workspace(mdp, OMaximization())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_OMax = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_OMax), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_OMax .>= 0.0)
                @test all(Vres_first_OMax .<= maximum(V))
                @test all(Vres_first_OMax .- epsilon .<= V_vertex)
                ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                @test Vres ≈ Vres_first_OMax
                ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                @test Vres ≈ Vres_first_OMax
            end
        end
    end
end

@testitem "bellman 2d partial dependence" tags =
    [:base, :bellman_2d_partial_dependence] begin
    using IntervalMDP
    using Random: MersenneTwister
    @testset for N in [Float32, Float64]
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 0 1 // 15 3 // 10 0 1 // 30 1 // 3 0 7 // 30 4 // 15 0 1 // 6 1 // 5 0 1 // 10 1 // 5 0 0 7 // 30 0 7 // 30 1 // 5 0 2 // 15 1 // 6 0;
                                1 // 5 4 // 15 0 1 // 10 1 // 5 0 3 // 10 3 // 10 0 1 // 10 1 // 15 0 3 // 10 3 // 10 0 7 // 30 1 // 5 0 1 // 10 1 // 5 0 1 // 5 1 // 30 0 1 // 5 3 // 10 0;
                                4 // 15 1 // 30 1 1 // 5 1 // 5 1 7 // 30 4 // 15 1 2 // 15 7 // 30 1 1 // 5 1 // 3 1 2 // 15 1 // 6 1 1 // 6 1 // 3 1 4 // 15 3 // 10 1 1 // 30 3 // 10 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 0 13 // 30 3 // 5 0 17 // 30 17 // 30 0 17 // 30 13 // 30 0 3 // 5 2 // 3 0 11 // 30 7 // 15 0 0 1 // 2 0 17 // 30 13 // 30 0 7 // 15 13 // 30 0;
                                8 // 15 1 // 2 0 3 // 5 7 // 15 0 8 // 15 17 // 30 0 2 // 3 17 // 30 0 11 // 30 7 // 15 0 19 // 30 19 // 30 0 13 // 15 1 // 2 0 17 // 30 13 // 30 0 3 // 5 11 // 30 0;
                                11 // 30 1 // 3 1 2 // 5 8 // 15 1 7 // 15 3 // 5 1 2 // 3 17 // 30 1 2 // 3 8 // 15 1 2 // 15 3 // 5 1 2 // 3 3 // 5 1 17 // 30 2 // 3 1 7 // 15 8 // 15 1
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
                                1 // 10 1 // 15 1 3 // 10 0 0 1 // 6 1 // 15 0 1 // 15 1 // 6 1 1 // 6 1 // 30 0 1 // 10 1 // 10 0 1 // 3 2 // 15 1 3 // 10 4 // 15 0 2 // 15 2 // 15 0;
                                3 // 10 1 // 5 0 3 // 10 2 // 15 1 0 1 // 30 0 0 1 // 15 0 1 // 30 7 // 30 1 1 // 30 1 // 15 0 7 // 30 1 // 15 0 1 // 6 1 // 30 1 1 // 10 1 // 15 0;
                                3 // 10 4 // 15 0 1 // 10 3 // 10 0 2 // 15 1 // 3 1 3 // 10 1 // 10 0 1 // 6 3 // 10 0 7 // 30 1 // 6 1 1 // 15 1 // 15 0 1 // 10 1 // 5 0 1 // 5 4 // 15 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 1 3 // 5 11 // 30 0 3 // 5 7 // 15 0 19 // 30 2 // 5 1 3 // 5 2 // 3 0 2 // 3 8 // 15 0 8 // 15 19 // 30 1 8 // 15 8 // 15 0 13 // 30 13 // 30 0;
                                1 // 3 13 // 30 0 11 // 30 2 // 5 1 2 // 3 2 // 3 0 0 13 // 30 0 1 // 2 17 // 30 1 17 // 30 1 // 3 0 2 // 5 1 // 3 0 13 // 30 11 // 30 1 8 // 15 1 // 3 0;
                                17 // 30 3 // 5 0 8 // 15 1 // 2 0 7 // 15 1 // 2 1 2 // 3 17 // 30 0 11 // 30 2 // 5 0 1 // 2 7 // 15 1 2 // 5 17 // 30 0 11 // 30 2 // 5 0 11 // 30 2 // 3 1
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
                                4 // 15 1 // 5 1 3 // 10 3 // 10 1 4 // 15 7 // 30 1 1 // 5 4 // 15 0 7 // 30 1 // 6 0 1 // 5 0 0 1 // 15 1 // 30 0 3 // 10 1 // 3 0 2 // 15 1 // 15 0;
                                2 // 15 4 // 15 0 1 // 10 1 // 30 0 7 // 30 2 // 15 0 1 // 15 1 // 30 1 3 // 10 1 // 3 1 1 // 5 1 // 10 1 2 // 15 1 // 30 0 2 // 15 4 // 15 0 0 4 // 15 0;
                                1 // 5 1 // 3 0 3 // 10 1 // 10 0 1 // 15 1 // 10 0 1 // 30 1 // 5 0 2 // 15 7 // 30 0 1 // 3 2 // 15 0 1 // 10 1 // 6 1 3 // 10 1 // 5 1 7 // 30 1 // 30 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 1 // 2 3 // 5 1 19 // 30 2 // 5 1 8 // 15 1 // 3 0 11 // 30 2 // 5 0 17 // 30 13 // 30 0 2 // 5 3 // 5 0 3 // 5 11 // 30 0 1 // 2 11 // 30 0;
                                3 // 5 2 // 3 0 13 // 30 19 // 30 0 1 // 3 2 // 5 0 17 // 30 7 // 15 1 11 // 30 3 // 5 1 19 // 30 7 // 15 1 2 // 5 8 // 15 0 17 // 30 11 // 30 0 19 // 30 13 // 30 0;
                                3 // 5 2 // 3 0 1 // 2 1 // 2 0 2 // 3 7 // 15 0 3 // 5 3 // 5 0 1 // 2 1 // 3 0 2 // 5 8 // 15 0 2 // 5 11 // 30 1 1 // 3 8 // 15 1 7 // 15 13 // 30 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(3, i, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 1 0 0 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 1 0 0 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 0 1 0 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 0 1 0 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 0 0 1 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 0 0 1 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 1 0 0 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 1 0 0 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 0 1 0 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 0 1 0 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 0 0 1 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 0 0 1 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 0 0 0 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 0 0 0 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 0 0 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 0 0 0 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 1 1 1 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 1 1 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 0 0 0 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 0 0 0 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 0 0 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 0 0 0 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 1 1 1 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 1 1 1 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 1 1 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 1 1 1 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 0 0 0 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 0 0 0 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 1 1 1 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 0 0 0 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 0 0 0 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 1 1 1 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 0 0 0 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 0 0 0 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, 3, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0 1 0 0 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0 0 1 0 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1 0 0 1 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0 1 0 0 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0 0 1 0 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1 0 0 1 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 1 1 1 0 0 0 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0 1 1 1 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 0 0 0 0 0 0 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 1 1 1 0 0 0 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0 1 1 1 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 0 0 0 0 0 0 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0 0 0 0 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0 0 0 0 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1 1 1 1 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1 1 1 1 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, j, 3) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
                    @test V ≈ V_implicit
                    @test k == k_implicit
                    @test res ≈ res_implicit
                end
            end
            @testset "4D abstraction" begin
                rng = MersenneTwister(995)
                prob_lower = [rand(rng, N, 3, 81) ./ N(3) for _ in 1:4]
                prob_upper = [(rand(rng, N, 3, 81) .+ N(1)) ./ N(3) for _ in 1:4]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    4,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(ambiguity_sets[i], (1, 2, 3, 4), (1,), (3, 3, 3, 3), (1,))
                        end
                    ),
                    4,
                )
                mdp = FactoredRobustMarkovDecisionProcess((3, 3, 3, 3), (1,), marginals)
                prop = FiniteTimeReachability([(3, 3, 3, 3)], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                (V_ortho, it_ortho, res_ortho) = solve(prob, alg)
                @test V_ortho[3, 3, 3, 3] ≈ one(N)
                @test all(V_ortho .>= zero(N))
                @test all(V_ortho .<= one(N))
                if !(bellman_algorithm(alg) isa VertexEnumeration)
                    prob_lower_simple = zeros(N, 81, 81)
                    prob_upper_simple = zeros(N, 81, 81)
                    lin = LinearIndices((3, 3, 3, 3))
                    act_idx = CartesianIndex(1)
                    for I in CartesianIndices((3, 3, 3, 3))
                        for J in CartesianIndices((3, 3, 3, 3))
                            marginal_ambiguity_sets = map((marginal->begin
                                marginal[act_idx, I]
                            end), marginals)
                            prob_lower_simple[lin[J], lin[I]] =
                                prod((lower(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
                            prob_upper_simple[lin[J], lin[I]] =
                                prod((upper(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
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
                    (V_direct, it_direct, res_direct) = solve(prob, alg)
                    @test V_direct[81] ≈ one(N)
                    @test all(V_ortho .≥ reshape(V_direct, 3, 3, 3, 3))
                end
            end
            @testset "synthesis" begin
                rng = MersenneTwister(3286)
                num_states_per_axis = 3
                num_axis = 3
                num_states = num_states_per_axis ^ num_axis
                num_actions = 2
                num_choices = num_states * num_actions
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = ntuple((_->begin
                    num_states_per_axis
                end), num_axis)
                action_vars = (num_actions,)
                prob_lower = [
                    rand(rng, N, num_states_per_axis, num_choices) ./ num_states_per_axis for _ in 1:num_axis
                ]
                prob_upper = [
                    (rand(rng, N, num_states_per_axis, num_choices) .+ N(1)) ./
                    num_states_per_axis for _ in 1:num_axis
                ]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    num_axis,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(
                                ambiguity_sets[i],
                                state_indices,
                                action_indices,
                                state_vars,
                                action_vars,
                            )
                        end
                    ),
                    num_axis,
                )
                mdp =
                    FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)
                prop = FiniteTimeReachability(
                    [(num_states_per_axis, num_states_per_axis, num_states_per_axis)],
                    10,
                )
                spec = Specification(prop, Pessimistic, Maximize)
                prob = ControlSynthesisProblem(mdp, spec)
                (policy, V, it, res) = solve(prob, alg)
                @test it == 10
                @test all(V .≥ 0.0)
                prob = VerificationProblem(mdp, spec, policy)
                (V_mc, k, res) = solve(prob, alg)
                @test V ≈ V_mc
            end
        end
        @testset "bellman 2d partial dependence" begin
            state_vars = (2, 3)
            action_vars = (1, 2)
            marginal1 = Marginal(
                IntervalAmbiguitySets(;
                    lower = N[
                        1 // 15 7 // 30 1 // 15 13 // 30 4 // 15 1 // 6;
                        2 // 5 7 // 30 1 // 30 11 // 30 2 // 15 1 // 10
                    ],
                    upper = N[
                        17 // 30 7 // 10 2 // 3 4 // 5 7 // 10 2 // 3;
                        9 // 10 13 // 15 9 // 10 5 // 6 4 // 5 14 // 15
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
                        1 // 30 1 // 3 1 // 6 1 // 15 2 // 5 2 // 15;
                        4 // 15 1 // 4 1 // 6 1 // 30 2 // 15 1 // 30;
                        2 // 15 7 // 30 1 // 10 7 // 30 7 // 15 1 // 5
                    ],
                    upper = N[
                        2 // 3 7 // 15 4 // 5 11 // 30 19 // 30 1 // 2;
                        23 // 30 4 // 5 23 // 30 3 // 5 7 // 10 8 // 15;
                        7 // 15 4 // 5 23 // 30 7 // 10 7 // 15 23 // 30
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
            V = N[3 13 18; 12 16 8]
            @testset "max/max" begin
                ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                V_vertex = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(V_vertex), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = true,)
                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_McCormick = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_McCormick), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = true,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_McCormick .>= 0.0)
                @test all(Vres_first_McCormick .<= maximum(V))
                @test all(Vres_first_McCormick .+ epsilon .>= V_vertex)
                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = true,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = true,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.construct_workspace(mdp, OMaximization())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_OMax = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_OMax), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = true,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_OMax .>= 0.0)
                @test all(Vres_first_OMax .<= maximum(V))
                @test all(Vres_first_OMax .+ epsilon .>= V_vertex)
                ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = true,)
                @test Vres ≈ Vres_first_OMax
                ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = true,)
                @test Vres ≈ Vres_first_OMax
            end
            @testset "min/max" begin
                ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                V_vertex = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(V_vertex), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = false,)
                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_McCormick = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_McCormick), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = false,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_McCormick .>= 0.0)
                @test all(Vres_first_McCormick .<= maximum(V))
                @test all(Vres_first_McCormick .+ epsilon .>= V_vertex)
                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = false,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = false,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.construct_workspace(mdp, OMaximization())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_OMax = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_OMax), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = false,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_OMax .>= 0.0)
                @test all(Vres_first_OMax .<= maximum(V))
                @test all(Vres_first_OMax .+ epsilon .>= V_vertex)
                ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = false,)
                @test Vres ≈ Vres_first_OMax
                ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,
                    maximize = false,)
                @test Vres ≈ Vres_first_OMax
            end
            @testset "min/min" begin
                ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                V_vertex = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(V_vertex), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = false,)
                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_McCormick = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_McCormick), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = false,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_McCormick .>= 0.0)
                @test all(Vres_first_McCormick .<= maximum(V))
                @test all(Vres_first_McCormick .- epsilon .<= V_vertex)
                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = false,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = false,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.construct_workspace(mdp, OMaximization())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_OMax = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_OMax), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = false,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_OMax .>= 0.0)
                @test all(Vres_first_OMax .<= maximum(V))
                @test all(Vres_first_OMax .- epsilon .<= V_vertex)
                ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = false,)
                @test Vres ≈ Vres_first_OMax
                ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = false,)
                @test Vres ≈ Vres_first_OMax
            end
            @testset "max/min" begin
                ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                V_vertex = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(V_vertex), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = true,)
                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_McCormick = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_McCormick), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = true,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_McCormick .>= 0.0)
                @test all(Vres_first_McCormick .<= maximum(V))
                @test all(Vres_first_McCormick .- epsilon .<= V_vertex)
                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = true,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = true,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.construct_workspace(mdp, OMaximization())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_OMax = zeros(N, 2, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_OMax), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = true,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_OMax .>= 0.0)
                @test all(Vres_first_OMax .<= maximum(V))
                @test all(Vres_first_OMax .- epsilon .<= V_vertex)
                ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = true,)
                @test Vres ≈ Vres_first_OMax
                ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,
                    maximize = true,)
                @test Vres ≈ Vres_first_OMax
            end
        end
    end
end

@testitem "bellman 3d" tags = [:base, :bellman_3d] begin
    using IntervalMDP
    using Random: MersenneTwister
    @testset for N in [Float32, Float64]
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 0 1 // 15 3 // 10 0 1 // 30 1 // 3 0 7 // 30 4 // 15 0 1 // 6 1 // 5 0 1 // 10 1 // 5 0 0 7 // 30 0 7 // 30 1 // 5 0 2 // 15 1 // 6 0;
                                1 // 5 4 // 15 0 1 // 10 1 // 5 0 3 // 10 3 // 10 0 1 // 10 1 // 15 0 3 // 10 3 // 10 0 7 // 30 1 // 5 0 1 // 10 1 // 5 0 1 // 5 1 // 30 0 1 // 5 3 // 10 0;
                                4 // 15 1 // 30 1 1 // 5 1 // 5 1 7 // 30 4 // 15 1 2 // 15 7 // 30 1 1 // 5 1 // 3 1 2 // 15 1 // 6 1 1 // 6 1 // 3 1 4 // 15 3 // 10 1 1 // 30 3 // 10 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 0 13 // 30 3 // 5 0 17 // 30 17 // 30 0 17 // 30 13 // 30 0 3 // 5 2 // 3 0 11 // 30 7 // 15 0 0 1 // 2 0 17 // 30 13 // 30 0 7 // 15 13 // 30 0;
                                8 // 15 1 // 2 0 3 // 5 7 // 15 0 8 // 15 17 // 30 0 2 // 3 17 // 30 0 11 // 30 7 // 15 0 19 // 30 19 // 30 0 13 // 15 1 // 2 0 17 // 30 13 // 30 0 3 // 5 11 // 30 0;
                                11 // 30 1 // 3 1 2 // 5 8 // 15 1 7 // 15 3 // 5 1 2 // 3 17 // 30 1 2 // 3 8 // 15 1 2 // 15 3 // 5 1 2 // 3 3 // 5 1 17 // 30 2 // 3 1 7 // 15 8 // 15 1
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
                                1 // 10 1 // 15 1 3 // 10 0 0 1 // 6 1 // 15 0 1 // 15 1 // 6 1 1 // 6 1 // 30 0 1 // 10 1 // 10 0 1 // 3 2 // 15 1 3 // 10 4 // 15 0 2 // 15 2 // 15 0;
                                3 // 10 1 // 5 0 3 // 10 2 // 15 1 0 1 // 30 0 0 1 // 15 0 1 // 30 7 // 30 1 1 // 30 1 // 15 0 7 // 30 1 // 15 0 1 // 6 1 // 30 1 1 // 10 1 // 15 0;
                                3 // 10 4 // 15 0 1 // 10 3 // 10 0 2 // 15 1 // 3 1 3 // 10 1 // 10 0 1 // 6 3 // 10 0 7 // 30 1 // 6 1 1 // 15 1 // 15 0 1 // 10 1 // 5 0 1 // 5 4 // 15 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 1 3 // 5 11 // 30 0 3 // 5 7 // 15 0 19 // 30 2 // 5 1 3 // 5 2 // 3 0 2 // 3 8 // 15 0 8 // 15 19 // 30 1 8 // 15 8 // 15 0 13 // 30 13 // 30 0;
                                1 // 3 13 // 30 0 11 // 30 2 // 5 1 2 // 3 2 // 3 0 0 13 // 30 0 1 // 2 17 // 30 1 17 // 30 1 // 3 0 2 // 5 1 // 3 0 13 // 30 11 // 30 1 8 // 15 1 // 3 0;
                                17 // 30 3 // 5 0 8 // 15 1 // 2 0 7 // 15 1 // 2 1 2 // 3 17 // 30 0 11 // 30 2 // 5 0 1 // 2 7 // 15 1 2 // 5 17 // 30 0 11 // 30 2 // 5 0 11 // 30 2 // 3 1
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
                                4 // 15 1 // 5 1 3 // 10 3 // 10 1 4 // 15 7 // 30 1 1 // 5 4 // 15 0 7 // 30 1 // 6 0 1 // 5 0 0 1 // 15 1 // 30 0 3 // 10 1 // 3 0 2 // 15 1 // 15 0;
                                2 // 15 4 // 15 0 1 // 10 1 // 30 0 7 // 30 2 // 15 0 1 // 15 1 // 30 1 3 // 10 1 // 3 1 1 // 5 1 // 10 1 2 // 15 1 // 30 0 2 // 15 4 // 15 0 0 4 // 15 0;
                                1 // 5 1 // 3 0 3 // 10 1 // 10 0 1 // 15 1 // 10 0 1 // 30 1 // 5 0 2 // 15 7 // 30 0 1 // 3 2 // 15 0 1 // 10 1 // 6 1 3 // 10 1 // 5 1 7 // 30 1 // 30 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 1 // 2 3 // 5 1 19 // 30 2 // 5 1 8 // 15 1 // 3 0 11 // 30 2 // 5 0 17 // 30 13 // 30 0 2 // 5 3 // 5 0 3 // 5 11 // 30 0 1 // 2 11 // 30 0;
                                3 // 5 2 // 3 0 13 // 30 19 // 30 0 1 // 3 2 // 5 0 17 // 30 7 // 15 1 11 // 30 3 // 5 1 19 // 30 7 // 15 1 2 // 5 8 // 15 0 17 // 30 11 // 30 0 19 // 30 13 // 30 0;
                                3 // 5 2 // 3 0 1 // 2 1 // 2 0 2 // 3 7 // 15 0 3 // 5 3 // 5 0 1 // 2 1 // 3 0 2 // 5 8 // 15 0 2 // 5 11 // 30 1 1 // 3 8 // 15 1 7 // 15 13 // 30 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(3, i, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 1 0 0 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 1 0 0 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 0 1 0 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 0 1 0 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 0 0 1 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 0 0 1 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 1 0 0 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 1 0 0 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 0 1 0 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 0 1 0 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 0 0 1 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 0 0 1 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 0 0 0 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 0 0 0 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 0 0 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 0 0 0 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 1 1 1 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 1 1 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 0 0 0 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 0 0 0 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 0 0 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 0 0 0 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 1 1 1 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 1 1 1 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 1 1 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 0 0 0 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 1 1 1 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 0 0 0 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 0 0 0 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 1 1 1 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 0 0 0 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 0 0 0 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 1 1 1 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 0 0 0 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 0 0 0 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, 3, j) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 0 0 1 0 0 1 0 0;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 0 1 0 0 1 0 0 1 0;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 0 0 1 0 0 1 0 0 1
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 1 0 0 1 0 0 1 0 0;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 0 1 0 0 1 0 0 1 0;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 0 0 1 0 0 1 0 0 1
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 1 1 1 0 0 0 0 0 0;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 0 0 0 1 1 1 0 0 0;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 0 0 0 0 0 0 1 1 1
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 1 1 1 0 0 0 0 0 0;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 0 0 0 1 1 1 0 0 0;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 0 0 0 0 0 0 1 1 1
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 0 0 0 0 0 0 0 0 0;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 0 0 0 0 0 0 0 0 0;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 1 1 1 1 1 1 1 1 1
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 0 0 0 0 0 0 0 0 0;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 1 1 1 1 1 1 1 1 1
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
                    marginal1 = Marginal(
                        IntervalAmbiguitySets(;
                            lower = N[
                                1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6;
                                1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10;
                                4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10
                            ],
                            upper = N[
                                7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30;
                                8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30;
                                11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15
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
                                1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15;
                                3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15;
                                3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15
                            ],
                            upper = N[
                                2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30;
                                1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3;
                                17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3
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
                                4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15;
                                2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15;
                                1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30
                            ],
                            upper = N[
                                3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30;
                                3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30;
                                3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30
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
                    prop = FiniteTimeSafety([(i, j, 3) for i in 1:3 for j in 1:3], 10)
                    spec = Specification(prop, Pessimistic, Maximize)
                    prob = VerificationProblem(mdp, spec)
                    implicit_prob = VerificationProblem(implicit_mdp, spec)
                    (V, k, res) = solve(prob, alg)
                    (V_implicit, k_implicit, res_implicit) = solve(implicit_prob, alg)
                    @test V ≈ V_implicit
                    @test k == k_implicit
                    @test res ≈ res_implicit
                end
            end
            @testset "4D abstraction" begin
                rng = MersenneTwister(995)
                prob_lower = [rand(rng, N, 3, 81) ./ N(3) for _ in 1:4]
                prob_upper = [(rand(rng, N, 3, 81) .+ N(1)) ./ N(3) for _ in 1:4]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    4,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(ambiguity_sets[i], (1, 2, 3, 4), (1,), (3, 3, 3, 3), (1,))
                        end
                    ),
                    4,
                )
                mdp = FactoredRobustMarkovDecisionProcess((3, 3, 3, 3), (1,), marginals)
                prop = FiniteTimeReachability([(3, 3, 3, 3)], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                prob = VerificationProblem(mdp, spec)
                (V_ortho, it_ortho, res_ortho) = solve(prob, alg)
                @test V_ortho[3, 3, 3, 3] ≈ one(N)
                @test all(V_ortho .>= zero(N))
                @test all(V_ortho .<= one(N))
                if !(bellman_algorithm(alg) isa VertexEnumeration)
                    prob_lower_simple = zeros(N, 81, 81)
                    prob_upper_simple = zeros(N, 81, 81)
                    lin = LinearIndices((3, 3, 3, 3))
                    act_idx = CartesianIndex(1)
                    for I in CartesianIndices((3, 3, 3, 3))
                        for J in CartesianIndices((3, 3, 3, 3))
                            marginal_ambiguity_sets = map((marginal->begin
                                marginal[act_idx, I]
                            end), marginals)
                            prob_lower_simple[lin[J], lin[I]] =
                                prod((lower(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
                            prob_upper_simple[lin[J], lin[I]] =
                                prod((upper(marginal_ambiguity_sets[i], J[i]) for i in 1:4))
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
                    (V_direct, it_direct, res_direct) = solve(prob, alg)
                    @test V_direct[81] ≈ one(N)
                    @test all(V_ortho .≥ reshape(V_direct, 3, 3, 3, 3))
                end
            end
            @testset "synthesis" begin
                rng = MersenneTwister(3286)
                num_states_per_axis = 3
                num_axis = 3
                num_states = num_states_per_axis ^ num_axis
                num_actions = 2
                num_choices = num_states * num_actions
                state_indices = (1, 2, 3)
                action_indices = (1,)
                state_vars = ntuple((_->begin
                    num_states_per_axis
                end), num_axis)
                action_vars = (num_actions,)
                prob_lower = [
                    rand(rng, N, num_states_per_axis, num_choices) ./ num_states_per_axis for _ in 1:num_axis
                ]
                prob_upper = [
                    (rand(rng, N, num_states_per_axis, num_choices) .+ N(1)) ./
                    num_states_per_axis for _ in 1:num_axis
                ]
                ambiguity_sets = ntuple(
                    (
                        i->begin
                            IntervalAmbiguitySets(;
                                lower = prob_lower[i],
                                upper = prob_upper[i],
                            )
                        end
                    ),
                    num_axis,
                )
                marginals = ntuple(
                    (
                        i->begin
                            Marginal(
                                ambiguity_sets[i],
                                state_indices,
                                action_indices,
                                state_vars,
                                action_vars,
                            )
                        end
                    ),
                    num_axis,
                )
                mdp =
                    FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)
                prop = FiniteTimeReachability(
                    [(num_states_per_axis, num_states_per_axis, num_states_per_axis)],
                    10,
                )
                spec = Specification(prop, Pessimistic, Maximize)
                prob = ControlSynthesisProblem(mdp, spec)
                (policy, V, it, res) = solve(prob, alg)
                @test it == 10
                @test all(V .≥ 0.0)
                prob = VerificationProblem(mdp, spec, policy)
                (V_mc, k, res) = solve(prob, alg)
                @test V ≈ V_mc
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
                        1 // 15 3 // 10 1 // 15 3 // 10 1 // 30 1 // 3 7 // 30 4 // 15 1 // 6 1 // 5 1 // 10 1 // 5 0 7 // 30 7 // 30 1 // 5 2 // 15 1 // 6 1 // 10 1 // 30 1 // 10 1 // 15 1 // 10 1 // 15 4 // 15 4 // 15 1 // 3;
                        1 // 5 4 // 15 1 // 10 1 // 5 3 // 10 3 // 10 1 // 10 1 // 15 3 // 10 3 // 10 7 // 30 1 // 5 1 // 10 1 // 5 1 // 5 1 // 30 1 // 5 3 // 10 1 // 5 1 // 5 1 // 10 1 // 30 4 // 15 1 // 10 1 // 5 1 // 6 7 // 30;
                        4 // 15 1 // 30 1 // 5 1 // 5 7 // 30 4 // 15 2 // 15 7 // 30 1 // 5 1 // 3 2 // 15 1 // 6 1 // 6 1 // 3 4 // 15 3 // 10 1 // 30 3 // 10 3 // 10 1 // 10 1 // 15 1 // 30 2 // 15 1 // 6 1 // 5 1 // 10 4 // 15
                    ],
                    upper = N[
                        7 // 15 17 // 30 13 // 30 3 // 5 17 // 30 17 // 30 17 // 30 13 // 30 3 // 5 2 // 3 11 // 30 7 // 15 0 1 // 2 17 // 30 13 // 30 7 // 15 13 // 30 17 // 30 13 // 30 2 // 5 2 // 5 2 // 3 2 // 5 17 // 30 2 // 5 19 // 30;
                        8 // 15 1 // 2 3 // 5 7 // 15 8 // 15 17 // 30 2 // 3 17 // 30 11 // 30 7 // 15 19 // 30 19 // 30 13 // 15 1 // 2 17 // 30 13 // 30 3 // 5 11 // 30 8 // 15 7 // 15 7 // 15 13 // 30 8 // 15 2 // 5 8 // 15 17 // 30 3 // 5;
                        11 // 30 1 // 3 2 // 5 8 // 15 7 // 15 3 // 5 2 // 3 17 // 30 2 // 3 8 // 15 2 // 15 3 // 5 2 // 3 3 // 5 17 // 30 2 // 3 7 // 15 8 // 15 2 // 5 2 // 5 11 // 30 17 // 30 17 // 30 1 // 2 2 // 5 19 // 30 13 // 30
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
                        1 // 10 1 // 15 3 // 10 0 1 // 6 1 // 15 1 // 15 1 // 6 1 // 6 1 // 30 1 // 10 1 // 10 1 // 3 2 // 15 3 // 10 4 // 15 2 // 15 2 // 15 1 // 6 7 // 30 1 // 15 2 // 15 1 // 10 1 // 3 7 // 30 1 // 30 7 // 30;
                        3 // 10 1 // 5 3 // 10 2 // 15 0 1 // 30 0 1 // 15 1 // 30 7 // 30 1 // 30 1 // 15 7 // 30 1 // 15 1 // 6 1 // 30 1 // 10 1 // 15 3 // 10 0 3 // 10 1 // 6 3 // 10 1 // 5 0 7 // 30 2 // 15;
                        3 // 10 4 // 15 1 // 10 3 // 10 2 // 15 1 // 3 3 // 10 1 // 10 1 // 6 3 // 10 7 // 30 1 // 6 1 // 15 1 // 15 1 // 10 1 // 5 1 // 5 4 // 15 1 // 15 1 // 3 2 // 15 1 // 15 1 // 5 1 // 5 1 // 15 7 // 30 1 // 15
                    ],
                    upper = N[
                        2 // 5 17 // 30 3 // 5 11 // 30 3 // 5 7 // 15 19 // 30 2 // 5 3 // 5 2 // 3 2 // 3 8 // 15 8 // 15 19 // 30 8 // 15 8 // 15 13 // 30 13 // 30 13 // 30 17 // 30 17 // 30 13 // 30 11 // 30 19 // 30 8 // 15 2 // 5 8 // 15;
                        1 // 3 13 // 30 11 // 30 2 // 5 2 // 3 2 // 3 0 13 // 30 1 // 2 17 // 30 17 // 30 1 // 3 2 // 5 1 // 3 13 // 30 11 // 30 8 // 15 1 // 3 1 // 2 8 // 15 8 // 15 8 // 15 8 // 15 2 // 5 3 // 5 2 // 3 13 // 30;
                        17 // 30 3 // 5 8 // 15 1 // 2 7 // 15 1 // 2 2 // 3 17 // 30 11 // 30 2 // 5 1 // 2 7 // 15 2 // 5 17 // 30 11 // 30 2 // 5 11 // 30 2 // 3 1 // 3 2 // 3 17 // 30 8 // 15 17 // 30 3 // 5 2 // 5 19 // 30 11 // 30
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
                        4 // 15 1 // 5 3 // 10 3 // 10 4 // 15 7 // 30 1 // 5 4 // 15 7 // 30 1 // 6 1 // 5 0 1 // 15 1 // 30 3 // 10 1 // 3 2 // 15 1 // 15 7 // 30 4 // 15 1 // 10 1 // 3 1 // 5 7 // 30 1 // 30 1 // 5 7 // 30;
                        2 // 15 4 // 15 1 // 10 1 // 30 7 // 30 2 // 15 1 // 15 1 // 30 3 // 10 1 // 3 1 // 5 1 // 10 2 // 15 1 // 30 2 // 15 4 // 15 0 4 // 15 1 // 5 4 // 15 1 // 10 1 // 10 1 // 3 7 // 30 3 // 10 1 // 3 3 // 10;
                        1 // 5 1 // 3 3 // 10 1 // 10 1 // 15 1 // 10 1 // 30 1 // 5 2 // 15 7 // 30 1 // 3 2 // 15 1 // 10 1 // 6 3 // 10 1 // 5 7 // 30 1 // 30 0 1 // 30 1 // 15 2 // 15 1 // 6 7 // 30 4 // 15 4 // 15 7 // 30
                    ],
                    upper = N[
                        3 // 5 17 // 30 1 // 2 3 // 5 19 // 30 2 // 5 8 // 15 1 // 3 11 // 30 2 // 5 17 // 30 13 // 30 2 // 5 3 // 5 3 // 5 11 // 30 1 // 2 11 // 30 2 // 3 17 // 30 3 // 5 7 // 15 19 // 30 1 // 2 3 // 5 1 // 3 19 // 30;
                        3 // 5 2 // 3 13 // 30 19 // 30 1 // 3 2 // 5 17 // 30 7 // 15 11 // 30 3 // 5 19 // 30 7 // 15 2 // 5 8 // 15 17 // 30 11 // 30 19 // 30 13 // 30 2 // 3 17 // 30 8 // 15 13 // 30 13 // 30 3 // 5 1 // 2 8 // 15 8 // 15;
                        3 // 5 2 // 3 1 // 2 1 // 2 2 // 3 7 // 15 3 // 5 3 // 5 1 // 2 1 // 3 2 // 5 8 // 15 2 // 5 11 // 30 1 // 3 8 // 15 7 // 15 13 // 30 0 2 // 5 11 // 30 19 // 30 19 // 30 2 // 5 1 // 2 7 // 15 7 // 15
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
            @testset "maximization" begin
                ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                V_vertex = zeros(N, 3, 3, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(V_vertex), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_McCormick = zeros(N, 3, 3, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_McCormick), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_McCormick .>= 0.0)
                @test all(Vres_first_McCormick .<= maximum(V))
                @test all(Vres_first_McCormick .+ epsilon .>= V_vertex)
                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.construct_workspace(mdp, OMaximization())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_OMax = zeros(N, 3, 3, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_OMax), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_OMax .>= 0.0)
                @test all(Vres_first_OMax .<= maximum(V))
                @test all(Vres_first_OMax .+ epsilon .>= V_vertex)
                ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                @test Vres ≈ Vres_first_OMax
                ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = true,)
                @test Vres ≈ Vres_first_OMax
            end
            @testset "minimization" begin
                ws = IntervalMDP.construct_workspace(mdp, VertexEnumeration())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                V_vertex = zeros(N, 3, 3, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(V_vertex), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                ws = IntervalMDP.construct_workspace(mdp, LPMcCormickRelaxation())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_McCormick = zeros(N, 3, 3, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_McCormick), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_McCormick .>= 0.0)
                @test all(Vres_first_McCormick .<= maximum(V))
                @test all(Vres_first_McCormick .- epsilon .<= V_vertex)
                ws = IntervalMDP.FactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.ThreadedFactoredIntervalMcCormickWorkspace(
                    mdp,
                    LPMcCormickRelaxation(),
                )
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_McCormick)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                @test Vres ≈ Vres_first_McCormick
                ws = IntervalMDP.construct_workspace(mdp, OMaximization())
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres_first_OMax = zeros(N, 3, 3, 3)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres_first_OMax), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                epsilon = if N == Float32
                    1.0e-5
                else
                    1.0e-8
                end
                @test all(Vres_first_OMax .>= 0.0)
                @test all(Vres_first_OMax .<= maximum(V))
                @test all(Vres_first_OMax .- epsilon .<= V_vertex)
                ws = IntervalMDP.FactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                @test Vres ≈ Vres_first_OMax
                ws = IntervalMDP.ThreadedFactoredIntervalOMaxWorkspace(mdp)
                strategy_cache = IntervalMDP.construct_strategy_cache(mdp)
                Vres = similar(Vres_first_OMax)
                IntervalMDP.bellman_q!(ws, strategy_cache, IntervalMDP.StateActionValueArray(Vres), IntervalMDP.StateValueArray(V), mdp; upper_bound = false,)
                @test Vres ≈ Vres_first_OMax
            end
        end
    end
end
