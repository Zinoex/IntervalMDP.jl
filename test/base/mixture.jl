using Revise, Test
using IntervalMDP, SparseArrays
using Random: MersenneTwister

for N in [Float32, Float64]
    @testset "N = $N" begin
        @testset "bellman 1d" begin
            prob1 = OrthogonalIntervalProbabilities(
                (
                    IntervalProbabilities(;
                        lower = N[
                            0.0 0.5
                            0.1 0.3
                            0.2 0.1
                        ],
                        upper = N[
                            0.5 0.7
                            0.6 0.5
                            0.7 0.3
                        ],
                    ),
                ),
                (Int32(2),),
            )
            prob2 = OrthogonalIntervalProbabilities(
                (
                    IntervalProbabilities(;
                        lower = N[
                            0.1 0.4
                            0.2 0.2
                            0.3 0.0
                        ],
                        upper = N[
                            0.4 0.6
                            0.5 0.4
                            0.6 0.2
                        ],
                    ),
                ),
                (Int32(2),),
            )
            weighting_probs = IntervalProbabilities(;
                lower = N[
                    0.3 0.5
                    0.4 0.3
                ],
                upper = N[
                    0.8 0.7
                    0.7 0.5
                ],
            )
            mixture_prob = MixtureIntervalProbabilities((prob1, prob2), weighting_probs)

            V = N[1.0, 2.0, 3.0]

            @testset "maximization" begin
                Vexpected = N[
                    (0.0 * 1 + 0.3 * 2 + 0.7 * 3) * 0.6 + (0.1 * 1 + 0.3 * 2 + 0.6 * 3) * 0.4,
                    (0.5 * 1 + 0.3 * 2 + 0.2 * 3) * 0.5 + (0.4 * 1 + 0.4 * 2 + 0.2 * 3) * 0.5,
                ]

                ws = IntervalMDP.construct_workspace(mixture_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = zeros(N, 2)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = true,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.MixtureWorkspace(mixture_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = similar(Vres)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = true,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.ThreadedMixtureWorkspace(mixture_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = similar(Vres)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = true,
                )
                @test Vres ≈ Vexpected
            end

            @testset "minimization" begin
                Vexpected = N[
                    (0.5 * 1 + 0.3 * 2 + 0.2 * 3) * 0.6 + (0.4 * 1 + 0.3 * 2 + 0.3 * 3) * 0.4,
                    (0.6 * 1 + 0.3 * 2 + 0.1 * 3) * 0.5 + (0.6 * 1 + 0.4 * 2 + 0.0 * 3) * 0.5,
                ]

                ws = IntervalMDP.construct_workspace(mixture_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = zeros(N, 2)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = false,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.MixtureWorkspace(mixture_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = similar(Vres)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = false,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.ThreadedMixtureWorkspace(mixture_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = similar(Vres)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = false,
                )
                @test Vres ≈ Vexpected
            end
        end

        @testset "sparse bellman 1d" begin
            prob1 = OrthogonalIntervalProbabilities(
                (
                    IntervalProbabilities(;
                        lower = sparse(N[
                            0.0 0.5
                            0.1 0.3
                            0.2 0.1
                        ]),
                        upper = sparse(N[
                            0.5 0.7
                            0.6 0.5
                            0.7 0.3
                        ]),
                    ),
                ),
                (Int32(2),),
            )
            prob2 = OrthogonalIntervalProbabilities(
                (
                    IntervalProbabilities(;
                        lower = sparse(N[
                            0.1 0.4
                            0.2 0.2
                            0.3 0.0
                        ]),
                        upper = sparse(N[
                            0.4 0.6
                            0.5 0.4
                            0.6 0.2
                        ]),
                    ),
                ),
                (Int32(2),),
            )

            # Weighting probabilities are treated the same way for sparse and dense matrices.
            # This choice is made to simplify the implementation and since the number of
            # mixtures is typically small, with few non-zero entries.
            weighting_probs = IntervalProbabilities(;
                lower = N[
                    0.3 0.5
                    0.4 0.3
                ],
                upper = N[
                    0.8 0.7
                    0.7 0.5
                ],
            )
            mixture_prob = MixtureIntervalProbabilities((prob1, prob2), weighting_probs)

            V = N[1.0, 2.0, 3.0]

            @testset "maximization" begin
                Vexpected = N[
                    (0.0 * 1 + 0.3 * 2 + 0.7 * 3) * 0.6 + (0.1 * 1 + 0.3 * 2 + 0.6 * 3) * 0.4,
                    (0.5 * 1 + 0.3 * 2 + 0.2 * 3) * 0.5 + (0.4 * 1 + 0.4 * 2 + 0.2 * 3) * 0.5,
                ]

                ws = IntervalMDP.construct_workspace(mixture_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = zeros(N, 2)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = true,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.MixtureWorkspace(mixture_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = similar(Vres)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = true,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.ThreadedMixtureWorkspace(mixture_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = similar(Vres)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = true,
                )
                @test Vres ≈ Vexpected
            end

            @testset "minimization" begin
                Vexpected = N[
                    (0.5 * 1 + 0.3 * 2 + 0.2 * 3) * 0.6 + (0.4 * 1 + 0.3 * 2 + 0.3 * 3) * 0.4,
                    (0.6 * 1 + 0.3 * 2 + 0.1 * 3) * 0.5 + (0.6 * 1 + 0.4 * 2 + 0.0 * 3) * 0.5,
                ]

                ws = IntervalMDP.construct_workspace(mixture_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = zeros(N, 2)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = false,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.MixtureWorkspace(mixture_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = similar(Vres)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = false,
                )
                @test Vres ≈ Vexpected

                ws = IntervalMDP.ThreadedMixtureWorkspace(mixture_prob, 1)
                strategy_cache = IntervalMDP.construct_strategy_cache(mixture_prob)
                Vres = similar(Vres)
                IntervalMDP._bellman_helper!(
                    ws,
                    strategy_cache,
                    Vres,
                    V,
                    mixture_prob,
                    stateptr(mixture_prob);
                    upper_bound = false,
                )
                @test Vres ≈ Vexpected
            end
        end

        @testset "synthesis" begin
            rng = MersenneTwister(3286)

            num_states_per_axis = 3
            num_extended_states_per_axis = num_states_per_axis + 1
            num_axis = 3
            num_states = num_states_per_axis^num_axis
            num_actions = 2
            num_choices = num_states * num_actions

            prob_lower1 = [
                rand(rng, N, num_extended_states_per_axis, num_choices) ./
                num_extended_states_per_axis for _ in 1:num_axis
            ]
            prob_upper1 = [
                (rand(rng, N, num_extended_states_per_axis, num_choices) .+ N(1.0)) ./
                num_extended_states_per_axis for _ in 1:num_axis
            ]

            probs1 = OrthogonalIntervalProbabilities(
                ntuple(
                    i -> IntervalProbabilities(;
                        lower = prob_lower1[i],
                        upper = prob_upper1[i],
                    ),
                    num_axis,
                ),
                (
                    Int32(num_states_per_axis),
                    Int32(num_states_per_axis),
                    Int32(num_states_per_axis),
                ),
            )

            prob_lower2 = [
                rand(rng, N, num_extended_states_per_axis, num_choices) ./
                num_extended_states_per_axis for _ in 1:num_axis
            ]
            prob_upper2 = [
                (rand(rng, N, num_extended_states_per_axis, num_choices) .+ N(1.0)) ./
                num_extended_states_per_axis for _ in 1:num_axis
            ]

            probs2 = OrthogonalIntervalProbabilities(
                ntuple(
                    i -> IntervalProbabilities(;
                        lower = prob_lower2[i],
                        upper = prob_upper2[i],
                    ),
                    num_axis,
                ),
                (
                    Int32(num_states_per_axis),
                    Int32(num_states_per_axis),
                    Int32(num_states_per_axis),
                ),
            )

            num_mixtures = 2
            weighting_probs = IntervalProbabilities(;
                lower = rand(rng, N, num_mixtures, num_choices) ./ num_mixtures,
                upper = (rand(rng, N, num_mixtures, num_choices) .+ N(1.0)) ./ num_mixtures,
            )

            mixture_probs = MixtureIntervalProbabilities((probs1, probs2), weighting_probs)

            stateptr = [Int32[1]; convert.(Int32, 1 .+ collect(1:num_states) .* 2)]
            mdp = MixtureIntervalMarkovDecisionProcess(mixture_probs, stateptr)

            prop = FiniteTimeReachability(
                [(
                    num_extended_states_per_axis,
                    num_extended_states_per_axis,
                    num_extended_states_per_axis,
                )],
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
