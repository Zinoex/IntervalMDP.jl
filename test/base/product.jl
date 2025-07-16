using Revise, Test
using IntervalMDP

@testset "construction product IMDP/DFA" begin
    # dfa
    T = UInt16[
        1 3 3
        2 1 3
        3 3 3
        1 1 1
    ]

    delta = TransitionFunction(T)
    istate = Int32(1)
    atomic_props = ["a", "b"]

    dfa = DFA(delta, istate, atomic_props)

    # imdp
    prob1 = IntervalProbabilities(;
        lower = [
            0.0 0.5
            0.1 0.3
            0.2 0.1
        ],
        upper = [
            0.5 0.7
            0.6 0.5
            0.7 0.3
        ],
    )

    prob2 = IntervalProbabilities(;
        lower = [
            0.1 0.2
            0.2 0.3
            0.3 0.4
        ],
        upper = [
            0.6 0.6
            0.5 0.5
            0.4 0.4
        ],
    )

    prob3 = IntervalProbabilities(; lower = [
        0.0
        0.0
        1.0
    ][:, :], upper = [
        0.0
        0.0
        1.0
    ][:, :])

    transition_probs = [prob1, prob2, prob3]
    istates = [Int32(1)]

    mdp = IntervalMarkovDecisionProcess(transition_probs, istates)

    @testset "good case" begin

        # labelling
        map = UInt16[1, 2, 3]
        lf = LabellingFunction(map)

        prodIMDP = ProductProcess(mdp, dfa, lf)

        @test markov_process(prodIMDP) == mdp
        @test automaton(prodIMDP) == dfa
        @test labelling_function(prodIMDP) == lf
    end

    @testset "IMDP state labelling func input mismatch" begin

        # labelling
        map = UInt16[1, 2]
        lf = LabellingFunction(map)

        @test_throws DimensionMismatch ProductProcess(mdp, dfa, lf)
    end

    @testset "DFA inputs labelling func output mismatch (more output than inputs)" begin
        T = UInt16[
            1 2 2
            2 1 2
        ]

        delta = TransitionFunction(T)

        istate = Int32(1)
        atomic_props = ["a"]
        dfa = DFA(delta, istate, atomic_props)

        # labelling
        map = UInt16[1, 2, 3]
        lf = LabellingFunction(map)

        @test_throws DimensionMismatch ProductProcess(mdp, dfa, lf)
    end
end

@testset "bellman" begin
    for N in [Float32, Float64, Rational{BigInt}]
        @testset "N = $N" begin
            prob = IntervalProbabilities(;
                lower = N[
                    0 5//10 0
                    1//10 3//10 0
                    2//10 1//10 1
                ],
                upper = N[
                    5//10 7//10 0
                    6//10 5//10 0
                    7//10 3//10 1
                ],
            )
            mc = IntervalMarkovChain(prob)

            # Product model - just simple reachability
            delta = TransitionFunction(Int32[
                1 2
                2 2
            ])
            istate = Int32(1)
            atomic_props = ["reach"]
            dfa = DFA(delta, istate, atomic_props)

            labelling = LabellingFunction(Int32[1, 1, 2])

            prod_proc = ProductProcess(mc, dfa, labelling)

            V = N[
                4 1
                2 3
                0 5
            ]

            Vres = IntervalMDP.bellman(V, prod_proc; upper_bound = false)

            @test Vres ≈ N[
                30//10 24//10
                33//10 2
                5 5
            ]
        end
    end
end

@testset "value iteration" begin
    for N in [Float32, Float64, Rational{BigInt}]
        @testset "N = $N" begin
            prob1 = IntervalProbabilities(;
                lower = N[
                    0//10 5//10
                    1//10 3//10
                    2//10 1//10
                ],
                upper = N[
                    5//10 7//10
                    6//10 5//10
                    7//10 3//10
                ],
            )

            prob2 = IntervalProbabilities(;
                lower = N[
                    1//10 2//10
                    2//10 3//10
                    3//10 4//10
                ],
                upper = N[
                    6//10 6//10
                    5//10 5//10
                    4//10 4//10
                ],
            )

            prob3 = IntervalProbabilities(; lower = N[
                0
                0
                1
            ][:, :], upper = N[
                0
                0
                1
            ][:, :])

            transition_probs = [prob1, prob2, prob3]
            mdp = IntervalMarkovDecisionProcess(transition_probs)

            # Product model - just simple reachability
            delta = TransitionFunction(Int32[
                1 2
                2 2
            ])
            istate = Int32(1)
            atomic_props = ["reach"]
            dfa = DFA(delta, istate, atomic_props)

            labelling = LabellingFunction(Int32[1, 1, 2])

            prod_proc = ProductProcess(mdp, dfa, labelling)

            @testset "finite time reachability" begin
                prop = FiniteTimeDFAReachability([2], 10)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = ControlSynthesisProblem(prod_proc, spec)

                policy, V_fixed_it1, k, res = solve(problem)

                @test all(V_fixed_it1 .>= 0)
                @test k == 10
                @test V_fixed_it1[:, 2] == N[1, 1, 1]

                problem = VerificationProblem(prod_proc, spec, policy)
                V_mc, k, res = solve(problem)

                @test V_fixed_it1 ≈ V_mc

                prop = FiniteTimeDFAReachability([2], 11)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = VerificationProblem(prod_proc, spec)

                V_fixed_it2, k, res = solve(problem)

                @test all(V_fixed_it2 .>= 0)
                @test k == 11
                @test V_fixed_it2[:, 2] == N[1, 1, 1]
                @test all(V_fixed_it2 .>= V_fixed_it1)
            end

            @testset "infinite time reachability" begin
                prop = InfiniteTimeDFAReachability([2], 1e-3)
                spec = Specification(prop, Pessimistic, Maximize)
                problem = ControlSynthesisProblem(prod_proc, spec)

                policy, V_conv, k, res = solve(problem)

                @test all(V_conv .>= 0)
                @test maximum(res) <= 1e-3
                @test V_conv[:, 2] == N[1, 1, 1]

                problem = VerificationProblem(prod_proc, spec, policy)
                V_mc, k, res = solve(problem)

                @test V_conv ≈ V_mc
            end
        end
    end
end
