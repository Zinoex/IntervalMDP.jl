using Revise, Test
using IntervalMDP

@testset "construction prodIMDP DFA" begin
    # dfa
    T = UInt16[
        1 3 3
        2 1 3
        3 3 3
    ]

    tr = TransitionFunction(T)
    iztates = Int32(1)
    aztates = Int32[3]
    alphabet = String["a", "b", "ab"]

    dfa = DFA(tr, iztates, aztates, alphabet)

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

        prodIMDP = ProductIntervalMarkovDecisionProcessDFA(mdp, dfa, lf)

        @test imdp(prodIMDP) == mdp
        @test automaton(prodIMDP) == dfa
        @test labelling_function(prodIMDP) == lf
    end

    @testset "IMDP state labelling func input mismatch" begin

        # labelling
        map = UInt16[1, 2]
        lf = LabellingFunction(map)

        @test_throws DimensionMismatch ProductIntervalMarkovDecisionProcessDFA(mdp, dfa, lf)
    end

    @testset "DFA inputs labelling func output mismatch (more output than inputs)" begin

        # labelling
        map = UInt16[1, 2, 4]
        lf = LabellingFunction(map)

        @test_throws DimensionMismatch ProductIntervalMarkovDecisionProcessDFA(mdp, dfa, lf)
    end
end
