using Revise, Test
using IntervalMDP

@testset "transition function" begin
    @testset "good case" begin
        T = UInt16[
            1 3 3
            2 1 3
            3 3 3
        ]

        tf = TransitionFunction(T)

        @test transition(tf) == T
    end

    @testset "not positive matrix vals" begin
        T = UInt16[
            1 3 3
            2 1 3
            3 0 3
        ]

        @test_throws ArgumentError TransitionFunction(T)
    end

    @testset "matrix val > |Z|" begin
        T = UInt16[
            1 3 3
            2 1 3
            3 4 3
        ]

        @test_throws ArgumentError TransitionFunction(T)
    end

    @testset "indexing" begin
        T = UInt16[
            1 3 3
            2 1 3
            3 3 3
        ]

        tf = TransitionFunction(T)

        @test tf[2, 1] == 3
        @test tf[1, 2] == 2
    end
end

@testset "dfa" begin
    atomic_props = ["a", "b"]
    map = Dict("" => 1, "a" => 2, "b" => 3, "ab" => 4)

    T = UInt16[
        1 3 3
        2 1 3
        3 3 3
        1 1 1
    ]

    delta = TransitionFunction(T)
    istate = Int32(1)

    @testset "labels" begin
        @test map == IntervalMDP.atomicpropositions2labels(atomic_props)
    end

    @testset "good case" begin
        dfa = DFA(delta, istate, atomic_props)
        Ns, Na = size(dfa)

        @test transition(dfa) == delta
        @test initial_state(dfa) == istate
        @test labelmap(dfa) == map
        @test Ns == 3
        @test Na == 4
    end

    @testset "alphabet size mismatch" begin
        @test_throws DimensionMismatch DFA(delta, istate, ["a", "b", "c"])
        @test_throws DimensionMismatch DFA(delta, istate, ["a"])
    end

    @testset "bad initial state" begin
        @test_throws ArgumentError DFA(delta, Int32(0), atomic_props)
        @test_throws ArgumentError DFA(delta, Int32(4), atomic_props)
    end

    @testset "indexing" begin
        dfa = DFA(delta, istate, atomic_props)

        @test dfa[2, "ab"] == 1
        @test dfa[2, 1] == 3
    end
end
