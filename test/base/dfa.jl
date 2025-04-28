using Revise, Test
using IntervalMDP

@testset "construction" begin
    alphabet = String["a", "b", "ab"]
    N = 3
    map = Dict{String, Int32}("a" => 1, "b" => 2, "ab" => 3)

    T = UInt16[
        1 3 3
        2 1 3
        3 3 3
    ]

    tr = TransitionFunction(T)
    is = Int32(1)
    as = Int32[3]

    @testset "alphabet indexing" begin
        @test (map, N) == alphabet2index(alphabet)
    end

    @testset "good case" begin
        dfa = DFA(tr, is, as, alphabet)
        Ns, Na = size(dfa)

        @test transition(dfa) == tr
        @test initial_state(dfa) == is
        @test accepting_states(dfa) == as
        @test alphabetptr(dfa) == map
        @test Ns == N
        @test Na == N
    end

    @testset "alphabet size mismatch" begin
        @test_throws DimensionMismatch DFA(tr, is, as, String["a", "b", "ab", "c"])
        @test_throws DimensionMismatch DFA(tr, is, as, String["a", "b"])
    end

    @testset "bad accepting states" begin
        @test_throws ArgumentError DFA(tr, is, Int32[0], alphabet)
        @test_throws ArgumentError DFA(tr, is, Int32[4], alphabet)
        @test_throws ArgumentError DFA(tr, is, Int32[1, 2, 3, 3], alphabet)
    end

    @testset "bad initial state" begin
        @test_throws ArgumentError DFA(tr, Int32(0), as, alphabet)
        @test_throws ArgumentError DFA(tr, Int32(4), as, alphabet)
    end

    @testset "indexing" begin
        dfa = DFA(tr, is, as, alphabet)

        z = 3
        @test dfa[2, "ab"] == z
        @test dfa[2, 3] == z
    end
end
