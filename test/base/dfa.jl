@testitem "base/dfa: transition function — good case" begin
    T = UInt16[
        1 3 3
        2 1 3
        3 3 3
    ]

    tf = TransitionFunction(T)

    @test transition(tf) == T
end

@testitem "base/dfa: transition function — not positive matrix vals" begin
    T = UInt16[
        1 3 3
        2 1 3
        3 0 3
    ]

    @test_throws ArgumentError TransitionFunction(T)
end

@testitem "base/dfa: transition function — matrix val > |Z|" begin
    T = UInt16[
        1 3 3
        2 1 3
        3 4 3
    ]

    @test_throws ArgumentError TransitionFunction(T)
end

@testitem "base/dfa: transition function — indexing" begin
    T = UInt16[
        1 3 3
        2 1 3
        3 3 3
    ]

    tf = TransitionFunction(T)

    @test tf[2, 1] == 3
    @test tf[1, 2] == 2
end

@testitem "base/dfa: dfa — labels" begin
    atomic_props = ["a", "b"]
    map = Dict("" => 1, "a" => 2, "b" => 3, "ab" => 4)

    @test map == IntervalMDP.atomicpropositions2labels(atomic_props)
end

@testitem "base/dfa: dfa — good case" begin
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

    dfa = DFA(delta, istate, atomic_props)
    Ns, Na = size(dfa)

    @test transition(dfa) == delta
    @test initial_state(dfa) == istate
    @test labelmap(dfa) == map
    @test Ns == 3
    @test Na == 4
end

@testitem "base/dfa: dfa — alphabet size mismatch" begin
    atomic_props = ["a", "b"]

    T = UInt16[
        1 3 3
        2 1 3
        3 3 3
        1 1 1
    ]

    delta = TransitionFunction(T)
    istate = Int32(1)

    @test_throws DimensionMismatch DFA(delta, istate, ["a", "b", "c"])
    @test_throws DimensionMismatch DFA(delta, istate, ["a"])
end

@testitem "base/dfa: dfa — bad initial state" begin
    atomic_props = ["a", "b"]

    T = UInt16[
        1 3 3
        2 1 3
        3 3 3
        1 1 1
    ]

    delta = TransitionFunction(T)

    @test_throws ArgumentError DFA(delta, Int32(0), atomic_props)
    @test_throws ArgumentError DFA(delta, Int32(4), atomic_props)
end

@testitem "base/dfa: dfa — indexing" begin
    atomic_props = ["a", "b"]

    T = UInt16[
        1 3 3
        2 1 3
        3 3 3
        1 1 1
    ]

    delta = TransitionFunction(T)
    istate = Int32(1)

    dfa = DFA(delta, istate, atomic_props)

    @test dfa[2, "ab"] == 1
    @test dfa[2, 1] == 3
end
