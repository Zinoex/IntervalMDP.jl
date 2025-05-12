using Revise, Test
using IntervalMDP

@testset "construction" begin
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
