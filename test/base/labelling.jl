using Revise, Test
using IntervalMDP

@testset "DeterministicLabelling" begin
    @testset "1d" begin
        a = UInt16[1, 3, 2, 2]
        lf = DeterministicLabelling(a)

        @test mapping(lf) == a
        @test size(lf) == (4,)
        @test num_labels(lf) == 3
        @test state_values(lf) == (4,)

        @test lf[1] == 1
        @test lf[2] == 3
        @test lf[3] == 2
        @test lf[4] == 2
    end

    @testset "2d" begin
        a = UInt16[
            1 3 2 2
            1 3 2 5
            1 3 4 2
        ]
        lf = DeterministicLabelling(a)

        @test mapping(lf) == a
        @test size(lf) == (3, 4)
        @test num_labels(lf) == 5
        @test state_values(lf) == (3, 4)

        @test lf[1, 1] == 1
        @test lf[1, 2] == 3
        @test lf[1, 3] == 2
        @test lf[1, 4] == 2

        @test lf[2, 1] == 1
        @test lf[2, 2] == 3
        @test lf[2, 3] == 2
        @test lf[2, 4] == 5

        @test lf[3, 1] == 1
        @test lf[3, 2] == 3
        @test lf[3, 3] == 4
        @test lf[3, 4] == 2
    end

    @testset "invalid labelling" begin
        @test_throws ArgumentError DeterministicLabelling(UInt16[0, 1, 2])
        @test_throws ArgumentError DeterministicLabelling(UInt16[1, 2, 4])
        @test_throws ArgumentError DeterministicLabelling(UInt16[1, 2, 3, 5])
    end
end

@testset "ProbabilisticLabelling" begin
    @testset "good case 1d" begin
        m = Float32[
            0.5 0.2 1 0.3
            0 0.7 0 0.4
            0.5 0.1 0 0.3
        ]

        dl = ProbabilisticLabelling(m)

        @test mapping(dl) == m
        @test size(dl) == (3, 4)
        @test size(dl, 1) == 3
        @test size(dl, 2) == 4
        @test num_labels(dl) == 3
        @test num_states(dl) == 4
        @test state_values(dl) == (4,)

        @test dl[1, 1] ≈ 0.5
        @test dl[1, 2] ≈ 0.2
        @test dl[1, 3] ≈ 1
        @test dl[1, 4] ≈ 0.3

        @test dl[2, 1] ≈ 0.0
        @test dl[2, 2] ≈ 0.7
        @test dl[2, 3] ≈ 0.0
        @test dl[2, 4] ≈ 0.4

        @test dl[3, 1] ≈ 0.5
        @test dl[3, 2] ≈ 0.1
        @test dl[3, 3] ≈ 0.0
        @test dl[3, 4] ≈ 0.3
    end

    @testset "invalid probabilities" begin
        m1 = Float32[
            0.5 1
            0.5 0
            0.1 0
        ]

        m2 = Float32[
            0.4 1
            0.5 0
            0 0
        ]

        @test_throws ArgumentError ProbabilisticLabelling(m1)
        @test_throws ArgumentError ProbabilisticLabelling(m2)
    end
end
