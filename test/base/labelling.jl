using Revise, Test
using IntervalMDP

@testset "1d" begin
    a = UInt16[1, 3, 2, 2]
    lf = LabellingFunction(a)

    @test mapping(lf) == a
    @test size(lf) == (4,)
    @test num_labels(lf) == 3

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
    lf = LabellingFunction(a)

    @test mapping(lf) == a
    @test size(lf) == (3, 4)
    @test num_labels(lf) == 5

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
    @test_throws ArgumentError LabellingFunction(UInt16[0, 1, 2])
    @test_throws ArgumentError LabellingFunction(UInt16[1, 2, 4])
    @test_throws ArgumentError LabellingFunction(UInt16[1, 2, 3, 5])
end
