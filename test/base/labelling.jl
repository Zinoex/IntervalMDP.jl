using Revise, Test
using IntervalMDP

@testset "construction" begin
    a = UInt16[1, 4, 8, 3]
    lf = LabellingFunction(a)
    Nin = 4
    Nout = 8

    @testset "count mapping" begin
        @test (Nin, Nout) == count_mapping(a)
    end

    @testset "good case" begin
        @test mapping(lf) == a
        @test (Nin, Nout) == size(lf)
    end

    @testset "indexing" begin
        @test lf[3] == 8
    end
end
