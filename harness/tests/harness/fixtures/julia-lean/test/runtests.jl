using Test, JuliaLean
@testset "JuliaLean fixture" begin
    @test bellman_step([1.0], ones(1, 1), [2.0]) == [3.0]
end
