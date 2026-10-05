using Test, JuliaConfig
@testset "config fixture" begin
    @test bellman_step([0.0], ones(1, 1), [1.0]) == [1.0]
end
