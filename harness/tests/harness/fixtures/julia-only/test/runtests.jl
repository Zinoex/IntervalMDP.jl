using Test, JuliaOnly
@testset "JuliaOnly fixture" begin
    P = [0.5 0.5; 0.0 1.0]
    @test bellman_step([1.0, 0.0], P, [0.0, 0.0]) == [1.0, 0.0]
    @test bellman_step([0.0, 0.0], P, [2.0, 4.0]) == [3.0, 4.0]
    @test bellman_step(Float32[1, 1], Float32[1 0; 0 1], Float32[1, 2]) isa Vector{Float32}
end
