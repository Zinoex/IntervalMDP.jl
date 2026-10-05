using Test, JuliaPerfGood
@testset "deterministic checks only" begin
    P = ones(2, 2); r = ones(2); v = ones(2)
    @test bellman_step(r, P, v) == [3.0, 3.0]
    bellman_step(r, P, v)  # warm up
    @test (@allocated bellman_step(r, P, v)) < 10_000   # allocation guard is deterministic, allowed
end
