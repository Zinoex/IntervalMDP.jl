using Test, JuliaPerfBad
@testset "brittle timing (must be flagged)" begin
    P = ones(2, 2)
    @test @elapsed(bellman_step(ones(2), P, ones(2))) < 0.5
    t = @elapsed bellman_step(ones(2), P, ones(2))
    @test t < 1.0
end
