using Test, JuliaFailing
@testset "JuliaFailing fixture (intentionally fails)" begin
    P = [0.5 0.5; 0.0 1.0]
    # Deliberately wrong expected value: drives the "QE fails -> no Ops" case.
    @test bellman_step([0.0, 0.0], P, [2.0, 4.0]) == [999.0, 4.0]
end
