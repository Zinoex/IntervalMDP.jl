#### Maximization
@testset "maximization" begin
    prob = IntervalMDP.cu(
        IntervalProbabilities(;
            lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
            upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
        ),
    )

    V = IntervalMDP.cu([1.0, 2.0, 3.0])

    Vres = bellman(V, prob; upper_bound = true)
    Vres = Vector(Vres)
    @test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

    # To GPU first
    prob = IntervalProbabilities(;
        lower = IntervalMDP.cu([0.0 0.5; 0.1 0.3; 0.2 0.1]),
        upper = IntervalMDP.cu([0.5 0.7; 0.6 0.5; 0.7 0.3]),
    )

    V = IntervalMDP.cu([1.0, 2.0, 3.0])

    Vres = bellman(V, prob; upper_bound = true)
    Vres = Vector(Vres)
    @test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]
end

#### Minimization
@testset "minimization" begin
    prob = IntervalMDP.cu(
        IntervalProbabilities(;
            lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
            upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
        ),
    )

    V = IntervalMDP.cu([1.0, 2.0, 3.0])

    Vres = bellman(V, prob; upper_bound = false)
    Vres = Vector(Vres)
    @test Vres ≈ [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]
end
