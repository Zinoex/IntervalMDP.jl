using Revise, Test
using IntervalMDP

prob = IntervalProbabilities(;
    lower = [0.0 0.5; 0.1 0.3; 0.2 0.1],
    upper = [0.5 0.7; 0.6 0.5; 0.7 0.3],
)

V = [1.0, 2.0, 3.0]

#### Maximization
@testset "maximization" begin
    Vres = bellman(V, prob; upper_bound = true)
    @test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

    Vres = similar(Vres)
    bellman!(Vres, V, prob; upper_bound = true)
    @test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

    ws = construct_workspace(prob)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
    @test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

    ws = IntervalMDP.DenseWorkspace(gap(prob), 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
    @test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

    ws = IntervalMDP.ThreadedDenseWorkspace(gap(prob), 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
    @test Vres ≈ [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]
end

#### Minimization
@testset "minimization" begin
    Vres = bellman(V, prob; upper_bound = false)
    @test Vres ≈ [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

    Vres = similar(Vres)
    bellman!(Vres, V, prob; upper_bound = false)
    @test Vres ≈ [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

    ws = construct_workspace(prob)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = false)
    @test Vres ≈ [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

    ws = IntervalMDP.DenseWorkspace(gap(prob), 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = false)
    @test Vres ≈ [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

    ws = IntervalMDP.ThreadedDenseWorkspace(gap(prob), 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = false)
    @test Vres ≈ [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]
end
