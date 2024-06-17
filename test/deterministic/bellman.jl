prob = Transitions(
    Int32[1, 3, 6],
    Int32[4, 10, 5, 6, 7],
    (15, 2)
)

V = collect(1.0:15.0)

#### Maximization
@testset "maximization" begin
    Vres = bellman(V, prob; upper_bound = true)
    @test Vres ≈ [10.0, 7.0]

    Vres = similar(Vres)
    bellman!(Vres, V, prob; upper_bound = true)
    @test Vres ≈ [10.0, 7.0]

    ws = construct_workspace(prob)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
    @test Vres ≈ [10.0, 7.0]

    ws = IntervalMDP.DeterministicWorkspace(prob, 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
    @test Vres ≈ [10.0, 7.0]

    ws = IntervalMDP.ThreadedDeterministicWorkspace(prob, 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
    @test Vres ≈ [10.0, 7.0]
end

#### Minimization
@testset "minimization" begin
    Vres = bellman(V, prob; upper_bound = true)
    @test Vres ≈ [10.0, 7.0]

    Vres = similar(Vres)
    bellman!(Vres, V, prob; upper_bound = true)
    @test Vres ≈ [10.0, 7.0]

    ws = construct_workspace(prob)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
    @test Vres ≈ [10.0, 7.0]

    ws = IntervalMDP.DeterministicWorkspace(prob, 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
    @test Vres ≈ [10.0, 7.0]

    ws = IntervalMDP.ThreadedDeterministicWorkspace(prob, 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    bellman!(ws, strategy_cache, Vres, V, prob; upper_bound = true)
    @test Vres ≈ [10.0, 7.0]
end