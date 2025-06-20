using Revise, Test
using IntervalMDP, SparseArrays

#### Maximization
@testset "maximization" begin
    prob = IntervalProbabilities(;
        lower = sparse_hcat(
            SparseVector(15, [4, 10], [0.1, 0.2]),
            SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
        ),
        upper = sparse_hcat(
            SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
            SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
        ),
    )

    V = collect(1.0:15.0)

    ws = construct_workspace(prob)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = zeros(Float64, 2)
    IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = true)
    @test Vres ≈ [0.3 * 4 + 0.7 * 10, 0.5 * 5 + 0.3 * 6 + 0.2 * 7]

    ws = IntervalMDP.SparseWorkspace(gap(prob), 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = true)
    @test Vres ≈ [0.3 * 4 + 0.7 * 10, 0.5 * 5 + 0.3 * 6 + 0.2 * 7]

    ws = IntervalMDP.ThreadedSparseWorkspace(gap(prob), 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = true)
    @test Vres ≈ [0.3 * 4 + 0.7 * 10, 0.5 * 5 + 0.3 * 6 + 0.2 * 7]
end

#### Minimization
@testset "minimization" begin
    prob = IntervalProbabilities(;
        lower = sparse_hcat(
            SparseVector(15, [4, 10], [0.1, 0.2]),
            SparseVector(15, [5, 6, 7], [0.5, 0.3, 0.1]),
        ),
        upper = sparse_hcat(
            SparseVector(15, [1, 4, 10], [0.5, 0.6, 0.7]),
            SparseVector(15, [5, 6, 7], [0.7, 0.5, 0.3]),
        ),
    )

    V = collect(1.0:15.0)

    ws = construct_workspace(prob)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = zeros(Float64, 2)
    IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = false)
    @test Vres ≈ [0.5 * 1 + 0.3 * 4 + 0.2 * 10, 0.6 * 5 + 0.3 * 6 + 0.1 * 7]

    ws = IntervalMDP.SparseWorkspace(gap(prob), 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = false)
    @test Vres ≈ [0.5 * 1 + 0.3 * 4 + 0.2 * 10, 0.6 * 5 + 0.3 * 6 + 0.1 * 7]

    ws = IntervalMDP.ThreadedSparseWorkspace(gap(prob), 1)
    strategy_cache = construct_strategy_cache(prob, NoStrategyConfig())
    Vres = similar(Vres)
    IntervalMDP._bellman_helper!(ws, strategy_cache, Vres, V, prob, stateptr(prob); upper_bound = false)
    @test Vres ≈ [0.5 * 1 + 0.3 * 4 + 0.2 * 10, 0.6 * 5 + 0.3 * 6 + 0.1 * 7]
end
