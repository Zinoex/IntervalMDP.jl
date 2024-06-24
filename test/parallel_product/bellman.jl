
prob1 = IntervalProbabilities(;
    lower = [
        0.0 0.5
        0.1 0.3
        0.2 0.1
    ],
    upper = [
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ],
)

prob2 = IntervalProbabilities(;
    lower = [
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ],
    upper = [
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ],
)

prob3 = IntervalProbabilities(; lower = [
    0.0
    0.0
    1.0
][:, :], upper = [
    0.0
    0.0
    1.0
][:, :])
dense_mdp = IntervalMarkovDecisionProcess([prob1, prob2, prob3])

prob1 = IntervalProbabilities(;
    lower = sparse([
        0.0 0.2
        0.1 0.4
        0.2 0.3
    ]),
    upper = sparse([
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ]),
)

prob2 = IntervalProbabilities(;
    lower = sparse([
        0.0
        1.0
        0.0
    ][:, :]),
    upper = sparse([
        0.0
        1.0
        0.0
    ][:, :]),
)

prob3 = IntervalProbabilities(;
    lower = sparse([
        0.1 0.1
        0.2 0.1
        0.3 0.1
    ]),
    upper = sparse([
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ]),
)
sparse_mdp = IntervalMarkovDecisionProcess([prob1, prob2, prob3])

product_mdp1 = ParallelProduct([dense_mdp, sparse_mdp])
ws1 = construct_workspace(product_mdp1)

product_mdp2 = ParallelProduct([sparse_mdp, dense_mdp])
ws2 = construct_workspace(product_mdp2)

strategy_cache = IntervalMDP.NoStrategyCache()

#### Minimization of upper bound
@testset "minimize upper bound" begin
    ##########################
    # First parallel product #
    ##########################
    V = [
        1.0 6.0 7.0
        8.0 5.0 2.0
        3.0 4.0 9.0
    ]

    # First IMDP
    Vdes1 = bellman([1.0, 8.0, 3.0], transition_prob(dense_mdp); upper_bound = true)
    Vdes2 = bellman([6.0, 5.0, 4.0], transition_prob(dense_mdp); upper_bound = true)
    Vdes3 = bellman([7.0, 2.0, 9.0], transition_prob(dense_mdp); upper_bound = true)

    # println(Vdes1)
    # println(Vdes2)
    # println(Vdes3)

    Vdes1 = Vdes1[[2, 4, 5]]
    Vdes2 = Vdes2[[1, 4, 5]]
    Vdes3 = Vdes3[[2, 4, 5]]
    Vdes = hcat(Vdes1, Vdes2, Vdes3)

    Vres = similar(V)
    bellman!(
        ws1.process_workspaces[1],
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.DenseParallelWorkspace(
        gap(transition_prob(dense_mdp)),
        num_states(dense_mdp),
        IntervalMDP.max_actions(dense_mdp),
        one(Int32),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.ThreadedDenseParallelWorkspace(
        gap(transition_prob(dense_mdp)),
        num_states(dense_mdp),
        IntervalMDP.max_actions(dense_mdp),
        one(Int32),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes

    # Second IMDP
    Vdes1 = bellman([1.0, 6.0, 7.0], transition_prob(sparse_mdp); upper_bound = true)
    Vdes2 = bellman([8.0, 5.0, 2.0], transition_prob(sparse_mdp); upper_bound = true)
    Vdes3 = bellman([3.0, 4.0, 9.0], transition_prob(sparse_mdp); upper_bound = true)

    # println(Vdes1)
    # println(Vdes2)
    # println(Vdes3)

    Vdes1 = Vdes1[[2, 3, 5]]
    Vdes2 = Vdes2[[2, 3, 4]]
    Vdes3 = Vdes3[[2, 3, 4]]
    Vdes = mapreduce(transpose, vcat, [Vdes1, Vdes2, Vdes3])

    Vres = similar(V)
    bellman!(
        ws1.process_workspaces[2],
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.SparseProductWorkspace(
        gap(transition_prob(sparse_mdp)),
        num_states(sparse_mdp),
        IntervalMDP.max_actions(sparse_mdp),
        Int32(2),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.ThreadedSparseProductWorkspace(
        gap(transition_prob(sparse_mdp)),
        num_states(sparse_mdp),
        IntervalMDP.max_actions(sparse_mdp),
        Int32(2),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes

    ##########################
    # Second parallel product #
    ##########################
    V = [
        1.0 6.0 7.0
        8.0 5.0 2.0
        3.0 4.0 9.0
    ]

    # First IMDP
    Vdes1 = bellman([1.0, 8.0, 3.0], transition_prob(sparse_mdp); upper_bound = true)
    Vdes2 = bellman([6.0, 5.0, 4.0], transition_prob(sparse_mdp); upper_bound = true)
    Vdes3 = bellman([7.0, 2.0, 9.0], transition_prob(sparse_mdp); upper_bound = true)

    # println(Vdes1)
    # println(Vdes2)
    # println(Vdes3)

    Vdes1 = Vdes1[[2, 3, 4]]
    Vdes2 = Vdes2[[2, 3, 4]]
    Vdes3 = Vdes3[[2, 3, 4]]
    Vdes = hcat(Vdes1, Vdes2, Vdes3)

    Vres = similar(V)
    bellman!(
        ws2.process_workspaces[1],
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.SparseProductWorkspace(
        gap(transition_prob(sparse_mdp)),
        num_states(sparse_mdp),
        IntervalMDP.max_actions(sparse_mdp),
        one(Int32),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.ThreadedSparseProductWorkspace(
        gap(transition_prob(sparse_mdp)),
        num_states(sparse_mdp),
        IntervalMDP.max_actions(sparse_mdp),
        one(Int32),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes

    # Second IMDP
    Vdes1 = bellman([1.0, 6.0, 7.0], transition_prob(dense_mdp); upper_bound = true)
    Vdes2 = bellman([8.0, 5.0, 2.0], transition_prob(dense_mdp); upper_bound = true)
    Vdes3 = bellman([3.0, 4.0, 9.0], transition_prob(dense_mdp); upper_bound = true)

    # println(Vdes1)
    # println(Vdes2)
    # println(Vdes3)

    Vdes1 = Vdes1[[2, 4, 5]]
    Vdes2 = Vdes2[[1, 4, 5]]
    Vdes3 = Vdes3[[2, 4, 5]]
    Vdes = mapreduce(transpose, vcat, [Vdes1, Vdes2, Vdes3])

    Vres = similar(V)
    bellman!(
        ws2.process_workspaces[2],
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.DenseParallelWorkspace(
        gap(transition_prob(dense_mdp)),
        num_states(dense_mdp),
        IntervalMDP.max_actions(dense_mdp),
        Int32(2),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.ThreadedDenseParallelWorkspace(
        gap(transition_prob(dense_mdp)),
        num_states(dense_mdp),
        IntervalMDP.max_actions(dense_mdp),
        Int32(2),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = true,
        maximize = false,
    )
    @test Vres ≈ Vdes
end

#### Maximization of lower bound
@testset "maximize lower bound" begin
    ##########################
    # First parallel product #
    ##########################
    V = [
        1.0 6.0 7.0
        8.0 5.0 2.0
        3.0 4.0 9.0
    ]

    # First IMDP
    Vdes1 = bellman([1.0, 8.0, 3.0], transition_prob(dense_mdp); upper_bound = false)
    Vdes2 = bellman([6.0, 5.0, 4.0], transition_prob(dense_mdp); upper_bound = false)
    Vdes3 = bellman([7.0, 2.0, 9.0], transition_prob(dense_mdp); upper_bound = false)

    # println(Vdes1)
    # println(Vdes2)
    # println(Vdes3)

    Vdes1 = Vdes1[[2, 4, 5]]
    Vdes2 = Vdes2[[2, 4, 5]]
    Vdes3 = Vdes3[[2, 4, 5]]
    Vdes = hcat(Vdes1, Vdes2, Vdes3)

    Vres = similar(V)
    bellman!(
        ws1.process_workspaces[1],
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.DenseParallelWorkspace(
        gap(transition_prob(dense_mdp)),
        num_states(dense_mdp),
        IntervalMDP.max_actions(dense_mdp),
        one(Int32),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.ThreadedDenseParallelWorkspace(
        gap(transition_prob(dense_mdp)),
        num_states(dense_mdp),
        IntervalMDP.max_actions(dense_mdp),
        one(Int32),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes

    # Second IMDP
    Vdes1 = bellman([1.0, 6.0, 7.0], transition_prob(sparse_mdp); upper_bound = false)
    Vdes2 = bellman([8.0, 5.0, 2.0], transition_prob(sparse_mdp); upper_bound = false)
    Vdes3 = bellman([3.0, 4.0, 9.0], transition_prob(sparse_mdp); upper_bound = false)

    # println(Vdes1)
    # println(Vdes2)
    # println(Vdes3)

    Vdes1 = Vdes1[[2, 3, 4]]
    Vdes2 = Vdes2[[2, 3, 4]]
    Vdes3 = Vdes3[[2, 3, 4]]
    Vdes = mapreduce(transpose, vcat, [Vdes1, Vdes2, Vdes3])

    Vres = similar(V)
    bellman!(
        ws1.process_workspaces[2],
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.SparseProductWorkspace(
        gap(transition_prob(sparse_mdp)),
        num_states(sparse_mdp),
        IntervalMDP.max_actions(sparse_mdp),
        Int32(2),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.ThreadedSparseProductWorkspace(
        gap(transition_prob(sparse_mdp)),
        num_states(sparse_mdp),
        IntervalMDP.max_actions(sparse_mdp),
        Int32(2),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes

    ##########################
    # Second parallel product #
    ##########################
    V = [
        1.0 6.0 7.0
        8.0 5.0 2.0
        3.0 4.0 9.0
    ]

    # First IMDP
    Vdes1 = bellman([1.0, 8.0, 3.0], transition_prob(sparse_mdp); upper_bound = false)
    Vdes2 = bellman([6.0, 5.0, 4.0], transition_prob(sparse_mdp); upper_bound = false)
    Vdes3 = bellman([7.0, 2.0, 9.0], transition_prob(sparse_mdp); upper_bound = false)

    # println(Vdes1)
    # println(Vdes2)
    # println(Vdes3)

    Vdes1 = Vdes1[[2, 3, 4]]
    Vdes2 = Vdes2[[2, 3, 4]]
    Vdes3 = Vdes3[[2, 3, 4]]
    Vdes = hcat(Vdes1, Vdes2, Vdes3)

    Vres = similar(V)
    bellman!(
        ws2.process_workspaces[1],
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.SparseProductWorkspace(
        gap(transition_prob(sparse_mdp)),
        num_states(sparse_mdp),
        IntervalMDP.max_actions(sparse_mdp),
        one(Int32),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.ThreadedSparseProductWorkspace(
        gap(transition_prob(sparse_mdp)),
        num_states(sparse_mdp),
        IntervalMDP.max_actions(sparse_mdp),
        one(Int32),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(sparse_mdp),
        stateptr(sparse_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes

    # Second IMDP
    Vdes1 = bellman([1.0, 6.0, 7.0], transition_prob(dense_mdp); upper_bound = false)
    Vdes2 = bellman([8.0, 5.0, 2.0], transition_prob(dense_mdp); upper_bound = false)
    Vdes3 = bellman([3.0, 4.0, 9.0], transition_prob(dense_mdp); upper_bound = false)

    # println(Vdes1)
    # println(Vdes2)
    # println(Vdes3)

    Vdes1 = Vdes1[[1, 4, 5]]
    Vdes2 = Vdes2[[2, 4, 5]]
    Vdes3 = Vdes3[[1, 4, 5]]
    Vdes = mapreduce(transpose, vcat, [Vdes1, Vdes2, Vdes3])

    Vres = similar(V)
    bellman!(
        ws2.process_workspaces[2],
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.DenseParallelWorkspace(
        gap(transition_prob(dense_mdp)),
        num_states(dense_mdp),
        IntervalMDP.max_actions(dense_mdp),
        Int32(2),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes

    ws_direct = IntervalMDP.ThreadedDenseParallelWorkspace(
        gap(transition_prob(dense_mdp)),
        num_states(dense_mdp),
        IntervalMDP.max_actions(dense_mdp),
        Int32(2),
    )
    Vres = similar(Vres)
    bellman!(
        ws_direct,
        strategy_cache,
        Vres,
        V,
        transition_prob(dense_mdp),
        stateptr(dense_mdp);
        upper_bound = false,
        maximize = true,
    )
    @test Vres ≈ Vdes
end
