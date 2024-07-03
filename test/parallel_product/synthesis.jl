using Revise, Test
using IntervalMDP, SparseArrays

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

product_mdp = ParallelProduct([dense_mdp, sparse_mdp])
ws = construct_workspace(product_mdp)

@testset "stationary" begin
    strategy_cache =
        construct_strategy_cache(product_mdp, IntervalMDP.StationaryStrategyConfig())

    #### Minimization of upper bound
    @testset "minimize upper bound" begin
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
            ws.process_workspaces[1],
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].strategy == [
            2 1 2
            4 4 4
            5 5 5
        ]

        strategy_cache.orthogonal_caches[1].strategy .= 0
        ws_direct = IntervalMDP.DenseParallelWorkspace(
            gap(transition_prob(dense_mdp)),
            num_states(dense_mdp),
            IntervalMDP.max_actions(dense_mdp),
            one(Int32),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].strategy == [
            2 1 2
            4 4 4
            5 5 5
        ]

        strategy_cache.orthogonal_caches[1].strategy .= 0
        ws_direct = IntervalMDP.ThreadedDenseParallelWorkspace(
            gap(transition_prob(dense_mdp)),
            num_states(dense_mdp),
            IntervalMDP.max_actions(dense_mdp),
            one(Int32),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].strategy == [
            2 1 2
            4 4 4
            5 5 5
        ]

        # Second IMDP
        Vdes1 = bellman([1.0, 6.0, 7.0], transition_prob(sparse_mdp); upper_bound = true)
        Vdes2 = bellman([8.0, 5.0, 2.0], transition_prob(sparse_mdp); upper_bound = true)
        Vdes3 = bellman([3.0, 4.0, 9.0], transition_prob(sparse_mdp); upper_bound = true)

        # println(Vdes1)
        # println(Vdes2)
        # println(Vdes3)

        Vdes1 = Vdes1[[2, 3, 4]]
        Vdes2 = Vdes2[[2, 3, 4]]
        Vdes3 = Vdes3[[2, 3, 4]]
        Vdes = mapreduce(transpose, vcat, [Vdes1, Vdes2, Vdes3])

        Vres = similar(V)
        bellman!(
            ws.process_workspaces[2],
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]

        strategy_cache.orthogonal_caches[2].strategy .= 0
        ws_direct = IntervalMDP.SparseProductWorkspace(
            gap(transition_prob(sparse_mdp)),
            num_states(sparse_mdp),
            IntervalMDP.max_actions(sparse_mdp),
            Int32(2),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]

        strategy_cache.orthogonal_caches[2].strategy .= 0
        ws_direct = IntervalMDP.ThreadedSparseProductWorkspace(
            gap(transition_prob(sparse_mdp)),
            num_states(sparse_mdp),
            IntervalMDP.max_actions(sparse_mdp),
            Int32(2),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]
    end

    #### Maximization of lower bound
    @testset "maximize lower bound" begin
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

        strategy_cache.orthogonal_caches[1].strategy .= 0
        Vres = similar(V)
        bellman!(
            ws.process_workspaces[1],
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].strategy == [
            2 2 2
            4 4 4
            5 5 5
        ]

        strategy_cache.orthogonal_caches[1].strategy .= 0
        ws_direct = IntervalMDP.DenseParallelWorkspace(
            gap(transition_prob(dense_mdp)),
            num_states(dense_mdp),
            IntervalMDP.max_actions(dense_mdp),
            one(Int32),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].strategy == [
            2 2 2
            4 4 4
            5 5 5
        ]

        strategy_cache.orthogonal_caches[1].strategy .= 0
        ws_direct = IntervalMDP.ThreadedDenseParallelWorkspace(
            gap(transition_prob(dense_mdp)),
            num_states(dense_mdp),
            IntervalMDP.max_actions(dense_mdp),
            one(Int32),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].strategy == [
            2 2 2
            4 4 4
            5 5 5
        ]

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

        strategy_cache.orthogonal_caches[2].strategy .= 0
        Vres = similar(V)
        bellman!(
            ws.process_workspaces[2],
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]

        strategy_cache.orthogonal_caches[2].strategy .= 0
        ws_direct = IntervalMDP.SparseProductWorkspace(
            gap(transition_prob(sparse_mdp)),
            num_states(sparse_mdp),
            IntervalMDP.max_actions(sparse_mdp),
            Int32(2),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]

        strategy_cache.orthogonal_caches[2].strategy .= 0
        ws_direct = IntervalMDP.ThreadedSparseProductWorkspace(
            gap(transition_prob(sparse_mdp)),
            num_states(sparse_mdp),
            IntervalMDP.max_actions(sparse_mdp),
            Int32(2),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]
    end

    @testset "value iteration" begin
        prop = InfiniteTimeReachability([(3, 3)], 1e-3)
        spec = Specification(prop, Pessimistic, Maximize)
        prob = Problem(product_mdp, spec)

        strategy, V, k, res = control_synthesis(prob)
        @test maximum(res) <= 1e-3
    end
end

@testset "time-varying" begin
    strategy_cache =
        construct_strategy_cache(product_mdp, IntervalMDP.TimeVaryingStrategyConfig())

    #### Minimization of upper bound
    @testset "minimize upper bound" begin
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
            ws.process_workspaces[1],
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].cur_strategy == [
            2 1 2
            4 4 4
            5 5 5
        ]

        strategy_cache.orthogonal_caches[1].cur_strategy .= 0
        ws_direct = IntervalMDP.DenseParallelWorkspace(
            gap(transition_prob(dense_mdp)),
            num_states(dense_mdp),
            IntervalMDP.max_actions(dense_mdp),
            one(Int32),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].cur_strategy == [
            2 1 2
            4 4 4
            5 5 5
        ]

        strategy_cache.orthogonal_caches[1].cur_strategy .= 0
        ws_direct = IntervalMDP.ThreadedDenseParallelWorkspace(
            gap(transition_prob(dense_mdp)),
            num_states(dense_mdp),
            IntervalMDP.max_actions(dense_mdp),
            one(Int32),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].cur_strategy == [
            2 1 2
            4 4 4
            5 5 5
        ]

        # Second IMDP
        Vdes1 = bellman([1.0, 6.0, 7.0], transition_prob(sparse_mdp); upper_bound = true)
        Vdes2 = bellman([8.0, 5.0, 2.0], transition_prob(sparse_mdp); upper_bound = true)
        Vdes3 = bellman([3.0, 4.0, 9.0], transition_prob(sparse_mdp); upper_bound = true)

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
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].cur_strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]

        strategy_cache.orthogonal_caches[2].cur_strategy .= 0
        ws_direct = IntervalMDP.SparseProductWorkspace(
            gap(transition_prob(sparse_mdp)),
            num_states(sparse_mdp),
            IntervalMDP.max_actions(sparse_mdp),
            Int32(2),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].cur_strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]

        strategy_cache.orthogonal_caches[2].cur_strategy .= 0
        ws_direct = IntervalMDP.ThreadedSparseProductWorkspace(
            gap(transition_prob(sparse_mdp)),
            num_states(sparse_mdp),
            IntervalMDP.max_actions(sparse_mdp),
            Int32(2),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = true,
            maximize = false,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].cur_strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]
    end

    #### Maximization of lower bound
    @testset "maximize lower bound" begin
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

        strategy_cache.orthogonal_caches[1].cur_strategy .= 0
        Vres = similar(V)
        bellman!(
            ws1.process_workspaces[1],
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].cur_strategy == [
            2 2 2
            4 4 4
            5 5 5
        ]

        strategy_cache.orthogonal_caches[1].cur_strategy .= 0
        ws_direct = IntervalMDP.DenseParallelWorkspace(
            gap(transition_prob(dense_mdp)),
            num_states(dense_mdp),
            IntervalMDP.max_actions(dense_mdp),
            one(Int32),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].cur_strategy == [
            2 2 2
            4 4 4
            5 5 5
        ]

        strategy_cache.orthogonal_caches[1].cur_strategy .= 0
        ws_direct = IntervalMDP.ThreadedDenseParallelWorkspace(
            gap(transition_prob(dense_mdp)),
            num_states(dense_mdp),
            IntervalMDP.max_actions(dense_mdp),
            one(Int32),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[1],
            Vres,
            V,
            transition_prob(dense_mdp),
            stateptr(dense_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[1].cur_strategy == [
            2 2 2
            4 4 4
            5 5 5
        ]

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

        strategy_cache.orthogonal_caches[2].cur_strategy .= 0
        Vres = similar(V)
        bellman!(
            ws1.process_workspaces[2],
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].cur_strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]

        strategy_cache.orthogonal_caches[2].cur_strategy .= 0
        ws_direct = IntervalMDP.SparseProductWorkspace(
            gap(transition_prob(sparse_mdp)),
            num_states(sparse_mdp),
            IntervalMDP.max_actions(sparse_mdp),
            Int32(2),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].cur_strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]

        strategy_cache.orthogonal_caches[2].cur_strategy .= 0
        ws_direct = IntervalMDP.ThreadedSparseProductWorkspace(
            gap(transition_prob(sparse_mdp)),
            num_states(sparse_mdp),
            IntervalMDP.max_actions(sparse_mdp),
            Int32(2),
        )
        Vres = similar(Vres)
        bellman!(
            ws_direct,
            strategy_cache.orthogonal_caches[2],
            Vres,
            V,
            transition_prob(sparse_mdp),
            stateptr(sparse_mdp);
            upper_bound = false,
            maximize = true,
        )
        @test Vres ≈ Vdes
        @test strategy_cache.orthogonal_caches[2].cur_strategy == [
            2 3 4
            2 3 4
            2 3 4
        ]
    end

    @testset "value iteration" begin
        prop = FiniteTimeReachability([(3, 3)], 10)
        spec = Specification(prop, Pessimistic, Maximize)
        prob = Problem(product_mdp, spec)

        strategy, V, k, res = control_synthesis(prob)
        @test k == 10
    end
end

@testset "nested" begin
    # Dense MDP
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

    transition_probs = [prob1, prob2, prob3]
    dense_mdp = IntervalMarkovDecisionProcess(transition_probs)

    # Sparse MDP
    prob1 = IntervalProbabilities(;
        lower = sparse([
            0.0 0.5
            0.1 0.3
            0.2 0.1
        ]),
        upper = sparse([
            0.5 0.7
            0.6 0.5
            0.7 0.3
        ]),
    )

    prob2 = IntervalProbabilities(;
        lower = sparse([
            0.1 0.2
            0.2 0.3
            0.3 0.4
        ]),
        upper = sparse([
            0.6 0.6
            0.5 0.5
            0.4 0.4
        ]),
    )

    prob3 = IntervalProbabilities(;
        lower = sparse([
            0.0
            0.0
            1.0
        ][:, :]),
        upper = sparse([
            0.0
            0.0
            1.0
        ][:, :]),
    )

    transition_probs = [prob1, prob2, prob3]
    sparse_mdp = IntervalMarkovDecisionProcess(transition_probs)

    prop = FiniteTimeReachability([(3, 3, 3)], 10)
    spec = Specification(prop, Pessimistic, Maximize)

    nested_product_dense_mdp =
        ParallelProduct([dense_mdp, ParallelProduct([dense_mdp, sparse_mdp])])
    problem = Problem(nested_product_dense_mdp, spec)
    strategy1, V_fixed_it1, k, _ = control_synthesis(problem)
    @test k == 10

    nested_product_sparse_mdp =
        ParallelProduct([ParallelProduct([dense_mdp, sparse_mdp]), dense_mdp])
    problem = Problem(nested_product_sparse_mdp, spec)
    strategy2, V_fixed_it2, k, _ = control_synthesis(problem)
    @test k == 10
    @test V_fixed_it1 ≈ V_fixed_it2
    @test strategy1[1] == strategy2[1][1]
    @test strategy1[2][1] == strategy2[1][2]
    @test strategy1[2][2] == strategy2[2]

    nested_product_mdp3 =
        ParallelProduct([sparse_mdp, ParallelProduct([dense_mdp, sparse_mdp])])
    problem = Problem(nested_product_mdp3, spec)
    strategy3, V_fixed_it3, k, _ = control_synthesis(problem)
    @test k == 10
    @test V_fixed_it1 ≈ V_fixed_it3
    @test strategy1[1] == strategy3[1]
    @test strategy1[2][1] == strategy3[2][1]
    @test strategy1[2][2] == strategy3[2][2]

    nested_product_mdp4 =
        ParallelProduct([ParallelProduct([dense_mdp, sparse_mdp]), sparse_mdp])
    problem = Problem(nested_product_mdp4, spec)
    strategy4, V_fixed_it4, k, _ = control_synthesis(problem)
    @test k == 10
    @test V_fixed_it1 ≈ V_fixed_it4
    @test strategy1[1] == strategy4[1][1]
    @test strategy1[2][1] == strategy4[1][2]
    @test strategy1[2][2] == strategy4[2]
end
