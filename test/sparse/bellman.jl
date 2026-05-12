@testitem "maximization" tags = [:sparse, :maximization] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        prob = IntervalAmbiguitySets(;
            lower = sparse_hcat(
                SparseVector(15, [4, 10], N[1 // 10, 2 // 10]),
                SparseVector(15, [5, 6, 7], N[5 // 10, 3 // 10, 1 // 10]),
            ),
            upper = sparse_hcat(
                SparseVector(15, [1, 4, 10], N[5 // 10, 6 // 10, 7 // 10]),
                SparseVector(15, [5, 6, 7], N[7 // 10, 5 // 10, 3 // 10]),
            ),
        )
        V = collect(N(1):N(15))
        @testset "maximization" begin
            ws = IntervalMDP.construct_workspace(prob)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = zeros(N, 2)
            IntervalMDP.bellman_v!(
                ws,
                strategy_cache,
                IntervalMDP.StateValueArray(Vres),
                IntervalMDP.StateValueArray(V),
                prob;
                upper_bound = true,
            )
            @test Vres ≈ N[82 // 10, 57 // 10]
            ws = IntervalMDP.DenseIntervalOMaxWorkspace(prob, 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP.bellman_v!(
                ws,
                strategy_cache,
                IntervalMDP.StateValueArray(Vres),
                IntervalMDP.StateValueArray(V),
                prob;
                upper_bound = true,
            )
            @test Vres ≈ N[82 // 10, 57 // 10]
            ws = IntervalMDP.ThreadedDenseIntervalOMaxWorkspace(prob, 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP.bellman_v!(
                ws,
                strategy_cache,
                IntervalMDP.StateValueArray(Vres),
                IntervalMDP.StateValueArray(V),
                prob;
                upper_bound = true,
            )
            @test Vres ≈ N[82 // 10, 57 // 10]
        end
    end
end

@testitem "minimization" tags = [:sparse, :minimization] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        prob = IntervalAmbiguitySets(;
            lower = sparse_hcat(
                SparseVector(15, [4, 10], N[1 // 10, 2 // 10]),
                SparseVector(15, [5, 6, 7], N[5 // 10, 3 // 10, 1 // 10]),
            ),
            upper = sparse_hcat(
                SparseVector(15, [1, 4, 10], N[5 // 10, 6 // 10, 7 // 10]),
                SparseVector(15, [5, 6, 7], N[7 // 10, 5 // 10, 3 // 10]),
            ),
        )
        V = collect(N(1):N(15))
        @testset "minimization" begin
            ws = IntervalMDP.construct_workspace(prob)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = zeros(N, 2)
            IntervalMDP.bellman_v!(
                ws,
                strategy_cache,
                IntervalMDP.StateValueArray(Vres),
                IntervalMDP.StateValueArray(V),
                prob;
                upper_bound = false,
            )
            @test Vres ≈ N[37 // 10, 55 // 10]
            ws = IntervalMDP.DenseIntervalOMaxWorkspace(prob, 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP.bellman_v!(
                ws,
                strategy_cache,
                IntervalMDP.StateValueArray(Vres),
                IntervalMDP.StateValueArray(V),
                prob;
                upper_bound = false,
            )
            @test Vres ≈ N[37 // 10, 55 // 10]
            ws = IntervalMDP.ThreadedDenseIntervalOMaxWorkspace(prob, 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP.bellman_v!(
                ws,
                strategy_cache,
                IntervalMDP.StateValueArray(Vres),
                IntervalMDP.StateValueArray(V),
                prob;
                upper_bound = false,
            )
            @test Vres ≈ N[37 // 10, 55 // 10]
        end
    end
end
