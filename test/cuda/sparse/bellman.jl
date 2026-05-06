@testitem "large matrices" tags = [:cuda, :large_matrices] begin
    using IntervalMDP, SparseArrays, CUDA
    using StatsBase
    using Random: MersenneTwister
    if CUDA.functional()
        for N in [Float32, Float64]
            @testset "N = $(N)" begin
                prob = IntervalAmbiguitySets(;
                    lower = sparse(
                        N[
                            0 1 // 6;
                            1 // 10 2 // 10;
                            2 // 10 1 // 10;
                            0 1 // 6;
                            1 // 10 2 // 10;
                            2 // 10 1 // 10
                        ],
                    ),
                    upper = sparse(
                        N[
                            5 // 10 7 // 10;
                            6 // 10 5 // 10;
                            7 // 10 3 // 10;
                            5 // 10 7 // 10;
                            6 // 10 5 // 10;
                            7 // 10 3 // 10
                        ],
                    ),
                )
                prob = IntervalMDP.cu(prob)
                V = IntervalMDP.cu(N[1, 2, 3, 4, 5, 6])
                @testset "maximization" begin
                    ws = IntervalMDP.construct_workspace(prob)
                    strategy_cache = IntervalMDP.construct_strategy_cache(prob)
                    Vres = CUDA.zeros(N, 2)
                    IntervalMDP.bellman_q!(
                        ws,
                        strategy_cache,
                        IntervalMDP.StateActionValueArray(Vres),
                        IntervalMDP.StateValueArray(V),
                        prob;
                        upper_bound = true,
                    )
                    Vres = IntervalMDP.cpu(Vres)
                    @test Vres ≈ N[49 // 10, 53 // 15]
                end
                @testset "minimization" begin
                    ws = IntervalMDP.construct_workspace(prob)
                    strategy_cache = IntervalMDP.construct_strategy_cache(prob)
                    Vres = CUDA.zeros(N, 2)
                    IntervalMDP.bellman_q!(
                        ws,
                        strategy_cache,
                        IntervalMDP.StateActionValueArray(Vres),
                        IntervalMDP.StateValueArray(V),
                        prob;
                        upper_bound = false,
                    )
                    Vres = IntervalMDP.cpu(Vres)
                    @test Vres ≈ N[29 // 10, 16 // 5]
                end
            end
        end
        @testset "large matrices" begin
            function sample_sparse_interval_ambiguity_sets(rng, n, m, nnz_per_column)
                prob_split = 1 / nnz_per_column
                rand_val_lower = rand(rng, Float64, nnz_per_column * m) .* prob_split
                rand_val_upper =
                    rand(rng, Float64, nnz_per_column * m) .* prob_split .+ prob_split
                rand_index = collect(1:nnz_per_column)
                row_vals = Vector{Int32}(undef, nnz_per_column * m)
                col_ptrs = Int32[1; collect(1:m) .* nnz_per_column .+ 1]
                for j in 1:m
                    StatsBase.seqsample_a!(rng, 1:n, rand_index)
                    sort!(rand_index)
                    row_vals[((j - 1) * nnz_per_column + 1):(j * nnz_per_column)] .=
                        rand_index
                end
                lower = SparseMatrixCSC{Float64, Int32}(
                    n,
                    m,
                    col_ptrs,
                    row_vals,
                    rand_val_lower,
                )
                upper = SparseMatrixCSC{Float64, Int32}(
                    n,
                    m,
                    col_ptrs,
                    row_vals,
                    rand_val_upper,
                )
                prob = IntervalAmbiguitySets(; lower = lower, upper = upper)
                V = rand(rng, Float64, n)
                cuda_prob = IntervalMDP.cu(prob)
                cuda_V = IntervalMDP.cu(V)
                return (prob, V, cuda_prob, cuda_V)
            end
            @testset "many columns" begin
                rng = MersenneTwister(55392)
                n = 100
                m = 5000000
                nnz_per_column = 10
                (prob, V, cuda_prob, cuda_V) =
                    sample_sparse_interval_ambiguity_sets(rng, n, m, nnz_per_column)
                ws = IntervalMDP.construct_workspace(prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(prob)
                V_cpu = zeros(Float64, m)
                IntervalMDP.bellman_q!(
                    ws,
                    strategy_cache,
                    IntervalMDP.StateActionValueArray(V_cpu),
                    IntervalMDP.StateValueArray(V),
                    prob;
                    upper_bound = false,
                )
                ws = IntervalMDP.construct_workspace(cuda_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(cuda_prob)
                V_gpu = CUDA.zeros(Float64, m)
                IntervalMDP.bellman_q!(
                    ws,
                    strategy_cache,
                    IntervalMDP.StateActionValueArray(V_gpu),
                    IntervalMDP.StateValueArray(cuda_V),
                    cuda_prob;
                    upper_bound = false,
                )
                V_gpu = IntervalMDP.cpu(V_gpu)
                @test V_cpu ≈ V_gpu
            end
            @testset "many non-zeros" begin
                rng = MersenneTwister(55392)
                n = 100000
                m = 10
                nnz_per_column = 800
                (prob, V, cuda_prob, cuda_V) =
                    sample_sparse_interval_ambiguity_sets(rng, n, m, nnz_per_column)
                ws = IntervalMDP.construct_workspace(prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(prob)
                V_cpu = zeros(Float64, m)
                IntervalMDP.bellman_q!(
                    ws,
                    strategy_cache,
                    IntervalMDP.StateActionValueArray(V_cpu),
                    IntervalMDP.StateValueArray(V),
                    prob;
                    upper_bound = false,
                )
                ws = IntervalMDP.construct_workspace(cuda_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(cuda_prob)
                V_gpu = CUDA.zeros(Float64, m)
                IntervalMDP.bellman_q!(
                    ws,
                    strategy_cache,
                    IntervalMDP.StateActionValueArray(V_gpu),
                    IntervalMDP.StateValueArray(cuda_V),
                    cuda_prob;
                    upper_bound = false,
                )
                V_gpu = IntervalMDP.cpu(V_gpu)
                @test V_cpu ≈ V_gpu
            end
            @testset "more non-zeros" begin
                rng = MersenneTwister(55392)
                n = 100000
                m = 10
                nnz_per_column = 4000
                (prob, V, cuda_prob, cuda_V) =
                    sample_sparse_interval_ambiguity_sets(rng, n, m, nnz_per_column)
                ws = IntervalMDP.construct_workspace(prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(prob)
                V_cpu = zeros(Float64, m)
                IntervalMDP.bellman_q!(
                    ws,
                    strategy_cache,
                    IntervalMDP.StateActionValueArray(V_cpu),
                    IntervalMDP.StateValueArray(V),
                    prob;
                    upper_bound = false,
                )
                ws = IntervalMDP.construct_workspace(cuda_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(cuda_prob)
                V_gpu = CUDA.zeros(Float64, m)
                IntervalMDP.bellman_q!(
                    ws,
                    strategy_cache,
                    IntervalMDP.StateActionValueArray(V_gpu),
                    IntervalMDP.StateValueArray(cuda_V),
                    cuda_prob;
                    upper_bound = false,
                )
                V_gpu = IntervalMDP.cpu(V_gpu)
                @test V_cpu ≈ V_gpu
            end
            @testset "even more non-zeros" begin
                rng = MersenneTwister(55392)
                n = 100000
                m = 10
                nnz_per_column = 6000
                (prob, V, cuda_prob, cuda_V) =
                    sample_sparse_interval_ambiguity_sets(rng, n, m, nnz_per_column)
                ws = IntervalMDP.construct_workspace(prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(prob)
                V_cpu = zeros(Float64, m)
                IntervalMDP.bellman_q!(
                    ws,
                    strategy_cache,
                    IntervalMDP.StateActionValueArray(V_cpu),
                    IntervalMDP.StateValueArray(V),
                    prob;
                    upper_bound = false,
                )
                ws = IntervalMDP.construct_workspace(cuda_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(cuda_prob)
                V_gpu = CUDA.zeros(Float64, m)
                IntervalMDP.bellman_q!(
                    ws,
                    strategy_cache,
                    IntervalMDP.StateActionValueArray(V_gpu),
                    IntervalMDP.StateValueArray(cuda_V),
                    cuda_prob;
                    upper_bound = false,
                )
                V_gpu = IntervalMDP.cpu(V_gpu)
                @test V_cpu ≈ V_gpu
            end
            @testset "most non-zeros" begin
                rng = MersenneTwister(55392)
                n = 100000
                m = 10
                nnz_per_column = 8000
                (prob, V, cuda_prob, cuda_V) =
                    sample_sparse_interval_ambiguity_sets(rng, n, m, nnz_per_column)
                ws = IntervalMDP.construct_workspace(prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(prob)
                V_cpu = zeros(Float64, m)
                IntervalMDP.bellman_q!(
                    ws,
                    strategy_cache,
                    IntervalMDP.StateActionValueArray(V_cpu),
                    IntervalMDP.StateValueArray(V),
                    prob;
                    upper_bound = false,
                )
                ws = IntervalMDP.construct_workspace(cuda_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(cuda_prob)
                V_gpu = CUDA.zeros(Float64, m)
                IntervalMDP.bellman_q!(
                    ws,
                    strategy_cache,
                    IntervalMDP.StateActionValueArray(V_gpu),
                    IntervalMDP.StateValueArray(cuda_V),
                    cuda_prob;
                    upper_bound = false,
                )
                V_gpu = IntervalMDP.cpu(V_gpu)
                @test V_cpu ≈ V_gpu
            end
            @testset "too many non-zeros" begin
                rng = MersenneTwister(55392)
                n = 100000
                m = 10
                nnz_per_column = 16000
                (prob, V, cuda_prob, cuda_V) =
                    sample_sparse_interval_ambiguity_sets(rng, n, m, nnz_per_column)
                ws = IntervalMDP.construct_workspace(cuda_prob)
                strategy_cache = IntervalMDP.construct_strategy_cache(cuda_prob)
                V_gpu = CUDA.zeros(Float64, m)
                @test_throws IntervalMDP.OutOfSharedMemory IntervalMDP.bellman_q!(
                    ws,
                    strategy_cache,
                    IntervalMDP.StateActionValueArray(V_gpu),
                    IntervalMDP.StateValueArray(cuda_V),
                    cuda_prob;
                    upper_bound = false,
                )
            end
        end
    end
end
