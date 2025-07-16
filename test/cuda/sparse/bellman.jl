using Revise, Test
using IntervalMDP, SparseArrays, CUDA
using StatsBase
using Random: MersenneTwister

for N in [Float32, Float64, Rational{BigInt}]
    @testset "N = $N" begin
        prob = IntervalProbabilities(;
            lower = sparse(N[0 1//2; 1//10 3//10; 2//10 1//10]),
            upper = sparse(N[5//10 7//10; 6//10 5//10; 7//10 3//10]),
        )

        V = N[1, 2, 3]

        #### Maximization
        @testset "maximization" begin
            ws = IntervalMDP.construct_workspace(prob)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = zeros(N, 2)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = true,
            )
            @test Vres ≈ N[27 // 10, 17 // 10] # [0.3 * 2 + 0.7 * 3, 0.5 * 1 + 0.3 * 2 + 0.2 * 3]

            ws = IntervalMDP.DenseWorkspace(gap(prob), 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = true,
            )
            @test Vres ≈ N[27 // 10, 17 // 10]

            ws = IntervalMDP.ThreadedDenseWorkspace(gap(prob), 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = true,
            )
            @test Vres ≈ N[27 // 10, 17 // 10]
        end

        #### Minimization
        @testset "minimization" begin
            ws = IntervalMDP.construct_workspace(prob)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = zeros(N, 2)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = false,
            )
            @test Vres ≈ N[17 // 10, 15 // 10]  # [0.5 * 1 + 0.3 * 2 + 0.2 * 3, 0.6 * 1 + 0.3 * 2 + 0.1 * 3]

            ws = IntervalMDP.DenseWorkspace(gap(prob), 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = false,
            )
            @test Vres ≈ N[17 // 10, 15 // 10]

            ws = IntervalMDP.ThreadedDenseWorkspace(gap(prob), 1)
            strategy_cache = IntervalMDP.construct_strategy_cache(prob)
            Vres = similar(Vres)
            IntervalMDP._bellman_helper!(
                ws,
                strategy_cache,
                Vres,
                V,
                prob,
                stateptr(prob);
                upper_bound = false,
            )
            @test Vres ≈ N[17 // 10, 15 // 10]
        end
    end
end

#### Large matrices
@testset "large matrices" begin
    function sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)
        prob_split = 1 / nnz_per_column

        rand_val_lower = rand(rng, Float64, nnz_per_column * m) .* prob_split
        rand_val_upper = rand(rng, Float64, nnz_per_column * m) .* prob_split .+ prob_split
        rand_index = collect(1:nnz_per_column)

        row_vals = Vector{Int32}(undef, nnz_per_column * m)
        col_ptrs = Int32[1; collect(1:m) .* nnz_per_column .+ 1]

        for j in 1:m
            StatsBase.seqsample_a!(rng, 1:n, rand_index)  # Select nnz_per_column elements from 1:n
            sort!(rand_index)

            row_vals[((j - 1) * nnz_per_column + 1):(j * nnz_per_column)] .= rand_index
        end

        lower = SparseMatrixCSC{Float64, Int32}(n, m, col_ptrs, row_vals, rand_val_lower)
        upper = SparseMatrixCSC{Float64, Int32}(n, m, col_ptrs, row_vals, rand_val_upper)

        prob = IntervalProbabilities(; lower = lower, upper = upper)
        V = rand(rng, Float64, n)

        cuda_prob = IntervalMDP.cu(prob)
        cuda_V = IntervalMDP.cu(V)

        return prob, V, cuda_prob, cuda_V
    end

    # Many columns
    @testset "many columns" begin
        rng = MersenneTwister(55392)

        n = 100
        m = 1000000  # It has to be greater than 8 * 2^16 to exceed maximum grid size
        nnz_per_column = 2
        prob, V, cuda_prob, cuda_V =
            sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)

        ws = IntervalMDP.construct_workspace(prob)
        strategy_cache = IntervalMDP.construct_strategy_cache(prob)
        V_cpu = zeros(Float64, m)
        IntervalMDP._bellman_helper!(
            ws,
            strategy_cache,
            V_cpu,
            V,
            prob,
            stateptr(prob);
            upper_bound = false,
        )

        ws = IntervalMDP.construct_workspace(cuda_prob)
        strategy_cache = IntervalMDP.construct_strategy_cache(cuda_prob)
        V_gpu = CUDA.zeros(Float64, m)
        IntervalMDP._bellman_helper!(
            ws,
            strategy_cache,
            V_gpu,
            cuda_V,
            cuda_prob,
            stateptr(cuda_prob);
            upper_bound = false,
        )
        V_gpu = Vector(V_gpu)  # Convert to CPU for testing

        @test V_cpu ≈ V_gpu
    end

    # Many non-zeros
    @testset "many non-zeros" begin
        rng = MersenneTwister(55392)

        n = 100000
        m = 10
        nnz_per_column = 1500   # It has to be greater than 187 to fill shared memory with 8 states per block.
        prob, V, cuda_prob, cuda_V =
            sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)

        ws = IntervalMDP.construct_workspace(prob)
        strategy_cache = IntervalMDP.construct_strategy_cache(prob)
        V_cpu = zeros(Float64, m)
        IntervalMDP._bellman_helper!(
            ws,
            strategy_cache,
            V_cpu,
            V,
            prob,
            stateptr(prob);
            upper_bound = false,
        )

        ws = IntervalMDP.construct_workspace(cuda_prob)
        strategy_cache = IntervalMDP.construct_strategy_cache(cuda_prob)
        V_gpu = CUDA.zeros(Float64, m)
        IntervalMDP._bellman_helper!(
            ws,
            strategy_cache,
            V_gpu,
            cuda_V,
            cuda_prob,
            stateptr(cuda_prob);
            upper_bound = false,
        )
        V_gpu = Vector(V_gpu)  # Convert to CPU for testing

        @test V_cpu ≈ V_gpu
    end

    # More non-zeros
    @testset "more non-zeros" begin
        rng = MersenneTwister(55392)

        n = 100000
        m = 10
        nnz_per_column = 4000   # It has to be greater than 3800 to exceed shared memory for ff implementation
        prob, V, cuda_prob, cuda_V =
            sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)

        ws = IntervalMDP.construct_workspace(prob)
        strategy_cache = IntervalMDP.construct_strategy_cache(prob)
        V_cpu = zeros(Float64, m)
        IntervalMDP._bellman_helper!(
            ws,
            strategy_cache,
            V_cpu,
            V,
            prob,
            stateptr(prob);
            upper_bound = false,
        )

        ws = IntervalMDP.construct_workspace(cuda_prob)
        strategy_cache = IntervalMDP.construct_strategy_cache(cuda_prob)
        V_gpu = CUDA.zeros(Float64, m)
        IntervalMDP._bellman_helper!(
            ws,
            strategy_cache,
            V_gpu,
            cuda_V,
            cuda_prob,
            stateptr(cuda_prob);
            upper_bound = false,
        )
        V_gpu = Vector(V_gpu)  # Convert to CPU for testing

        @test V_cpu ≈ V_gpu
    end

    # Most non-zeros
    @testset "most non-zeros" begin
        rng = MersenneTwister(55392)

        n = 100000
        m = 10
        nnz_per_column = 6000   # It has to be greater than 5800 to exceed shared memory for fi implementation
        prob, V, cuda_prob, cuda_V =
            sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)

        ws = IntervalMDP.construct_workspace(prob)
        strategy_cache = IntervalMDP.construct_strategy_cache(prob)
        V_cpu = zeros(Float64, m)
        IntervalMDP._bellman_helper!(
            ws,
            strategy_cache,
            V_cpu,
            V,
            prob,
            stateptr(prob);
            upper_bound = false,
        )

        ws = IntervalMDP.construct_workspace(cuda_prob)
        strategy_cache = IntervalMDP.construct_strategy_cache(cuda_prob)
        V_gpu = CUDA.zeros(Float64, m)
        IntervalMDP._bellman_helper!(
            ws,
            strategy_cache,
            V_gpu,
            cuda_V,
            cuda_prob,
            stateptr(cuda_prob);
            upper_bound = false,
        )
        V_gpu = Vector(V_gpu)  # Convert to CPU for testing

        @test V_cpu ≈ V_gpu
    end

    # Too many non-zeros
    @testset "too many non-zeros" begin
        rng = MersenneTwister(55392)

        n = 100000
        m = 10
        nnz_per_column = 8000   # It has to be greater than 7800 to exceed shared memory for ii implementation
        prob, V, cuda_prob, cuda_V =
            sample_sparse_interval_probabilities(rng, n, m, nnz_per_column)

        ws = IntervalMDP.construct_workspace(cuda_prob)
        strategy_cache = IntervalMDP.construct_strategy_cache(cuda_prob)
        V_gpu = CUDA.zeros(Float64, m)
        @test_throws IntervalMDP.OutOfSharedMemory IntervalMDP._bellman_helper!(
            ws,
            strategy_cache,
            V_gpu,
            cuda_V,
            cuda_prob,
            stateptr(cuda_prob);
            upper_bound = false,
        )
    end
end
