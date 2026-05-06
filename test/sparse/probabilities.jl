@testitem "getters" tags = [:sparse, :getters] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "getters" begin
            l = sparse(N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10])
            u = sparse(N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10])
            prob = IntervalAmbiguitySets(; lower = l, upper = u)
            @test length(prob) == 2
            @test num_sets(prob) == 2
            @test num_target(prob) == 3
            res = sum(upper, prob)
            @test res == N[6 // 5, 11 // 10, 1]
            io = IOBuffer()
            show(io, MIME("text/plain"), prob)
            str = String(take!(io))
            @test occursin("IntervalAmbiguitySets", str)
            @test occursin("Storage type: ", str)
            @test occursin("CSC{$(N), Int64}", str)
            @test occursin("Number of target states: 3", str)
            @test occursin("Number of ambiguity sets: 2", str)
            @test occursin("Maximum support size: 3", str)
            @test occursin("Number of non-zeros: 6", str)
        end
    end
end

@testitem "vertex enumerator" tags = [:sparse, :vertex_enumerator] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "vertex enumerator" begin
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10])
            upper = sparse(N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10])
            prob = IntervalAmbiguitySets(; lower = lower, upper = upper)
            ambiguity_set = prob[1]
            verts = IntervalMDP.vertices(ambiguity_set)
            @test length(verts) == 6
            expected_verts = N[
                5 // 10 3 // 10 2 // 10;
                5 // 10 1 // 10 4 // 10;
                2 // 10 6 // 10 2 // 10;
                0 6 // 10 4 // 10;
                2 // 10 1 // 10 7 // 10;
                0 3 // 10 7 // 10
            ]
            @test length(verts) ≥ size(expected_verts, 1)
            @test all((any((v2->begin
                v1 ≈ v2
            end), verts) for v1 in eachrow(expected_verts)))
            ambiguity_set = prob[2]
            verts = IntervalMDP.vertices(ambiguity_set)
            @test length(verts) <= 6
            expected_verts =
                N[6 // 10 3 // 10 1 // 10; 5 // 10 4 // 10 1 // 10; 5 // 10 3 // 10 2 // 10]
            @test length(verts) ≥ size(expected_verts, 1)
            @test all((any((v2->begin
                v1 ≈ v2
            end), verts) for v1 in eachrow(expected_verts)))
        end
    end
end

@testitem "check vs no check" tags = [:sparse, :check_vs_no_check] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "check vs no check" begin
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10])
            upper = sparse(N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10])
            gap = upper - lower
            prob = IntervalAmbiguitySets(; lower = lower, upper = upper)
            prob_no_check = IntervalAmbiguitySets(lower, gap, Val{false}())
            @test prob.lower == prob_no_check.lower
            @test prob.gap == prob_no_check.gap
        end
    end
end

@testitem "dimension mismatch" tags = [:sparse, :dimension_mismatch] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "dimension mismatch" begin
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10])
            upper = sparse(N[5 // 10 7 // 10; 6 // 10 5 // 10])
            @test_throws DimensionMismatch IntervalAmbiguitySets(;
                lower = lower,
                upper = upper,
            )
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10])
            gap = sparse(N[5 // 10 7 // 10; 6 // 10 5 // 10])
            @test_throws DimensionMismatch IntervalAmbiguitySets(lower, gap)
        end
    end
end

@testitem "structure mismatch" tags = [:sparse, :structure_mismatch] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "structure mismatch" begin
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10])
            gap = sparse(N[5 // 10 7 // 10; 6 // 10 5 // 10; 0 0])
            @test_throws DimensionMismatch IntervalAmbiguitySets(lower, gap)
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10])
            gap = sparse(N[5 // 10 7 // 10; 6 // 10 5 // 10; 0 1 // 10])
            @test_throws DimensionMismatch IntervalAmbiguitySets(lower, gap)
        end
    end
end

@testitem "negative lower bound" tags = [:sparse, :negative_lower_bound] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "negative lower bound" begin
            lower = sparse(N[0 1 // 2; -1 // 10 3 // 10; 2 // 10 1 // 10])
            upper = sparse(N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10])
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end

@testitem "lower bound greater than one" tags = [:sparse, :lower_bound_greater_than_one] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "lower bound greater than one" begin
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 11 // 10])
            upper = sparse(N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10])
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end

@testitem "lower greater than upper" tags = [:sparse, :lower_greater_than_upper] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "lower greater than upper" begin
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10])
            upper = sparse(N[5 // 10 7 // 10; 6 // 10 2 // 10; 7 // 10 3 // 10])
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end

@testitem "upper bound greater than one" tags = [:sparse, :upper_bound_greater_than_one] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "upper bound greater than one" begin
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10])
            upper = sparse(N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 13 // 10])
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end

@testitem "sum lower greater than one" tags = [:sparse, :sum_lower_greater_than_one] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "sum lower greater than one" begin
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 6 // 10 1 // 2])
            upper = sparse(N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 1 // 2])
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end

@testitem "sum upper less than one" tags = [:sparse, :sum_upper_less_than_one] begin
    using IntervalMDP, SparseArrays
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "sum upper less than one" begin
            lower = sparse(N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10])
            upper = sparse(N[1 // 10 7 // 10; 2 // 10 5 // 10; 3 // 10 6 // 10])
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end
