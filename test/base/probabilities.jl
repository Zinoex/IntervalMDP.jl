@testitem "getters" tags = [:base, :getters] begin
    using IntervalMDP
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "getters" begin
            l = N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10]
            u = N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10]
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
            @test occursin("Storage type: Matrix{$(N)}", str)
            @test occursin("Number of target states: 3", str)
            @test occursin("Number of ambiguity sets: 2", str)
        end
    end
end

@testitem "vertex enumerator" tags = [:base, :vertex_enumerator] begin
    using IntervalMDP
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "vertex enumerator" begin
            lower = N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10]
            upper = N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10]
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

@testitem "check vs no check" tags = [:base, :check_vs_no_check] begin
    using IntervalMDP
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "check vs no check" begin
            lower = N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10]
            upper = N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10]
            gap = upper - lower
            prob = IntervalAmbiguitySets(; lower = lower, upper = upper)
            prob_no_check = IntervalAmbiguitySets(lower, gap, Val{false}())
            @test prob.lower == prob_no_check.lower
            @test prob.gap == prob_no_check.gap
        end
    end
end

@testitem "dimension mismatch" tags = [:base, :dimension_mismatch] begin
    using IntervalMDP
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "dimension mismatch" begin
            lower = N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10]
            upper = N[5 // 10 7 // 10; 6 // 10 5 // 10]
            @test_throws DimensionMismatch IntervalAmbiguitySets(;
                lower = lower,
                upper = upper,
            )
            lower = N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10]
            gap = N[5 // 10 7 // 10; 6 // 10 5 // 10]
            @test_throws DimensionMismatch IntervalAmbiguitySets(lower, gap)
        end
    end
end

@testitem "negative lower bound" tags =
    [:base, :negative_lower_bound] begin
    using IntervalMDP
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "negative lower bound" begin
            lower = N[0 1 // 2; -1 // 10 3 // 10; 2 // 10 1 // 10]
            upper = N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10]
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end

@testitem "lower bound greater than one" tags =
    [:base, :lower_bound_greater_than_one] begin
    using IntervalMDP
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "lower bound greater than one" begin
            lower = N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 11 // 10]
            upper = N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10]
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end

@testitem "lower greater than upper" tags =
    [:base, :lower_greater_than_upper] begin
    using IntervalMDP
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "lower greater than upper" begin
            lower = N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10]
            upper = N[5 // 10 7 // 10; 6 // 10 2 // 10; 7 // 10 3 // 10]
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end

@testitem "upper bound greater than one" tags =
    [:base, :upper_bound_greater_than_one] begin
    using IntervalMDP
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "upper bound greater than one" begin
            lower = N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10]
            upper = N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 13 // 10]
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end

@testitem "sum lower greater than one" tags =
    [:base, :sum_lower_greater_than_one] begin
    using IntervalMDP
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "sum lower greater than one" begin
            lower = N[0 1 // 2; 1 // 10 3 // 10; 6 // 10 1 // 2]
            upper = N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 1 // 2]
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end

@testitem "sum upper less than one" tags =
    [:base, :sum_upper_less_than_one] begin
    using IntervalMDP
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "sum upper less than one" begin
            lower = N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10]
            upper = N[1 // 10 7 // 10; 2 // 10 5 // 10; 3 // 10 6 // 10]
            @test_throws ArgumentError IntervalAmbiguitySets(; lower = lower, upper = upper)
        end
    end
end

@testitem "marginal" tags = [:base, :marginal] begin
    using IntervalMDP
    @testset for N in [Float32, Float64, Rational{BigInt}]
        @testset "marginal" begin
            N = Float64
            l = N[0 1 // 2; 1 // 10 3 // 10; 2 // 10 1 // 10]
            u = N[5 // 10 7 // 10; 6 // 10 5 // 10; 7 // 10 3 // 10]
            prob = IntervalAmbiguitySets(; lower = l, upper = u)
            @test_throws ArgumentError Marginal(prob, (-1,), (1,), (2,), (1,))
            @test_throws ArgumentError Marginal(prob, (1,), (-1,), (2,), (1,))
            @test_throws ArgumentError Marginal(prob, (1,), (1,), (-2,), (1,))
            @test_throws ArgumentError Marginal(prob, (1,), (1,), (2,), (-1,))
            @test_throws ArgumentError Marginal(prob, (1,), (1,), (3,), (1,))
            marg = Marginal(prob, (1,), (1,), (2,), (1,))
            amb_set = marg[CartesianIndex(1), CartesianIndex(1)]
            @test amb_set == prob[1]
            amb_set = marg[CartesianIndex(1), CartesianIndex(2)]
            @test amb_set == prob[2]
            @test state_variables(marg) == (1,)
            @test action_variables(marg) == (1,)
            @test source_shape(marg) == (2,)
            @test action_shape(marg) == (1,)
            @test ambiguity_sets(marg) == prob
            io = IOBuffer()
            show(io, MIME("text/plain"), marg)
            str = String(take!(io))
            @test occursin("Marginal", str)
            @test occursin("Conditional variables: states = (1,), actions = (1,)", str)
            @test occursin("Ambiguity set type: Interval", str)
            marg = Marginal(prob, (1,), (1,), (1,), (2,))
            amb_set = marg[CartesianIndex(1), CartesianIndex(1)]
            @test amb_set == prob[1]
            amb_set = marg[CartesianIndex(2), CartesianIndex(1)]
            @test amb_set == prob[2]
        end
    end
end
