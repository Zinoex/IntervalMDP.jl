using Revise, Test
using IntervalMDP

@testset for N in [Float32, Float64, Rational{BigInt}]
    @testset "getters" begin
        l = N[0 1//2; 1//10 3//10; 2//10 1//10]
        u = N[5//10 7//10; 6//10 5//10; 7//10 3//10]

        prob = IntervalAmbiguitySets(;lower = l, upper = u)

        @test length(prob) == 2
        @test num_sets(prob) == 2
        @test num_target(prob) == 3
        
        res = sum(upper, prob) # Test iteration and upper
        @test res == N[6//5, 11//10, 1]

        io = IOBuffer()
        show(io, MIME("text/plain"), prob)
        str = String(take!(io))
        @test occursin("IntervalAmbiguitySets", str)
        @test occursin("Storage type: Matrix{$N}", str)
        @test occursin("Number of target states: 3", str)
        @test occursin("Number of ambiguity sets: 2", str)
    end

    @testset "vertex enumerator" begin
        lower = N[0 1//2; 1//10 3//10; 2//10 1//10]
        upper = N[5//10 7//10; 6//10 5//10; 7//10 3//10]

        prob = IntervalAmbiguitySets(;lower = lower, upper = upper)

        ambiguity_set = prob[1] # First ambiguity set
        verts = IntervalMDP.vertices(ambiguity_set)
        @test length(verts) == 6

        expected_verts = N[
            5//10 3//10 2//10
            5//10 1//10 4//10
            2//10 6//10 2//10
            0 6//10 4//10
            2//10 1//10 7//10
            0 3//10 7//10
        ]
        @test length(verts) ≥ size(expected_verts, 1)  # at least the unique ones
        @test all(any(v2 -> v1 ≈ v2, verts) for v1 in eachrow(expected_verts))

        ambiguity_set = prob[2] # Second ambiguity set
        verts = IntervalMDP.vertices(ambiguity_set)
        @test length(verts) <= 6  # = number of permutations of 3 elements 

        expected_verts = N[  # duplicates due to budget < gap for all elements
            6 // 10 3//10 1//10
            5//10 4//10 1//10
            5//10 3//10 2//10
        ]
        @test length(verts) ≥ size(expected_verts, 1)  # at least the unique ones
        @test all(any(v2 -> v1 ≈ v2, verts) for v1 in eachrow(expected_verts))
    end
    
    @testset "check vs no check" begin
        lower = N[0 1//2; 1//10 3//10; 2//10 1//10]
        upper = N[5//10 7//10; 6//10 5//10; 7//10 3//10]
        gap = upper - lower

        prob = IntervalAmbiguitySets(;lower = lower, upper = upper)
        prob_no_check = IntervalAmbiguitySets(lower, gap, Val{false}())

        @test prob.lower == prob_no_check.lower
        @test prob.gap == prob_no_check.gap
    end

    @testset "dimension mismatch" begin
        lower = N[0 1//2; 1//10 3//10; 2//10 1//10]
        upper = N[5//10 7//10; 6//10 5//10] # Wrong size

        @test_throws DimensionMismatch IntervalAmbiguitySets(;lower = lower, upper = upper)

        lower = N[0 1//2; 1//10 3//10; 2//10 1//10]
        gap = N[5//10 7//10; 6//10 5//10] # Wrong size

        @test_throws DimensionMismatch IntervalAmbiguitySets(lower, gap)
    end

    @testset "negative lower bound" begin
        lower = N[0 1//2; -1//10 3//10; 2//10 1//10] # Negative entry
        upper = N[5//10 7//10; 6//10 5//10; 7//10 3//10]

        @test_throws ArgumentError IntervalAmbiguitySets(;lower = lower, upper = upper)
    end

    @testset "lower bound greater than one" begin
        lower = N[0 1//2; 1//10 3//10; 2//10 11//10]
        upper = N[5//10 7//10; 6//10 5//10; 7//10 3//10]

        @test_throws ArgumentError IntervalAmbiguitySets(;lower = lower, upper = upper)
    end

    @testset "lower greater than upper" begin
        lower = N[0 1//2; 1//10 3//10; 2//10 1//10]
        upper = N[5//10 7//10; 6//10 2//10; 7//10 3//10] # Lower bound greater than upper bound

        @test_throws ArgumentError IntervalAmbiguitySets(;lower = lower, upper = upper)
    end

    @testset "upper bound greater than one" begin
        lower = N[0 1//2; 1//10 3//10; 2//10 1//10]
        upper = N[5//10 7//10; 6//10 5//10; 7//10 13//10] # Entry greater than 1

        @test_throws ArgumentError IntervalAmbiguitySets(;lower = lower, upper = upper)
    end

    @testset "sum lower greater than one" begin
        lower = N[0 1//2; 1//10 3//10; 6//10 1//2] # Column sums to more than 1
        upper = N[5//10 7//10; 6//10 5//10; 7//10 1//2]

        @test_throws ArgumentError IntervalAmbiguitySets(;lower = lower, upper = upper)
    end

    @testset "sum upper less than one" begin
        lower = N[0 1//2; 1//10 3//10; 2//10 1//10]
        upper = N[1//10 7//10; 2//10 5//10; 3//10 6//10] # Column sums to less than 1

        @test_throws ArgumentError IntervalAmbiguitySets(;lower = lower, upper = upper)
    end
end