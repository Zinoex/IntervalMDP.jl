using Revise, Test
using IntervalMDP

@testset "Orthogonal abstraction" begin
    using LazySets
    using SpecialFunctions: erf
    using LinearAlgebra: I

    function IMDP_orthogonal_abstraction()
        A = 0.9 * I
        B = 0.7 * I
        sigma = 2.0

        X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
        X1 = Interval(-10.0, 10.0)
        X2 = Interval(-10.0, 10.0)
        U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])

        reach_region = Hyperrectangle(; low = [4.0, -6.0], high = [10.0, -2.0])

        l = [5, 5]
        X1_split = split(X1, l[1])
        X2_split = split(X2, l[2])

        X_split = Matrix{LazySet}(undef, l[1], l[2])
        for j in 1:l[2]
            for i in 1:l[1]
                x1 = X1_split[i]
                x2 = X2_split[j]
                X_split[i, j] = Hyperrectangle(
                    [center(x1)[1], center(x2)[1]],
                    [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]],
                )
            end
        end

        U_split = split(U, [3, 3])

        transition_prob(x, v_lower, v_upper) =
            0.5 * erf((x - v_upper) / (sigma * sqrt(2.0)), (x - v_lower) / (sigma * sqrt(2.0)))

        probs1 = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
        probs2 = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
        stateptr = Int32[1]

        for source2 in 1:(l[2] + 1)
            for source1 in 1:(l[1] + 1)
                if source1 == 1 || source2 == 1
                    probs1_lower = zeros(l[1] + 1, 1)
                    probs1_upper = zeros(l[1] + 1, 1)

                    probs1_upper[source1, 1] = 1
                    probs1_lower[source1, 1] = 1

                    probs2_lower = zeros(l[2] + 1, 1)
                    probs2_upper = zeros(l[2] + 1, 1)

                    probs2_upper[source2, 1] = 1
                    probs2_lower[source2, 1] = 1

                    push!(
                        probs1,
                        IntervalProbabilities(; lower = probs1_lower, upper = probs1_upper),
                    )
                    push!(
                        probs2,
                        IntervalProbabilities(; lower = probs2_lower, upper = probs2_upper),
                    )
                else
                    Xij = X_split[source1 - 1, source2 - 1]

                    for u in U_split
                        Xij_u = A * Xij + B * u
                        Xij_u = box_approximation(Xij_u)

                        probs1_lower = zeros(l[1] + 1, 1)
                        probs1_upper = zeros(l[1] + 1, 1)

                        for target1 in 1:(l[1] + 1)
                            if target1 == 1
                                probs1_upper[target1, 1] =
                                    max(
                                        1 -
                                        transition_prob(low(Xij_u)[1], low(X)[1], high(X)[1]),
                                        1 -
                                        transition_prob(high(Xij_u)[1], low(X)[1], high(X)[1]),
                                    ) + eps(Float64)
                                probs1_lower[target1, 1] = min(
                                    1 - transition_prob(
                                        center(Xij_u)[1],
                                        low(X)[1],
                                        high(X)[1],
                                    ),
                                    1 - transition_prob(low(Xij_u)[1], low(X)[1], high(X)[1]),
                                    1 - transition_prob(high(Xij_u)[1], low(X)[1], high(X)[1]),
                                )
                            else
                                probs1_upper[target1, 1] = max(
                                    transition_prob(
                                        center(Xij_u)[1],
                                        low(X1_split[target1 - 1])[1],
                                        high(X1_split[target1 - 1])[1],
                                    ),
                                    transition_prob(
                                        low(Xij_u)[1],
                                        low(X1_split[target1 - 1])[1],
                                        high(X1_split[target1 - 1])[1],
                                    ),
                                    transition_prob(
                                        high(Xij_u)[1],
                                        low(X1_split[target1 - 1])[1],
                                        high(X1_split[target1 - 1])[1],
                                    ),
                                )
                                probs1_lower[target1, 1] = min(
                                    transition_prob(
                                        low(Xij_u)[1],
                                        low(X1_split[target1 - 1])[1],
                                        high(X1_split[target1 - 1])[1],
                                    ),
                                    transition_prob(
                                        high(Xij_u)[1],
                                        low(X1_split[target1 - 1])[1],
                                        high(X1_split[target1 - 1])[1],
                                    ),
                                )
                            end
                        end

                        probs2_lower = zeros(l[2] + 1, 1)
                        probs2_upper = zeros(l[2] + 1, 1)

                        for target2 in 1:(l[2] + 1)
                            if target2 == 1
                                probs2_upper[target2, 1] =
                                    max(
                                        1 -
                                        transition_prob(low(Xij_u)[2], low(X)[2], high(X)[2]),
                                        1 -
                                        transition_prob(high(Xij_u)[2], low(X)[2], high(X)[2]),
                                    ) + eps(Float64)
                                probs2_lower[target2, 1] = min(
                                    1 - transition_prob(
                                        center(Xij_u)[2],
                                        low(X)[2],
                                        high(X)[2],
                                    ),
                                    1 - transition_prob(low(Xij_u)[2], low(X)[2], high(X)[2]),
                                    1 - transition_prob(high(Xij_u)[2], low(X)[2], high(X)[2]),
                                )
                            else
                                probs2_upper[target2, 1] = max(
                                    transition_prob(
                                        center(Xij_u)[2],
                                        low(X2_split[target2 - 1])[1],
                                        high(X2_split[target2 - 1])[1],
                                    ),
                                    transition_prob(
                                        low(Xij_u)[2],
                                        low(X2_split[target2 - 1])[1],
                                        high(X2_split[target2 - 1])[1],
                                    ),
                                    transition_prob(
                                        high(Xij_u)[2],
                                        low(X2_split[target2 - 1])[1],
                                        high(X2_split[target2 - 1])[1],
                                    ),
                                )
                                probs2_lower[target2, 1] = min(
                                    transition_prob(
                                        low(Xij_u)[2],
                                        low(X2_split[target2 - 1])[1],
                                        high(X2_split[target2 - 1])[1],
                                    ),
                                    transition_prob(
                                        high(Xij_u)[2],
                                        low(X2_split[target2 - 1])[1],
                                        high(X2_split[target2 - 1])[1],
                                    ),
                                )
                            end
                        end

                        push!(
                            probs1,
                            IntervalProbabilities(; lower = probs1_lower, upper = probs1_upper),
                        )
                        push!(
                            probs2,
                            IntervalProbabilities(; lower = probs2_lower, upper = probs2_upper),
                        )
                    end
                end

                push!(stateptr, length(probs1) + 1)
            end
        end

        probs1, _ = IntervalMDP.interval_prob_hcat(probs1)
        probs2, _ = IntervalMDP.interval_prob_hcat(probs2)
        probs = OrthogonalIntervalProbabilities(
            (probs1, probs2),
            (Int32(l[1] + 1), Int32(l[2] + 1)),
        )
        pmdp = OrthogonalIntervalMarkovDecisionProcess(probs, stateptr)

        reach = Tuple{Int32, Int32}[]
        avoid = Tuple{Int32, Int32}[]

        for j in 1:(l[2] + 1)
            for i in 1:(l[1] + 1)
                if j == 1 || i == 1
                    push!(avoid, (i, j))
                elseif X_split[i - 1, j - 1] ⊆ reach_region
                    push!(reach, (i, j))
                end
            end
        end

        return pmdp, reach, avoid
    end

    function IMDP_direct_abstraction()
        A = 0.9I(2)
        B = 0.7I(2)
        sigma = 2.0

        X = Hyperrectangle(; low = [-10.0, -10.0], high = [10.0, 10.0])
        X1 = Interval(-10.0, 10.0)
        X2 = Interval(-10.0, 10.0)
        U = Hyperrectangle(; low = [-1.0, -1.0], high = [1.0, 1.0])

        reach_region = Hyperrectangle(; low = [4.0, -6.0], high = [10.0, -2.0])

        l = [5, 5]
        X1_split = split(X1, l[1])
        X2_split = split(X2, l[2])

        X_split = Matrix{LazySet}(undef, l[1] + 1, l[2] + 1)
        for j in 1:(l[2] + 1)
            for i in 1:(l[1] + 1)
                if i == 1 && j == 1
                    X_split[i, j] = CartesianProduct(
                        Complement(Interval(low(X, 1), high(X, 1))),
                        Complement(Interval(low(X, 2), high(X, 2))),
                    )
                elseif i == 1
                    x2 = X2_split[j - 1]
                    X_split[i, j] = CartesianProduct(
                        Complement(Interval(low(X, 1), high(X, 1))),
                        Interval(low(x2, 1), high(x2, 1)),
                    )
                elseif j == 1
                    x1 = X1_split[i - 1]
                    X_split[i, j] = CartesianProduct(
                        Interval(low(x1, 1), high(x1, 1)),
                        Complement(Interval(low(X, 2), high(X, 2))),
                    )
                else
                    x1 = X1_split[i - 1]
                    x2 = X2_split[j - 1]
                    X_split[i, j] = Hyperrectangle(
                        [center(x1)[1], center(x2)[1]],
                        [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]],
                    )
                end
            end
        end

        U_split = split(U, [3, 3])

        transition_prob(x, v_lower, v_upper) =
            0.5 * erf((x - v_upper) / (sigma * sqrt(2.0)), (x - v_lower) / (sigma * sqrt(2.0)))

        probs = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
        for source2 in 1:(l[2] + 1)
            for source1 in 1:(l[1] + 1)
                source = (source2 - 1) * (l[1] + 1) + source1

                probs_lower = Vector{Float64}[]
                probs_upper = Vector{Float64}[]

                if source1 == 1 || source2 == 1
                    prob_upper = zeros(prod(l .+ 1))
                    prob_lower = zeros(prod(l .+ 1))

                    prob_upper[source] = 1
                    prob_lower[source] = 1

                    push!(probs_lower, prob_lower)
                    push!(probs_upper, prob_upper)
                else
                    Xij = X_split[source1, source2]

                    for u in U_split
                        Xij_u = A * Xij + B * u
                        box_Xij_u = box_approximation(Xij_u)

                        prob_upper = zeros(prod(l .+ 1))
                        prob_lower = zeros(prod(l .+ 1))

                        for target2 in 1:(l[2] + 1)
                            for target1 in 1:(l[1] + 1)
                                Xij_target = X_split[target1, target2]
                                target = (target2 - 1) * (l[1] + 1) + target1

                                if target1 == 1 && target2 == 1
                                    prob_upper[target] =
                                        max(
                                            1 - transition_prob(
                                                low(box_Xij_u)[1],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                            1 - transition_prob(
                                                high(box_Xij_u)[1],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                        ) * max(
                                            1 - transition_prob(
                                                low(box_Xij_u)[2],
                                                low(X)[2],
                                                high(X)[2],
                                            ),
                                            1 - transition_prob(
                                                high(box_Xij_u)[2],
                                                low(X)[2],
                                                high(X)[2],
                                            ),
                                        )
                                    prob_lower[target] =
                                        min(
                                            1 - transition_prob(
                                                center(box_Xij_u)[1],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                            1 - transition_prob(
                                                low(box_Xij_u)[1],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                            1 - transition_prob(
                                                high(box_Xij_u)[1],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                        ) * min(
                                            1 - transition_prob(
                                                center(box_Xij_u)[2],
                                                low(X)[2],
                                                high(X)[2],
                                            ),
                                            1 - transition_prob(
                                                low(box_Xij_u)[2],
                                                low(X)[2],
                                                high(X)[2],
                                            ),
                                            1 - transition_prob(
                                                high(box_Xij_u)[2],
                                                low(X)[2],
                                                high(X)[2],
                                            ),
                                        )
                                elseif target1 == 1
                                    prob_upper[target] =
                                        max(
                                            1 - transition_prob(
                                                low(box_Xij_u)[1],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                            1 - transition_prob(
                                                high(box_Xij_u)[1],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                        ) * max(
                                            transition_prob(
                                                center(box_Xij_u)[2],
                                                low(Xij_target.Y)[1],
                                                high(Xij_target.Y)[1],
                                            ),
                                            transition_prob(
                                                low(box_Xij_u)[2],
                                                low(Xij_target.Y)[1],
                                                high(Xij_target.Y)[1],
                                            ),
                                            transition_prob(
                                                high(box_Xij_u)[2],
                                                low(Xij_target.Y)[1],
                                                high(Xij_target.Y)[1],
                                            ),
                                        )
                                    prob_lower[target] =
                                        min(
                                            1 - transition_prob(
                                                center(box_Xij_u)[1],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                            1 - transition_prob(
                                                low(box_Xij_u)[1],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                            1 - transition_prob(
                                                high(box_Xij_u)[1],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                        ) * min(
                                            transition_prob(
                                                low(box_Xij_u)[2],
                                                low(Xij_target.Y)[1],
                                                high(Xij_target.Y)[1],
                                            ),
                                            transition_prob(
                                                high(box_Xij_u)[2],
                                                low(Xij_target.Y)[1],
                                                high(Xij_target.Y)[1],
                                            ),
                                        )
                                elseif target2 == 1
                                    prob_upper[target] =
                                        max(
                                            transition_prob(
                                                center(box_Xij_u)[1],
                                                low(Xij_target.X)[1],
                                                high(Xij_target.X)[1],
                                            ),
                                            transition_prob(
                                                low(box_Xij_u)[1],
                                                low(Xij_target.X)[1],
                                                high(Xij_target.X)[1],
                                            ),
                                            transition_prob(
                                                high(box_Xij_u)[1],
                                                low(Xij_target.X)[1],
                                                high(Xij_target.X)[1],
                                            ),
                                        ) * max(
                                            1 - transition_prob(
                                                low(box_Xij_u)[2],
                                                low(X)[2],
                                                high(X)[2],
                                            ),
                                            1 - transition_prob(
                                                high(box_Xij_u)[2],
                                                low(X)[2],
                                                high(X)[2],
                                            ),
                                        )
                                    prob_lower[target] =
                                        min(
                                            transition_prob(
                                                low(box_Xij_u)[1],
                                                low(Xij_target.X)[1],
                                                high(Xij_target.X)[1],
                                            ),
                                            transition_prob(
                                                high(box_Xij_u)[1],
                                                low(Xij_target.X)[1],
                                                high(Xij_target.X)[1],
                                            ),
                                        ) * min(
                                            1 - transition_prob(
                                                center(box_Xij_u)[2],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                            1 - transition_prob(
                                                low(box_Xij_u)[2],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                            1 - transition_prob(
                                                high(box_Xij_u)[2],
                                                low(X)[1],
                                                high(X)[1],
                                            ),
                                        )
                                else
                                    prob_upper[target] =
                                        max(
                                            transition_prob(
                                                center(box_Xij_u)[1],
                                                low(Xij_target)[1],
                                                high(Xij_target)[1],
                                            ),
                                            transition_prob(
                                                low(box_Xij_u)[1],
                                                low(Xij_target)[1],
                                                high(Xij_target)[1],
                                            ),
                                            transition_prob(
                                                high(box_Xij_u)[1],
                                                low(Xij_target)[1],
                                                high(Xij_target)[1],
                                            ),
                                        ) * max(
                                            transition_prob(
                                                center(box_Xij_u)[2],
                                                low(Xij_target)[2],
                                                high(Xij_target)[2],
                                            ),
                                            transition_prob(
                                                low(box_Xij_u)[2],
                                                low(Xij_target)[2],
                                                high(Xij_target)[2],
                                            ),
                                            transition_prob(
                                                high(box_Xij_u)[2],
                                                low(Xij_target)[2],
                                                high(Xij_target)[2],
                                            ),
                                        )
                                    prob_lower[target] =
                                        min(
                                            transition_prob(
                                                low(box_Xij_u)[1],
                                                low(Xij_target)[1],
                                                high(Xij_target)[1],
                                            ),
                                            transition_prob(
                                                high(box_Xij_u)[1],
                                                low(Xij_target)[1],
                                                high(Xij_target)[1],
                                            ),
                                        ) * min(
                                            transition_prob(
                                                low(box_Xij_u)[2],
                                                low(Xij_target)[2],
                                                high(Xij_target)[2],
                                            ),
                                            transition_prob(
                                                high(box_Xij_u)[2],
                                                low(Xij_target)[2],
                                                high(Xij_target)[2],
                                            ),
                                        )
                                end
                            end
                        end

                        push!(probs_lower, prob_lower)
                        push!(probs_upper, prob_upper)
                    end
                end

                prob = IntervalProbabilities(;
                    lower = reduce(hcat, probs_lower),
                    upper = reduce(hcat, probs_upper),
                )
                push!(probs, prob)
            end
        end
        mdp = IntervalMarkovDecisionProcess(probs)

        reach = Int32[]
        avoid = Int32[]

        for source2 in 1:(l[2] + 1)
            for source1 in 1:(l[1] + 1)
                Xij = X_split[source1, source2]
                source = (source2 - 1) * (l[1] + 1) + source1

                if source1 == 1 || source2 == 1
                    push!(avoid, source)
                elseif Xij ⊆ reach_region
                    push!(reach, source)
                end
            end
        end

        return mdp, reach, avoid
    end

    # Orthogonal abstraction
    pmdp, reach_set, avoid_set = IMDP_orthogonal_abstraction()

    prop = FiniteTimeReachAvoid(reach_set, avoid_set, 10)
    spec = Specification(prop, Pessimistic, Maximize)
    prob_ortho = Problem(pmdp, spec)

    V_ortho, it_ortho, res_ortho = value_iteration(prob_ortho)

    # Direct abstraction
    mdp, reach_set, avoid_set = IMDP_direct_abstraction()

    prop = FiniteTimeReachAvoid(reach_set, avoid_set, 10)
    spec = Specification(prop, Pessimistic, Maximize)
    prob_direct = Problem(mdp, spec)

    V_direct, it_direct, res_direct = value_iteration(prob_direct)

    @test it_ortho == it_direct
    @test all(V_ortho .≥ reshape(V_direct, 6, 6))
end


# 3-D abstraction
@testset "3D abstraction" begin
    prob_lower = [rand(3, 27) ./ 3 for _ in 1:3]
    prob_upper = [(rand(3, 27) .+ 1) ./ 3 for _ in 1:3]

    probs = OrthogonalIntervalProbabilities(
        ntuple(i -> IntervalProbabilities(; lower = prob_lower[i], upper = prob_upper[i]), 3),
        (Int32(3), Int32(3), Int32(3)),
    )

    stateptr = Int32.(collect(1:28))
    mdp = OrthogonalIntervalMarkovDecisionProcess(probs, stateptr)

    prop = FiniteTimeReachability([(3, 3, 3)], 10)
    spec = Specification(prop, Pessimistic, Maximize)
    prob = Problem(mdp, spec)

    V, it, res = value_iteration(prob)
    @test V[3, 3, 3] ≈ 1.0
    @test minimum(V) > 0.0
end