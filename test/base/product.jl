using LazySets
using SpecialFunctions: erf
using LinearAlgebra: I

function IMDP_abstraction()
    A = 0.9 * I
    B = 0.7 * I
    sigma = 2.0

    X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
    X1 = Interval(-10.0, 10.0)
    X2 = Interval(-10.0, 10.0)
    U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])

    reach_region = Hyperrectangle(; low=[4.0, -6.0], high=[10.0, -2.0])

    # TODO: Change this to be asymmetric - requires changing the reach region.
    l = [5, 5]
    X1_split = split(X1, l[1])
    X2_split = split(X2, l[2])

    X_split = Matrix{LazySet}(undef, l[1], l[2])
    for j in 1:l[2]
        for i in 1:l[1]
            x1 = X1_split[i]
            x2 = X2_split[j]
            X_split[i, j] = Hyperrectangle([center(x1)[1], center(x2)[1]], [radius_hyperrectangle(x1)[1], radius_hyperrectangle(x2)[1]])
        end
    end

    U_split = split(U, [3, 3])

    transition_prob(x, v_lower, v_upper) = 0.5 * erf((x - v_upper) / (sigma * sqrt(2.0)), (x - v_lower) / (sigma * sqrt(2.0)))

    probs1 = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
    probs2 = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
    stateptr = Int32[1]

    for source2 in 1:l[2] + 1
        for source1 in 1:l[1] + 1
            if source1 == 1 || source2 == 1
                probs1_lower = zeros(l[1] + 1, 1)
                probs1_upper = zeros(l[1] + 1, 1)

                probs1_upper[source1, 1] = 1
                probs1_lower[source1, 1] = 1
                
                probs2_lower = zeros(l[2] + 1, 1)
                probs2_upper = zeros(l[2] + 1, 1)

                probs2_upper[source2, 1] = 1
                probs2_lower[source2, 1] = 1

                push!(probs1, IntervalProbabilities(; lower=probs1_lower, upper=probs1_upper))
                push!(probs2, IntervalProbabilities(; lower=probs2_lower, upper=probs2_upper))
            else
                for u in U_split
                    Xij_u = A * X_split[source1 - 1, source2 - 1] + B * u
                    Xij_u = box_approximation(Xij_u)
                    
                    probs1_lower = zeros(l[1] + 1, 1)
                    probs1_upper = zeros(l[1] + 1, 1)

                    for target1 in 1:l[1] + 1
                        if target1 == 1
                            probs1_upper[target1, 1] = max(
                                1 - transition_prob(low(Xij_u)[1], low(X)[1], high(X)[1]),
                                1 - transition_prob(high(Xij_u)[1], low(X)[1], high(X)[1])
                            ) + eps(Float64)
                            probs1_lower[target1, 1] = min(
                                1 - transition_prob(center(Xij_u)[1], low(X)[1], high(X)[1]),
                                1 - transition_prob(low(Xij_u)[1], low(X)[1], high(X)[1]),
                                1 - transition_prob(high(Xij_u)[1], low(X)[1], high(X)[1])
                            )
                        else
                            probs1_upper[target1, 1] = max(
                                transition_prob(center(Xij_u)[1], low(X1_split[target1 - 1])[1], high(X1_split[target1 - 1])[1]),
                                transition_prob(low(Xij_u)[1], low(X1_split[target1 - 1])[1], high(X1_split[target1 - 1])[1]),
                                transition_prob(high(Xij_u)[1], low(X1_split[target1 - 1])[1], high(X1_split[target1 - 1])[1])
                            )
                            probs1_lower[target1, 1] = min(
                                transition_prob(low(Xij_u)[1], low(X1_split[target1 - 1])[1], high(X1_split[target1 - 1])[1]),
                                transition_prob(high(Xij_u)[1], low(X1_split[target1 - 1])[1], high(X1_split[target1 - 1])[1])
                            )
                        end
                    end
                    
                    probs2_lower = zeros(l[2] + 1, 1)
                    probs2_upper = zeros(l[2] + 1, 1)

                    for target2 in 1:l[2] + 1
                        if target2 == 1
                            probs2_upper[target2, 1] = max(
                                1 - transition_prob(low(Xij_u)[2], low(X)[2], high(X)[2]),
                                1 - transition_prob(high(Xij_u)[2], low(X)[2], high(X)[2])
                            ) + eps(Float64)
                            probs2_lower[target2, 1] = min(
                                1 - transition_prob(center(Xij_u)[2], low(X)[2], high(X)[2]),
                                1 - transition_prob(low(Xij_u)[2], low(X)[2], high(X)[2]),
                                1 - transition_prob(high(Xij_u)[2], low(X)[2], high(X)[2])
                            )
                        else
                            probs2_upper[target2, 1] = max(
                                transition_prob(center(Xij_u)[2], low(X2_split[target2 - 1])[1], high(X2_split[target2 - 1])[1]),
                                transition_prob(low(Xij_u)[2], low(X2_split[target2 - 1])[1], high(X2_split[target2 - 1])[1]),
                                transition_prob(high(Xij_u)[2], low(X2_split[target2 - 1])[1], high(X2_split[target2 - 1])[1])
                            )
                            probs2_lower[target2, 1] = min(
                                transition_prob(low(Xij_u)[2], low(X2_split[target2 - 1])[1], high(X2_split[target2 - 1])[1]),
                                transition_prob(high(Xij_u)[2], low(X2_split[target2 - 1])[1], high(X2_split[target2 - 1])[1])
                            )
                        end
                    end

                    push!(probs1, IntervalProbabilities(; lower=probs1_lower, upper=probs1_upper))
                    push!(probs2, IntervalProbabilities(; lower=probs2_lower, upper=probs2_upper))
                end
            end

            push!(stateptr, length(probs1) + 1)
        end
    end

    probs1, _ = IntervalMDP.interval_prob_hcat(probs1)
    probs2, _ = IntervalMDP.interval_prob_hcat(probs2)
    probs = ProductIntervalProbabilities((probs1, probs2), (Int32(l[1] + 1), Int32(l[2] + 1)))
    pmdp = ProductIntervalMarkovDecisionProcess(probs, stateptr)

    reach = Tuple{Int32, Int32}[]
    avoid = Tuple{Int32, Int32}[]
    
    for j in 1:l[2] + 1
        for i in 1:l[1] + 1
            if j == 1 || i == 1
                push!(avoid, (i, j))
            elseif X_split[i - 1, j - 1] ⊆ reach_region
                push!(reach, (i, j))
            end
        end
    end

    return pmdp, reach, avoid
end

pmdp, reach_set, avoid_set = IMDP_abstraction()

prop = FiniteTimeReachAvoid(reach_set, avoid_set, 10)
spec = Specification(prop, Pessimistic, Maximize)
prob = Problem(pmdp, spec)

V, k, it = value_iteration(prob)
