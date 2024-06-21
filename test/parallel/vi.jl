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

# Value iteration of orthogonal component
prop = FiniteTimeReachability([3], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(dense_mdp, spec)
V_fixed_it_ortho, k, _ = value_iteration(problem)
V_fixed_it_ortho = V_fixed_it_ortho .* reshape(V_fixed_it_ortho, 1, 3)

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

# Dense/Dense Parallel Product
product_mdp = ParallelProduct([dense_mdp, dense_mdp])

# Finite time reachability
prop = FiniteTimeReachability([(3, 3)], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(product_mdp, spec)
V_fixed_it, k, _ = value_iteration(problem)
@test k == 10
@test V_fixed_it_ortho ≈ V_fixed_it

# Dense/Sparse Parallel Product
product_mdp = ParallelProduct([dense_mdp, sparse_mdp])

# Finite time reachability
prop = FiniteTimeReachability([(3, 3)], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(product_mdp, spec)
V_fixed_it, k, _ = value_iteration(problem)
@test k == 10
@test V_fixed_it_ortho ≈ V_fixed_it

# Sparse/Dense Parallel Product
product_mdp = ParallelProduct([sparse_mdp, dense_mdp])

# Finite time reachability
prop = FiniteTimeReachability([(3, 3)], 10)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(product_mdp, spec)
V_fixed_it, k, _ = value_iteration(problem)
@test k == 10
@test V_fixed_it_ortho ≈ V_fixed_it

# Sparse/Sparse Parallel Product
product_mdp = ParallelProduct([sparse_mdp, sparse_mdp])

# Infinite time reachability
prop = InfiniteTimeReachability([(3, 3)], 1e-6)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(product_mdp, spec)
V_fixed_it, k, res = value_iteration(problem)
@test maximum(res) < 1e-6

######################
# Nested composition #
######################
prop = FiniteTimeReachability([(3, 3, 3)], 10)
spec = Specification(prop, Pessimistic, Maximize)

nested_product_mdp1 = ParallelProduct([dense_mdp, ParallelProduct([dense_mdp, sparse_mdp])])
problem = Problem(nested_product_mdp1, spec)
V_fixed_it1, k, _ = value_iteration(problem)
@test k == 10

nested_product_mdp2 = ParallelProduct([ParallelProduct([dense_mdp, sparse_mdp]), dense_mdp])
problem = Problem(nested_product_mdp2, spec)
V_fixed_it2, k, _ = value_iteration(problem)
@test k == 10
@test V_fixed_it1 ≈ V_fixed_it2

nested_product_mdp3 =
    ParallelProduct([sparse_mdp, ParallelProduct([dense_mdp, sparse_mdp])])
problem = Problem(nested_product_mdp3, spec)
V_fixed_it3, k, _ = value_iteration(problem)
@test k == 10
@test V_fixed_it1 ≈ V_fixed_it3

nested_product_mdp4 =
    ParallelProduct([ParallelProduct([dense_mdp, sparse_mdp]), sparse_mdp])
problem = Problem(nested_product_mdp4, spec)
V_fixed_it4, k, _ = value_iteration(problem)
@test k == 10
@test V_fixed_it1 ≈ V_fixed_it4

#################################
# Test against concrete product #
#################################
prob1 = IntervalProbabilities(; lower = [
    0.2
    0.1
][:, :], upper = [
    0.7
    0.5
][:, :])

prob2 = IntervalProbabilities(; lower = [
    0.0
    1.0
][:, :], upper = [
    0.0
    1.0
][:, :])

transition_probs = [prob1, prob2]
dense_mdp = IntervalMarkovDecisionProcess(transition_probs)

# Value iteration of orthogonal component
prop = FiniteTimeReachability([2], 5)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(dense_mdp, spec)
V_fixed_it_ortho, k, _ = value_iteration(problem)
V_fixed_it_ortho = V_fixed_it_ortho .* reshape(V_fixed_it_ortho, 1, 2)

product_mdp = ParallelProduct([dense_mdp, dense_mdp])

# Finite time reachability
prop = FiniteTimeReachability([(2, 2)], 5)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(product_mdp, spec)
V_fixed_it_abstract, k, _ = value_iteration(problem)
@test k == 5
@test V_fixed_it_ortho ≈ V_fixed_it_abstract

function compute_concrete_product_mdp(
    mdp1::IntervalMarkovDecisionProcess,
    mdp2::IntervalMarkovDecisionProcess,
)
    prob1 = transition_prob(mdp1)
    prob2 = transition_prob(mdp2)

    transition_probs = IntervalProbabilities{Float64, Vector{Float64}, Matrix{Float64}}[]
    for s1 in 1:num_states(mdp1)
        for s2 in 1:num_states(mdp2)
            state_lower = Vector{Float64}[]
            state_upper = Vector{Float64}[]

            for a1 in stateptr(mdp1)[s1]:(stateptr(mdp1)[s1 + 1] - 1)
                for a2 in stateptr(mdp2)[s2]:(stateptr(mdp2)[s2 + 1] - 1)
                    l1_a1, u1_a1 = lower(prob1)[:, a1], upper(prob1)[:, a1]
                    l2_a2, u2_a2 = lower(prob2)[:, a2], upper(prob2)[:, a2]

                    l = kron(l1_a1, l2_a2)
                    u = kron(u1_a1, u2_a2)

                    push!(state_lower, l)
                    push!(state_upper, u)
                end
            end

            state_lower = reduce(hcat, state_lower)
            state_upper = reduce(hcat, state_upper)
            prob = IntervalProbabilities(; lower = state_lower, upper = state_upper)

            push!(transition_probs, prob)
        end
    end

    return IntervalMarkovDecisionProcess(transition_probs)
end

concrete_product_mdp = compute_concrete_product_mdp(dense_mdp, dense_mdp)

# Finite time reachability
prop = FiniteTimeReachability([2 * 2], 5)
spec = Specification(prop, Pessimistic, Maximize)
problem = Problem(concrete_product_mdp, spec)
V_fixed_it_concrete, k, _ = value_iteration(problem)
@test k == 5
@test vec(V_fixed_it_abstract) ≥ V_fixed_it_concrete
