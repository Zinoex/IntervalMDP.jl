# Case-matrix row "Product IMDP × DFA".
#
# Underlying IMDP (dense or sparse) × the fixed 4-state DFA of
# `random_dfa_product`. Timed: workspace construction, steady-state `bellman!`
# on the product (value function of shape (n, 4)), and `solve` with
# RobustValueIteration on InfiniteTimeDFAReachability([3], 1e-6),
# Pessimistic/Maximize.
#
# Dense n = 10⁴ is not included: the dense underlying IMDP alone needs
# 1.6 GB (1 action) / 6.4 GB (4 actions), which does not fit next to the
# desktop workload on this 15 GB host (see README "Deviations").

function product_case(; storage, n, a, nnz = 0, suites)
    name = "product-" * imdp_name(storage, n, a, nnz) * "-dfa4"
    k = storage === :dense ? n : nnz
    meta = Dict{String, Any}(
        "family" => "product",
        "storage" => string(storage),
        "states" => n,
        "dfa_states" => 4,
        "actions" => a,
        "columns" => n * a,
        "nnz_per_column" => k,
        "nnz" => k * n * a,
    )
    build = function (T, be)
        rng = StableRNG(case_seed(name))
        mdp = random_imdp(rng; n, actions = a, storage, nnz, T)
        prod, accept = random_dfa_product(rng, be.to_dev(mdp))
        V = input_values(rng, prod, T)
        spec = Specification(InfiniteTimeDFAReachability([accept], T(CONV_EPS)), Pessimistic, Maximize)
        problems = Dict(
            "solve_dfa" => SolveSpec(VerificationProblem(prod, spec), RobustValueIteration(OMaximization()), :converged, CONV_EPS),
        )
        return Ctx(prod, OMaximization(), be.to_dev(V), problems)
    end
    entries = [E_ws(), E_bellman(), E_solve("solve_dfa")]
    return register!(Case(name, "Product IMDP x DFA", suites, meta, build, entries, [Float64], DataType[]))
end

for a in (1, 4)
    product_case(; storage = :dense, n = 1_000, a, suites = ["full", "quick"])
    product_case(; storage = :sparse, n = 1_000, a, nnz = 10, suites = ["full"])
    product_case(; storage = :sparse, n = 10_000, a, nnz = 10, suites = ["full"])
end
