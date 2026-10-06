# Case-matrix rows "IMDP dense" and "IMDP sparse".
#
# Timed: workspace construction, steady-state `bellman!`, `solve` with
# RobustValueIteration on InfiniteTimeReachability (reach = 20% of the states),
# and `solve` with IntervalValueIteration on InfiniteTimeReachAvoid (reach 10%,
# avoid 10%). All Pessimistic/Maximize, ε_conv = 1e-6.

function imdp_name(storage, n, a, nnz)
    storage === :dense && return "imdp-dense-n$(n)-a$(a)"
    return "imdp-sparse-n$(n)-nnz$(nnz)-a$(a)"
end

function imdp_problems(rng, dm, n, T)
    reach = random_states(rng, (n,), 0.2)
    reach2 = random_states(rng, (n,), 0.1)
    avoid = random_states(rng, (n,), 0.1; exclude = Set(reach2))
    rvi = VerificationProblem(
        dm,
        Specification(InfiniteTimeReachability(reach, T(CONV_EPS)), Pessimistic, Maximize),
    )
    ivi = VerificationProblem(
        dm,
        Specification(
            InfiniteTimeReachAvoid(reach2, avoid, T(CONV_EPS)),
            Pessimistic,
            Maximize,
        ),
    )
    return Dict(
        "solve_rvi" => SolveSpec(rvi, RobustValueIteration(OMaximization()), :converged, CONV_EPS),
        "solve_ivi" => SolveSpec(ivi, IntervalValueIteration(OMaximization()), :converged, CONV_EPS),
    )
end

function imdp_case(; storage, n, a, nnz = 0, suites, solve_budget = 4.0, bellman_budget = 2.0, solve_min = 5)
    name = imdp_name(storage, n, a, nnz)
    k = storage === :dense ? n : nnz
    meta = Dict{String, Any}(
        "family" => "imdp",
        "storage" => string(storage),
        "states" => n,
        "actions" => a,
        "columns" => n * a,
        "nnz_per_column" => k,
        "nnz" => k * n * a,
    )
    build = function (T, be)
        rng = StableRNG(case_seed(name))
        mdp = random_imdp(rng; n, actions = a, storage, nnz, T)
        V = input_values(rng, mdp, T)
        dm = be.to_dev(mdp)
        return Ctx(dm, OMaximization(), be.to_dev(V), imdp_problems(rng, dm, n, T))
    end
    entries = [
        E_ws(),
        E_bellman(; budget = bellman_budget),
        E_solve("solve_rvi"; budget = solve_budget, min = solve_min),
        E_solve("solve_ivi"; budget = solve_budget, min = solve_min),
    ]
    return register!(
        Case(name, storage === :dense ? "IMDP dense" : "IMDP sparse", suites, meta, build, entries, [Float64], [Float64, Float32]),
    )
end

for n in (100, 1_000, 4_000), a in (1, 4)
    suites = ["full"]
    n <= 1_000 && push!(suites, "quick")
    (n == 4_000) && push!(suites, "scaling")
    imdp_case(; storage = :dense, n, a, suites)
end

for n in (10_000, 100_000), nnz in (10, 100), a in (1, 4)
    suites = ["full"]
    n == 10_000 && nnz == 10 && push!(suites, "quick")
    n == 100_000 && a == 1 && push!(suites, "scaling")
    imdp_case(; storage = :sparse, n, a, nnz, suites)
end

# Size-scaling series (bellman! only; see REPORT.md "Scaling").
function imdp_size_case(; storage, n, a = 1, nnz = 0)
    name = "size-" * imdp_name(storage, n, a, nnz)
    k = storage === :dense ? n : nnz
    meta = Dict{String, Any}(
        "family" => "imdp",
        "storage" => string(storage),
        "states" => n,
        "actions" => a,
        "columns" => n * a,
        "nnz_per_column" => k,
        "nnz" => k * n * a,
    )
    build = function (T, be)
        rng = StableRNG(case_seed(name))
        mdp = random_imdp(rng; n, actions = a, storage, nnz, T)
        V = input_values(rng, mdp, T)
        return Ctx(be.to_dev(mdp), OMaximization(), be.to_dev(V), Dict{String, SolveSpec}())
    end
    return register!(
        Case(name, storage === :dense ? "IMDP dense (size scaling)" : "IMDP sparse (size scaling)", ["sizes"], meta, build, [E_bellman()], [Float64], DataType[]),
    )
end

for n in (250, 500, 1_000, 2_000, 4_000, 8_000)
    imdp_size_case(; storage = :dense, n)
end
for n in (1_000, 10_000, 100_000, 1_000_000)
    imdp_size_case(; storage = :sparse, n, nnz = 10)
end
for n in (1_000, 10_000, 100_000)
    imdp_size_case(; storage = :sparse, n, nnz = 100)
end
