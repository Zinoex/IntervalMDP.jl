# Case-matrix row "fIMDP" (factored IMDP; dense and sparse marginals).
#
# Every marginal depends on all state variables and the action variable.
# Timed: workspace construction, steady-state `bellman!`, and `solve` with
# RobustValueIteration on InfiniteTimeReachability (reach 20%, ε = 1e-6,
# Pessimistic/Maximize), for each Bellman algorithm.
#
# Sizes were chosen from a sizing probe at 1 thread (README "Case sizing"):
#   OMaximization            dense  v2×d10, v2×d50, v3×d10;  sparse v2×d50 (k=10), v3×d20 (k=5)
#   LPMcCormickRelaxation    dense  v2×d10 (a=1);            sparse v2×d10 (k=4), v3×d10 (k=3, bellman only)
#   VertexEnumeration        sparse v2×d10 (k=4), v3×d10 (k=3, bellman only)
# Vertex enumeration on dense 10-value marginals did not finish one bellman! call
# in 10 minutes (the vertex count of a 10-dimensional interval set is huge), so it
# runs on small supports only, as the spec allows. McCormick and vertex
# enumeration on v3×d10 take ≈1.3 s / 0.34 s per bellman! call; their solve
# (≈100 iterations) is left out to keep the suite within its time budget.

const FACTORED_ALGS = Dict(
    "omax" => OMaximization(),
    "mccormick" => LPMcCormickRelaxation(),
    "vertex" => VertexEnumeration(),
)

function fimdp_case(; nvars, nvals, a, support = nothing, alg = "omax", suites, solve = true, solve_budget = 4.0)
    storage = isnothing(support) ? "dense" : "sparse"
    name = "fimdp-$(storage)-v$(nvars)-d$(nvals)" * (isnothing(support) ? "" : "-k$(support)") * "-a$(a)-$(alg)"
    k = isnothing(support) ? nvals : support
    meta = Dict{String, Any}(
        "family" => "fimdp",
        "storage" => storage,
        "state_vars" => nvars,
        "values_per_var" => nvals,
        "states" => nvals^nvars,
        "actions" => a,
        "support_per_marginal" => k,
        "bellman_alg" => alg,
    )
    balg = FACTORED_ALGS[alg]
    build = function (T, be)
        rng = StableRNG(case_seed(name))
        mdp = random_fimdp(rng; nvars, nvals, actions = a, support, T)
        V = input_values(rng, mdp, T)
        dm = be.to_dev(mdp)
        problems = Dict{String, SolveSpec}()
        if solve
            reach = random_states(rng, ntuple(_ -> nvals, nvars), 0.2)
            spec = Specification(InfiniteTimeReachability(reach, T(CONV_EPS)), Pessimistic, Maximize)
            problems["solve_rvi"] = SolveSpec(VerificationProblem(dm, spec), RobustValueIteration(balg), :converged, CONV_EPS)
        end
        return Ctx(dm, balg, be.to_dev(V), problems)
    end
    entries = [E_ws(), E_bellman()]
    solve && push!(entries, E_solve("solve_rvi"; budget = solve_budget))
    # CUDA: only O-max has a CUDA implementation; sparse marginals are excluded
    # because the CUDA kernel returns all-zero values for them (Finding B-3).
    cuda = (alg == "omax" && isnothing(support)) ? [Float64, Float32] : DataType[]
    return register!(Case(name, "fIMDP " * alg, suites, meta, build, entries, [Float64], cuda))
end

for a in (1, 4)
    # O-maximization (recursive), dense marginals
    fimdp_case(; nvars = 2, nvals = 10, a, suites = ["full", "quick"])
    fimdp_case(; nvars = 2, nvals = 50, a, suites = ["full"])
    fimdp_case(; nvars = 3, nvals = 10, a, suites = ["full"])
    # O-maximization, sparse marginals
    fimdp_case(; nvars = 2, nvals = 50, support = 10, a, suites = ["full"])
    fimdp_case(; nvars = 3, nvals = 20, support = 5, a, suites = ["full"])
    # McCormick / vertex enumeration on small supports
    fimdp_case(; nvars = 2, nvals = 10, support = 4, a, alg = "mccormick", suites = ["full", "quick"])
    fimdp_case(; nvars = 2, nvals = 10, support = 4, a, alg = "vertex", suites = ["full", "quick"])
end
fimdp_case(; nvars = 2, nvals = 10, a = 1, alg = "mccormick", suites = ["full"])
fimdp_case(; nvars = 3, nvals = 10, support = 3, a = 1, alg = "mccormick", suites = ["full"], solve = false)
fimdp_case(; nvars = 3, nvals = 10, support = 3, a = 1, alg = "vertex", suites = ["full"], solve = false)
