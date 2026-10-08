# Case-matrix row "Control synthesis" (middle sizes, 4 actions).
#
# solve(ControlSynthesisProblem) with RobustValueIteration:
#   "solve_cs_stationary"   InfiniteTimeReachability (reach 20%), ε = 1e-6 → StationaryStrategy
#   "solve_cs_timevarying"  FiniteTimeReachability (reach 20%), horizon 10 → TimeVaryingStrategy
# The correctness check compares values, iteration counts and the strategy; a
# differing strategy is accepted only if its policy-evaluated value matches the
# reference value within tolerance (ties).

const CS_HORIZON = 10

function synthesis_case(; storage, n, a = 4, nnz = 0, suites)
    name = "cs-" * imdp_name(storage, n, a, nnz)
    k = storage === :dense ? n : nnz
    meta = Dict{String, Any}(
        "family" => "control-synthesis",
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
        dm = be.to_dev(mdp)
        reach = random_states(rng, (n,), 0.2)
        inf_spec = Specification(
            InfiniteTimeReachability(reach, T(CONV_EPS)),
            Pessimistic,
            Maximize,
        )
        fin_spec = Specification(
            FiniteTimeReachability(reach, CS_HORIZON),
            Pessimistic,
            Maximize,
        )
        problems = Dict(
            "solve_cs_stationary" => SolveSpec(
                ControlSynthesisProblem(dm, inf_spec),
                RobustValueIteration(OMaximization()),
                :converged,
                CONV_EPS,
            ),
            "solve_cs_timevarying" => SolveSpec(
                ControlSynthesisProblem(dm, fin_spec),
                RobustValueIteration(OMaximization()),
                :finite,
                Float64(CS_HORIZON),
            ),
        )
        return Ctx(dm, OMaximization(), nothing, problems)
    end
    entries = [E_solve("solve_cs_stationary"), E_solve("solve_cs_timevarying")]
    return register!(
        Case(name, "Control synthesis", suites, meta, build, entries, [Float64], [Float64]),
    )
end

synthesis_case(; storage = :dense, n = 1_000, suites = ["full", "quick"])
synthesis_case(; storage = :sparse, n = 10_000, nnz = 100, suites = ["full"])
