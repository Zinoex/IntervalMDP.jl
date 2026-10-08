# Case-matrix row "Real model": test/data/multiObj_robotIMDP.{sta,tra,lab,pctl}
# (PRISM format; sparse IMDP, InfiniteTimeReachability Pessimistic/Maximize as
# stored in the .pctl file). Timed: workspace, bellman!, verification solve and
# control-synthesis solve with the default algorithm.

using IntervalMDP.Data

const REAL_MODEL_PATH = joinpath(@__DIR__, "..", "..", "test", "data", "multiObj_robotIMDP")

function real_case()
    name = "real-multiObj_robotIMDP"
    meta = Dict{String, Any}(
        "family" => "real",
        "storage" => "sparse",
        "source" => "test/data/multiObj_robotIMDP.{sta,tra,lab,pctl}",
    )
    build = function (T, be)
        T == Float64 || error("the real model is stored in Float64")
        cs = read_prism_file(REAL_MODEL_PATH)
        mdp = be.to_dev(system(cs))
        spec = specification(cs)
        prop = system_property(spec)
        eps = Float64(convergence_eps(prop))
        rng = StableRNG(case_seed(name))
        V = input_values(rng, mdp, T)
        problems = Dict(
            "solve_rvi" => SolveSpec(
                VerificationProblem(mdp, spec),
                RobustValueIteration(OMaximization()),
                :converged,
                eps,
            ),
            "solve_cs_stationary" => SolveSpec(
                ControlSynthesisProblem(mdp, spec),
                RobustValueIteration(OMaximization()),
                :converged,
                eps,
            ),
        )
        return Ctx(mdp, OMaximization(), be.to_dev(V), problems)
    end
    entries = [E_ws(), E_bellman(), E_solve("solve_rvi"), E_solve("solve_cs_stationary")]
    return register!(
        Case(
            name,
            "Real model",
            ["full", "quick"],
            meta,
            build,
            entries,
            [Float64],
            [Float64],
        ),
    )
end

real_case()
