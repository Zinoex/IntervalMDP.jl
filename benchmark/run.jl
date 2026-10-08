# IntervalMDP.jl benchmark suite — entry point.
#
#   julia --project=benchmark --threads=<T> benchmark/run.jl \
#       --suite <full|quick|scaling|sizes|...> [--backend cpu|cuda] [--eltype Float64|Float32|all] \
#       --out benchmark/results/<file>.json [--filter REGEX] [--entries a,b] [--pin compact|none] \
#       [--write-reference] [--budget-scale X] [--max-entry-seconds S]
#
# See benchmark/README.md for the meaning of the options and of the output.

const USAGE = """
usage: julia --project=benchmark --threads=T benchmark/run.jl --suite NAME --out FILE.json [options]
  --suite NAME[,NAME]   suites to run (full, quick, scaling, sizes); a case runs if it is in any of them
  --backend cpu|cuda    default cpu
  --eltype T            Float64 | Float32 | all (default: Float64 on cpu, all on cuda)
  --out FILE            output JSON path (required unless --reference-only)
  --filter REGEX        only cases whose name matches REGEX
  --entries LIST        only these entries (comma separated, e.g. bellman,solve_rvi)
  --pin compact|none    thread pinning (default compact: Julia thread i -> CPU i-1)
  --write-reference     store outcomes as the reference; refused (exit 3, no override) unless src/ and ext/ have no
                        committed or uncommitted difference to the base ref (BENCH_BASE_REF, default 20fc03b)
  --budget-scale X      multiply all sampling budgets by X (default 1)
  --max-entry-seconds S cap on the sampling time of one entry (default 90)
  --reference-only      compute outcomes only, skip timing; with --write-reference: write the reference, without:
                        check every outcome against it. --out is optional (default benchmark/results/logs/)
  --clock-retries N     re-measure an entry up to N times when the clock deviates (default 3; 0 = record only)
  --list                list the selected cases and exit (without --suite: all registered cases)
"""

function parse_args(args)
    opts = Dict{String, Any}(
        "suite" => nothing,
        "backend" => "cpu",
        "eltype" => nothing,
        "out" => nothing,
        "filter" => nothing,
        "entries" => nothing,
        "pin" => "compact",
        "write-reference" => false,
        "budget-scale" => 1.0,
        "max-entry-seconds" => 90.0,
        "list" => false,
        "reference-only" => false,
        "clock-retries" => "3",
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("--write-reference", "--list", "--reference-only")
            opts[a[3:end]] = true
            i += 1
        elseif startswith(a, "--") && i < length(args)
            key = a[3:end]
            haskey(opts, key) || (println(stderr, "unknown option $a\n", USAGE); exit(2))
            opts[key] = args[i + 1]
            i += 2
        else
            println(stderr, "cannot parse argument $a\n", USAGE)
            exit(2)
        end
    end
    isnothing(opts["suite"]) && opts["list"] && (opts["suite"] = "*")  # --list alone: every registered case
    isnothing(opts["suite"]) && (println(stderr, USAGE); exit(2))
    opts["budget-scale"] = parse(Float64, string(opts["budget-scale"]))
    opts["max-entry-seconds"] = parse(Float64, string(opts["max-entry-seconds"]))
    return opts
end

const OPTS = parse_args(ARGS)

# --- Thread pinning first (before anything else runs on the threads) ---------
using ThreadPinning
include(joinpath(@__DIR__, "lib", "environment.jl"))
if OPTS["pin"] == "compact"
    pin_compact!()
elseif OPTS["pin"] != "none"
    error("unknown --pin $(OPTS["pin"])")
end

using IntervalMDP, BenchmarkTools, JSON, Statistics, LinearAlgebra, Random, SparseArrays
using StableRNGs, Printf, Dates

const BACKEND_NAME = Symbol(OPTS["backend"])
if BACKEND_NAME === :cuda
    @eval using CUDA, cuSPARSE, Adapt, GPUArrays
end

include(joinpath(@__DIR__, "lib", "reference.jl"))
include(joinpath(@__DIR__, "lib", "clock.jl"))
include(joinpath(@__DIR__, "lib", "measure.jl"))
include(joinpath(@__DIR__, "cases", "registry.jl"))

const BACKEND = if BACKEND_NAME === :cpu
    CPU_BACKEND
elseif BACKEND_NAME === :cuda
    CUDA.functional() || error("CUDA backend requested but CUDA.functional() == false")
    Backend(:cuda, x -> IntervalMDP.cu(x), x -> Array(x), () -> CUDA.synchronize())
else
    error("unknown backend $(BACKEND_NAME)")
end

function select_cases(opts)
    suites = split(opts["suite"], ',')
    re = isnothing(opts["filter"]) ? nothing : Regex(opts["filter"])
    sel = Case[]
    for c in CASES
        "*" in suites || any(s -> s in c.suites, suites) || continue
        isnothing(re) || occursin(re, c.name) || continue
        BACKEND_NAME === :cuda && isempty(c.eltypes_cuda) && continue
        push!(sel, c)
    end
    return sel
end

function case_eltypes(c::Case, opts)
    avail = BACKEND_NAME === :cuda ? c.eltypes_cuda : c.eltypes_cpu
    e = opts["eltype"]
    (isnothing(e) || e == "all") && return BACKEND_NAME === :cuda || e == "all" ? avail : intersect(avail, [Float64])
    T = e == "Float32" ? Float32 : e == "Float64" ? Float64 : error("unknown eltype $e")
    return intersect(avail, [T])
end

function main()
    opts = OPTS
    cases = select_cases(opts)
    if opts["list"]
        for c in cases
            println(rpad(c.name, 48), rpad(c.row, 28), join([e.name for e in c.entries], ","))
        end
        return
    end
    if opts["write-reference"]
        # Reference values come from the base ref only; there is no override.
        ok, reason = base_ref_guard(REPO_ROOT, BASE_REF)
        ok || (println(stderr, "refusing --write-reference: ", reason); exit(3))
    end
    if isnothing(opts["out"])
        # --reference-only runs need no timing file; keep their record in the (gitignored) logs dir.
        opts["reference-only"] || (println(stderr, USAGE); exit(2))
        sha = _cmd(`git -C $REPO_ROOT rev-parse --short HEAD`; default = "nogit")
        opts["out"] = joinpath(@__DIR__, "results", "logs",
            "$(opts["write-reference"] ? "reference-write" : "reference-check")-$(sha)-$(BACKEND_NAME).json")
    end
    entry_filter = isnothing(opts["entries"]) ? nothing : split(opts["entries"], ',')

    env = environment_block(;
        pinning = opts["pin"] == "compact" ? "compact: default-pool thread i -> CPU i-1, interactive/main thread -> CPU 0 (ThreadPinning.jl)" : "none (OS scheduler)",
        gpu = BACKEND_NAME === :cuda ? gpu_info_cuda(CUDA) : gpu_info_unqueried(),
    )

    CLOCK_RETRIES[] = parse(Int, string(opts["clock-retries"]))
    init_clock_reference!()
    out = Dict{String, Any}(
        "schema" => "intervalmdp-benchmark/1",
        "suite" => opts["suite"],
        "backend" => string(BACKEND_NAME),
        "options" => Dict(k => string(v) for (k, v) in opts),
        "parameters" => measurement_parameters(opts),
        "environment" => env,
        "results" => Any[],
    )
    t_start = time()
    mkpath(dirname(abspath(opts["out"])))

    for c in cases, T in case_eltypes(c, opts)
        dir = refdir(BACKEND_NAME, T)
        index = load_index(dir)
        println("case $(c.name) [$T, $(BACKEND_NAME), $(Threads.nthreads()) threads]")
        ctx = try
            c.build(T, BACKEND)
        catch e
            @error "build failed for $(c.name)" exception = (e, catch_backtrace())
            push!(out["results"], Dict("case" => c.name, "row" => c.row, "entry" => "build", "eltype" => string(T), "valid" => false, "error" => sprint(showerror, e)))
            continue
        end
        for e in c.entries
            !isnothing(entry_filter) && !(e.name in entry_filter) && continue
            if BACKEND_NAME === :cuda && e.name == "solve_ivi"
                # Finding B-2 (REPORT.md): IVI throws on CUDA models (scalar indexing in
                # max_initial_gap, src/interval_value_iteration.jl). Not timed.
                push!(out["results"], Dict("case" => c.name, "row" => c.row, "entry" => e.name, "eltype" => string(T), "backend" => "cuda",
                    "unsupported" => "IntervalValueIteration throws on CUDA (scalar indexing in max_initial_gap); Finding B-2", "valid" => true))
                println("  solve_ivi              UNSUPPORTED on CUDA (Finding B-2)")
                continue
            end
            res = measure_entry(c, e, ctx, T, BACKEND, opts)
            key = "$(c.name)/$(e.name)"
            outcome = pop!(res, "_outcome", Outcome())
            policy_eval = pop!(res, "_policy_eval", nothing)
            if haskey(res, "error")
                res["valid"] = false
            elseif opts["write-reference"]
                write_reference!(index, dir, key, outcome, T)
                res["correctness"] = Dict("status" => outcome.kind === :none ? "not-applicable" : "reference-written")
                res["valid"] = true
            else
                chk = check_reference(index, dir, key, outcome, T; policy_eval)
                res["correctness"] = chk
                res["valid"] = chk["status"] in ("pass", "not-applicable")
                chk["status"] == "fail" && @warn "correctness check FAILED for $key: $(get(chk, "reason", ""))"
                chk["status"] == "no-reference" && @warn "no reference for $key"
            end
            res["case"] = c.name
            res["row"] = c.row
            res["meta"] = c.meta
            res["eltype"] = string(T)
            res["backend"] = string(BACKEND_NAME)
            push!(out["results"], res)
            println(@sprintf("  %-22s median %12.3f µs  min %12.3f µs  samples %5d  allocs %d  %s", e.name, get(res, "median_ns", NaN) / 1e3, get(res, "min_ns", NaN) / 1e3, get(res, "samples", 0), get(res, "allocs", -1), get(get(res, "correctness", Dict()), "status", get(res, "error", ""))) * (haskey(res, "clock_probe_ratio") ? @sprintf("  clock %.3f%s", res["clock_probe_ratio"], res["clock_state"] == "nominal" ? "" : " DEVIATING") : ""))
            flush(stdout)
            write_json(opts["out"], out)
        end
        opts["write-reference"] && save_index(dir, index)
        ctx = nothing
        GC.gc(true)
    end

    nres = length(out["results"])
    ninvalid = count(r -> !get(r, "valid", false), out["results"])
    out["summary"] = Dict(
        "entries" => nres,
        "invalid" => ninvalid,
        "unsupported" => count(r -> haskey(r, "unsupported"), out["results"]),
        "clock_deviating" => count(r -> get(r, "clock_state", "") == "deviating", out["results"]),
        "clock_retried" => count(r -> get(r, "clock_attempts", 1) > 1, out["results"]),
        "no_reference" => count(r -> get(get(r, "correctness", Dict()), "status", "") == "no-reference", out["results"]),
        "wall_time_s" => time() - t_start,
        "finished_utc" => string(Dates.now(Dates.UTC)),
    )
    write_json(opts["out"], out)
    @info "wrote $(opts["out"]): $nres entries, $ninvalid invalid, $(round(time() - t_start; digits = 1)) s"
    return ninvalid == 0
end

function write_json(path, obj)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON.json(io, obj; pretty = 1)
    end
    mv(tmp, path; force = true)
end

ok = main()
exit(ok === false ? 1 : 0)
