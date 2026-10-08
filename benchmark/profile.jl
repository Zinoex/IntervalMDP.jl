# Profile one benchmark case and write a Markdown summary (§ Phase 0 Deliverables, 3).
#
#   julia --project=benchmark --threads=T benchmark/profile.jl --case NAME \
#       [--entries bellman,solve_rvi] [--backend cpu|cuda] [--eltype Float64] \
#       [--seconds 6] [--out benchmark/profiles/NAME.md] [--append]
#
# Per entry it records:
#   * CPU sampling profile (Profile stdlib): top frames by self time ("overhead")
#     and by inclusive count, plus a pruned call tree;
#   * allocation profile (Profile.Allocs, one call): totals, by type, by the
#     innermost IntervalMDP frame;
#   * steady-state allocations of one call (@allocated after warm-up);
#   * JET.@report_opt (runtime dispatch / optimisation failures) restricted to
#     IntervalMDP code;
#   * on CUDA: the CUDA.@profile trace summary;
#   * the correctness check of the profiled call against the stored reference
#     (lib/reference.jl, same check as run.jl); a failing entry must not be used.
# Cases without a `bellman` entry (control synthesis) get the pseudo-entry
# `bellman_cs`: steady-state allocation, Profile.Allocs and JET of one `bellman!`
# call with the strategy cache `solve(ControlSynthesisProblem)` uses (runs by
# default; select it alone with `--entries bellman_cs`).
# Only summaries are saved, never raw dumps.

const POPTS = let
    o = Dict{String, Any}(
        "case" => nothing,
        "entries" => nothing,
        "backend" => "cpu",
        "eltype" => "Float64",
        "seconds" => "6",
        "out" => nothing,
        "append" => false,
        "pin" => "compact",
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--append"
            o["append"] = true
            i += 1
        else
            o[a[3:end]] = ARGS[i + 1]
            i += 2
        end
    end
    isnothing(o["case"]) && error("--case NAME is required")
    o
end

using ThreadPinning
include(joinpath(@__DIR__, "lib", "environment.jl"))
POPTS["pin"] == "compact" && pin_compact!()

using IntervalMDP, BenchmarkTools, JSON, Statistics, LinearAlgebra, Random, SparseArrays
using StableRNGs, Printf, Dates, Profile, JET

const BACKEND_NAME = Symbol(POPTS["backend"])
if BACKEND_NAME === :cuda
    @eval using CUDA, cuSPARSE, Adapt, GPUArrays
    @eval cuda_profile(f) = CUDA.@profile f()   # separate @eval: the macro needs CUDA loaded
end

include(joinpath(@__DIR__, "lib", "reference.jl"))
include(joinpath(@__DIR__, "lib", "clock.jl"))
include(joinpath(@__DIR__, "lib", "measure.jl"))
include(joinpath(@__DIR__, "cases", "registry.jl"))

const BACKEND =
    BACKEND_NAME === :cuda ?
    Backend(
        :cuda,
        x -> IntervalMDP.cu(x),
        x -> Array(x),
        () -> Base.invokelatest(getfield(Main, :CUDA).synchronize),
    ) : CPU_BACKEND

function capture(f)
    io = IOBuffer()
    ctx = IOContext(io, :displaysize => (2000, 240), :color => false)
    f(ctx)
    return String(take!(io))
end

function head_lines(s, n)
    ls = split(s, '\n')
    length(ls) <= n && return s
    return join(ls[1:n], '\n') * "\n… ($(length(ls) - n) more lines truncated)"
end

function sampling_profile(f, seconds)
    Profile.clear()
    Profile.init(; n = 10^7, delay = 0.0005)
    t0 = time()
    ncalls = 0
    @profile while time() - t0 < seconds
        f()
        ncalls += 1
    end
    # Main thread (runs the serial parts; may be in the interactive pool on
    # Julia ≥ 1.12) + the default-pool compute threads. Idle frames (wait) of
    # threads with nothing to do are still listed and must be read as idle time.
    tids = sort(unique(vcat(1, collect(Threads.threadpooltids(:default)))))  # main thread + compute threads
    selfs = capture(
        io -> Profile.print(
            io;
            format = :flat,
            sortedby = :overhead,
            C = false,
            mincount = 5,
            noisefloor = 0,
            threads = tids,
        ),
    )
    nsamples = _total_samples(selfs)
    incl = capture(
        io -> Profile.print(
            io;
            format = :flat,
            sortedby = :count,
            C = false,
            mincount = 5,
            threads = tids,
        ),
    )
    tree = capture(
        io -> Profile.print(
            io;
            format = :tree,
            C = false,
            maxdepth = 40,
            mincount = max(5, nsamples ÷ 100),
            noisefloor = 2,
            threads = tids,
        ),
    )
    return (; ncalls, nsamples, selfs, incl, tree)
end

# Total samples = sum of the Overhead column of a flat profile.
function _total_samples(s)
    tot = 0
    for l in split(s, '\n')
        f = split(strip(l))
        length(f) >= 2 || continue
        a, b = tryparse(Int, f[1]), tryparse(Int, f[2])
        (a === nothing || b === nothing) && continue
        tot += b
    end
    return tot
end

# Keep the lines of a flat profile with the largest first column.
function top_flat(s, n; col = 1)
    lines = split(s, '\n')
    hdr = findfirst(l -> occursin("Count", l) && occursin("Function", l), lines)
    isnothing(hdr) && return head_lines(s, n + 5)
    body = filter(
        l -> !isempty(strip(l)) && tryparse(Int, first(split(strip(l)))) !== nothing,
        lines[(hdr + 2):end],
    )
    key(l) = parse(Int, split(strip(l))[col])
    sort!(body; by = key, rev = true)
    return join(vcat(lines[hdr:(hdr + 1)], body[1:min(n, length(body))]), '\n')
end

function allocs_summary(f)
    nalloc = Base.@allocations f()
    rate = nalloc <= 200_000 ? 1.0 : 200_000 / nalloc
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = rate f()
    res = Profile.Allocs.fetch()
    allocs = res.allocs
    total = sum((a.size for a in allocs); init = 0)
    bytype = Dict{String, Tuple{Int, Int}}()
    bysite = Dict{String, Tuple{Int, Int}}()
    for a in allocs
        t = string(a.type)
        c, b = get(bytype, t, (0, 0))
        bytype[t] = (c + 1, b + a.size)
        site = "(no IntervalMDP frame)"
        for fr in a.stacktrace
            file = string(fr.file)
            if occursin("IntervalMDP/src", file) || occursin("IntervalMDP/ext", file)
                site = "$(fr.func) @ $(replace(file, r".*IntervalMDP/" => "")):$(fr.line)"
                break
            end
        end
        c, b = get(bysite, site, (0, 0))
        bysite[site] = (c + 1, b + a.size)
    end
    Profile.Allocs.clear()
    return (; n = length(allocs), total, bytype, bysite, rate, nalloc)
end

function table(d::Dict, n)
    rows = sort(collect(d); by = x -> x[2][2], rev = true)
    io = IOBuffer()
    println(io, "| site / type | count | bytes |")
    println(io, "|---|---:|---:|")
    for (k, (c, b)) in rows[1:min(n, length(rows))]
        println(io, "| `", replace(k, "|" => "\\|"), "` | ", c, " | ", b, " |")
    end
    return String(take!(io))
end

function jet_summary(f)
    rep = JET.report_opt(f, (); target_modules = (IntervalMDP,))
    reports = JET.get_reports(rep)
    txt = capture(io -> show(io, rep))
    return (; n = length(reports), txt)
end

# Control-synthesis cases have no `bellman` entry in the registry. Report the
# steady-state allocation of one `bellman!` call with the strategy cache that
# `solve(ControlSynthesisProblem)` builds (construct_strategy_cache(problem)), on
# fixed input values. Its result is checked against `bellman!` with the default
# cache (construct_strategy_cache(model)) and a fresh workspace: with
# maximize = true both give the same values.
function cs_bellman_probe(io, ctx, T)
    for pname in sort(collect(keys(ctx.problems)))
        prob = ctx.problems[pname].problem
        prob isa ControlSynthesisProblem || continue
        mp = IntervalMDP.system(prob)
        V = BACKEND.to_dev(input_values(StableRNG(20251006), mp, T))
        Vres = similar(V, Int.(IntervalMDP.source_shape(mp)))
        Vchk = similar(Vres)
        ws = IntervalMDP.construct_workspace(mp, ctx.alg)
        sc = IntervalMDP.construct_strategy_cache(prob)
        f =
            () -> begin
                IntervalMDP.bellman!(ws, sc, Vres, V, mp; upper_bound = false, maximize = true)
                BACKEND.sync()
                Vres
            end
        IntervalMDP.bellman!(
            IntervalMDP.construct_workspace(mp, ctx.alg),
            IntervalMDP.construct_strategy_cache(mp),
            Vchk,
            V,
            mp;
            upper_bound = false,
            maximize = true,
        )
        f();
        f()
        err = maximum(abs.(Array(Vres) .- Array(Vchk)))
        GC.gc()
        a1 = @allocated f()
        tb = @elapsed f()
        println(
            io,
            "### Entry `bellman_cs` (strategy cache of `$(pname)`: `$(nameof(typeof(sc)))`)\n",
        )
        println(
            io,
            @sprintf(
                "- one call after warm-up: %.3f ms, %d bytes allocated (`@allocated`), `Base.@allocations` = %d",
                tb * 1e3,
                a1,
                Base.@allocations(f())
            )
        )
        println(
            io,
            "- **steady-state `bellman!` allocation: ",
            a1 == 0 ? "0 bytes (meets the zero-allocation goal)" :
            "$(a1) bytes per call (does NOT meet the zero-allocation goal)",
            "**",
        )
        println(
            io,
            "- correctness: ",
            err == 0 ? "pass" : "FAIL",
            " (max |ΔV| vs `bellman!` with the default strategy cache = $(err); no stored reference exists for this pseudo-entry)\n",
        )
        report_allocs_jet(io, f)
    end
end

function report_allocs_jet(io, f)
    al = allocs_summary(f)
    println(io, "#### Allocation profile (`Profile.Allocs`, one call)\n")
    println(
        io,
        @sprintf(
            "`Base.@allocations` = %d per call; sample_rate = %.4g; %d allocations (%d bytes) recorded%s.\n",
            al.nalloc,
            al.rate,
            al.n,
            al.total,
            al.rate < 1 ?
            " — counts below are samples, multiply by 1/sample_rate for totals" : ""
        )
    )
    if al.n > 0
        println(io, "By innermost IntervalMDP frame:\n")
        println(io, table(al.bysite, 12))
        println(io, "By type:\n")
        println(io, table(al.bytype, 10))
    end
    println(io, "#### JET.@report_opt (target_modules = (IntervalMDP,))\n")
    js = try
        jet_summary(f)
    catch err
        (; n = -1, txt = "JET failed: " * first(sprint(showerror, err), 500))
    end
    println(io, "$(js.n) report(s).\n")
    println(io, "```\n", head_lines(js.txt, 60), "\n```\n")
end

function main()
    cname = POPTS["case"]
    idx = findfirst(c -> c.name == cname, CASES)
    isnothing(idx) && error("unknown case $cname")
    c = CASES[idx]
    T = POPTS["eltype"] == "Float32" ? Float32 : Float64
    ctx = c.build(T, BACKEND)
    entries =
        isnothing(POPTS["entries"]) ? [e.name for e in c.entries if e.name != "workspace"] :
        split(POPTS["entries"], ',')
    if isnothing(POPTS["entries"]) &&
       !any(e -> e.name == "bellman", c.entries) &&
       c.row == "Control synthesis"
        push!(entries, "bellman_cs")
    end
    seconds = parse(Float64, POPTS["seconds"])
    out =
        isnothing(POPTS["out"]) ? joinpath(@__DIR__, "profiles", cname * ".md") :
        POPTS["out"]
    mkpath(dirname(out))
    git = git_info()
    io = IOBuffer()
    if !POPTS["append"] || !isfile(out)
        println(io, "# Profile: `$cname`\n")
        println(io, "Case-matrix row: **$(c.row)**. Metadata: `", JSON.json(c.meta), "`.\n")
        println(
            io,
            "Generated by `benchmark/profile.jl` (summaries only; raw profiles are not stored).\n",
        )
    end
    println(io, "## Run: backend $(BACKEND_NAME), $(T), $(Threads.nthreads()) thread(s)\n")
    println(
        io,
        "- date (UTC): $(Dates.now(Dates.UTC)); git $(git["short_sha"]) (src/ext dirty: $(git["src_ext_dirty"]))",
    )
    println(
        io,
        "- Julia $(VERSION); pinning: $(POPTS["pin"]) (thread→CPU: interactive $(ThreadPinning.getcpuids(; threadpool = :interactive)), default $(ThreadPinning.getcpuids(; threadpool = :default))); sampling delay 0.5 ms, $(seconds) s per entry",
    )
    println(io)
    refd = refdir(BACKEND_NAME, T)
    refindex = load_index(refd)
    for en in entries
        if en == "bellman_cs"
            cs_bellman_probe(io, ctx, T)
            continue
        end
        e = c.entries[findfirst(x -> x.name == en, c.entries)]
        f, outf, extra = entry_closure(c, e, ctx, T, BACKEND)
        pef = pop!(extra, "_policy_eval_factory", nothing)
        f()
        r = f()
        chk = check_reference(
            refindex,
            refd,
            "$(cname)/$(en)",
            outf(r),
            T;
            policy_eval = isnothing(pef) ? nothing : pef(r),
        )
        println(io, "### Entry `$(en)`\n")
        println(io, "Descriptive fields: `", JSON.json(extra), "`\n")
        println(
            io,
            "- correctness check vs stored reference (`",
            relpath(refd, dirname(@__DIR__)),
            "`): **",
            chk["status"],
            "**",
            haskey(chk, "max_abs_diff") ?
            @sprintf(
                " (max |ΔV| = %.3g, tolerance %.3g)",
                chk["max_abs_diff"],
                chk["tolerance"]
            ) : "",
            haskey(chk, "reason") ? " — " * chk["reason"] : "",
        )
        GC.gc()
        a1 = @allocated f()
        tb = @elapsed f()
        println(
            io,
            @sprintf(
                "- one call after warm-up: %.3f ms, %d bytes allocated (`@allocated`)",
                tb * 1e3,
                a1
            )
        )
        trial = run(@benchmarkable($f()); samples = 1, evals = 1)
        println(
            io,
            "- BenchmarkTools single call: allocs = $(trial.allocs), memory = $(trial.memory) bytes",
        )
        if en == "bellman"
            println(
                io,
                "- **steady-state `bellman!` allocation: ",
                a1 == 0 ? "0 bytes (meets the zero-allocation goal)" :
                "$(a1) bytes per call (does NOT meet the zero-allocation goal)",
                "**",
            )
        end
        println(io)

        if BACKEND_NAME === :cpu
            sp = sampling_profile(f, seconds)
            println(io, "#### CPU sampling profile\n")
            println(
                io,
                "$(sp.ncalls) calls profiled, $(sp.nsamples) samples on the compute threads (default threadpool; mincount 5 for listed frames). Self-time share of a frame = Overhead / $(sp.nsamples).\n",
            )
            println(
                io,
                "Top frames by **self time** (`Overhead` column = samples whose leaf is this frame):\n",
            )
            println(io, "```\n", top_flat(sp.selfs, 20; col = 2), "\n```\n")
            println(io, "Top frames by **inclusive** count:\n")
            println(io, "```\n", top_flat(sp.incl, 25; col = 1), "\n```\n")
            println(
                io,
                "<details><summary>Pruned call tree (frames with ≥1% of samples)</summary>\n",
            )
            println(io, "```\n", head_lines(sp.tree, 120), "\n```\n</details>\n")
        else
            println(io, "#### CUDA.@profile trace\n")
            txt = try
                capture(
                    io2 -> show(
                        io2,
                        MIME"text/plain"(),
                        Base.invokelatest(cuda_profile, f),
                    ),
                )
            catch err
                "CUDA.@profile failed: " * first(sprint(showerror, err), 500)
            end
            println(io, "```\n", head_lines(txt, 80), "\n```\n")
        end

        report_allocs_jet(io, f)
        flush(stdout)
    end
    open(out, POPTS["append"] ? "a" : "w") do f
        write(f, String(take!(io)))
    end
    println("wrote $out")
end

main()
