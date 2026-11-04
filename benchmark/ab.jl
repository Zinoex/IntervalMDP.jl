# Interleaved A/B benchmark of two git refs (§ Evidence Protocol: A B A B …, at least
# 3 rounds, separate Julia processes).
#
#   julia --project=benchmark benchmark/ab.jl --a BASE_REF --b CAND_REF \
#       --filter REGEX --entries LIST [--threads T] [--backend cpu|cuda] [--eltype T] \
#       [--suite full] [--rounds 3] [--label NAME] [--out-dir DIR] [--work-dir DIR] \
#       [--run-args "--budget-scale 2 --clock-retries 3"] [--keep-worktrees]
#
# What it does:
#   1. Resolves both refs to commits and checks each out into its own detached git
#      worktree under --work-dir (default: a fresh temporary directory; removed at the
#      end unless --keep-worktrees).
#   2. Overlays the benchmark harness of the *invoking* checkout (benchmark/run.jl,
#      lib/, cases/, Project.toml, Manifest.toml) onto both worktrees and links its
#      local benchmark/reference/, so both sides run the identical harness and
#      reference values and differ only in the package source (src/, ext/, …) of the
#      ref. The harness identity (commit, `git status --porcelain -- benchmark`, and a
#      SHA-256 over the copied files) is written to the manifest.
#   3. Precompiles each side once, then runs round 1 A, round 1 B, round 2 A, … each
#      in a fresh process: `julia --project=<wt>/benchmark --threads=T
#      <wt>/benchmark/run.jl --suite S --filter F --entries E --out <round file>`.
#      Each run's output goes to <out-dir>/<label>-r<i>-<A|B>.log.
#   4. Writes, in <out-dir> (default benchmark/results/logs/ab/<label>/, gitignored):
#        <label>-r<i>-A.json / -B.json   per-round results (run.jl schema; compare.jl reads them)
#        <label>-manifest.json           refs, commits, rounds, files, harness identity, command lines
#        <label>-summary.md / .json      compare.jl --base A1,A2,… --cand B1,B2,… (multi-round verdicts)
#        <label>-spread-A.md / -B.md     compare.jl --spread over the A runs / the B runs (noise per side;
#                                        also .json)
#      The summary is the multi-round evidence: compare.jl calls a speedup/regression only
#      if the pooled median ratio passes the threshold AND it holds in every round (or
#      Mann–Whitney p < 0.01), and CLOCK when the clock probes of the sides differ.
#      An "Evidence check" section is appended to the summary (Phase 1 rule, REPORT.md
#      § 10.2): every entry whose A-side spread is > 5% is listed as "A-side spread > 5%:
#      not evidence", and any round whose log has a "not settled" clock reference is listed.
#
# Resumable: a round file that already exists and is complete (has a "summary" block)
# is not re-run, so an interrupted A/B continues where it stopped (same --label and
# --out-dir). A run that fails is retried once, then the A/B stops with an error.
#
# Always pass --filter and --entries (benchmark/README.md: never run the whole suite by
# accident). Exit status: 0 if compare.jl finds no regression/invalid entry, 1 otherwise,
# 2 on usage errors.

using JSON, SHA, Dates, Printf

const BENCH_DIR = @__DIR__
const REPO = normpath(joinpath(BENCH_DIR, ".."))
const HARNESS_FILES = ["run.jl", "lib", "cases", "Project.toml", "Manifest.toml"]

const USAGE = """
usage: julia --project=benchmark benchmark/ab.jl --a REF --b REF --filter REGEX --entries LIST [options]
  --a REF / --b REF     git refs (sha, branch, tag) of the baseline (A) and the candidate (B)
  --filter REGEX        run.jl --filter (required)
  --entries LIST        run.jl --entries, comma separated (required)
  --threads T           Julia threads of each run (default 1)
  --backend cpu|cuda    run.jl --backend (default cpu)
  --eltype T            run.jl --eltype (default: run.jl's default)
  --suite NAME          run.jl --suite (default full)
  --rounds N            interleaved rounds, ≥ 3 (default 3)
  --label NAME          file prefix (default ab-<a>-<b>-<backend>-t<T>)
  --out-dir DIR         output directory (default benchmark/results/logs/ab/<label>, gitignored)
  --work-dir DIR        where the worktrees are created (default: new temporary directory)
  --run-args "ARGS"     extra run.jl arguments, e.g. "--budget-scale 2 --clock-retries 3"
  --keep-worktrees      do not remove the worktrees at the end
"""

function parse_args(args)
    o = Dict{String, Any}(
        "a" => nothing,
        "b" => nothing,
        "filter" => nothing,
        "entries" => nothing,
        "threads" => "1",
        "backend" => "cpu",
        "eltype" => nothing,
        "suite" => "full",
        "rounds" => "3",
        "label" => nothing,
        "out-dir" => nothing,
        "work-dir" => nothing,
        "run-args" => "",
        "keep-worktrees" => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--keep-worktrees"
            o["keep-worktrees"] = true
            i += 1
        elseif startswith(a, "--") && haskey(o, a[3:end]) && i < length(args)
            o[a[3:end]] = args[i + 1]
            i += 2
        else
            println(stderr, "cannot parse argument $a\n", USAGE)
            exit(2)
        end
    end
    for k in ("a", "b", "filter", "entries")
        isnothing(o[k]) && (println(stderr, "--$k is required\n", USAGE); exit(2))
    end
    o["rounds"] = parse(Int, o["rounds"])
    o["rounds"] >= 3 ||
        (println(stderr, "--rounds must be ≥ 3 (Evidence Protocol)\n", USAGE); exit(2))
    return o
end

git(args...) = readchomp(`git -C $REPO $args`)

resolve(ref) = git("rev-parse", "--verify", "$(ref)^{commit}")

function harness_identity()
    h = SHA.SHA256_CTX()
    files = String[]
    for f in HARNESS_FILES
        p = joinpath(BENCH_DIR, f)
        if isdir(p)
            for (root, _, fs) in walkdir(p), x in sort(fs)
                push!(files, joinpath(root, x))
            end
        else
            push!(files, p)
        end
    end
    sort!(files)
    for f in files
        SHA.update!(h, codeunits(relpath(f, BENCH_DIR)))
        SHA.update!(h, read(f))
    end
    return Dict(
        "commit" => git("rev-parse", "HEAD"),
        "benchmark_status" => git("status", "--porcelain", "--", "benchmark"),
        "committed" => isempty(git("status", "--porcelain", "--", "benchmark")),
        "files_sha256" => bytes2hex(SHA.digest!(h)),
    )
end

function make_worktree(dir, sha)
    isdir(dir) && run(`git -C $REPO worktree remove --force $dir`)
    run(`git -C $REPO worktree add --detach --quiet $dir $sha`)
    wb = joinpath(dir, "benchmark")
    mkpath(wb)
    for f in HARNESS_FILES
        src = joinpath(BENCH_DIR, f)
        dst = joinpath(wb, f)
        ispath(dst) && rm(dst; recursive = true, force = true)
        cp(src, dst)
    end
    ref = joinpath(BENCH_DIR, "reference")
    dref = joinpath(wb, "reference")
    ispath(dref) && rm(dref; recursive = true, force = true)
    isdir(ref) && symlink(ref, dref)
    return wb
end

complete(path) = isfile(path) && try
    haskey(JSON.parsefile(path), "summary")
catch
    false
end

function run_logged(cmd, log)
    open(log, "a") do io
        println(io, "\n# ", Dates.now(), "  ", cmd)
        flush(io)
        return success(pipeline(ignorestatus(cmd); stdout = io, stderr = io))
    end
end

function run_cmd(o, wb, out)
    julia = Base.julia_cmd()
    args = [
        "--suite",
        o["suite"],
        "--backend",
        o["backend"],
        "--filter",
        o["filter"],
        "--entries",
        o["entries"],
        "--out",
        out,
    ]
    isnothing(o["eltype"]) || append!(args, ["--eltype", o["eltype"]])
    append!(args, split(o["run-args"]))
    return `$julia --project=$wb --threads=$(o["threads"]) $(joinpath(wb, "run.jl")) $args`
end

function compare(o, files_a, files_b, prefix)
    julia = Base.julia_cmd()
    cmp = joinpath(BENCH_DIR, "compare.jl")
    log = prefix * "-compare.log"
    ok = run_logged(
        `$julia --project=$BENCH_DIR $cmp --base $(join(files_a, ',')) --cand $(join(files_b, ',')) --md $(prefix * "-summary.md") --json $(prefix * "-summary.json")`,
        log,
    )
    for (side, fs) in (("A", files_a), ("B", files_b))
        run_logged(
            `$julia --project=$BENCH_DIR $cmp --spread $fs --md $(prefix * "-spread-$side.md") --json $(prefix * "-spread-$side.json")`,
            log,
        )
    end
    return ok
end

# Phase 1 evidence rule (REPORT.md § 10.2): an entry counts only if the A-side spread
# over this A/B's own rounds is ≤ 5% and no round logged a "NOT settled" clock reference.
const A_SPREAD_MAX = 0.05

function evidence_check(prefix, rounds)
    sp = "$prefix-spread-A.json"
    noisy =
        isfile(sp) ? filter(r -> r["spread"] > A_SPREAD_MAX, JSON.parsefile(sp)) : nothing
    unsettled = [
        "r$r-$side" for
        r in 1:rounds, side in ("A", "B") if isfile("$prefix-r$r-$side.log") &&
        occursin(r"not settled"i, read("$prefix-r$r-$side.log", String))
    ]
    io = IOBuffer()
    println(io, "\n### Evidence check (A-side spread ≤ 5%, clock reference settled)\n")
    isnothing(noisy) && println(io, "- `$sp` missing: not evidence")
    for r in something(noisy, [])
        println(
            io,
            @sprintf(
                "- %s / %s / %s: A-side spread > 5%%: not evidence (%.1f%%)",
                r["case"],
                r["entry"],
                r["eltype"],
                100r["spread"]
            )
        )
    end
    isempty(unsettled) || println(
        io,
        "- clock reference not settled in $(join(unsettled, ", ")): not evidence (re-run in a steadier session)",
    )
    pass = !isnothing(noisy) && isempty(noisy) && isempty(unsettled)
    pass && println(io, "- passed: every A-side spread ≤ 5%, every clock reference settled")
    open(io2 -> write(io2, String(take!(io))), "$prefix-summary.md", "a")
    return Dict(
        "passed" => pass,
        "a_spread_over_5pct" =>
            isnothing(noisy) ? "missing" :
            [
                Dict("case" => r["case"], "entry" => r["entry"], "spread" => r["spread"])
                for r in noisy
            ],
        "clock_not_settled" => unsettled,
    )
end

function main()
    o = parse_args(ARGS)
    sha_a, sha_b = resolve(o["a"]), resolve(o["b"])
    label = something(
        o["label"],
        "ab-$(sha_a[1:7])-$(sha_b[1:7])-$(o["backend"])-t$(o["threads"])",
    )
    outdir = abspath(
        something(o["out-dir"], joinpath(BENCH_DIR, "results", "logs", "ab", label)),
    )
    mkpath(outdir)
    prefix = joinpath(outdir, label)
    workdir =
        isnothing(o["work-dir"]) ? mktempdir(; cleanup = false) : abspath(o["work-dir"])
    mkpath(workdir)
    harness = harness_identity()
    harness["committed"] ||
        @warn "benchmark harness has uncommitted changes; recorded in the manifest" harness["benchmark_status"]

    manifest = Dict{String, Any}(
        "schema" => "intervalmdp-ab/1",
        "label" => label,
        "a" => Dict("ref" => o["a"], "commit" => sha_a),
        "b" => Dict("ref" => o["b"], "commit" => sha_b),
        "rounds" => o["rounds"],
        "order" => "A B per round, rounds in sequence (A1 B1 A2 B2 …)",
        "options" => o,
        "harness" => harness,
        "started" => string(Dates.now(Dates.UTC)),
        "runs" => Any[],
    )
    sides = Dict(
        "A" => make_worktree(joinpath(workdir, "A"), sha_a),
        "B" => make_worktree(joinpath(workdir, "B"), sha_b),
    )
    files = Dict("A" => String[], "B" => String[])
    ok = false
    try
        for side in ("A", "B")
            wb = sides[side]
            log = "$prefix-precompile-$side.log"
            run_logged(
                `$(Base.julia_cmd()) --project=$wb -e "using Pkg; Pkg.instantiate(); using IntervalMDP"`,
                log,
            ) || error("precompiling side $side failed; see $log")
        end
        for r in 1:o["rounds"], side in ("A", "B")
            out = "$prefix-r$r-$side.json"
            push!(files[side], out)
            cmd = run_cmd(o, sides[side], out)
            entry =
                Dict("round" => r, "side" => side, "file" => out, "command" => string(cmd))
            if complete(out)
                entry["status"] = "reused"
            else
                t0 = time()
                good = run_logged(cmd, "$prefix-r$r-$side.log")
                good = good || run_logged(cmd, "$prefix-r$r-$side.log")
                (good && complete(out)) ||
                    error("round $r side $side failed; see $prefix-r$r-$side.log")
                entry["status"] = "ran"
                entry["wall_s"] = time() - t0
            end
            push!(manifest["runs"], entry)
            println(@sprintf("round %d %s: %s", r, side, entry["status"]))
            flush(stdout)
            write("$prefix-manifest.json", JSON.json(manifest; pretty = 1))
        end
        ok = compare(o, files["A"], files["B"], prefix)
        manifest["finished"] = string(Dates.now(Dates.UTC))
        manifest["summary"] = Dict(
            "md" => "$prefix-summary.md",
            "json" => "$prefix-summary.json",
            "spread_md" => ["$prefix-spread-A.md", "$prefix-spread-B.md"],
            "no_regression_no_invalid" => ok,
            "evidence_check" => evidence_check(prefix, o["rounds"]),
        )
        write("$prefix-manifest.json", JSON.json(manifest; pretty = 1))
        println(read("$prefix-summary.md", String))
    finally
        if !o["keep-worktrees"]
            for side in ("A", "B")
                run(
                    ignorestatus(
                        `git -C $REPO worktree remove --force $(joinpath(workdir, side))`,
                    ),
                )
            end
            run(`git -C $REPO worktree prune`)
            isnothing(o["work-dir"]) && rm(workdir; recursive = true, force = true)
        end
    end
    return ok
end

exit(main() ? 0 : 1)
