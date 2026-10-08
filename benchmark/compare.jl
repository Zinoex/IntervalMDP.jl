# Compare benchmark result files (§ Evidence Protocol).
#
#   A/B, one file each:
#     julia --project=benchmark benchmark/compare.jl BASE.json CAND.json
#   Interleaved rounds (A B A B …; files in round order):
#     julia --project=benchmark benchmark/compare.jl --base A1.json,A2.json,A3.json --cand B1.json,B2.json,B3.json
#   Run-to-run spread of repeated runs of the same code (noise bound):
#     julia --project=benchmark benchmark/compare.jl --spread R1.json R2.json [R3.json …]
#   Options: --md FILE (also write the Markdown table to FILE), --json FILE,
#            --threshold-speedup 0.05, --threshold-regression 0.03, --noise 0.05
#
# Verdicts (per case/entry/eltype, comparing medians):
#   speedup     candidate median ≥ 5% faster AND (faster by ≥ 5% in every round OR
#               Mann–Whitney U two-sided p < 0.01 on the pooled samples)
#   regression  candidate median > 3% slower under the same test
#   no change   otherwise
#   INVALID     a correctness check failed or is missing in either file
#   CLOCK       the difference would count, but the clock probes of the two sides
#               differ by > 5% or an entry was measured with a deviating clock
#               (lib/clock.jl); not usable as evidence — re-measure
# Allocation increases of steady-state `bellman` entries are flagged (ALLOC+).

using JSON, Statistics, Printf

function parse_cli(args)
    o = Dict{String, Any}("base" => String[], "cand" => String[], "spread" => String[], "md" => nothing, "json" => nothing,
        "threshold-speedup" => 0.05, "threshold-regression" => 0.03, "noise" => 0.05)
    pos = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--base" || a == "--cand"
            o[a[3:end]] = String.(split(args[i + 1], ','))
            i += 2
        elseif a == "--spread"
            i += 1
            while i <= length(args) && !startswith(args[i], "--")
                push!(o["spread"], args[i])
                i += 1
            end
        elseif a in ("--md", "--json")
            o[a[3:end]] = args[i + 1]
            i += 2
        elseif a in ("--threshold-speedup", "--threshold-regression", "--noise")
            o[a[3:end]] = parse(Float64, args[i + 1])
            i += 2
        else
            push!(pos, a)
            i += 1
        end
    end
    if isempty(o["spread"]) && isempty(o["base"])
        length(pos) == 2 || error("usage: compare.jl BASE.json CAND.json | --base a,b --cand c,d | --spread r1 r2 ...")
        o["base"] = [pos[1]]
        o["cand"] = [pos[2]]
    end
    return o
end

load(path) = JSON.parsefile(path; dicttype = Dict{String, Any})

key(r) = (r["case"], r["entry"], get(r, "eltype", "Float64"))

function index_results(d)
    idx = Dict{Tuple{String, String, String}, Dict{String, Any}}()
    ref = get(get(get(d, "parameters", Dict()), "clock_guard", Dict()), "reference_probe_ns", nothing)
    for r in d["results"]
        haskey(r, "median_ns") || continue
        # absolute clock probe (ns) of this entry, comparable across files
        isnothing(ref) || (r["_file_clock"] = Dict("ref" => ref))
        idx[key(r)] = r
    end
    return idx
end

# --- Mann–Whitney U (two-sided, normal approximation with tie correction) ----
function erfc_approx(x)
    # Numerical Recipes erfcc (fractional error < 1.2e-7)
    z = abs(x)
    t = 1 / (1 + 0.5z)
    r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
        t * (-0.82215223 + t * 0.17087277)))))))))
    return x >= 0 ? r : 2 - r
end

function mann_whitney_p(a::AbstractVector, b::AbstractVector)
    n1, n2 = length(a), length(b)
    (n1 < 2 || n2 < 2) && return NaN
    all_ = vcat([(x, 1) for x in a], [(x, 2) for x in b])
    sort!(all_; by = first)
    N = n1 + n2
    ranks = zeros(N)
    tie_term = 0.0
    i = 1
    while i <= N
        j = i
        while j < N && all_[j + 1][1] == all_[i][1]
            j += 1
        end
        r = (i + j) / 2
        for k in i:j
            ranks[k] = r
        end
        t = j - i + 1
        tie_term += t^3 - t
        i = j + 1
    end
    R1 = sum(ranks[k] for k in 1:N if all_[k][2] == 1)
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = sqrt(n1 * n2 / 12 * ((N + 1) - tie_term / (N * (N - 1))))
    sigma == 0 && return 1.0
    z = (U1 - mu - 0.5 * sign(U1 - mu)) / sigma
    return erfc_approx(abs(z) / sqrt(2))
end

valid(r) = get(r, "valid", false) == true

function ab(o)
    bases = [index_results(load(f)) for f in o["base"]]
    cands = [index_results(load(f)) for f in o["cand"]]
    length(bases) == length(cands) || error("--base and --cand need the same number of rounds")
    ks = sort(collect(intersect((keys(b) for b in vcat(bases, cands))...)))
    rows = Any[]
    s, g = o["threshold-speedup"], o["threshold-regression"]
    for k in ks
        bs = [b[k] for b in bases]
        cs = [c[k] for c in cands]
        bt = reduce(vcat, [Float64.(r["times_ns"]) for r in bs])
        ct = reduce(vcat, [Float64.(r["times_ns"]) for r in cs])
        mb, mc = median(bt), median(ct)
        ratio = mc / mb
        rounds = [r2["median_ns"] / r1["median_ns"] for (r1, r2) in zip(bs, cs)]
        p = mann_whitney_p(bt, ct)
        clk(r) = get(r, "clock_probe_ratio", 1.0) * get(get(r, "_file_clock", Dict()), "ref", 1.0)
        cb = median([clk(r) for r in bs]); cc = median([clk(r) for r in cs])
        clock_mismatch = abs(cc / cb - 1) > 0.05 || any(r -> get(r, "clock_state", "nominal") != "nominal", vcat(bs, cs))
        verdict = if !(all(valid, bs) && all(valid, cs))
            "INVALID"
        elseif clock_mismatch && (ratio <= 1 - s || ratio >= 1 + g)
            "CLOCK (not evidence)"
        elseif ratio <= 1 - s && (all(<=(1 - s), rounds) || p < 0.01)
            "speedup"
        elseif ratio >= 1 + g && (all(>=(1 + g), rounds) || p < 0.01)
            "regression"
        else
            "no change"
        end
        alloc_b = maximum(r -> r["allocs"], bs)
        alloc_c = maximum(r -> r["allocs"], cs)
        flag = (k[2] == "bellman" && alloc_c > alloc_b) ? "ALLOC+" : ""
        push!(rows, (k, mb, mc, ratio, rounds, p, verdict, alloc_b, alloc_c, flag))
    end
    io = IOBuffer()
    println(io, "| case | entry | eltype | base median | cand median | ratio | per-round ratios | MWU p | verdict | allocs base→cand |")
    println(io, "|---|---|---|---:|---:|---:|---|---:|---|---|")
    for (k, mb, mc, ratio, rounds, p, verdict, ab_, ac, flag) in rows
        println(io, @sprintf("| %s | %s | %s | %s | %s | %.3f | %s | %.2g | %s | %d→%d %s |", k[1], k[2], k[3], fmt_time(mb), fmt_time(mc), ratio,
            join([@sprintf("%.3f", r) for r in rounds], ", "), p, verdict, ab_, ac, flag))
    end
    nspeed = count(r -> r[7] == "speedup", rows)
    nreg = count(r -> r[7] == "regression", rows)
    ninv = count(r -> r[7] == "INVALID", rows)
    nclk = count(r -> startswith(r[7], "CLOCK"), rows)
    println(io, "\n$(length(rows)) entries: $nspeed speedup, $nreg regression, $ninv invalid, $nclk CLOCK (not evidence), $(length(rows) - nspeed - nreg - ninv - nclk) no change.")
    missing_ = setdiff(union((keys(b) for b in vcat(bases, cands))...), ks)
    isempty(missing_) || println(io, "Not in all files (skipped): ", join(["$(m[1])/$(m[2])/$(m[3])" for m in sort(collect(missing_))], ", "))
    out = String(take!(io))
    print(out)
    isnothing(o["md"]) || write(o["md"], out)
    if !isnothing(o["json"])
        open(o["json"], "w") do f
            JSON.json(f, [Dict("case" => r[1][1], "entry" => r[1][2], "eltype" => r[1][3], "base_median_ns" => r[2], "cand_median_ns" => r[3],
                "ratio" => r[4], "round_ratios" => r[5], "mwu_p" => r[6], "verdict" => r[7], "allocs_base" => r[8], "allocs_cand" => r[9]) for r in rows]; pretty = 1)
        end
    end
    return nreg == 0 && ninv == 0
end

function fmt_time(ns)
    ns < 1e3 && return @sprintf("%.0f ns", ns)
    ns < 1e6 && return @sprintf("%.2f µs", ns / 1e3)
    ns < 1e9 && return @sprintf("%.2f ms", ns / 1e6)
    return @sprintf("%.3f s", ns / 1e9)
end

function spread(o)
    runs = [index_results(load(f)) for f in o["spread"]]
    ks = sort(collect(intersect((keys(r) for r in runs)...)))
    io = IOBuffer()
    println(io, "| case | entry | eltype | samples | medians (per run) | spread | clock probe spread | status |")
    println(io, "|---|---|---|---|---|---:|---:|---|")
    rows = Any[]
    for k in ks
        meds = [r[k]["median_ns"] for r in runs]
        sp = (maximum(meds) - minimum(meds)) / minimum(meds)
        clks = [get(r[k], "clock_probe_ratio", NaN) * get(get(r[k], "_file_clock", Dict()), "ref", NaN) for r in runs]
        csp = (maximum(clks) - minimum(clks)) / minimum(clks)
        st = sp > o["noise"] ? "NOISY" : "ok"
        any(r -> get(r[k], "clock_state", "nominal") != "nominal", runs) && (st *= " (clock deviating)")
        all(r -> valid(r[k]), runs) || (st *= " INVALID")
        push!(rows, (k, [r[k]["samples"] for r in runs], meds, sp, st))
        println(io, @sprintf("| %s | %s | %s | %s | %s | %.1f%% | %.1f%% | %s |", k[1], k[2], k[3], join(string.([r[k]["samples"] for r in runs]), "/"),
            join(fmt_time.(meds), " / "), 100sp, 100csp, st))
    end
    noisy = filter(r -> startswith(r[5], "NOISY"), rows)
    println(io, "\n$(length(rows)) entries, $(length(noisy)) with spread > $(round(100o["noise"]; digits = 1))%.")
    sps = [r[4] for r in rows]
    isempty(sps) || println(io, @sprintf("spread: median %.2f%%, 90th percentile %.2f%%, max %.2f%%", 100median(sps), 100quantile(sps, 0.9), 100maximum(sps)))
    out = String(take!(io))
    print(out)
    isnothing(o["md"]) || write(o["md"], out)
    if !isnothing(o["json"])
        open(o["json"], "w") do f
            JSON.json(f, [Dict("case" => r[1][1], "entry" => r[1][2], "eltype" => r[1][3], "samples" => r[2], "medians_ns" => r[3], "spread" => r[4], "status" => r[5]) for r in rows]; pretty = 1)
        end
    end
    return isempty(noisy)
end

o = parse_cli(ARGS)
ok = isempty(o["spread"]) ? ab(o) : spread(o)
exit(ok ? 0 : 1)
