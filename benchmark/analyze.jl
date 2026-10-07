# Scaling and roofline tables from result files (§ Phase 0 Deliverables, 4).
#
#   julia --project=benchmark benchmark/analyze.jl \
#       --strong benchmark/results/scaling-20fc03b-cpu-t{1,2,4,...}.json \
#       --sizes  benchmark/results/sizes-20fc03b-cpu-t1.json,benchmark/results/sizes-20fc03b-cpu-t16.json \
#       --roofline benchmark/results/baseline-20fc03b-cpu-t1.json,benchmark/results/baseline-20fc03b-cpu-t16.json \
#       --stream benchmark/results/stream-t1.json,benchmark/results/stream-t16.json \
#       --out benchmark/results/scaling-roofline-20fc03b.md
#
# Traffic/work model for one O-maximization `bellman!` call (per nonzero of the
# ambiguity sets, "nnz" = n·n·a dense, k·n·a sparse; product: ×DFA states):
#   dense : bytes = 16·nnz (lower read fully by the dot + gap, upper bound; the gap
#           walk stops when the budget is used, roughly half the column here)
#   sparse: bytes = 20·nnz (lower 8 + gap 8 + Int32 row index 4) + 4·columns
#   flops = 4·nnz (dot: 2/entry, gap walk: ≤2/entry); sorting is not counted
# Classification (per case, using the measured roofs at the same thread count):
#   memory-bound   achieved bandwidth ≥ 50% of the DRAM read roof (data > L3) or
#                  of the L2 read roof (data ≤ L3)
#   overhead-bound median < 100 µs at 1 thread, or parallel efficiency < 25%
#                  at 16 threads
#   compute/latency-bound otherwise (sorting, branchy scalar loops, gathers)

using JSON, Statistics, Printf

function cli()
    o = Dict{String, Vector{String}}()
    out = nothing
    i = 1
    while i <= length(ARGS)
        k = ARGS[i][3:end]
        if k == "out"
            out = ARGS[i + 1]
            i += 2
            continue
        end
        vals = String[]
        i += 1
        while i <= length(ARGS) && !startswith(ARGS[i], "--")
            append!(vals, split(ARGS[i], ','))
            i += 1
        end
        o[k] = filter(!isempty, vals)
    end
    return o, out
end

load(p) = JSON.parsefile(p; dicttype = Dict{String, Any})
nthreads_of(d) = d["environment"]["threads"]["nthreads"]

function traffic(meta)
    fam = get(meta, "family", "")
    fam in ("imdp", "product") || return nothing
    nnz = Float64(meta["nnz"]) * (fam == "product" ? meta["dfa_states"] : 1)
    cols = Float64(meta["columns"]) * (fam == "product" ? meta["dfa_states"] : 1)
    bytes = meta["storage"] == "dense" ? 16nnz : 20nnz + 4cols
    return (; bytes, flops = 4nnz, nnz)
end

fmt_t(ns) = ns < 1e3 ? @sprintf("%.0f ns", ns) : ns < 1e6 ? @sprintf("%.1f µs", ns / 1e3) : ns < 1e9 ? @sprintf("%.2f ms", ns / 1e6) : @sprintf("%.2f s", ns / 1e9)

function main()
    o, out = cli()
    io = IOBuffer()
    roofs = Dict{Int, Dict{String, Any}}()
    for f in get(o, "stream", String[])
        d = load(f)
        roofs[d["nthreads"]] = d["results"]
    end
    println(io, "## Machine roofs (measured, benchmark/stream.jl)\n")
    println(io, "| threads | DRAM read GB/s | DRAM copy GB/s | DRAM triad GB/s | L2 read GB/s | in-L1 FMA GFLOP/s |")
    println(io, "|---:|---:|---:|---:|---:|---:|")
    for t in sort(collect(keys(roofs)))
        r = roofs[t]
        println(io, @sprintf("| %d | %.1f | %.1f | %.1f | %.1f | %.1f |", t, r["dram_read_GBs"], r["dram_copy_GBs"], r["dram_triad_GBs"], r["l2_read_GBs"], r["l1_fma_GFLOPs"]))
    end
    println(io)

    # Strong scaling
    strong = [load(f) for f in get(o, "strong", String[])]
    t1med = Dict{Tuple{String, String}, Float64}()
    if !isempty(strong)
        sort!(strong; by = nthreads_of)
        println(io, "## Strong scaling (fixed size, `bellman!` steady state)\n")
        cases = sort(unique([(r["case"], r["entry"]) for d in strong for r in d["results"] if haskey(r, "median_ns")]))
        ts = nthreads_of.(strong)
        println(io, "| case | entry | ", join(["t=$t" for t in ts], " | "), " |")
        println(io, "|---|---|", repeat("---:|", length(ts)))
        for (c, e) in cases
            meds = [begin
                k = findfirst(r -> r["case"] == c && r["entry"] == e && haskey(r, "median_ns"), d["results"])
                isnothing(k) ? NaN : d["results"][k]["median_ns"]
            end for d in strong]
            base = meds[1]
            t1med[(c, e)] = base
            cells = [@sprintf("%s (×%.2f, eff %.0f%%)", fmt_t(m), base / m, 100 * base / m / t) for (m, t) in zip(meds, ts)]
            println(io, "| $c | $e | ", join(cells, " | "), " |")
        end
        println(io, "\nCell = median (speed-up vs 1 thread, parallel efficiency = speed-up / threads). Threads are pinned compactly: t ≤ 6 P-cores only; t = 8 adds 2 E-cores; t = 14 all P+E; t = 16 adds the 2 LP-E cores.\n")
    end

    # Size scaling
    sizes = [load(f) for f in get(o, "sizes", String[])]
    if !isempty(sizes)
        println(io, "## Size scaling (`bellman!`, 1 action)\n")
        println(io, "| case | threads | n | nnz | median | ns per nnz | achieved GB/s (model) | GFLOP/s (model) |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
        for d in sizes, r in sort(d["results"]; by = r -> (r["meta"]["storage"], get(r["meta"], "nnz_per_column", 0), r["meta"]["states"]))
            haskey(r, "median_ns") || continue
            tr = traffic(r["meta"])
            isnothing(tr) && continue
            m = r["median_ns"]
            println(io, @sprintf("| %s | %d | %d | %d | %s | %.2f | %.1f | %.2f |", r["case"], nthreads_of(d), r["meta"]["states"], tr.nnz, fmt_t(m), m / tr.nnz, tr.bytes / m, tr.flops / m))
        end
        println(io)
    end

    # Roofline on baseline files
    rl = [load(f) for f in get(o, "roofline", String[])]
    if !isempty(rl)
        println(io, "## Roofline estimate (`bellman!` entries of the baseline)\n")
        println(io, "| case | threads | median | data MiB | GB/s (model) | % of roof | GFLOP/s (model) | AI flop/B | allocs/call | class |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        base1 = Dict{String, Float64}()
        for d in rl
            nthreads_of(d) == 1 || continue
            for r in d["results"]
                r["entry"] == "bellman" && haskey(r, "median_ns") && (base1[r["case"]] = r["median_ns"])
            end
        end
        for d in rl
            T = nthreads_of(d)
            roof = get(roofs, T, nothing)
            for r in d["results"]
                (r["entry"] == "bellman" && haskey(r, "median_ns")) || continue
                tr = traffic(r["meta"])
                isnothing(tr) && continue
                m = r["median_ns"]
                gbs = tr.bytes / m
                mib = tr.bytes / 2^20
                ref = isnothing(roof) ? NaN : (mib > 24 ? roof["dram_read_GBs"] : roof["l2_read_GBs"])
                frac = gbs / ref
                eff = haskey(base1, r["case"]) ? base1[r["case"]] / m / T : NaN
                cls = if frac >= 0.5
                    "memory-bound"
                elseif (T == 1 && m < 1e5) || (T >= 16 && eff < 0.25)
                    "overhead-bound"
                else
                    "compute/latency-bound"
                end
                println(io, @sprintf("| %s | %d | %s | %.1f | %.1f | %.0f%% (%s) | %.2f | %.2f | %d | %s |", r["case"], T, fmt_t(m), mib, gbs, 100frac,
                    mib > 24 ? "DRAM" : "L2", tr.flops / m, tr.flops / tr.bytes, r["allocs"], cls))
            end
        end
        println(io, "\nAI = arithmetic intensity of the model (4 flop per 16–20 bytes ≈ 0.2–0.25 flop/B): far below the ridge point of this machine (in-L1 FMA roof / DRAM roof ≈ tens of flop/B), so the O-max kernel can only be memory-, latency- or overhead-bound, never FLOP-bound.\n")
    end
    s = String(take!(io))
    print(s)
    isnothing(out) || write(out, s)
end

main()
