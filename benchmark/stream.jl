# Machine roofs for the roofline estimate (§ Phase 0 Deliverables, 4).
#
#   julia --project=benchmark --threads=T benchmark/stream.jl --out benchmark/results/stream-tT.json
#
# Measures, with the same compact thread pinning as run.jl:
#   * DRAM bandwidth with STREAM-like kernels on 256 MiB arrays (≫ 24 MiB L3):
#     read (sum), copy (a = b), triad (a = b + s·c); bytes counted without
#     write-allocate traffic (as in STREAM);
#   * L2-resident read bandwidth (1 MiB per thread);
#   * an in-L1 FMA throughput "practical compute roof" (dot of two 1 KiB vectors, @simd).
# Best of 10 repetitions. Each thread works on its own contiguous chunk
# (Threads.@threads :static), so pages are first-touched by their thread.

using ThreadPinning
using JSON, Printf, Dates, LinearAlgebra
include(joinpath(@__DIR__, "lib", "environment.jl"))
include(joinpath(@__DIR__, "lib", "clock.jl"))
pin_compact!()

out = let i = findfirst(==("--out"), ARGS)
    isnothing(i) ? joinpath(@__DIR__, "results", "stream-t$(Threads.nthreads()).json") : ARGS[i + 1]
end

const NT = Threads.nthreads()

function chunks(n)
    len, rem = divrem(n, NT)
    return [((t - 1) * len + min(t - 1, rem) + 1):(t * len + min(t, rem)) for t in 1:NT]
end

function par_read(a, cs)
    s = zeros(NT * 8)
    Threads.@threads :static for t in 1:NT
        acc = 0.0
        @inbounds @simd for i in cs[t]
            acc += a[i]
        end
        s[8t] = acc
    end
    return sum(s)
end

function par_copy!(a, b, cs)
    Threads.@threads :static for t in 1:NT
        @inbounds @simd for i in cs[t]
            a[i] = b[i]
        end
    end
end

function par_triad!(a, b, c, s, cs)
    Threads.@threads :static for t in 1:NT
        @inbounds @simd for i in cs[t]
            a[i] = b[i] + s * c[i]
        end
    end
end

function par_init!(a, v, cs)
    Threads.@threads :static for t in 1:NT
        @inbounds for i in cs[t]
            a[i] = v
        end
    end
end

function best(f; reps = 10)
    f()
    return minimum(@elapsed(f()) for _ in 1:reps)
end

function l1_fma(reps)
    s = zeros(NT * 8)
    Threads.@threads :static for t in 1:NT
        x = rand(128)
        y = rand(128)
        acc = 0.0
        for _ in 1:reps
            acc += _dot(x, y)
        end
        s[8t] = acc
    end
    return sum(s)
end

@inline function _dot(x, y)
    a1 = a2 = a3 = a4 = 0.0
    @inbounds for i in 1:4:length(x)
        a1 = muladd(x[i], y[i], a1)
        a2 = muladd(x[i + 1], y[i + 1], a2)
        a3 = muladd(x[i + 2], y[i + 2], a3)
        a4 = muladd(x[i + 3], y[i + 3], a4)
    end
    return a1 + a2 + a3 + a4
end

N = 2^25  # 256 MiB per Float64 array
a = Vector{Float64}(undef, N); b = Vector{Float64}(undef, N); c = Vector{Float64}(undef, N)
cs = chunks(N)
par_init!(a, 1.0, cs); par_init!(b, 2.0, cs); par_init!(c, 0.5, cs)

res = Dict{String, Any}()
res["clock_probe_ns_at_start"] = clock_probe()   # ≈197 µs at 5.1 GHz, ≈436 µs at the 2.3 GHz cap (lib/clock.jl)
t = best(() -> par_read(b, cs));         res["dram_read_GBs"] = 8N / t / 1e9
t = best(() -> par_copy!(a, b, cs));     res["dram_copy_GBs"] = 16N / t / 1e9
t = best(() -> par_triad!(a, b, c, 3.0, cs)); res["dram_triad_GBs"] = 24N / t / 1e9
a = b = c = nothing; GC.gc()

M = 2^17 * NT  # 1 MiB per thread → L2-resident
x = Vector{Float64}(undef, M); csx = chunks(M); par_init!(x, 1.0, csx)
t = best(() -> (for _ in 1:20; par_read(x, csx); end)); res["l2_read_GBs"] = 20 * 8M / t / 1e9

reps = 200_000
t = best(() -> l1_fma(reps); reps = 5)
res["l1_fma_GFLOPs"] = NT * reps * 2 * 128 / t / 1e9

res["notes"] = "Bytes exclude write-allocate traffic. l1_fma is a practical in-cache compute roof (4 independent FMA chains, @simd-free unrolled), not the theoretical peak."
res["theoretical"] = Dict(
    "P_core_DP_GFLOPs" => "16 DP flop/cycle (2×256-bit FMA) × 5.1 GHz ≈ 81.6 per P-core (6 P-cores)",
    "E_core_DP_GFLOPs" => "≈16 DP flop/cycle (4×128-bit FMA, Skymont) × 4.4 GHz ≈ 70 per E-core (8 E-cores); LP-E at 2.5 GHz ≈ 40 (2 cores)",
    "DRAM" => "LPDDR5X-8400 (Arrow Lake-H platform maximum, 128-bit bus) ⇒ ≈134 GB/s theoretical; actual DIMM speed not verified (no root/dmidecode)",
)
env = environment_block(; pinning = "compact (pin_compact!)", gpu = gpu_info_unqueried())
mkpath(dirname(abspath(out)))
open(out, "w") do io
    JSON.json(io, Dict("environment" => env, "nthreads" => NT, "results" => res); pretty = 1)
end
for k in sort([k for (k, v) in res if v isa Number]); println(rpad(k, 18), @sprintf("%10.2f", res[k])); end
println("wrote $out")
