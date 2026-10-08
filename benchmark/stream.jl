# Machine roofs for the roofline estimate (§ Phase 0 Deliverables, 4).
#
#   julia --project=benchmark --threads=T benchmark/stream.jl --out benchmark/results/stream-tT.json
#
# Measures, with the same compact thread pinning as run.jl:
#   * DRAM bandwidth with STREAM-like kernels on 256 MiB arrays (≫ 24 MiB L3):
#     read (sum), copy (a = b), triad (a = b + s·c); bytes counted without
#     write-allocate traffic (as in STREAM);
#   * L2-resident read bandwidth (1 MiB per thread) and L3-resident read bandwidth
#     (12 MiB in total), each pass repeated inside the threaded loop;
#   * an in-L1 FMA throughput "practical compute roof" (dot of two 1 KiB vectors, @simd).
# Best of 10 repetitions per round; 3 rounds (each with a clock probe); roof =
# maximum over the rounds, all rounds stored under "rounds". Each thread works on its own contiguous chunk
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

# Cache-resident read: the repetitions run inside the threaded loop, so one task
# spawn is amortised over `reps` passes (a spawn per pass dominated the old L2 figure).
function par_read_rep(a, cs, reps)
    s = zeros(NT * 8)
    Threads.@threads :static for t in 1:NT
        acc = 0.0
        for _ in 1:reps
            @inbounds @simd for i in cs[t]
                acc += a[i]
            end
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
M = 2^17 * NT  # 1 MiB per thread → L2-resident (P-core L2 3 MiB, E-core cluster L2 4 MiB / 4 cores)
x = Vector{Float64}(undef, M); csx = chunks(M); par_init!(x, 1.0, csx)
L = 3 * 2^19  # 12 MiB in total (half of the 24 MiB L3), split over the threads → L3-resident
y = Vector{Float64}(undef, L); csy = chunks(L); par_init!(y, 1.0, csy)
const FMA_REPS = 200_000

# One round of all roofs, each the best of 10 (FMA: 5) repetitions.
function measure_round()
    r = Dict{String, Float64}()
    r["clock_probe_ns"] = clock_probe()   # ≈197 µs at 5.1 GHz, ≈436 µs at the 2.3 GHz cap (lib/clock.jl)
    t = best(() -> par_read(b, cs));              r["dram_read_GBs"] = 8N / t / 1e9
    t = best(() -> par_copy!(a, b, cs));          r["dram_copy_GBs"] = 16N / t / 1e9
    t = best(() -> par_triad!(a, b, c, 3.0, cs)); r["dram_triad_GBs"] = 24N / t / 1e9
    t = best(() -> par_read_rep(x, csx, 200));    r["l2_read_GBs"] = 200 * 8M / t / 1e9
    t = best(() -> par_read_rep(y, csy, 50));     r["llc_read_GBs"] = 50 * 8L / t / 1e9
    t = best(() -> l1_fma(FMA_REPS); reps = 5);   r["l1_fma_GFLOPs"] = NT * FMA_REPS * 2 * 128 / t / 1e9
    return r
end

# Roof = maximum over ROUNDS rounds (temporal noise, clock switches); every round is stored.
const ROUNDS = 3
rounds = [(i > 1 && sleep(2); measure_round()) for i in 1:ROUNDS]
res = Dict{String, Any}()
for k in ("dram_read_GBs", "dram_copy_GBs", "dram_triad_GBs", "l2_read_GBs", "llc_read_GBs", "l1_fma_GFLOPs")
    res[k] = maximum(r[k] for r in rounds)
end
res["clock_probe_ns_at_start"] = rounds[1]["clock_probe_ns"]
res["clock_probe_ns_at_end"] = clock_probe()   # same state as at start ⇒ roofs belong to one clock state
res["rounds"] = rounds

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
