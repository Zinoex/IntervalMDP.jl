# Clock probe and guard.
#
# On the reference laptop the P-core clock was observed to switch between
# ≈5 GHz and a firmware cap of ≈2.3 GHz for minutes at a time (REPORT.md,
# "Environment"), which shifts compute-bound timings by ≈2.1× and cannot be
# averaged away with more samples. Every entry is therefore bracketed by a clock
# probe: a fixed chain of dependent integer multiply–adds whose time is
# proportional to 1/frequency (only ratios are used; the compiler may combine
# steps, so no absolute GHz is derived). The reference is taken at the start of the run
# under the same thread load as the trials, after the clock has settled (see
# `init_clock_reference!`). If the probe taken right before a trial deviates by more
# than CLOCK_TOL from the reference, the entry is re-measured (after a pause) up to CLOCK_RETRIES times,
# and the attempt with the smallest deviation is kept. Every result records
# its probes and `clock_state` ("nominal" or "deviating"),
# and compare.jl refuses to call a difference a speedup/regression when the
# clock states of the two sides differ.

const CLOCK_ITERS = 2_000_000
const CLOCK_TOL = 0.05
const CLOCK_RETRIES = Ref(3)   # run.jl --clock-retries
const CLOCK_PAUSE_S = 20.0

@noinline function _lcg_chain(n, x::UInt64)
    for _ in 1:n
        x = x * 0x5851f42d4c957f2d + 0x14057b7ef767814f
    end
    return x
end

"""
Median time (ns) of 5 probe chains on the calling thread, after ≈30 ms of busy
spinning so that the core has left any idle/low-frequency state (without the
spin, probes taken right after a pause or a GC read up to 50% slow).
"""
function clock_probe()
    t_spin = time_ns()
    while time_ns() - t_spin < 30_000_000
        _lcg_chain(10_000, UInt64(t_spin))
    end
    ts = Float64[]
    for _ in 1:5
        t0 = time_ns()
        x = _lcg_chain(CLOCK_ITERS, UInt64(time_ns()))
        t1 = time_ns()
        x == 0x1 && print("")   # keep the result alive
        push!(ts, t1 - t0)
    end
    sort!(ts)
    return ts[3]
end

const CLOCK_REF = Ref(NaN)

# Reference protocol (carry-over item 17 of the 0h spec; benchmark/README.md
# "Clock reference"). Until 1a the reference was the median of 20 probes taken
# right at process start, unloaded: at 1 thread while the core was still boosting
# (≈197 µs against ≈436 µs per probe later in the run) and at 8/16 threads without
# the all-core load under which the trial probes run (≈1.28× slower), so nearly
# every 8/16-thread entry and some 1-thread entries were flagged CLOCK.
# Now every reference probe follows a burst of the trial thread load (all
# `Threads.nthreads(:default)` threads busy, as in a threaded `bellman!`; one thread
# at --threads=1), first in a settle phase until CLOCK_SETTLE_WINDOW consecutive
# probes agree within CLOCK_SETTLE_TOL (or CLOCK_SETTLE_MAX_S has passed), then the
# reference is the median of `n` further load+probe rounds. The reference is logged
# as "NOT settled" if the settle phase timed out or the range (max − min)/median of
# those `n` probes exceeds CLOCK_SETTLE_TOL; such a run is not evidence.
const CLOCK_LOAD_S = 1.0
const CLOCK_SETTLE_WINDOW = 5
const CLOCK_SETTLE_TOL = 0.02
const CLOCK_SETTLE_MIN_S = 10.0
const CLOCK_SETTLE_MAX_S = 60.0
const CLOCK_REF_INFO = Dict{String, Any}()

"Busy-spin all default-pool threads for `seconds` (the thread load of a trial)."
function thread_load!(seconds)
    nt = Threads.nthreads(:default)
    Threads.@threads :static for _ in 1:nt
        t0 = time_ns()
        x = UInt64(t0)
        while time_ns() - t0 < seconds * 1e9
            x = _lcg_chain(10_000, x)
        end
        x == 0x1 && print("")
    end
    return nothing
end

"One reference sample: the trial thread load for `load_s`, then a probe (as before a trial)."
loaded_probe(load_s = CLOCK_LOAD_S) = (thread_load!(load_s); clock_probe())

function init_clock_reference!(; n = 9, load_s = CLOCK_LOAD_S)
    t_start = time()
    settle = Float64[]
    while true
        push!(settle, loaded_probe(load_s))
        w =
            length(settle) >= CLOCK_SETTLE_WINDOW ?
            settle[(end - CLOCK_SETTLE_WINDOW + 1):end] : nothing
        settled =
            !isnothing(w) &&
            time() - t_start >= CLOCK_SETTLE_MIN_S &&
            maximum(w) / minimum(w) - 1 <= CLOCK_SETTLE_TOL
        (settled || time() - t_start >= CLOCK_SETTLE_MAX_S) && break
    end
    settled_s = time() - t_start
    ps = sort([loaded_probe(load_s) for _ in 1:n])
    CLOCK_REF[] = ps[cld(n, 2)]   # median under trial load, after the clock has settled
    # Flag only (1a cycle 2): the settle window can pass while the clock still switches
    # state during the reference probes (a later session showed 196.5–425.6 µs within
    # one run). If the range of the reference probes, (max − min)/median, exceeds
    # CLOCK_SETTLE_TOL, the reference is reported as not settled; the reference value
    # and the measurement procedure are unchanged.
    window_settled = settled_s < CLOCK_SETTLE_MAX_S
    ref_range = (last(ps) - first(ps)) / CLOCK_REF[]
    merge!(
        CLOCK_REF_INFO,
        Dict(
            "protocol" => "loaded-settled",
            "load_threads" => Threads.nthreads(:default),
            "load_s" => load_s,
            "settle_probes_ns" => settle,
            "settle_s" => settled_s,
            "settle_window_settled" => window_settled,
            "reference_range" => ref_range,
            "settled" => window_settled && ref_range <= CLOCK_SETTLE_TOL,
            "reference_probes_ns" => ps,
            "reference_ns" => CLOCK_REF[],
        ),
    )
    ref_range <= CLOCK_SETTLE_TOL ||
        @warn "clock reference not settled: reference probe range " *
              "$(round(100ref_range; digits = 1))% of the median exceeds " *
              "$(round(100CLOCK_SETTLE_TOL; digits = 1))% (clock-state switch during the " *
              "reference probes); results of this run are not evidence"
    @info "clock reference $(round(CLOCK_REF[] / 1e3; digits = 1)) µs under $(Threads.nthreads(:default))-thread load " *
          "(settle $(length(settle)) probes in $(round(settled_s; digits = 1)) s, " *
          "$(CLOCK_REF_INFO["settled"] ? "settled" : "NOT settled"); first settle probe " *
          "$(round(first(settle) / 1e3; digits = 1)) µs; reference probe range " *
          "$(round(first(ps) / 1e3; digits = 1))–$(round(last(ps) / 1e3; digits = 1)) µs, " *
          "range $(round(100ref_range; digits = 1))%)"
    return CLOCK_REF[]
end

clock_deviation(p) = p / CLOCK_REF[] - 1
