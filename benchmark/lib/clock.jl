# Clock probe and guard.
#
# On the reference laptop the P-core clock was observed to switch between
# ≈5 GHz and a firmware cap of ≈2.3 GHz for minutes at a time (REPORT.md,
# "Environment"), which shifts compute-bound timings by ≈2.1× and cannot be
# averaged away with more samples. Every entry is therefore bracketed by a clock
# probe: a fixed chain of dependent integer multiply–adds whose time is
# proportional to 1/frequency (only ratios are used; the compiler may combine
# steps, so no absolute GHz is derived). The reference is the median of 20 probes at
# the start of the run (the prevailing clock state). If the probe taken right before a trial deviates by more
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

function init_clock_reference!(; n = 20)
    clock_probe()
    ps = sort([clock_probe() for _ in 1:n])
    CLOCK_REF[] = ps[cld(n, 2)]   # median: the prevailing clock state at the start of the run
    return CLOCK_REF[]
end

clock_deviation(p) = p / CLOCK_REF[] - 1
