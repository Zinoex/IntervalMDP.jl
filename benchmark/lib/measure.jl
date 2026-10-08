# Timing of one entry with BenchmarkTools (§ Benchmark Suite, "Separate what is timed").
#
# Protocol per entry:
#   1. warm-up call (compiles; its result is the correctness outcome);
#   2. estimate t of one call (the warm-up time is reused if it exceeds 5 s);
#   3. samples = clamp(round(budget / t), min_samples, 10_000), then capped so
#      that samples · t ≤ max-entry-seconds, but never below 3;
#      evals = ceil(50 µs / t) for calls shorter than 50 µs, else 1;
#   4. BenchmarkTools `run` with gctrial = true, gcsample = false.
# Every entry is idempotent (bellman! overwrites Vres; solve allocates its own
# state), so repeated evaluations measure the same work.

const MAX_SAMPLES = 10_000
const MIN_SAMPLES_FLOOR = 3
const SHORT_CALL_S = 50e-6
const WARMUP_REUSE_S = 5.0
const ITERATION_GUARD = 20_000
const CLOCK_RETRY_MAX_S = 30.0

function measurement_parameters(opts)
    return Dict(
        "BenchmarkTools" => string(pkgversion(BenchmarkTools)),
        "budget_scale" => opts["budget-scale"],
        "max_entry_seconds" => opts["max-entry-seconds"],
        "max_samples" => MAX_SAMPLES,
        "min_samples_floor" => MIN_SAMPLES_FLOOR,
        "short_call_threshold_s" => SHORT_CALL_S,
        "warmup_reuse_threshold_s" => WARMUP_REUSE_S,
        "gctrial" => true,
        "gcsample" => false,
        "clock_guard" => Dict(
            "probe_iterations" => CLOCK_ITERS,
            "tolerance" => CLOCK_TOL,
            "retries" => CLOCK_RETRIES[],
            "pause_s" => CLOCK_PAUSE_S,
            "retry_only_if_sampling_s_below" => CLOCK_RETRY_MAX_S,
            "reference_probe_ns" => CLOCK_REF[],
        ),
        "default_budgets" => Dict(
            "workspace" => "3 s, ≥50 samples",
            "bellman" => "4 s, ≥40 samples",
            "solve_*" => "4 s, ≥5 samples (case-specific overrides in benchmark/cases)",
        ),
        "bellman_call" => "IntervalMDP.bellman!(ws, construct_strategy_cache(model), Vres, V, model; upper_bound = false, maximize = true)",
    )
end

workspace_type(ws) = string(nameof(typeof(ws)))
workspace_type(ws::IntervalMDP.ProductWorkspace) =
    "ProductWorkspace{" * workspace_type(ws.underlying_workspace) * "}"

# Flatten a strategy to Int32 actions (all action-variable components).
strategy_vector(s::StationaryStrategy) = Int32[x for t in vec(Array(s.strategy)) for x in t]
strategy_vector(s::TimeVaryingStrategy) =
    Int32[x for st in s.strategy for t in vec(Array(st)) for x in t]

host_vec(be, A) = Float64.(vec(be.to_host(A)))

"""
    entry_closure(c, e, ctx, be)

Return `(f, outcome_fn, extra)` where `f()` is the timed call, `outcome_fn(result)`
builds the correctness `Outcome` and `extra` holds descriptive fields.
"""
function entry_closure(c, e, ctx, T, be)
    model, alg = ctx.model, ctx.alg
    if e.name == "workspace"
        f = () -> IntervalMDP.construct_workspace(model, alg)
        return f, r -> Outcome(), Dict{String, Any}()
    elseif e.name == "bellman"
        ws = IntervalMDP.construct_workspace(model, alg)
        sc = IntervalMDP.construct_strategy_cache(model)
        V = ctx.V
        Vres = similar(V, Int.(IntervalMDP.source_shape(model)))
        f =
            () -> begin
                IntervalMDP.bellman!(
                    ws,
                    sc,
                    Vres,
                    V,
                    model;
                    upper_bound = false,
                    maximize = true,
                )
                be.sync()
                Vres
            end
        out = r -> Outcome(host_vec(be, r), -1, Int32[], :bellman, 0.0)
        return f,
        out,
        Dict{String, Any}(
            "workspace_type" => workspace_type(ws),
            "strategy_cache" => string(nameof(typeof(sc))),
        )
    else
        sp = ctx.problems[e.name]
        prob, mc = sp.problem, sp.alg
        f = () -> begin
            sol = solve(prob, mc)
            be.sync()
            sol
        end
        out = function (sol)
            vals = host_vec(be, value_function(sol))
            strat =
                sol isa IntervalMDP.ControlSynthesisSolution ?
                strategy_vector(strategy(sol)) : Int32[]
            return Outcome(vals, num_iterations(sol), strat, sp.kind, sp.eps)
        end
        extra = Dict{String, Any}(
            "problem" => string(nameof(typeof(prob))),
            "property" => string(nameof(typeof(system_property(specification(prob))))),
            "algorithm" => string(nameof(typeof(mc))),
            "workspace_type" => workspace_type(
                IntervalMDP.construct_workspace(
                    IntervalMDP.system(prob),
                    IntervalMDP.bellman_algorithm(mc),
                ),
            ),
        )
        if prob isa ControlSynthesisProblem
            extra["_policy_eval_factory"] = function (sol)
                return () -> begin
                    vp = VerificationProblem(
                        IntervalMDP.system(prob),
                        specification(prob),
                        strategy(sol),
                    )
                    host_vec(be, value_function(solve(vp, mc)))
                end
            end
        end
        return f, out, extra
    end
end

# Guarded first call for solves: abort runaway (non-converging) iterations.
function guarded_first_call(f, e, ctx)
    e.name in ("workspace", "bellman") && return f()
    sp = ctx.problems[e.name]
    guard =
        (args...) -> (
            last(args) > ITERATION_GUARD &&
            error("more than $ITERATION_GUARD iterations; aborted")
        )
    sol = solve(sp.problem, sp.alg; callback = guard)
    return sol
end

function measure_entry(c, e, ctx, T, be, opts)
    res = Dict{String, Any}("entry" => e.name)
    try
        f, outcome_fn, extra = entry_closure(c, e, ctx, T, be)
        pef = pop!(extra, "_policy_eval_factory", nothing)
        merge!(res, extra)

        GC.gc()
        t_warm = @elapsed r = guarded_first_call(f, e, ctx)
        be.sync()
        res["_outcome"] = outcome_fn(r)
        isnothing(pef) || (res["_policy_eval"] = pef(r))
        if e.name != "workspace" && e.name != "bellman"
            res["iterations"] = num_iterations(r)
        end
        r = nothing
        res["warmup_s"] = t_warm
        opts["reference-only"] && return res

        t_est = if t_warm > WARMUP_REUSE_S
            t_warm
        else
            GC.gc()
            @elapsed f()
        end
        budget = e.budget * opts["budget-scale"]
        evals = t_est < SHORT_CALL_S ? ceil(Int, SHORT_CALL_S / max(t_est, 1e-9)) : 1
        t_sample = t_est * evals
        samples = clamp(round(Int, budget / t_sample), e.min_samples, MAX_SAMPLES)
        if samples * t_sample > opts["max-entry-seconds"]
            samples =
                max(MIN_SAMPLES_FLOOR, floor(Int, opts["max-entry-seconds"] / t_sample))
        end
        seconds = samples * t_sample * 3 + 5   # never the binding limit

        b = @benchmarkable $f()
        # Clock guard (lib/clock.jl): bracket the trial with clock probes and
        # re-measure (after a pause) when the clock deviates from the run reference.
        # Entries with more than CLOCK_RETRY_MAX_S of sampling are not retried.
        best = nothing
        attempts = 0
        for attempt in 0:CLOCK_RETRIES[]
            attempts += 1
            p0 = clock_probe()
            tr = run(
                b;
                samples = samples,
                evals = evals,
                seconds = seconds,
                gctrial = true,
                gcsample = false,
            )
            p1 = clock_probe()
            # Only the probe *before* the trial decides: it measures the ambient clock
            # state. The probe right after a long memory-heavy trial is often ≈28%
            # slower (package power shared with the memory traffic) and is recorded
            # for information only.
            dev = abs(clock_deviation(p0))
            if isnothing(best) || dev < best.dev
                best = (; trial = tr, p0, p1, dev)
            end
            (
                dev <= CLOCK_TOL ||
                samples * t_sample > CLOCK_RETRY_MAX_S ||
                attempt == CLOCK_RETRIES[]
            ) && break
            @warn "clock deviates by $(round(100dev; digits = 1))% from the run reference; re-measuring $(c.name)/$(e.name) in $(CLOCK_PAUSE_S) s"
            sleep(CLOCK_PAUSE_S)
        end
        trial = best.trial
        res["clock_probe_before_ns"] = best.p0
        res["clock_probe_after_ns"] = best.p1
        res["clock_probe_ratio"] = best.p0 / CLOCK_REF[]
        res["clock_state"] = best.dev <= CLOCK_TOL ? "nominal" : "deviating"
        res["clock_after_deviation"] = clock_deviation(best.p1)
        res["clock_attempts"] = attempts

        times = trial.times
        gct = trial.gctimes
        res["samples"] = length(times)
        res["evals"] = evals
        res["median_ns"] = median(times)
        res["min_ns"] = minimum(times)
        res["mean_ns"] = mean(times)
        res["std_ns"] = length(times) > 1 ? std(times) : 0.0
        res["max_ns"] = maximum(times)
        res["allocs"] = trial.allocs
        res["memory_bytes"] = trial.memory
        res["gc_median_ns"] = median(gct)
        res["gc_mean_ns"] = mean(gct)
        res["gc_fraction_of_mean"] = mean(gct) / mean(times)
        res["times_ns"] = times
        res["gctimes_ns"] = gct
        res["warmup_s"] = t_warm
        res["estimate_s"] = t_est
    catch err
        @error "entry $(c.name)/$(e.name) failed" exception = (err, catch_backtrace())
        res["error"] = first(sprint(showerror, err), 2000)
    end
    return res
end
