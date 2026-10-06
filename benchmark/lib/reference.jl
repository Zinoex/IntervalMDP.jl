# Reference values and correctness checks (§ Benchmark Suite, "Correctness check
# inside the suite"; tolerances from § Julia Behavior & Tests).
#
# Layout: benchmark/reference/<backend>-<eltype>/
#   index.json                      metadata per "<case>/<entry>" key
#   <case>__<entry>.values.f64      raw little-endian Float64 values (column-major)
#   <case>__<entry>.strategy.i32    raw little-endian Int32 actions (control synthesis)
# The values are written once from the base ref (`run.jl --write-reference`) and
# every later run is checked against them.

using JSON

const REFERENCE_ROOT = joinpath(@__DIR__, "..", "reference")

# Vectors longer than this are stored as the stride subsample v[1:stride:end]
# (stride = cld(length, MAX_REF_LEN)); this only affects the n = 10⁶ size case.
const MAX_REF_LEN = 2^17

refstride(n) = n > MAX_REF_LEN ? cld(n, MAX_REF_LEN) : 1

refdir(backend, T) = joinpath(REFERENCE_ROOT, "$(backend)-$(T)")
_fname(key) = replace(key, "/" => "__")

"""
    Outcome

The observable result of one timed entry: values (as Float64 on the host),
iteration count (`-1` if not applicable), strategy (Int32 actions, empty if
not applicable), and the tolerance metadata used to check it.
"""
struct Outcome
    values::Vector{Float64}
    iterations::Int
    strategy::Vector{Int32}
    kind::Symbol      # :bellman, :converged, :finite, :none
    eps::Float64      # convergence threshold (converged) or horizon (finite)
end

Outcome() = Outcome(Float64[], -1, Int32[], :none, 0.0)

"""
    tolerance(o::Outcome, T)

‖V_new − V_ref‖∞ bound:
- single `bellman!` call: 1e-12 (Float64);
- converged infinite-horizon solve: 10·ε_conv;
- finite-horizon solve with horizon H: H·1e-12 (the per-step bound accumulated
  over H non-expansive steps);
- Float32 (any kind): max(1e-5, the bound above).
"""
function tolerance(o::Outcome, T)
    base = if o.kind === :bellman
        1e-12
    elseif o.kind === :converged
        10 * o.eps
    elseif o.kind === :finite
        o.eps * 1e-12
    else
        0.0
    end
    return T == Float32 ? max(1e-5, base) : base
end

iteration_slack(o::Outcome) = o.kind === :converged ? 1 : 0

function load_index(dir)
    path = joinpath(dir, "index.json")
    isfile(path) || return Dict{String, Any}()
    return JSON.parsefile(path; dicttype = Dict{String, Any})
end

function save_index(dir, index)
    mkpath(dir)
    open(joinpath(dir, "index.json"), "w") do io
        JSON.json(io, index; pretty = 2)
    end
end

function write_reference!(index, dir, key, o::Outcome, T)
    o.kind === :none && return nothing
    mkpath(dir)
    stride = refstride(length(o.values))
    write(joinpath(dir, _fname(key) * ".values.f64"), htol.(o.values[1:stride:end]))
    if !isempty(o.strategy)
        write(joinpath(dir, _fname(key) * ".strategy.i32"), htol.(o.strategy))
    end
    index[key] = Dict(
        "length" => length(o.values),
        "stride" => stride,
        "iterations" => o.iterations,
        "strategy_length" => length(o.strategy),
        "kind" => string(o.kind),
        "eps_or_horizon" => o.eps,
        "tolerance" => tolerance(o, T),
        "eltype" => string(T),
    )
    return nothing
end

function read_values(dir, key, n)
    v = Vector{Float64}(undef, n)
    read!(joinpath(dir, _fname(key) * ".values.f64"), v)
    return ltoh.(v)
end

function read_strategy(dir, key, n)
    v = Vector{Int32}(undef, n)
    read!(joinpath(dir, _fname(key) * ".strategy.i32"), v)
    return ltoh.(v)
end

"""
    check_reference(index, dir, key, o, T; policy_eval = nothing)

Compare an outcome with the stored reference. Returns a Dict with
`status ∈ ("pass", "fail", "no-reference", "not-applicable")`, the measured
‖ΔV‖∞, the tolerance, iteration counts and the number of strategy mismatches.
Strategy mismatches alone do not fail the check if `policy_eval` (a function
returning the value of the new strategy) shows that the new strategy achieves the
reference value within tolerance (ties, § Julia Behavior & Tests).
"""
function check_reference(index, dir, key, o::Outcome, T; policy_eval = nothing)
    o.kind === :none && return Dict("status" => "not-applicable")
    haskey(index, key) || return Dict("status" => "no-reference")
    meta = index[key]
    tol = tolerance(o, T)
    res = Dict{String, Any}("tolerance" => tol)
    if meta["length"] != length(o.values)
        res["status"] = "fail"
        res["reason"] = "length mismatch $(length(o.values)) vs reference $(meta["length"])"
        return res
    end
    stride = get(meta, "stride", 1)
    vals = o.values[1:stride:end]
    ref = read_values(dir, key, length(vals))
    err = isempty(ref) ? 0.0 : maximum(abs.(vals .- ref))
    stride > 1 && (res["compared_entries"] = "every $(stride)th entry ($(length(vals)) of $(length(o.values)))")
    res["max_abs_diff"] = err
    ok = err <= tol
    reasons = String[]
    err <= tol || push!(reasons, "‖ΔV‖∞ = $err > $tol")
    if o.iterations >= 0
        res["iterations"] = o.iterations
        res["reference_iterations"] = meta["iterations"]
        if abs(o.iterations - meta["iterations"]) > iteration_slack(o)
            ok = false
            push!(reasons, "iterations $(o.iterations) vs reference $(meta["iterations"])")
        end
    end
    if meta["strategy_length"] > 0
        sref = read_strategy(dir, key, meta["strategy_length"])
        nm = length(o.strategy) == length(sref) ? count(o.strategy .!= sref) : -1
        res["strategy_mismatches"] = nm
        if nm != 0
            if isnothing(policy_eval)
                ok = false
                push!(reasons, "strategy differs in $nm entries and no policy evaluation available")
            else
                vpol = policy_eval()[1:stride:end]
                perr = maximum(abs.(vpol .- ref))
                res["policy_eval_max_abs_diff"] = perr
                if perr > tol
                    ok = false
                    push!(reasons, "strategy differs in $nm entries and its value deviates by $perr > $tol")
                end
            end
        end
    end
    res["status"] = ok ? "pass" : "fail"
    isempty(reasons) || (res["reason"] = join(reasons, "; "))
    return res
end
