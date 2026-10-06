# Case registry for the benchmark suite.
#
# A `Case` has a stable name, the case-matrix row it belongs to, the suites it is
# part of, size metadata (used by the roofline analysis), a `build(T, backend)`
# function returning a `Ctx`, and the list of timed entries.
#
# Entry names:
#   "workspace"          IntervalMDP.construct_workspace(model, alg)
#   "bellman"            IntervalMDP.bellman!(ws, cache, Vres, V, model; upper_bound=false, maximize=true)
#                        with pre-built workspace and strategy cache (steady state)
#   anything else        CommonSolve.solve(problem, mc_alg) for ctx.problems[entry]

struct Entry
    name::String
    budget::Float64     # target seconds of sampling
    min_samples::Int
end

struct Case
    name::String
    row::String
    suites::Vector{String}
    meta::Dict{String, Any}
    build::Function                 # (T, backend) -> Ctx
    entries::Vector{Entry}
    eltypes_cpu::Vector{DataType}
    eltypes_cuda::Vector{DataType}  # empty = not run on CUDA
end

struct SolveSpec
    problem::Any
    alg::Any                        # model checking algorithm
    kind::Symbol                    # :converged or :finite
    eps::Float64                    # ε_conv or horizon
end

struct Ctx
    model::Any
    alg::Any                        # Bellman algorithm
    V::Any                          # input value function for "bellman"
    problems::Dict{String, SolveSpec}
end

struct Backend
    name::Symbol
    to_dev::Function                # model or array → device
    to_host::Function               # array → Array
    sync::Function
end

const CPU_BACKEND = Backend(:cpu, identity, x -> Array(x), () -> nothing)

const CASES = Case[]

function register!(c::Case)
    any(x -> x.name == c.name, CASES) && error("duplicate case name $(c.name)")
    push!(CASES, c)
    return c
end

# Default entry budgets (seconds) and minimum samples; see README "Parameters".
E_ws(; budget = 3.0, min = 50) = Entry("workspace", budget, min)
E_bellman(; budget = 4.0, min = 40) = Entry("bellman", budget, min)
E_solve(name; budget = 4.0, min = 5) = Entry(name, budget, min)

const CONV_EPS = 1e-6

"""Value function input for `bellman`: seeded uniform values of the state shape."""
function input_values(rng, model, T)
    shape = Int.(IntervalMDP.state_values(model))
    return T.(rand(rng, shape...))
end

include("generators.jl")
include("imdp.jl")
include("fimdp.jl")
include("product.jl")
include("synthesis.jl")
include("real.jl")
