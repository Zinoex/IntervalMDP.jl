
"""
    construct_workspace(sys::StochasticProcess)

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

The workspace type is determined by the system type, the type (including device) and size of the ambiguity sets,
as well as the number of threads available.
"""
function construct_workspace end

struct ProductWorkspace{W, MT <: AbstractArray}
    underlying_workspace::W
    intermediate_values::MT
end

function construct_workspace(proc::ProductProcess, alg=default_bellman_algorithm(proc); kwargs...)
    mp = markov_process(proc)
    underlying_workspace = construct_workspace(mp, alg; kwargs...)
    intermediate_values = arrayfactory(mp, valuetype(mp), state_values(mp))

    return ProductWorkspace(underlying_workspace, intermediate_values)
end

construct_workspace(mdp::FactoredRMDP, alg=default_bellman_algorithm(mdp); kwargs...) = construct_workspace(mdp, modeltype(mdp), alg; kwargs...)

function construct_workspace(
    sys::FactoredRMDP,
    ::IsIMDP,
    ::OMaximization;
    threshold = 10,
    kwargs...
)
    prob = ambiguity_sets(marginals(sys)[1])
    return construct_workspace(prob, OMaximization(); threshold = threshold, num_actions = num_actions(sys), kwargs...)
end

# Dense
struct DenseIntervalOMaxWorkspace{T <: Real}
    budget::Vector{T}
    scratch::Vector{Int32}
    permutation::Vector{Int32}
    actions::Vector{T}
end

function DenseIntervalOMaxWorkspace(ambiguity_set::IntervalAmbiguitySets{R}, nactions) where {R <: Real}
    budget = 1 .- vec(sum(ambiguity_set.lower; dims = 1))
    scratch = Vector{Int32}(undef, num_target(ambiguity_set))
    perm = Vector{Int32}(undef, num_target(ambiguity_set))
    actions = Vector{R}(undef, nactions)
    return DenseIntervalOMaxWorkspace(budget, scratch, perm, actions)
end

permutation(ws::DenseIntervalOMaxWorkspace) = ws.permutation
scratch(ws::DenseIntervalOMaxWorkspace) = ws.scratch

struct ThreadedDenseIntervalOMaxWorkspace{T <: Real}
    thread_workspaces::Vector{DenseIntervalOMaxWorkspace{T}}
end

function ThreadedDenseIntervalOMaxWorkspace(ambiguity_set::IntervalAmbiguitySets{R}, nactions) where {R <: Real}
    budget = 1 .- vec(sum(ambiguity_set.lower; dims = 1))
    scratch = Vector{Int32}(undef, num_target(ambiguity_set))
    perm = Vector{Int32}(undef, num_target(ambiguity_set))

    workspaces = [
        DenseIntervalOMaxWorkspace(budget, scratch, perm, Vector{R}(undef, nactions)) for
        _ in 1:Threads.nthreads()
    ]
    return ThreadedDenseIntervalOMaxWorkspace(workspaces)
end

Base.getindex(ws::ThreadedDenseIntervalOMaxWorkspace, i) = ws.thread_workspaces[i]

## permutation and scratch space is shared across threads
permutation(ws::ThreadedDenseIntervalOMaxWorkspace) = permutation(first(ws.thread_workspaces))
scratch(ws::ThreadedDenseIntervalOMaxWorkspace) = scratch(first(ws.thread_workspaces))

function construct_workspace(
    prob::IntervalAmbiguitySets{R, MR},
    ::OMaximization = default_bellman_algorithm(prob);
    threshold = 10, num_actions = 1, kwargs...
) where {R, MR <: AbstractMatrix{R}}
    if Threads.nthreads() == 1 || num_sets(prob) <= threshold
        return DenseIntervalOMaxWorkspace(prob, num_actions)
    else
        return ThreadedDenseIntervalOMaxWorkspace(prob, num_actions)
    end
end

# Sparse
struct SparseIntervalOMaxWorkspace{T <: Real}
    budget::Vector{T}
    scratch::Vector{Tuple{T, T}}
    values_gaps::Vector{Tuple{T, T}}
    actions::Vector{T}
end

function SparseIntervalOMaxWorkspace(ambiguity_sets::IntervalAmbiguitySets{R}, nactions) where {R <: Real}
    max_support = maximum(supportsize, ambiguity_sets)

    budget = 1 .- vec(sum(ambiguity_sets.lower; dims = 1))
    scratch = Vector{Tuple{R, R}}(undef, max_support)
    values_gaps = Vector{Tuple{R, R}}(undef, max_support)
    actions = Vector{R}(undef, nactions)
    return SparseIntervalOMaxWorkspace(budget, scratch, values_gaps, actions)
end

scratch(ws::SparseIntervalOMaxWorkspace) = ws.scratch

struct ThreadedSparseIntervalOMaxWorkspace{T <: Real}
    thread_workspaces::Vector{SparseIntervalOMaxWorkspace{T}}
end

function ThreadedSparseIntervalOMaxWorkspace(ambiguity_sets::IntervalAmbiguitySets, nactions)
    nthreads = Threads.nthreads()
    thread_workspaces = [SparseIntervalOMaxWorkspace(ambiguity_sets, nactions) for _ in 1:nthreads]
    return ThreadedSparseIntervalOMaxWorkspace(thread_workspaces)
end

Base.getindex(ws::ThreadedSparseIntervalOMaxWorkspace, i) = ws.thread_workspaces[i]

function construct_workspace(
    prob::IntervalAmbiguitySets{R, MR},
    ::OMaximization = default_bellman_algorithm(prob);
    threshold = 10,
    num_actions = 1,
    kwargs...
) where {R, MR <: AbstractSparseMatrix{R}}
    if Threads.nthreads() == 1 || num_sets(prob) <= threshold
        return SparseIntervalOMaxWorkspace(prob, num_actions)
    else
        return ThreadedSparseIntervalOMaxWorkspace(prob, num_actions)
    end
end

# Factored interval McCormick workspace
struct FactoredIntervalMcCormickWorkspace{M <: JuMP.Model, T <: Real, AT <: AbstractArray{T}}
    model::M
    actions::AT
end

function FactoredIntervalMcCormickWorkspace(sys, alg)
    model = JuMP.Model(alg.lp_solver)
    JuMP.set_silent(model)
    set_string_names_on_creation(model, false)

    actions = Array{valuetype(sys)}(undef, action_shape(sys))

    return FactoredIntervalMcCormickWorkspace(model, actions)
end

struct ThreadedFactoredIntervalMcCormickWorkspace{M <: JuMP.Model, T <: Real, AT <: AbstractArray{T}}
    thread_workspaces::Vector{FactoredIntervalMcCormickWorkspace{M, T, AT}}
end

function ThreadedFactoredIntervalMcCormickWorkspace(sys, alg)
    nthreads = Threads.nthreads()
    thread_workspaces = [FactoredIntervalMcCormickWorkspace(sys, alg) for _ in 1:nthreads]
    return ThreadedFactoredIntervalMcCormickWorkspace(thread_workspaces)
end
Base.getindex(ws::ThreadedFactoredIntervalMcCormickWorkspace, i) = ws.thread_workspaces[i]

function construct_workspace(
    sys::FactoredRMDP,
    ::Union{IsFIMDP, IsIMDP},
    alg::LPMcCormickRelaxation;
    threshold = 10,
    kwargs...
)
    if Threads.nthreads() == 1 || num_states(sys) <= threshold
        return FactoredIntervalMcCormickWorkspace(sys, alg)
    else
        return ThreadedFactoredIntervalMcCormickWorkspace(sys, alg)
    end
end

# Factored interval o-max workspace
struct FactoredIntervalOMaxWorkspace{N, M, T <: Real, AT <: AbstractArray{T}}
    expectation_cache::NTuple{M, Vector{T}}
    values_gaps::Vector{Tuple{T, T}}
    scratch::Vector{Tuple{T, T}}
    budgets::NTuple{N, Vector{T}}
    actions::AT
end

function FactoredIntervalOMaxWorkspace(sys::FactoredRMDP)
    N = length(marginals(sys))
    R = valuetype(sys)

    max_support_per_marginal = Tuple(maximum(map(length ∘ support, ambiguity_sets(marginal))) for marginal in marginals(sys))
    max_support = maximum(max_support_per_marginal)

    expectation_cache = NTuple{N - 1, Vector{R}}(Vector{R}(undef, n) for n in max_support_per_marginal[2:end])
    values_gaps = Vector{Tuple{R, R}}(undef, max_support)
    scratch = Vector{Tuple{R, R}}(undef, max_support)

    budgets = ntuple(r -> one(R) .- vec(sum(ambiguity_sets(sys[r]).lower; dims = 1)), N)
    actions = Array{R}(undef, action_shape(sys))

    return FactoredIntervalOMaxWorkspace(expectation_cache, values_gaps, scratch, budgets, actions)
end
scratch(ws::FactoredIntervalOMaxWorkspace) = ws.scratch

struct ThreadedFactoredIntervalOMaxWorkspace{N, M, T <: Real, AT <: AbstractArray{T}}
    thread_workspaces::Vector{FactoredIntervalOMaxWorkspace{N, M, T, AT}}
end

function ThreadedFactoredIntervalOMaxWorkspace(sys::FactoredRMDP)
    nthreads = Threads.nthreads()
    thread_workspaces = [FactoredIntervalOMaxWorkspace(sys) for _ in 1:nthreads]
    return ThreadedFactoredIntervalOMaxWorkspace(thread_workspaces)
end
Base.getindex(ws::ThreadedFactoredIntervalOMaxWorkspace, i) = ws.thread_workspaces[i]

function construct_workspace(
    sys::FactoredRMDP,
    ::IsFIMDP,
    ::OMaximization;
    threshold = 10,
    kwargs...
)
    if Threads.nthreads() == 1 || num_states(sys) <= threshold
        return FactoredIntervalOMaxWorkspace(sys)
    else
        return ThreadedFactoredIntervalOMaxWorkspace(sys)
    end
end

# Factored vertex iterator workspace
struct FactoredVertexIteratorWorkspace{N, T, AT <: AbstractArray{T}}
    result_vectors::NTuple{N, Vector{T}}
    actions::AT
end

function FactoredVertexIteratorWorkspace(sys::FactoredRMDP)
    N = length(marginals(sys))
    R = valuetype(sys)

    result_vectors = ntuple(r -> Vector{R}(undef, state_values(sys, r)), N)
    actions = Array{valuetype(sys)}(undef, action_shape(sys))

    return FactoredVertexIteratorWorkspace(result_vectors, actions)
end

struct ThreadedFactoredVertexIteratorWorkspace{N, T, AT <: AbstractArray{T}}
    thread_workspaces::Vector{FactoredVertexIteratorWorkspace{N, T, AT}}
end

function ThreadedFactoredVertexIteratorWorkspace(sys::FactoredRMDP)
    nthreads = Threads.nthreads()
    thread_workspaces = [FactoredVertexIteratorWorkspace(sys) for _ in 1:nthreads]
    return ThreadedFactoredVertexIteratorWorkspace(thread_workspaces)
end

Base.getindex(ws::ThreadedFactoredVertexIteratorWorkspace, i) = ws.thread_workspaces[i]

function construct_workspace(
    sys::FactoredRMDP,
    ::Union{IsFIMDP, IsIMDP},
    ::VertexEnumeration;
    threshold = 10,
    kwargs...
)
    if Threads.nthreads() == 1 || num_states(sys) <= threshold
        return FactoredVertexIteratorWorkspace(sys)
    else
        return ThreadedFactoredVertexIteratorWorkspace(sys)
    end
end