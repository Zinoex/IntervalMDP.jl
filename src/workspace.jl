
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

function construct_workspace(proc::ProductProcess, alg; kwargs...)
    mp = markov_process(proc)
    underlying_workspace = construct_workspace(mp, alg; kwargs...)
    intermediate_values = arrayfactory(mp, valuetype(mp), state_variables(mp))

    return ProductWorkspace(underlying_workspace, intermediate_values)
end

abstract type IMDPWorkspace end

# Dense
struct DenseIntervalOMaxWorkspace{T <: Real} <: IMDPWorkspace
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

struct ThreadedDenseIntervalOMaxWorkspace{T <: Real} <: IMDPWorkspace
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
    ::OMaximization;
    threshold = 10, kwargs...
) where {R, MR <: AbstractMatrix{R}}
    if Threads.nthreads() == 1 || num_sets(prob) <= threshold
        return DenseIntervalOMaxWorkspace(prob, 1)
    else
        return ThreadedDenseIntervalOMaxWorkspace(prob, 1)
    end
end

function construct_workspace(
    sys::FactoredRMDP{N, M, <:Tuple{<:Marginal{<:IntervalAmbiguitySets{R, MR}}}},
    ::OMaximization;
    threshold = 10,
    kwargs...
) where {N, M, R, MR <: AbstractMatrix{R}}
    prob = sys.transition[1].ambiguity_sets
    if Threads.nthreads() == 1 || num_states(sys) <= threshold
        return DenseIntervalOMaxWorkspace(prob, num_actions(sys))
    else
        return ThreadedDenseIntervalOMaxWorkspace(prob, num_actions(sys))
    end
end

# Sparse
struct SparseIntervalOMaxWorkspace{T <: Real} <: IMDPWorkspace
    budget::Vector{T}
    scratch::Vector{Tuple{T, T}}
    values_gaps::Vector{Tuple{T, T}}
    actions::Vector{T}
end

function SparseIntervalOMaxWorkspace(ambiguity_sets::IntervalAmbiguitySets{R}, nactions) where {R <: Real}
    max_support = maximum(nnz, ambiguity_sets)

    budget = 1 .- vec(sum(ambiguity_sets.lower; dims = 1))
    scratch = Vector{Tuple{R, R}}(undef, max_support)
    values_gaps = Vector{Tuple{R, R}}(undef, max_support)
    actions = Vector{R}(undef, nactions)
    return SparseIntervalOMaxWorkspace(budget, scratch, values_gaps, actions)
end

scratch(ws::SparseIntervalOMaxWorkspace) = ws.scratch

struct ThreadedSparseIntervalOMaxWorkspace{T <: Real} <: IMDPWorkspace
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
    ::OMaximization;
    threshold = 10,
    kwargs...
) where {R, MR <: AbstractSparseMatrix{R}}
    if Threads.nthreads() == 1 || num_sets(prob) <= threshold
        return SparseIntervalOMaxWorkspace(prob, 1)
    else
        return ThreadedSparseIntervalOMaxWorkspace(prob, 1)
    end
end

function construct_workspace(
    sys::FactoredRMDP{N, M, <:Tuple{<:Marginal{<:IntervalAmbiguitySets{R, MR}}}},
    ::OMaximization;
    threshold = 10,
) where {N, M, R, MR <: AbstractSparseMatrix{R}}
    prob = sys.transition[1].ambiguity_sets
    if Threads.nthreads() == 1 || num_states(sys) <= threshold
        return SparseIntervalOMaxWorkspace(prob, num_actions(sys))
    else
        return ThreadedSparseIntervalOMaxWorkspace(prob, num_actions(sys))
    end
end

# Factored interval McCormick workspace
struct FactoredIntervalMcCormickWorkspace{M <: JuMP.Model, T <: Real, AT <: AbstractArray{T}}
    model::M
    actions::AT
end

function FactoredIntervalMcCormickWorkspace(sys, alg)
    model = JuMP.Model(alg.lp_optimizer)
    JuMP.set_silent(model)
    set_string_names_on_creation(model, false)

    actions = Array{valuetype(sys)}(undef, action_shape(sys))

    return FactoredIntervalMcCormickWorkspace(model, actions)
end

struct ThreadedFactoredIntervalMcCormickWorkspace{M <: JuMP.Model, T <: Real, AT <: AbstractArray{T}} <: IMDPWorkspace
    thread_workspaces::Vector{FactoredIntervalMcCormickWorkspace{M, T, AT}}
end

function ThreadedFactoredIntervalMcCormickWorkspace(sys, alg)
    nthreads = Threads.nthreads()
    thread_workspaces = [FactoredIntervalMcCormickWorkspace(sys, alg) for _ in 1:nthreads]
    return ThreadedFactoredIntervalMcCormickWorkspace(thread_workspaces)
end
Base.getindex(ws::ThreadedFactoredIntervalMcCormickWorkspace, i) = ws.thread_workspaces[i]

function construct_workspace(
    sys::FactoredRMDP{N, M, <:NTuple{N, <:Marginal{<:IntervalAmbiguitySets}}},
    alg::LPMcCormickRelaxation;
    threshold = 10,
    kwargs...
) where {N, M}
    if Threads.nthreads() == 1 || num_states(sys) <= threshold
        return FactoredIntervalMcCormickWorkspace(sys, alg)
    else
        return ThreadedFactoredIntervalMcCormickWorkspace(sys, alg)
    end
end