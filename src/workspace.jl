
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

function construct_workspace(proc::ProductProcess)
    mp = markov_process(proc)
    underlying_workspace = construct_workspace(mp)
    intermediate_values = arrayfactory(mp, valuetype(mp), state_variables(mp))

    return ProductWorkspace(underlying_workspace, intermediate_values)
end

# Dense
struct DenseIntervalWorkspace{T <: Real}
    budget::Vector{T}
    scratch::Vector{Int32}
    permutation::Vector{Int32}
    actions::Vector{T}
end

function DenseIntervalWorkspace(ambiguity_set::IntervalAmbiguitySets{R}, nactions) where {R <: Real}
    budget = 1 .- vec(sum(ambiguity_set.lower; dims = 1))
    scratch = Vector{Int32}(undef, num_target(ambiguity_set))
    perm = Vector{Int32}(undef, num_target(ambiguity_set))
    actions = Vector{R}(undef, nactions)
    return DenseIntervalWorkspace(budget, scratch, perm, actions)
end

permutation(ws::DenseIntervalWorkspace) = ws.permutation
scratch(ws::DenseIntervalWorkspace) = ws.scratch

struct ThreadedDenseIntervalWorkspace{T <: Real}
    thread_workspaces::Vector{DenseIntervalWorkspace{T}}
end

function ThreadedDenseIntervalWorkspace(ambiguity_set::IntervalAmbiguitySets{R}, nactions) where {R <: Real}
    budget = 1 .- vec(sum(ambiguity_set.lower; dims = 1))
    scratch = Vector{Int32}(undef, num_target(ambiguity_set))
    perm = Vector{Int32}(undef, num_target(ambiguity_set))

    workspaces = [
        DenseIntervalWorkspace(budget, scratch, perm, Vector{R}(undef, nactions)) for
        _ in 1:Threads.nthreads()
    ]
    return ThreadedDenseIntervalWorkspace(workspaces)
end

Base.getindex(ws::ThreadedDenseIntervalWorkspace, i) = ws.thread_workspaces[i]

## permutation and scratch space is shared across threads
permutation(ws::ThreadedDenseIntervalWorkspace) = permutation(first(ws.thread_workspaces))
scratch(ws::ThreadedDenseIntervalWorkspace) = scratch(first(ws.thread_workspaces))

function construct_workspace(
    prob::IntervalAmbiguitySets{R, MR};
    threshold = 10,
) where {R, MR <: AbstractMatrix{R}}
    if Threads.nthreads() == 1 || num_sets(prob) <= threshold
        return DenseIntervalWorkspace(prob, 1)
    else
        return ThreadedDenseIntervalWorkspace(prob, 1)
    end
end

function construct_workspace(
    sys::FactoredRMDP{N, M, <:Tuple{<:Marginal{<:IntervalAmbiguitySets{R, MR}}}};
    threshold = 10,
) where {N, M, R, MR <: AbstractMatrix{R}}
    prob = sys.transition[1].ambiguity_sets
    if Threads.nthreads() == 1 || num_states(sys) <= threshold
        return DenseIntervalWorkspace(prob, num_actions(sys))
    else
        return ThreadedDenseIntervalWorkspace(prob, num_actions(sys))
    end
end

# Sparse
struct SparseIntervalWorkspace{T <: Real}
    budget::Vector{T}
    scratch::Vector{Tuple{T, T}}
    values_gaps::Vector{Tuple{T, T}}
    actions::Vector{T}
end

function SparseIntervalWorkspace(ambiguity_sets::IntervalAmbiguitySets{R}, nactions) where {R <: Real}
    max_support = maximum(nnz, ambiguity_sets)

    budget = 1 .- vec(sum(ambiguity_sets.lower; dims = 1))
    scratch = Vector{Tuple{R, R}}(undef, max_support)
    values_gaps = Vector{Tuple{R, R}}(undef, max_support)
    actions = Vector{R}(undef, nactions)
    return SparseIntervalWorkspace(budget, scratch, values_gaps, actions)
end

scratch(ws::SparseIntervalWorkspace) = ws.scratch

struct ThreadedSparseIntervalWorkspace{T}
    thread_workspaces::Vector{SparseIntervalWorkspace{T}}
end

function ThreadedSparseIntervalWorkspace(ambiguity_sets::IntervalAmbiguitySets, nactions)
    nthreads = Threads.nthreads()
    thread_workspaces = [SparseIntervalWorkspace(ambiguity_sets, nactions) for _ in 1:nthreads]
    return ThreadedSparseIntervalWorkspace(thread_workspaces)
end

Base.getindex(ws::ThreadedSparseIntervalWorkspace, i) = ws.thread_workspaces[i]

function construct_workspace(
    prob::IntervalAmbiguitySets{R, MR};
    threshold = 10,
) where {R, MR <: AbstractSparseMatrix{R}}
    if Threads.nthreads() == 1 || num_sets(prob) <= threshold
        return SparseIntervalWorkspace(prob, 1)
    else
        return ThreadedSparseIntervalWorkspace(prob, 1)
    end
end

function construct_workspace(
    sys::FactoredRMDP{N, M, <:Tuple{<:Marginal{<:IntervalAmbiguitySets{R, MR}}}};
    threshold = 10,
) where {N, M, R, MR <: AbstractSparseMatrix{R}}
    prob = sys.transition[1].ambiguity_sets
    if Threads.nthreads() == 1 || num_states(sys) <= threshold
        return SparseIntervalWorkspace(prob, num_actions(sys))
    else
        return ThreadedSparseIntervalWorkspace(prob, num_actions(sys))
    end
end
