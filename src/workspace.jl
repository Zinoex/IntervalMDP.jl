
function construct_workspace end

"""
    construct_workspace(mp::SimpleIntervalMarkovProcess)

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
construct_workspace(mp::SimpleIntervalMarkovProcess) =
    construct_workspace(transition_prob(mp, 1), max_actions(mp))

"""
    construct_workspace(prob::IntervalProbabilities)

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
construct_workspace(prob::IntervalProbabilities, max_actions = 1) =
    construct_workspace(gap(prob), max_actions)

# Dense
struct DenseWorkspace{T <: Real}
    permutation::Vector{Int32}
    actions::Vector{T}
end

function DenseWorkspace(p::AbstractMatrix{T}, max_actions) where {T <: Real}
    n = size(p, 1)
    perm = Vector{Int32}(undef, n)
    actions = Vector{T}(undef, max_actions)
    return DenseWorkspace(perm, actions)
end

struct ThreadedDenseWorkspace{T <: Real}
    permutation::Vector{Int32}
    actions::Vector{Vector{T}}
end

function ThreadedDenseWorkspace(p::AbstractMatrix{T}, max_actions) where {T <: Real}
    n = size(p, 1)
    perm = Vector{Int32}(undef, n)
    actions = [Vector{T}(undef, max_actions) for _ in 1:Threads.nthreads()]
    return ThreadedDenseWorkspace(perm, actions)
end

function construct_workspace(p::AbstractMatrix, max_actions; threshold = 10)
    if Threads.nthreads() == 1 || size(p, 2) <= threshold
        return DenseWorkspace(p, max_actions)
    else
        return ThreadedDenseWorkspace(p, max_actions)
    end
end

# Sparse
struct SparseWorkspace{T <: Real}
    values_gaps::Vector{Tuple{T, T}}
    actions::Vector{T}
end

function SparseWorkspace(p::AbstractSparseMatrix{T}, max_actions) where {T <: Real}
    max_nonzeros = maximum(map(nnz, eachcol(p)))
    values_gaps = Vector{Tuple{T, T}}(undef, max_nonzeros)
    actions = Vector{T}(undef, max_actions)
    return SparseWorkspace(values_gaps, actions)
end

struct ThreadedSparseWorkspace{T}
    thread_workspaces::Vector{SparseWorkspace{T}}
end

function ThreadedSparseWorkspace(p::AbstractSparseMatrix, max_actions)
    nthreads = Threads.nthreads()
    thread_workspaces = [SparseWorkspace(p, max_actions) for _ in 1:nthreads]
    return ThreadedSparseWorkspace(thread_workspaces)
end

function construct_workspace(p::AbstractSparseMatrix, max_actions; threshold = 10)
    if Threads.nthreads() == 1 || size(p, 2) <= threshold
        return SparseWorkspace(p, max_actions)
    else
        return ThreadedSparseWorkspace(p, max_actions)
    end
end
