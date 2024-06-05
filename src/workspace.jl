
function construct_workspace end

"""
    construct_workspace(mp::IntervalMarkovProcess, policy_cache::AbstractPolicyCache = NoPolicyCache(mp))

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

If policy_cache is not provided, a NoPolicyCache is used by default, implying that no policy is stored.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
construct_workspace(mp::IntervalMarkovProcess, policy_cache::AbstractPolicyCache = NoPolicyCache(mp)) = construct_workspace(transition_prob(mp), policy_cache)

"""
    construct_workspace(prob::IntervalProbabilities, policy_cache::AbstractPolicyCache = NoPolicyCache(mp))

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

If policy_cache is not provided, a NoPolicyCache is used by default, implying that no policy is stored.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
construct_workspace(prob::IntervalProbabilities, policy_cache::AbstractPolicyCache = NoPolicyCache(prob)) = construct_workspace(gap(prob), policy_cache)


###################
# Dense workspace #
###################
abstract type AbstractDenseWorkspace end

struct DenseWorkspace{T <: Real, C <: AbstractPolicyCache} <: AbstractDenseWorkspace
    permutation::Vector{Int32}
    actions::Vector{T}
    policy_cache::C
end

function DenseWorkspace(p::AbstractMatrix{T}, policy_cache) where {T <: Real}
    n = size(p, 1)
    perm = collect(UnitRange{Int32}(1, n))
    actions = Vector{T}(undef, max_actions(policy_cache))
    return DenseWorkspace(perm, actions, policy_cache)
end

struct ThreadedDenseWorkspace{T <: Real, C <: AbstractPolicyCache} <: AbstractDenseWorkspace
    permutation::Vector{Int32}
    actions::Vector{Vector{T}}
    policy_cache::C
end

function ThreadedDenseWorkspace(p::AbstractMatrix{T}, policy_cache) where {T <: Real}
    n = size(p, 1)
    perm = collect(UnitRange{Int32}(1, n))
    actions = [Vector{T}(undef, max_actions(policy_cache)) for _ in 1:Threads.nthreads()]
    return ThreadedDenseWorkspace(perm, actions, policy_cache)
end

function construct_workspace(p::AbstractMatrix, policy_cache; threshold = 10)
    if Threads.nthreads() == 1 || size(p, 2) <= threshold
        return DenseWorkspace(p, policy_cache)
    else
        return ThreadedDenseWorkspace(p, policy_cache)
    end
end

####################
# Sparse workspace #
####################
struct SparseWorkspace{T <: Real, C <: AbstractPolicyCache}
    values_gaps::Vector{Tuple{T, T}}
    actions::Vector{T}
    policy_cache::C
end

function SparseWorkspace(p::AbstractSparseMatrix{T}, policy_cache) where {T <: Real}
    max_nonzeros = maximum(map(nnz, eachcol(p)))
    values_gaps = Vector{Tuple{T, T}}(undef, max_nonzeros)
    actions = Vector{T}(undef, max_actions(policy_cache))
    return SparseWorkspace(values_gaps, actions, policy_cache)
end

struct ThreadedSparseWorkspace{T, C <: AbstractPolicyCache}
    thread_workspaces::Vector{SparseWorkspace{T, C}}
    policy_cache::C
end

function ThreadedSparseWorkspace(p::AbstractSparseMatrix, policy_cache)
    nthreads = Threads.nthreads()
    thread_workspaces = [SparseWorkspace(p, policy_cache) for _ in 1:nthreads]
    return ThreadedSparseWorkspace(thread_workspaces, policy_cache)
end

function construct_workspace(p::AbstractSparseMatrix, policy_cache; threshold = 10)
    if Threads.nthreads() == 1 || size(p, 2) <= threshold
        return SparseWorkspace(p, policy_cache)
    else
        return ThreadedSparseWorkspace(p, policy_cache)
    end
end
