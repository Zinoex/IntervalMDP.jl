
function construct_workspace end

"""
    construct_workspace(mp::SimpleIntervalMarkovProcess, policy_cache::AbstractPolicyCache = NoPolicyCache(mp))

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

If policy_cache is not provided, a NoPolicyCache is used by default, implying that no policy is stored.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
construct_workspace(mp::SimpleIntervalMarkovProcess, policy_cache::AbstractPolicyCache = NoPolicyCache()) = construct_workspace(transition_prob(mp, 1), policy_cache, max_actions(mp))

"""
    construct_workspace(prob::IntervalProbabilities, policy_cache::AbstractPolicyCache = NoPolicyCache(mp))

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

If policy_cache is not provided, a NoPolicyCache is used by default, implying that no policy is stored.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
construct_workspace(prob::IntervalProbabilities, policy_cache::AbstractPolicyCache = NoPolicyCache(), max_actions=1) = construct_workspace(gap(prob), policy_cache, max_actions)

"""
    construct_workspace(mp::ParallelProduct, policy_cache::AbstractPolicyCache = NoPolicyCache(mp))

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

If policy_cache is not provided, a NoPolicyCache is used by default, implying that no policy is stored.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
construct_workspace(mp::ParallelProduct, policy_cache::AbstractPolicyCache = NoPolicyCache()) = construct_workspace(first_transition_prob(mp), product_num_states(mp), policy_cache, max_actions(mp))

############################
# Dense workspace - simple #
############################
struct DenseWorkspace{T <: Real, C <: AbstractPolicyCache}
    permutation::Vector{Int32}
    actions::Vector{T}
    policy_cache::C
end

function DenseWorkspace(p::AbstractMatrix{T}, policy_cache, max_actions) where {T <: Real}
    n = size(p, 1)
    perm = Vector{Int32}(undef, n)
    actions = Vector{T}(undef, max_actions)
    return DenseWorkspace(perm, actions, policy_cache)
end

struct ThreadedDenseWorkspace{T <: Real, C <: AbstractPolicyCache}
    permutation::Vector{Int32}
    actions::Vector{Vector{T}}
    policy_cache::C
end

function ThreadedDenseWorkspace(p::AbstractMatrix{T}, policy_cache, max_actions) where {T <: Real}
    n = size(p, 1)
    perm = Vector{Int32}(undef, n)
    actions = [Vector{T}(undef, max_actions) for _ in 1:Threads.nthreads()]
    return ThreadedDenseWorkspace(perm, actions, policy_cache)
end

function construct_workspace(p::AbstractMatrix, policy_cache, max_actions; threshold = 10)
    if Threads.nthreads() == 1 || size(p, 2) <= threshold
        return DenseWorkspace(p, policy_cache, max_actions)
    else
        return ThreadedDenseWorkspace(p, policy_cache, max_actions)
    end
end

#############################
# Dense workspace - product #
#############################
struct DenseProductWorkspace{T <: Real, C <: AbstractPolicyCache}
    permutation::Vector{Int32}
    actions::Vector{T}
    policy_cache::C
end

function DenseProductWorkspace(::AbstractMatrix{T}, max_dim, policy_cache, max_actions) where {T <: Real}
    perm = Vector{Int32}(undef, max_dim)
    actions = Vector{T}(undef, max_actions)
    return DenseProductWorkspace(perm, actions, policy_cache)
end

struct ThreadedDenseProductWorkspace{T <: Real, C <: AbstractPolicyCache}
    thread_workspaces::Vector{DenseProductWorkspace{T, C}}
    policy_cache::C
end

function ThreadedDenseProductWorkspace(p::AbstractMatrix{T}, max_dim, policy_cache, max_actions) where {T <: Real}
    thread_workspaces = [DenseProductWorkspace(p, max_dim, policy_cache, max_actions) for _ in 1:Threads.nthreads()]
    return ThreadedDenseWorkspace(thread_workspaces, policy_cache, max_actions)
end

function construct_workspace(p::AbstractMatrix, product_num_states, policy_cache, max_actions)
    max_dim = product_num_states |> recursiveflatten |> maximum

    if Threads.nthreads() == 1
        return DenseProductWorkspace(p, max_dim, policy_cache, max_actions)
    else
        return ThreadedDenseProductWorkspace(p, max_dim, policy_cache, max_actions)
    end
end

#############################
# Sparse workspace - simple #
#############################
struct SparseWorkspace{T <: Real, C <: AbstractPolicyCache}
    values_gaps::Vector{Tuple{T, T}}
    actions::Vector{T}
    policy_cache::C
end

function SparseWorkspace(p::AbstractSparseMatrix{T}, policy_cache, max_actions) where {T <: Real}
    max_nonzeros = maximum(map(nnz, eachcol(p)))
    values_gaps = Vector{Tuple{T, T}}(undef, max_nonzeros)
    actions = Vector{T}(undef, max_actions)
    return SparseWorkspace(values_gaps, actions, policy_cache)
end

struct ThreadedSparseWorkspace{T, C <: AbstractPolicyCache}
    thread_workspaces::Vector{SparseWorkspace{T, C}}
    policy_cache::C
end

function ThreadedSparseWorkspace(p::AbstractSparseMatrix, policy_cache, max_actions)
    nthreads = Threads.nthreads()
    thread_workspaces = [SparseWorkspace(p, policy_cache, max_actions) for _ in 1:nthreads]
    return ThreadedSparseWorkspace(thread_workspaces, policy_cache)
end

function construct_workspace(p::AbstractSparseMatrix, policy_cache, max_actions; threshold = 10)
    if Threads.nthreads() == 1 || size(p, 2) <= threshold
        return SparseWorkspace(p, policy_cache, max_actions)
    else
        return ThreadedSparseWorkspace(p, policy_cache, max_actions)
    end
end

##############################
# Sparse workspace - product #
##############################
struct SparseProductWorkspace{T <: Real, C <: AbstractPolicyCache}
    values_gaps::Vector{Tuple{T, T}}
    actions::Vector{T}
    policy_cache::C
end

function SparseProductWorkspace(::AbstractSparseMatrix{T}, max_dim, policy_cache, max_actions) where {T <: Real}
    values_gaps = Vector{Tuple{T, T}}(undef, max_dim)
    actions = Vector{T}(undef, max_actions)
    return SparseProductWorkspace(values_gaps, actions, policy_cache)
end

struct ThreadedSparseProductWorkspace{T, C <: AbstractPolicyCache}
    thread_workspaces::Vector{SparseProductWorkspace{T, C}}
    policy_cache::C
end

function ThreadedSparseProductWorkspace(p::AbstractSparseMatrix, max_dim, policy_cache, max_actions)
    nthreads = Threads.nthreads()
    thread_workspaces = [SparseProductWorkspace(p, max_dim, policy_cache, max_actions) for _ in 1:nthreads]
    return ThreadedSparseProductWorkspace(thread_workspaces, policy_cache)
end

function construct_workspace(p::AbstractSparseMatrix, product_num_states, policy_cache, max_actions; threshold = 10)
    max_dim = product_num_states |> recursiveflatten |> maximum

    if Threads.nthreads() == 1 || size(p, 2) <= threshold
        return SparseProductWorkspace(p, max_dim, policy_cache, max_actions)
    else
        return ThreadedSparseProductWorkspace(p, max_dim, policy_cache, max_actions)
    end
end
