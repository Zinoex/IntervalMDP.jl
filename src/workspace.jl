
function construct_workspace end

struct ProductWorkspace{W, MT <: AbstractArray}
    underlying_workspace::W
    intermediate_values::MT
end

"""
    construct_workspace(proc::ProductProcess)

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

The underlying workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
function construct_workspace(proc::ProductProcess)
    mp = markov_process(proc)
    underlying_workspace = construct_workspace(mp)
    intermediate_values = arrayfactory(mp, valuetype(mp), product_num_states(mp))

    return ProductWorkspace(underlying_workspace, intermediate_values)
end

"""
    construct_workspace(mp::IntervalMarkovProcess)

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
construct_workspace(mp::IntervalMarkovProcess) =
    construct_workspace(transition_prob(mp), max_actions(mp))

# Dense
struct DenseWorkspace{T <: Real}
    scratch::Vector{Int32}
    permutation::Vector{Int32}
    actions::Vector{T}
end

function DenseWorkspace(p::AbstractMatrix{T}, max_actions) where {T <: Real}
    n = size(p, 1)
    scratch = Vector{Int32}(undef, n)
    perm = Vector{Int32}(undef, n)
    actions = Vector{T}(undef, max_actions)
    return DenseWorkspace(scratch, perm, actions)
end

permutation(ws::DenseWorkspace) = ws.permutation
scratch(ws::DenseWorkspace) = ws.scratch

struct ThreadedDenseWorkspace{T <: Real}
    thread_workspaces::Vector{DenseWorkspace{T}}
end

function ThreadedDenseWorkspace(p::AbstractMatrix{T}, max_actions) where {T <: Real}
    n = size(p, 1)
    scratch = Vector{Int32}(undef, n)
    perm = Vector{Int32}(undef, n)

    workspaces = [
        DenseWorkspace(scratch, perm, Vector{T}(undef, max_actions)) for
        _ in 1:Threads.nthreads()
    ]
    return ThreadedDenseWorkspace(workspaces)
end

Base.getindex(ws::ThreadedDenseWorkspace, i) = ws.thread_workspaces[i]

## permutation and scratch space is shared across threads
permutation(ws::ThreadedDenseWorkspace) = permutation(first(ws.thread_workspaces))
scratch(ws::ThreadedDenseWorkspace) = scratch(first(ws.thread_workspaces))

"""
    construct_workspace(prob::IntervalProbabilities)

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
function construct_workspace(
    prob::IntervalProbabilities{R, VR, MR},
    max_actions = 1;
    threshold = 10,
) where {R, VR, MR <: AbstractMatrix{R}}
    if Threads.nthreads() == 1 || size(gap(prob), 2) <= threshold
        return DenseWorkspace(gap(prob), max_actions)
    else
        return ThreadedDenseWorkspace(gap(prob), max_actions)
    end
end

# Sparse
struct SparseWorkspace{T <: Real}
    scratch::Vector{Tuple{T, T}}
    values_gaps::Vector{Tuple{T, T}}
    actions::Vector{T}
end

function SparseWorkspace(p::AbstractSparseMatrix{T}, max_actions) where {T <: Real}
    max_nonzeros = maximum(map(nnz, eachcol(p)))
    scratch = Vector{Tuple{T, T}}(undef, max_nonzeros)
    values_gaps = Vector{Tuple{T, T}}(undef, max_nonzeros)
    actions = Vector{T}(undef, max_actions)
    return SparseWorkspace(scratch, values_gaps, actions)
end

scratch(ws::SparseWorkspace) = ws.scratch

struct ThreadedSparseWorkspace{T}
    thread_workspaces::Vector{SparseWorkspace{T}}
end

function ThreadedSparseWorkspace(p::AbstractSparseMatrix, max_actions)
    nthreads = Threads.nthreads()
    thread_workspaces = [SparseWorkspace(p, max_actions) for _ in 1:nthreads]
    return ThreadedSparseWorkspace(thread_workspaces)
end

Base.getindex(ws::ThreadedSparseWorkspace, i) = ws.thread_workspaces[i]

function construct_workspace(
    prob::IntervalProbabilities{R, VR, MR},
    max_actions = 1;
    threshold = 10,
) where {R, VR, MR <: AbstractSparseMatrix{R}}
    if Threads.nthreads() == 1 || size(gap(prob), 2) <= threshold
        return SparseWorkspace(gap(prob), max_actions)
    else
        return ThreadedSparseWorkspace(gap(prob), max_actions)
    end
end

## Orthogonal
abstract type SimpleOrthogonalWorkspace end

# Dense
struct DenseOrthogonalWorkspace{N, M, T <: Real} <: SimpleOrthogonalWorkspace
    expectation_cache::NTuple{N, Vector{T}}
    first_level_perm::Array{Int32, M}
    permutation::Vector{Int32}
    scratch::Vector{Int32}
    actions::Vector{T}
end

function DenseOrthogonalWorkspace(
    p::OrthogonalIntervalProbabilities{N, <:IntervalProbabilities{R}},
    max_actions,
) where {N, R}
    pns = num_target(p)
    nmax = maximum(pns)

    first_level_perm = Array{Int32}(undef, pns)
    perm = Vector{Int32}(undef, nmax)
    scratch = Vector{Int32}(undef, nmax)
    expectation_cache = NTuple{N - 1, Vector{R}}(Vector{R}(undef, n) for n in pns[2:end])
    actions = Vector{R}(undef, max_actions)
    return DenseOrthogonalWorkspace(
        expectation_cache,
        first_level_perm,
        perm,
        scratch,
        actions,
    )
end
permutation(ws::DenseOrthogonalWorkspace) = ws.permutation
scratch(ws::DenseOrthogonalWorkspace) = ws.scratch
first_level_perm(ws::DenseOrthogonalWorkspace) = ws.first_level_perm
actions(ws::DenseOrthogonalWorkspace) = ws.actions

struct ThreadedDenseOrthogonalWorkspace{N, M, T}
    thread_workspaces::Vector{DenseOrthogonalWorkspace{N, M, T}}
end

function ThreadedDenseOrthogonalWorkspace(
    p::OrthogonalIntervalProbabilities{N, <:IntervalProbabilities{R}},
    max_actions,
) where {N, R}
    nthreads = Threads.nthreads()
    pns = num_target(p)
    nmax = maximum(pns)

    first_level_perm = Array{Int32}(undef, pns)

    workspaces = map(1:nthreads) do _
        perm = Vector{Int32}(undef, nmax)
        scratch = Vector{Int32}(undef, nmax)
        expectation_cache =
            NTuple{N - 1, Vector{R}}(Vector{R}(undef, n) for n in pns[2:end])
        actions = Vector{R}(undef, max_actions)
        return DenseOrthogonalWorkspace(
            expectation_cache,
            first_level_perm,
            perm,
            scratch,
            actions,
        )
    end

    return ThreadedDenseOrthogonalWorkspace(workspaces)
end

Base.getindex(ws::ThreadedDenseOrthogonalWorkspace, i) = ws.thread_workspaces[i]

"""
    construct_workspace(prob::OrthogonalIntervalProbabilities)

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
function construct_workspace(
    p::OrthogonalIntervalProbabilities{N, <:IntervalProbabilities{R, VR, MR}},
    max_actions = 1,
) where {N, R, VR, MR <: AbstractMatrix{R}}
    if Threads.nthreads() == 1
        return DenseOrthogonalWorkspace(p, max_actions)
    else
        return ThreadedDenseOrthogonalWorkspace(p, max_actions)
    end
end

# Sparse
struct SparseOrthogonalWorkspace{N, T <: Real} <: SimpleOrthogonalWorkspace
    expectation_cache::NTuple{N, Vector{T}}
    values_gaps::Vector{Tuple{T, T}}
    scratch::Vector{Tuple{T, T}}
    actions::Vector{T}
end
scratch(ws::SparseOrthogonalWorkspace) = ws.scratch
actions(ws::SparseOrthogonalWorkspace) = ws.actions

function SparseOrthogonalWorkspace(
    p::OrthogonalIntervalProbabilities{N, <:IntervalProbabilities{R, VR, MR}},
    max_actions,
) where {N, R, VR, MR <: AbstractSparseMatrix{R}}
    max_nonzeros_per_prob = [maximum(map(nnz, eachcol(gap(pᵢ)))) for pᵢ in p]
    max_nonzeros = maximum(max_nonzeros_per_prob)

    scratch = Vector{Tuple{R, R}}(undef, max_nonzeros)
    values_gaps = Vector{Tuple{R, R}}(undef, max_nonzeros)
    expectation_cache =
        NTuple{N - 1, Vector{R}}(Vector{R}(undef, n) for n in max_nonzeros_per_prob[2:end])
    actions = Vector{R}(undef, max_actions)

    return SparseOrthogonalWorkspace(expectation_cache, values_gaps, scratch, actions)
end

struct ThreadedSparseOrthogonalWorkspace{N, T}
    thread_workspaces::Vector{SparseOrthogonalWorkspace{N, T}}
end

function ThreadedSparseOrthogonalWorkspace(p::OrthogonalIntervalProbabilities, max_actions)
    nthreads = Threads.nthreads()
    thread_workspaces = [SparseOrthogonalWorkspace(p, max_actions) for _ in 1:nthreads]

    return ThreadedSparseOrthogonalWorkspace(thread_workspaces)
end

Base.getindex(ws::ThreadedSparseOrthogonalWorkspace, i) = ws.thread_workspaces[i]

"""
    construct_workspace(prob::OrthogonalIntervalProbabilities)

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
function construct_workspace(
    p::OrthogonalIntervalProbabilities{N, <:IntervalProbabilities{R, VR, MR}},
    max_actions = 1,
) where {N, R, VR, MR <: AbstractSparseMatrix{R}}
    if Threads.nthreads() == 1
        return SparseOrthogonalWorkspace(p, max_actions)
    else
        return ThreadedSparseOrthogonalWorkspace(p, max_actions)
    end
end

## Mixture

# Sequential
struct MixtureWorkspace{W <: SimpleOrthogonalWorkspace, R}
    orthogonal_workspace::W
    mixture_cache::Vector{R}
    scratch::Vector{Int32}
    permutation::Vector{Int32}
end
permutation(ws::MixtureWorkspace) = ws.permutation
scratch(ws::MixtureWorkspace) = ws.scratch
actions(ws::MixtureWorkspace) = actions(ws.orthogonal_workspace)

function MixtureWorkspace(
    p::MixtureIntervalProbabilities{
        N,
        <:OrthogonalIntervalProbabilities{M, <:IntervalProbabilities{R, VR, MR}},
    },
    max_actions,
) where {N, M, R, VR, MR <: AbstractMatrix{R}}
    mixture_cache = Vector{R}(undef, N)
    scratch = Vector{Int32}(undef, N)
    permutation = Vector{Int32}(undef, N)
    return MixtureWorkspace(
        DenseOrthogonalWorkspace(first(p), max_actions),
        mixture_cache,
        scratch,
        permutation,
    )
end

function MixtureWorkspace(
    p::MixtureIntervalProbabilities{
        N,
        <:OrthogonalIntervalProbabilities{M, <:IntervalProbabilities{R, VR, MR}},
    },
    max_actions,
) where {N, M, R, VR, MR <: AbstractSparseMatrix{R}}
    mixture_cache = Vector{R}(undef, N)
    mixture_scratch = Vector{Int32}(undef, N)
    mixture_permutation = Vector{Int32}(undef, N)

    max_nonzeros_per_prob = map(1:M) do l
        return maximum(1:N) do k
            return maximum(map(nnz, eachcol(gap(p[k], l))))
        end
    end
    max_nonzeros = maximum(max_nonzeros_per_prob)

    scratch = Vector{Tuple{R, R}}(undef, max_nonzeros)
    values_gaps = Vector{Tuple{R, R}}(undef, max_nonzeros)
    expectation_cache =
        NTuple{M - 1, Vector{R}}(Vector{R}(undef, n) for n in max_nonzeros_per_prob[2:end])
    actions = Vector{R}(undef, max_actions)

    return MixtureWorkspace(
        SparseOrthogonalWorkspace(expectation_cache, values_gaps, scratch, actions),
        mixture_cache,
        mixture_scratch,
        mixture_permutation,
    )
end

# Threaded
struct ThreadedMixtureWorkspace{W <: SimpleOrthogonalWorkspace, V <: MixtureWorkspace{W}}
    thread_workspaces::Vector{V}
end

function ThreadedMixtureWorkspace(
    p::MixtureIntervalProbabilities{
        N,
        <:OrthogonalIntervalProbabilities{M, <:IntervalProbabilities{R, VR, MR}},
    },
    max_actions,
) where {N, M, R, VR, MR <: AbstractMatrix{R}}
    nthreads = Threads.nthreads()
    pns = num_target(p)
    nmax = maximum(pns)

    first_level_perm = Array{Int32}(undef, pns)

    workspaces = map(1:nthreads) do _
        mixture_cache = Vector{R}(undef, N)
        mixture_scratch = Vector{Int32}(undef, N)
        mixture_permutation = Vector{Int32}(undef, N)

        perm = Vector{Int32}(undef, nmax)
        scratch = Vector{Int32}(undef, nmax)
        expectation_cache =
            NTuple{M - 1, Vector{R}}(Vector{R}(undef, n) for n in pns[2:end])
        actions = Vector{R}(undef, max_actions)
        return MixtureWorkspace(
            DenseOrthogonalWorkspace(
                expectation_cache,
                first_level_perm,
                perm,
                scratch,
                actions,
            ),
            mixture_cache,
            mixture_scratch,
            mixture_permutation,
        )
    end

    return ThreadedMixtureWorkspace(workspaces)
end

function ThreadedMixtureWorkspace(
    p::MixtureIntervalProbabilities{
        N,
        <:OrthogonalIntervalProbabilities{M, <:IntervalProbabilities{R, VR, MR}},
    },
    max_actions,
) where {N, M, R, VR, MR <: AbstractSparseMatrix{R}}
    nthreads = Threads.nthreads()

    max_nonzeros_per_prob = map(1:M) do l
        return maximum(1:N) do k
            return maximum(map(nnz, eachcol(gap(p[k], l))))
        end
    end
    max_nonzeros = maximum(max_nonzeros_per_prob)

    workspaces = map(1:nthreads) do _
        mixture_cache = Vector{R}(undef, N)
        mixture_scratch = Vector{Int32}(undef, N)
        mixture_permutation = Vector{Int32}(undef, N)

        scratch = Vector{Tuple{R, R}}(undef, max_nonzeros)
        values_gaps = Vector{Tuple{R, R}}(undef, max_nonzeros)
        expectation_cache = NTuple{M - 1, Vector{R}}(
            Vector{R}(undef, n) for n in max_nonzeros_per_prob[2:end]
        )
        actions = Vector{R}(undef, max_actions)
        return MixtureWorkspace(
            SparseOrthogonalWorkspace(expectation_cache, values_gaps, scratch, actions),
            mixture_cache,
            mixture_scratch,
            mixture_permutation,
        )
    end

    return ThreadedMixtureWorkspace(workspaces)
end

Base.getindex(ws::ThreadedMixtureWorkspace, i) = ws.thread_workspaces[i]

"""
    construct_workspace(prob::MixtureIntervalProbabilities)

Construct a workspace for computing the Bellman update, given a value function.
If the Bellman update is used in a hot-loop, it is more efficient to use this function
to preallocate the workspace and reuse across iterations.

The workspace type is determined by the type and size of the transition probability matrix,
as well as the number of threads available.
"""
function construct_workspace(
    p::MixtureIntervalProbabilities{
        N,
        <:OrthogonalIntervalProbabilities{M, <:IntervalProbabilities{R, VR, MR}},
    },
    max_actions = 1,
) where {N, M, R, VR, MR <: Union{AbstractMatrix{R}, AbstractSparseMatrix{R}}}
    if Threads.nthreads() == 1
        return MixtureWorkspace(p, max_actions)
    else
        return ThreadedMixtureWorkspace(p, max_actions)
    end
end
