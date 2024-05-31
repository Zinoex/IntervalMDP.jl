construct_workspace(prob::IntervalProbabilities) = construct_workspace(gap(prob))

###################
# Dense workspace #
###################
abstract type AbstractDenseWorkspace end

struct DenseWorkspace <: AbstractDenseWorkspace
    permutation::Vector{Int32}
end

function DenseWorkspace(p::AbstractMatrix)
    n = size(p, 1)
    return DenseWorkspace(collect(UnitRange{Int32}(1, n)))
end

struct ThreadedDenseWorkspace <: AbstractDenseWorkspace
    permutation::Vector{Int32}
end

function ThreadedDenseWorkspace(p::AbstractMatrix)
    n = size(p, 1)
    return ThreadedDenseWorkspace(collect(UnitRange{Int32}(1, n)))
end

function construct_workspace(p::AbstractMatrix; threshold = 10)
    if Threads.nthreads() == 1 || size(p, 2) <= threshold
        return DenseWorkspace(p)
    else
        return ThreadedDenseWorkspace(p)
    end
end

####################
# Sparse workspace #
####################
struct SparseWorkspace{Tv}
    values_gaps::Vector{Tuple{Tv, Tv}}
end

function SparseWorkspace(p::AbstractSparseMatrix{Tv}) where {Tv}
    max_nonzeros = maximum(map(nnz, eachcol(p)))
    values_gaps = Vector{Tuple{Tv, Tv}}(undef, max_nonzeros)
    return SparseWorkspace(values_gaps)
end

struct ThreadedSparseWorkspace{Tv}
    thread_workspaces::Vector{SparseWorkspace{Tv}}
end

function ThreadedSparseWorkspace(p::AbstractSparseMatrix)
    nthreads = Threads.nthreads()
    thread_workspaces = [SparseWorkspace(p) for _ in 1:nthreads]
    return ThreadedSparseWorkspace(thread_workspaces)
end

function construct_workspace(p::AbstractSparseMatrix; threshold = 10)
    if Threads.nthreads() == 1 || size(p, 2) <= threshold
        return SparseWorkspace(p)
    else
        return ThreadedSparseWorkspace(p)
    end
end