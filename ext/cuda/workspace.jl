abstract type AbstractCuWorkspace end

###################
# Dense workspace #
###################
struct CuDenseWorkspace <: AbstractCuWorkspace
    max_actions::Int32
end

IntervalMDP.construct_workspace(::AbstractGPUMatrix, max_actions) =
    CuDenseWorkspace(max_actions)

####################
# Sparse workspace #
####################
struct CuSparseWorkspace <: AbstractCuWorkspace
    max_nonzeros::Int32
    max_actions::Int32
end

function CuSparseWorkspace(p::AbstractCuSparseMatrix, max_actions)
    max_nonzeros = maximum(nnz, eachcol(p))
    return CuSparseWorkspace(max_nonzeros, max_actions)
end

IntervalMDP.construct_workspace(p::AbstractCuSparseMatrix, max_actions) =
    CuSparseWorkspace(p, max_actions)
