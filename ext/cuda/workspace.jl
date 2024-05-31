abstract type AbstractCuWorkspace end

###################
# Dense workspace #
###################
struct CuDenseWorkspace <: AbstractCuWorkspace end

IntervalMDP.construct_workspace(p::AbstractGPUMatrix) = CuDenseWorkspace()

####################
# Sparse workspace #
####################
struct CuSparseWorkspace <: AbstractCuWorkspace 
    max_nonzeros::Int32
end

function CuSparseWorkspace(p::AbstractCuSparseMatrix)
    max_nonzeros = maximum(nnz, eachcol(p))
    return CuSparseWorkspace(max_nonzeros)
end

IntervalMDP.construct_workspace(p::AbstractCuSparseMatrix) = CuSparseWorkspace(p)
